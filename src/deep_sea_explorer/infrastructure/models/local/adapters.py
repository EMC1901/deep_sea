"""Offline-only adapters for the four models verified in S7."""

from __future__ import annotations

import io
import json
import math
import re
import threading
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CaptureDecision, CountItem, ModelHealth
from deep_sea_explorer.domain.report_material import compact_report_material

from .errors import InvalidModelInput, ModelLoadFailure, ModelNotConfigured, ModelOutputInvalid


FRAME_DECISION_PROMPT = """
Classify this image and return exactly one JSON object, without Markdown or explanatory text.
Use this schema:
{
  "is_deepsea": true,
  "is_typical": true,
  "category": "bio",
  "description": "short description",
  "organisms": [{"name": "organism name", "count": 1}],
  "env_features": [{"name": "environment feature", "count": 1}]
}
category must be "bio" or "env". organisms and env_features must always be JSON arrays.
Every array item must be an object containing a non-empty string name and an integer count.
Use an empty array when no item is present.
""".strip()

VIDEO_DESCRIPTION_PROMPT = """
请用简体中文简要描述这段深海视频中的当前场景，只输出一至三句连续正文。
描述应优先说明可见生物、底质或环境特征及明显变化，不要使用标题、列表或 Markdown。
输出必须以中文为主；若物种、设备、地质构造等专有名词没有通用中文译名，可以保留其原文，
但其余叙述必须使用简体中文。
""".strip()


class LocalAdapter:
    def __init__(self, name: str, model_path: str) -> None:
        self.name = name
        self.model_path = Path(model_path) if model_path else None
        self._resource: Any | None = None
        self._state = "not_loaded"
        self._lock = threading.Lock()

    def health(self) -> ModelHealth:
        return ModelHealth(self._state == "ready", self._state)

    def load(self) -> None:
        if self._resource is not None:
            return
        with self._lock:
            if self._resource is not None:
                return
            if self.model_path is None or not self.model_path.is_dir():
                self._state = "error"
                raise ModelNotConfigured(f"{self.name} model path is not configured")
            self._state = "loading"
            try:
                self._resource = self._load_resource()
            except Exception as error:
                self._state = "error"
                if isinstance(error, ModelNotConfigured):
                    raise
                raise ModelLoadFailure(f"failed to load local {self.name} model") from error
            self._state = "ready"

    def unload(self) -> None:
        with self._lock:
            self._resource = None
            if self._state == "ready":
                self._state = "not_loaded"

    def _load_resource(self) -> Any:
        raise NotImplementedError

    @property
    def resource(self) -> Any:
        if self._resource is None:
            raise ModelLoadFailure(f"local {self.name} model is not loaded")
        return self._resource


class QwenAdapter(LocalAdapter):
    def __init__(self, model_path: str) -> None:
        super().__init__("qwen", model_path)

    def _load_resource(self) -> tuple[Any, Any, Any]:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore[import-not-found]

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        model.eval()
        processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        return torch, model, processor

    def describe_video(self, video_path: Path) -> str:
        description = self._generate(
            [
                {"type": "video", "video": self._video_frames(video_path)},
                {"type": "text", "text": VIDEO_DESCRIPTION_PROMPT},
            ]
        )
        if _is_chinese_description(description):
            return description
        rewritten = self._generate(
            [
                {
                    "type": "text",
                    "text": (
                        "请将下面的深海场景描述改写为一至三句简体中文。"
                        "除没有通用中文译名的专有名词可保留原文外，其余内容必须使用中文。"
                        "只输出改写后的正文，不要解释、标题、列表或 Markdown。\n"
                        f"待改写内容：{description}"
                    ),
                }
            ]
        )
        if not _is_chinese_description(rewritten):
            raise ModelOutputInvalid("local qwen video description is not Chinese")
        return rewritten

    def answer(self, video_path: Path, question: str) -> str:
        if not question.strip():
            raise InvalidModelInput("question must not be empty")
        return self._generate(
            [{"type": "video", "video": self._video_frames(video_path)}, {"type": "text", "text": question.strip()}]
        )

    def answer_stream(self, video_path: Path, question: str) -> Iterator[str]:
        if not question.strip():
            raise InvalidModelInput("question must not be empty")
        _, model, processor = self.resource
        inputs = self._inputs(
            [{"type": "video", "video": self._video_frames(video_path)}, {"type": "text", "text": question.strip()}],
            processor,
        )
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                model.generate(**inputs, streamer=streamer, max_new_tokens=256, do_sample=False)
            except Exception as error:  # model errors must be re-raised in the request thread
                errors.append(error)
            finally:
                streamer.end()

        worker = threading.Thread(target=generate, name="qwen-local-answer", daemon=True)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()
        if errors:
            raise errors[0]

    def summarize_report(self, material: dict[str, object]) -> str:
        content = json.dumps(
            compact_report_material(material),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        prompt = (
            "你是深海探测任务分析专家。请根据下面的 JSON 材料撰写一段中文“智能任务总结”。"
            "无论材料使用何种语言，都必须使用简体中文输出。总结应覆盖生物发现、底质与环境特征、"
            "监测过程及异常、指挥官与系统的交互情况，并给出一至两项后续建议。"
            "只输出一段连续正文，不要标题、列表、编号、Markdown 或星号，控制在 300 至 500 个汉字。"
            f"\n任务材料：{content}"
        )
        return _clean_report_summary(
            self._generate([{"type": "text", "text": prompt}], max_new_tokens=512)
        )

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        if not image_path.is_file():
            raise InvalidModelInput("image file does not exist")
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        raw = self._generate(
            [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": FRAME_DECISION_PROMPT,
                },
            ]
        )
        return _capture_decision(raw)

    def _generate(
        self,
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
    ) -> str:
        _, model, processor = self.resource
        inputs = self._inputs(content, processor)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = inputs["input_ids"].shape[1]
        text = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
        if not text:
            raise ModelOutputInvalid("local qwen returned an empty response")
        return text

    @staticmethod
    def _inputs(content: list[dict[str, object]], processor: Any) -> Any:
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt", "padding": True}
        images = [item["image"] for item in content if item["type"] == "image"]
        videos = [item["video"] for item in content if item["type"] == "video"]
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        return processor(**kwargs).to("cuda")

    @staticmethod
    def _video_frames(video_path: Path) -> Any:
        if not video_path.is_file():
            raise InvalidModelInput("video file does not exist")
        import cv2  # type: ignore[import-not-found]
        import numpy as np

        capture = cv2.VideoCapture(str(video_path))
        frames: list[Any] = []
        try:
            total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            sample_count = min(total, 8)
            indexes = (
                {0}
                if sample_count == 1
                else {round(index * (total - 1) / (sample_count - 1)) for index in range(sample_count)}
            )
            position = 0
            while len(frames) < 8:
                ok, frame = capture.read()
                if not ok:
                    break
                if position in indexes:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                position += 1
        finally:
            capture.release()
        if not frames:
            raise InvalidModelInput("video has no readable frames")
        # The Qwen3-VL video processor needs at least two temporal frames.
        # Browser monitoring uploads one still frame per capture, so preserve
        # that contract by representing a still as two identical frames.
        if len(frames) == 1:
            frames.append(frames[0].copy())
        return np.stack(frames)


class ImageAdapter(LocalAdapter):
    def __init__(self, model_path: str) -> None:
        super().__init__("image", model_path)

    def _load_resource(self) -> Any:
        import torch  # type: ignore[import-not-found]
        from diffusers import StableDiffusionXLPipeline  # type: ignore[import-not-found]

        return StableDiffusionXLPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")

    def generate(self, prompt: str) -> bytes:
        if not prompt.strip() or len(prompt) > 1_000:
            raise InvalidModelInput("image prompt must contain 1 to 1000 characters")
        image = self.resource(
            prompt=prompt.strip(), height=512, width=512, num_inference_steps=20, guidance_scale=7.5
        ).images[0]
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        value = output.getvalue()
        if not value.startswith(b"\xff\xd8"):
            raise ModelOutputInvalid("local image model did not return a JPEG")
        return value


class EmbeddingAdapter(LocalAdapter):
    def __init__(self, name: str, model_path: str, dimension: int, trust_remote_code: bool) -> None:
        super().__init__(name, model_path)
        self.dimension = dimension
        self.trust_remote_code = trust_remote_code

    def _load_resource(self) -> Any:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        return SentenceTransformer(
            str(self.model_path),
            device="cuda",
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise InvalidModelInput("embedding texts must be non-empty strings")
        values = self.resource.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        if values.shape != (len(texts), self.dimension) or not math.isfinite(float(values.sum())):
            raise ModelOutputInvalid(f"local {self.name} embeddings are invalid")
        return [[float(value) for value in row] for row in values]


def _capture_decision(raw: str) -> CaptureDecision:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ModelOutputInvalid("local qwen frame decision is not JSON")
    try:
        value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            raise TypeError("frame decision must be an object")
        category = CaptureType(str(value["category"]).strip().lower())
        description = value["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("frame decision description must be a non-empty string")
        return CaptureDecision(
            _boolean(value["is_deepsea"], "is_deepsea"),
            _boolean(value["is_typical"], "is_typical"),
            category,
            description.strip(),
            _count_items(value.get("organisms")),
            _count_items(value.get("env_features")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelOutputInvalid("local qwen frame decision is invalid") from error


def _count_items(values: object) -> tuple[CountItem, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("count items must be a list")
    result: list[CountItem] = []
    for item in values:
        if isinstance(item, str):
            name, count = item.strip(), 1
        elif isinstance(item, dict):
            name = item.get("name")
            count = item.get("count", 1)
            if not isinstance(name, str):
                raise ValueError("count item name must be a string")
        else:
            raise ValueError("count item must be an object")
        if not name:
            continue
        if isinstance(count, bool):
            raise ValueError("count item count must be an integer")
        result.append(CountItem(name, int(count)))
    return tuple(result)


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} must be a boolean")


def _clean_report_summary(value: str) -> str:
    text = re.sub(r"[#*_`>]+", "", value)
    text = re.sub(r"^\s*(?:[-•]|\d+[.、])\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 500:
        text = text[:500].rstrip("，。；、:： ") + "。"
    return text


def _is_chinese_description(value: str) -> bool:
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", value))
    latin_count = len(re.findall(r"[A-Za-z]", value))
    language_count = chinese_count + latin_count
    return chinese_count >= 4 and (
        language_count == 0 or chinese_count / language_count >= 0.3
    )
