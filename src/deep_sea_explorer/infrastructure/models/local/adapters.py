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
from deep_sea_explorer.services.key_frame_detection import SurveyEventEvaluation
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

SURVEY_EVENT_PROMPT = """
你是一名深海调查事件审核员。输入图像按顺序给出：第一张是上一次已确认的场景参考图，
第二张是当前候选图。只比较两张图中可见且具有调查价值的变化；不要把它们当作视频帧序列。

已知目标检测和场景变化指标只是候选信号，不能据此臆测不可见的物种或地貌。忽略轻微镜头抖动、
整体亮度闪动、悬浮颗粒、压缩噪声以及同一目标的普通位移。只有明确出现新要素、数量/状态显著变化，
或底质、微地形发生重大可见变化时，才将 survey_value 设为 true。

description 是给值班调查人员阅读的当前场景描述，不是变化说明。使用“该场景包含……”或“场景显示……”
这样的自然科学调查句式，尽可能列出当前候选图中可辨认的生物类群、附生群落、底质和微地貌。
不要出现“与参考图相比”“上一次”“前后”“变化”“新出现”“候选图”“图像 1/2”等比较或处理过程措辞。

只输出一个 JSON 对象，不要 Markdown、代码块或任何额外文字，且所有 name 与 description 必须使用简体中文：
{
  "survey_value": true,
  "event_type": "new_element",
  "scene_changed": true,
  "new_elements": [{"category": "organism", "name": "名称", "is_new": true}],
  "description": "一至两句简体中文的当前场景生态描述。",
  "confidence": 0.0
}

event_type 只能是 new_element、major_scene_change、none。
new_elements 最多列出 5 项当前候选图中可辨认的代表性调查要素，同一种要素只能列一次；
category 只能是 organism、seabed_substrate、micro_topography、other。它们用于样本归档与统计，
除确有把握外不要臆测未清晰可见的要素。
如果没有调查价值，必须返回 survey_value=false、event_type="none"、new_elements=[]，并简要说明未确认变化。
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
    def __init__(self, model_path: str, adapter_path: str = "") -> None:
        super().__init__("qwen", model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None

    def _load_resource(self) -> tuple[Any, Any, Any]:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore[import-not-found]

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path), dtype=torch.bfloat16, local_files_only=True
        )
        processor_path = self.model_path
        if self.adapter_path is not None:
            if not self.adapter_path.is_dir():
                raise ModelNotConfigured("qwen adapter path is not configured")
            try:
                from peft import PeftModel  # type: ignore[import-not-found]
            except ImportError as error:
                raise ModelNotConfigured("qwen adapter requires the peft package") from error
            model = PeftModel.from_pretrained(
                model, str(self.adapter_path), is_trainable=False, local_files_only=True
            )
            processor_path = self.adapter_path
        model = model.to("cuda")
        model.eval()
        processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True)
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

    def evaluate_survey_event(self, reference_image: Path | None, current_image: Path, metadata: dict[str, object]) -> SurveyEventEvaluation:
        if not current_image.is_file():
            raise InvalidModelInput("candidate image does not exist")
        content: list[dict[str, object]] = []
        if reference_image is not None and reference_image.is_file():
            from PIL import Image  # type: ignore[import-not-found]
            with Image.open(reference_image) as source:
                content.extend(
                    [
                        {"type": "text", "text": "图像 1：上一次已确认的场景参考图。"},
                        {"type": "image", "image": source.convert("RGB")},
                    ]
                )
        from PIL import Image  # type: ignore[import-not-found]
        with Image.open(current_image) as source:
            content.extend(
                [
                    {"type": "text", "text": "图像 2：当前候选图。"},
                    {"type": "image", "image": source.convert("RGB")},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": SURVEY_EVENT_PROMPT
                + "\n候选检测元数据（仅用于辅助比对）："
                + json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            }
        )
        raw = self._generate(content, max_new_tokens=256)
        return _survey_event_evaluation(raw)

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


def _survey_event_evaluation(raw: str) -> SurveyEventEvaluation:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ModelOutputInvalid("survey event response is not JSON")
    try:
        value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("survey event must be an object")
        survey_value = _boolean(value["survey_value"], "survey_value")
        # The second-round LoRA was trained without this redundant field. A
        # valuable event necessarily represents a confirmed visible change.
        scene_changed = _boolean(value.get("scene_changed", survey_value), "scene_changed")
        event_type = str(value["event_type"])
        if event_type not in {"new_element", "major_scene_change", "none"}:
            raise ValueError("survey event type is invalid")
        normalized = _survey_elements(
            value.get("new_elements", []), "new element", include_new_flag=True, strict=True
        )
        # The selected LoRA can emit `none` alongside verified new elements.
        # The structured evidence is sufficient to resolve only this contradiction.
        if survey_value and event_type == "none" and normalized:
            event_type = "new_element"
        observed = _survey_elements(
            value.get("observed_elements", normalized),
            "observed element",
            include_new_flag=False,
            strict=False,
        )
        description = value.get("description")
        confidence = value.get("confidence")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("survey event description is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("survey event confidence is invalid")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("survey event confidence is outside range")
        if survey_value and (event_type == "none" or not (normalized or scene_changed)):
            raise ValueError("accepted survey event has no visible change")
        if not survey_value and event_type != "none":
            raise ValueError("rejected survey event must have type none")
        return SurveyEventEvaluation(
            survey_value,
            event_type,
            scene_changed,
            tuple(normalized),
            description.strip(),
            float(confidence),
            tuple(observed),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelOutputInvalid(f"survey event response is invalid: {error}") from error


def _survey_elements(
    values: object,
    label: str,
    *,
    include_new_flag: bool,
    strict: bool,
) -> list[dict[str, object]]:
    if not isinstance(values, list):
        raise ValueError(f"{label}s must be a list")
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for element in values:
        if not isinstance(element, dict):
            if strict:
                raise ValueError(f"{label} must be an object")
            continue
        category, name = element.get("category"), element.get("name")
        normalized_category = _survey_category(category)
        if normalized_category is None:
            if strict:
                raise ValueError(f"{label} category is invalid")
            normalized_category = "other"
        if not isinstance(name, str) or not name.strip():
            if strict:
                raise ValueError(f"{label} name is invalid")
            continue
        key = (normalized_category, name.strip())
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, object] = {"category": normalized_category, "name": key[1]}
        if include_new_flag:
            item["is_new"] = _boolean(element.get("is_new", True), "is_new")
        normalized.append(item)
    return normalized


def _survey_category(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "organism": "organism",
        "bio": "organism",
        "biological": "organism",
        "生物": "organism",
        "seabed_substrate": "seabed_substrate",
        "substrate": "seabed_substrate",
        "seabed": "seabed_substrate",
        "底质": "seabed_substrate",
        "沉积物": "seabed_substrate",
        "micro_topography": "micro_topography",
        "topography": "micro_topography",
        "地形": "micro_topography",
        "微地貌": "micro_topography",
        "other": "other",
        "environment": "other",
        "环境": "other",
        "其他": "other",
    }
    return mapping.get(normalized)


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
