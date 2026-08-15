"""Offline-only adapters for the four models verified in S7."""

from __future__ import annotations

import io
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CaptureDecision, CountItem, ModelHealth, MonitoringAnalysis, MonitoringTagMatch
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

MONITORING_FRAME_PROMPT = """
你是深海影像科学解译员。独立分析这一张图，只输出一个 JSON 对象，不要 Markdown、推理过程或其他文字。
必须先根据图像中可直接观察到的证据，分别形成三类标签，然后再依据这些标签和图像生成客观描述。

严格使用此结构：
{
  "organisms": [{"name": "可见生物类群或形态名称", "count": 1}],
  "substrates": [{"name": "可见底质名称", "count": 1}],
  "geomorphologies": [{"name": "可见地貌名称", "count": 1}],
  "description": "一至三句简体中文科学解说式客观描述"
}

分类规则：
- organisms 只能包含可见生物，如鱼类、甲壳类、珊瑚、海绵、藻类、水螅珊瑚等；禁止放入底质或地貌。
- substrates 只能包含底面物质，如沙、泥、沙泥、岩石、砾石、未固结沉积物；禁止放入生物或地貌。
- geomorphologies 只能包含可辨认的地形结构，如平坦海床、坡地、沟槽、岩脊、凹地、岩壁；禁止放入生物或底质。
- 同一标签只能出现在一个数组；三个数组始终存在。未能从图像明确确认的类别必须输出空数组，不得猜测物种级名称。
- 每项 name 为简体中文短标签；count 为图中可辨认的数量。对底质和地貌可用 1 表示该特征可见。

描述规则：
- description 必须依据上述三个数组及图像证据，使用自然、连贯的科学解说式语言。
- 只陈述可观察的底质、生物类型、形态特征、数量及空间分布关系；不使用文学化或评价性措辞，不说明用途，不作图像证据之外的推断。
- 不要出现“可能”“推测”“似乎”“丰富”“优美”“壮观”等不确定或评价性表达；不要报告处理步骤、标签数组或 JSON 字段名。

语言风格参考：
“这片海底以柔软的沙泥底质为主，分布有附着性海绵、杯状海绵、大型石珊瑚、黑珊瑚和八放珊瑚。水螅珊瑚和丛生的丝状大型藻类也分布在底质表面，局部区域可见微藻覆盖。”
“柔软的沙泥底质上分布着石珊瑚和扇形黑色八放珊瑚，其中部分八放珊瑚具有白色脉纹。周围还分布有附着性红色钙质大型藻类、丛生的丝状藻类、直立海绵和水螅珊瑚，并可见少量鱼类活动。”
“该区域以岩石底质为主，其间夹杂未固结的沙泥斑块。底质上分布有黄色杯状管海绵、绿色块状海绵、结壳型钙质大型红藻、丛生的丝状大型藻类、黑色八放珊瑚和块状石珊瑚，同时可见直立海绵、壳状海绵以及微藻。”
""".strip()

CONSTRAINED_TAG_MATCH_PROMPT = """
你是深海影像标签匹配器。只依据当前图片证据，从下面给出的候选标签中精确匹配标签；不得创造、翻译、改写或输出候选表以外的标签。只输出一个 JSON 对象，不要 Markdown、推理过程或其他文字。
结构固定为：
{"organisms":[{"name":"候选原样标签","count":1}],"substrates":[{"name":"候选原样标签","count":1}],"geomorphologies":[{"name":"候选原样标签","count":1}],"unknown_categories":["bio"]}
organisms 只能使用生物候选；substrates 只能使用底质候选；geomorphologies 只能使用地貌候选。每个 name 必须逐字符等于候选项。没有明确可见证据时数组为空。若图中该类别确有可见证据、但本批候选没有匹配项，在 unknown_categories 中使用 bio、substrate 或 geomorphology；没有相应证据时绝不添加未知类别。
候选标签：
""".strip()

MONITORING_DESCRIPTION_PROMPT = """
你是深海影像科学解译员。根据当前图片和已经后端确认的标签及标签描述，生成一至三句简体中文科学调查式客观描述。标签描述仅是识别参考：不得照抄图片中不可见的特征，也不得补充标签之外的生物、底质或地貌。只陈述当前图片可观察到的底质与地貌、生物类型、形态、数量和空间分布关系；禁止文学化表达、评价、用途判断和证据不足的推断。只输出连续描述正文，不要标题、列表、JSON 或 Markdown。
已确认标签及知识库描述：
""".strip()

SURVEY_EVENT_PROMPT = """
你是一名深海调查事件审核员。输入图像按顺序给出：第一张是上一次已确认的场景参考图，
第二张是当前候选图。后续图像（如有）是带有真实标签的检索范例，只能辅助识别候选图中的
可见要素，绝不能直接复制其标签、物种名称、数量或调查结论。只比较前两张图中可见且具有
调查价值的变化；不要把它们当作视频帧序列。

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
除确有把握外不要臆测未清晰可见的要素。CATAMI 标签中的 Biota 必须输出为 organism，
Substrate 必须输出为 seabed_substrate，Bedforms、Relief、No bedforms、Flat、High 和
Low / moderate 必须输出为 micro_topography；绝不能在 JSON 中输出 biota、substrate、
bedforms、relief 或任何未列出的类别名称。

输出前必须逐项检查 new_elements：每一个 category 必须精确等于 organism、
seabed_substrate、micro_topography 或 other 之一。禁止输出 CATAMI 原始根类、中文类别名
（例如“生物”“底质”“微地貌”）或任何近义词；若不能映射到这四个精确值，则不要输出该要素。
如果没有调查价值，必须返回 survey_value=false、event_type="none"、new_elements=[]，并简要说明未确认变化。
""".strip()

SURVEY_EVENT_REPAIR_PROMPT = """
Re-emit the following survey-event model response as exactly one valid JSON object.
Do not add any observation, element, quantity, or conclusion that is not already present.
This is a syntax and schema repair only: preserve supported values where possible and remove
unsupported elements. Output no Markdown or explanation.

Use exactly this schema:
{
  "survey_value": false,
  "event_type": "none",
  "scene_changed": false,
  "new_elements": [],
  "description": "",
  "confidence": 0.0
}

event_type must be one of new_element, major_scene_change, none. Every new_elements item must
have category exactly organism, seabed_substrate, micro_topography, or other; name must be a
non-empty string; is_new must be true or false. If the supplied response cannot support an
accepted event, return survey_value=false, event_type="none", scene_changed=false, and an empty
new_elements array. Keep description concise and use a confidence from 0.0 to 1.0.

Response to repair:
""".strip()


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    image_path: Path
    similarity: float
    labels: dict[str, list[str]]
    survey_labels: dict[str, list[str]]


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

    def answer(self, question: str) -> str:
        if not question.strip():
            raise InvalidModelInput("question must not be empty")
        return self._generate([{"type": "text", "text": question.strip()}])

    def answer_stream(self, question: str) -> Iterator[str]:
        if not question.strip():
            raise InvalidModelInput("question must not be empty")
        _, model, processor = self.resource
        inputs = self._inputs(
            [{"type": "text", "text": question.strip()}],
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

    def describe_knowledge_base_label(self, image_path: Path, prompt: str, *, retry_sample: bool = False) -> str:
        """Describe one labelled exemplar with the caller-provided prompt verbatim."""
        if not image_path.is_file():
            raise InvalidModelInput("knowledge-base image does not exist")
        if not isinstance(prompt, str) or not prompt:
            raise InvalidModelInput("knowledge-base prompt is empty")
        return self._generate(
            [
                {"type": "image", "image": self._load_rgb_image(image_path)},
                {"type": "text", "text": prompt},
            ],
            max_new_tokens=512,
            direct_response=True,
            do_sample=retry_sample,
        ).strip()

    def match_monitoring_tags(self, image_path: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        if not image_path.is_file():
            raise InvalidModelInput("monitoring image does not exist")
        normalized = {
            field: [value for value in candidates.get(field, ()) if isinstance(value, str) and value]
            for field in ("organisms", "substrates", "geomorphologies")
        }
        raw = self._generate(
            [
                {"type": "image", "image": self._load_rgb_image(image_path)},
                {"type": "text", "text": CONSTRAINED_TAG_MATCH_PROMPT + "\n" + json.dumps(normalized, ensure_ascii=False)},
            ],
            max_new_tokens=256,
            direct_response=True,
        )
        return _monitoring_tag_match(raw)

    def describe_monitoring_frame(self, image_path: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        if not image_path.is_file():
            raise InvalidModelInput("monitoring image does not exist")
        reference = {
            "organisms": [item.name for item in tags.organisms],
            "substrates": [item.name for item in tags.substrates],
            "geomorphologies": [item.name for item in tags.geomorphologies],
            "label_descriptions": descriptions,
        }
        raw = self._generate(
            [{"type": "image", "image": self._load_rgb_image(image_path)}, {"type": "text", "text": MONITORING_DESCRIPTION_PROMPT + "\n" + json.dumps(reference, ensure_ascii=False)}],
            max_new_tokens=256,
            direct_response=True,
        )
        return _monitoring_description(raw, tags.organisms, tags.substrates, tags.geomorphologies)

    def evaluate_survey_event(self, reference_image: Path | None, current_image: Path, metadata: dict[str, object]) -> SurveyEventEvaluation:
        if not current_image.is_file():
            raise InvalidModelInput("candidate image does not exist")
        content: list[dict[str, object]] = []
        if reference_image is not None and reference_image.is_file():
            content.extend(
                [
                    {"type": "text", "text": "图像 1：上一次已确认的场景参考图。"},
                    {"type": "image", "image": self._load_rgb_image(reference_image)},
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "图像 2：当前候选图。"},
                {"type": "image", "image": self._load_rgb_image(current_image)},
            ]
        )
        for position, example in enumerate(_retrieval_examples(metadata), start=3):
            try:
                image = self._load_rgb_image(example.image_path, max_size=512)
            except (OSError, ValueError):
                continue
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"图像 {position}：检索到的相似标注范例，仅作辅助识别。"
                            "必须独立观察当前候选图，不能直接复制范例标签或推断结论。"
                            f"相似度：{example.similarity:.3f}。"
                            f"已核验的真实标签提示：{_retrieval_hint_text(example.survey_labels)}。"
                        ),
                    },
                    {"type": "image", "image": image},
                ]
            )
        prompt_metadata = {
            key: value for key, value in metadata.items() if key != "retrieval_context"
        }
        content.append(
            {
                "type": "text",
                "text": SURVEY_EVENT_PROMPT
                + "\n候选检测元数据（仅用于辅助比对）："
                + json.dumps(prompt_metadata, ensure_ascii=False, separators=(",", ":")),
            }
        )
        # Qwen3-VL otherwise spends the response budget on a hidden reasoning
        # trace. Survey decisions need a compact machine-readable response.
        raw = self._generate(content, max_new_tokens=384, direct_response=True)
        try:
            return _survey_event_evaluation(raw)
        except ModelOutputInvalid:
            # Reformat only failed structured output; normal event semantics still use the parser below.
            repaired = self._generate(
                [{"type": "text", "text": f"{SURVEY_EVENT_REPAIR_PROMPT}\n{raw}"}],
                max_new_tokens=384,
                direct_response=True,
            )
            try:
                return _survey_event_evaluation(repaired)
            except ModelOutputInvalid:
                # A malformed answer cannot establish a survey event. Returning
                # the existing conservative no-event result keeps the session
                # queue healthy without weakening the acceptance criteria.
                return SurveyEventEvaluation(False, "none", False, (), "未确认有效变化。", 0.0, ())

    @staticmethod
    def _load_rgb_image(image_path: Path, *, max_size: int | None = None) -> Any:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if max_size is not None:
                image.thumbnail((max_size, max_size))
            return image.copy()

    def _generate(
        self,
        content: list[dict[str, object]],
        *,
        max_new_tokens: int = 256,
        direct_response: bool = False,
        do_sample: bool = False,
    ) -> str:
        _, model, processor = self.resource
        inputs = self._inputs(content, processor, direct_response=direct_response)
        generation_args: dict[str, object] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            # Low-temperature retry changes only decoding, leaving the supplied prompt untouched.
            generation_args.update(temperature=0.2, top_p=0.9)
        generated = model.generate(**inputs, **generation_args)
        input_length = inputs["input_ids"].shape[1]
        text = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
        if not text:
            raise ModelOutputInvalid("local qwen returned an empty response")
        return text

    @staticmethod
    def _inputs(
        content: list[dict[str, object]],
        processor: Any,
        *,
        direct_response: bool = False,
    ) -> Any:
        messages = [{"role": "user", "content": content}]
        template_args: dict[str, object] = {"tokenize": False, "add_generation_prompt": True}
        if direct_response:
            template_args["enable_thinking"] = False
        try:
            prompt = processor.apply_chat_template(messages, **template_args)
        except TypeError:
            # Keep test doubles and older processors usable. Production Qwen3-VL
            # supports ``enable_thinking`` and therefore uses direct JSON mode.
            template_args.pop("enable_thinking", None)
            prompt = processor.apply_chat_template(messages, **template_args)
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


def _retrieval_examples(metadata: dict[str, object]) -> list[RetrievalExample]:
    """Accept only the normalized retrieval context emitted by MonitoringService."""

    values = metadata.get("retrieval_context")
    if not isinstance(values, list):
        return []
    examples: list[RetrievalExample] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        image_path = value.get("image_path")
        similarity = value.get("similarity")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            continue
        labels = _retrieval_label_mapping(value.get("labels"))
        survey_labels = _retrieval_label_mapping(value.get("survey_labels"))
        if not labels:
            continue
        examples.append(
            RetrievalExample(Path(image_path), float(similarity), labels, survey_labels)
        )
    return examples


def _retrieval_label_mapping(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for category, labels in value.items():
        if not isinstance(category, str) or not isinstance(labels, list):
            continue
        normalized = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
        if normalized:
            result[category] = normalized
    return result


def _retrieval_hint_text(labels: dict[str, list[str]]) -> str:
    """Keep multi-image prompts compact and avoid CATAMI root names in JSON fields."""

    values: list[str] = []
    for category, paths in labels.items():
        leaves: list[str] = []
        for path in paths:
            leaf = path.rsplit(" > ", 1)[-1]
            if leaf not in leaves:
                leaves.append(leaf)
            if len(leaves) == 3:
                break
        if leaves:
            values.append(f"{category}: {', '.join(leaves)}")
    return "；".join(values) or "无"


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


def _monitoring_tag_match(raw: str) -> MonitoringTagMatch:
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("not JSON")
        payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("not object")
        unknown = payload.get("unknown_categories", [])
        if not isinstance(unknown, list) or any(item not in {"bio", "substrate", "geomorphology"} for item in unknown):
            raise ValueError("unknown categories invalid")
        return MonitoringTagMatch(
            _tag_items(payload.get("organisms", [])),
            _tag_items(payload.get("substrates", [])),
            _tag_items(payload.get("geomorphologies", [])),
            tuple(dict.fromkeys(unknown)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelOutputInvalid("monitoring tag match must be valid JSON") from error


def _tag_items(values: object) -> tuple[CountItem, ...]:
    seen: set[str] = set()
    result: list[CountItem] = []
    for item in _count_items(values):
        name = " ".join(item.name.split()).strip("，,；;。.")
        key = name.casefold()
        if name and key not in seen and 0 < item.count <= 1_000_000:
            seen.add(key)
            result.append(CountItem(name, item.count))
    return tuple(result)


def _monitoring_analysis(raw: str) -> MonitoringAnalysis:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ModelOutputInvalid("monitoring analysis is not JSON")
    try:
        value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            raise TypeError("monitoring analysis must be an object")
        description = value.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("monitoring description is invalid")
        raw_organisms = _count_items(value.get("organisms"))
        raw_substrates = _count_items(value.get("substrates"))
        raw_geomorphologies = _count_items(value.get("geomorphologies"))
        _ensure_distinct_monitoring_tags(raw_organisms, raw_substrates, raw_geomorphologies)
        organisms = _monitoring_items(raw_organisms, "organisms")
        substrates = _monitoring_items(raw_substrates, "substrates")
        geomorphologies = _monitoring_items(raw_geomorphologies, "geomorphologies")
        return MonitoringAnalysis(
            _monitoring_description(description, organisms, substrates, geomorphologies),
            organisms,
            (),
            substrates,
            geomorphologies,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelOutputInvalid("monitoring analysis is invalid") from error


_PROHIBITED_MONITORING_DESCRIPTION_PHRASES = (
    "可能",
    "推测",
    "似乎",
    "估计",
    "丰富",
    "优美",
    "场景壮观",
    "壮观",
    "整洁",
    "有序",
    "适合",
    "用于",
)


def _monitoring_description(
    raw_description: str,
    organisms: tuple[CountItem, ...],
    substrates: tuple[CountItem, ...],
    geomorphologies: tuple[CountItem, ...],
) -> str:
    """Keep the public memo factual even when a model adds evaluative wording."""
    if not organisms and not substrates and not geomorphologies:
        return "画面中未确认可归类的生物、底质或地貌特征。"
    description = " ".join(raw_description.split())
    for phrase in _PROHIBITED_MONITORING_DESCRIPTION_PHRASES:
        description = description.replace(phrase, "")
    description = re.sub(r"[，,]\s*[。.]", "。", description)
    description = re.sub(r"\s+", "", description).strip("，,；;。.")
    return description + "。" if description else "画面中可见已确认的生物、底质或地貌特征。"


_MONITORING_TAG_RULES = {
    "organisms": {
        "required": (),
        "forbidden": ("沙", "泥", "砾", "底质", "海床", "坡", "沟", "岩脊", "凹地", "地貌"),
    },
    "substrates": {
        "required": ("沙", "泥", "砾", "岩", "沉积", "底质"),
        "forbidden": ("鱼", "虾", "蟹", "珊瑚", "海绵", "藻", "水螅"),
    },
    "geomorphologies": {
        "required": ("海床", "坡", "沟", "脊", "凹", "岩壁", "地形", "地貌", "平坦"),
        "forbidden": ("鱼", "虾", "蟹", "珊瑚", "海绵", "藻", "水螅", "沙泥"),
    },
}


def _monitoring_items(values: object | tuple[CountItem, ...], category: str) -> tuple[CountItem, ...]:
    """Normalize category-local Qwen tags and discard labels from another category."""
    items = values if isinstance(values, tuple) else _count_items(values)
    rules = _MONITORING_TAG_RULES[category]
    seen: set[str] = set()
    cleaned: list[CountItem] = []
    for item in items:
        name = " ".join(item.name.split()).strip("，,；;。.")
        key = name.casefold()
        has_required = not rules["required"] or any(token in name for token in rules["required"])
        has_forbidden = any(token in name for token in rules["forbidden"])
        if not name or key in seen or not has_required or has_forbidden or item.count < 1 or item.count > 1_000_000:
            continue
        seen.add(key)
        cleaned.append(CountItem(name, item.count))
    return tuple(cleaned)


def _ensure_distinct_monitoring_tags(*groups: tuple[CountItem, ...]) -> None:
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = item.name.casefold()
            if key in seen:
                raise ValueError("monitoring label appears in multiple categories")
            seen.add(key)


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
        "biota": "organism",
        "no_visible_biota": "organism",
        "生物": "organism",
        "seabed_substrate": "seabed_substrate",
        "substrate": "seabed_substrate",
        "seabed": "seabed_substrate",
        "底质": "seabed_substrate",
        "沉积物": "seabed_substrate",
        "micro_topography": "micro_topography",
        "topography": "micro_topography",
        "bedforms": "micro_topography",
        "no_bedforms": "micro_topography",
        "relief": "micro_topography",
        "bioturbation": "micro_topography",
        "flat": "micro_topography",
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
        name: object
        count: object
        if isinstance(item, str):
            name, count = item.strip(), 1
        elif isinstance(item, dict):
            name = item.get("name")
            count = item.get("count", 1)
            if not isinstance(name, str):
                raise ValueError("count item name must be a string")
        else:
            raise ValueError("count item must be an object")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(count, bool) or not isinstance(count, (int, float, str)):
            raise ValueError("count item count must be an integer")
        try:
            result.append(CountItem(name, int(count)))
        except (TypeError, ValueError) as error:
            raise ValueError("count item count must be an integer") from error
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
