"""Vision gateway backed by a localhost llama-server process.

The Flask application never loads GGUF tensors.  llama-server owns the model,
projector, CUDA allocations and lifecycle; this adapter only sends compatible
OpenAI-style multimodal messages and retains the existing validation helpers.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from deep_sea_explorer.config import Settings
from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, MonitoringTagMatch, StreamEvent
from deep_sea_explorer.domain.report_material import compact_report_material
from deep_sea_explorer.infrastructure.models.local.adapters import (
    CONSTRAINED_TAG_MATCH_PROMPT,
    FRAME_DECISION_PROMPT,
    MONITORING_DESCRIPTION_PROMPT,
    SURVEY_EVENT_PROMPT,
    VIDEO_DESCRIPTION_PROMPT,
    QwenAdapter,
    _capture_decision,
    _clean_report_summary,
    _is_chinese_description,
    _monitoring_description,
    _monitoring_tag_match,
    _survey_event_evaluation,
)
from deep_sea_explorer.services.key_frame_detection import SurveyEventEvaluation


class LlamaCppVisionGateway:
    """Translate the established vision gateway contract to llama-server."""

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.base_url = settings.gguf_server_url.rstrip("/")
        self.model_path = Path(settings.gguf_model_path)
        self.mmproj_path = Path(settings.gguf_mmproj_path)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                settings.gguf_read_timeout_seconds,
                connect=settings.gguf_connect_timeout_seconds,
            )
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> ModelHealth:
        if not self.model_path.is_file():
            return ModelHealth(False, "gguf text model file is unavailable", False, False)
        if not self.mmproj_path.is_file():
            return ModelHealth(False, "gguf vision projector file is unavailable", True, False)
        try:
            health = self._client.get(f"{self.base_url}/health")
            props = self._client.get(f"{self.base_url}/props")
            health.raise_for_status()
            props.raise_for_status()
            payload = props.json()
            modalities = payload.get("modalities", {}) if isinstance(payload, dict) else {}
            vision_ready = bool(isinstance(modalities, dict) and modalities.get("vision") is True)
            text_ready = bool(isinstance(health.json(), dict) and health.json().get("status") == "ok")
            return ModelHealth(
                text_ready and vision_ready,
                "gguf text=ready vision=ready" if vision_ready else "gguf text=ready vision=unavailable",
                text_ready,
                vision_ready,
            )
        except (httpx.HTTPError, ValueError):
            return ModelHealth(False, "llama-server unavailable", False, False)

    @staticmethod
    def _image_part(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise ModelUnavailableError("vision image file is unavailable")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}

    @staticmethod
    def _video_parts(video_path: Path) -> list[dict[str, object]]:
        """Submit a temporal first/last-frame pair within the configured vision context."""
        frames = QwenAdapter._video_frames(video_path)
        # Qwen3.6 images consume at least 1024 tokens each.  The deployment
        # deliberately uses a 4096-token context on a 24GB GPU, so forwarding
        # all eight legacy samples could overflow context before generation.
        # A first/last pair remains genuine multi-image temporal input.
        if len(frames) > 2:
            frames = frames[[0, -1]]
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise ModelUnavailableError("Pillow is unavailable for video frames") from error
        parts: list[dict[str, object]] = []
        for frame in frames:
            from io import BytesIO

            output = BytesIO()
            Image.fromarray(frame).save(output, format="JPEG", quality=90)
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return parts

    def _chat(self, content: Sequence[dict[str, object]], *, max_tokens: int = 256) -> str:
        payload = {
            "messages": [{"role": "user", "content": list(content)}],
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3.6's template emits an empty thinking block with this flag.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            body: Any = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            text = message.get("content") if isinstance(message, dict) else None
        except (httpx.HTTPError, ValueError, IndexError, TypeError) as error:
            raise ModelUnavailableError("llama-server multimodal request failed") from error
        if not isinstance(text, str) or not text.strip() or "<think>" in text:
            raise ModelUnavailableError("llama-server returned an invalid direct response")
        return text.strip()

    def describe_video(self, video_path: Path) -> str:
        text = self._chat([*self._video_parts(video_path), {"type": "text", "text": VIDEO_DESCRIPTION_PROMPT}])
        if _is_chinese_description(text):
            return text
        rewritten = self._chat(
            [{"type": "text", "text": "请将下面的深海场景描述改写为一至三句简体中文。只输出改写后的正文。\n待改写内容：" + text}]
        )
        if not _is_chinese_description(rewritten):
            raise ModelUnavailableError("llama-server video description is not Chinese")
        return rewritten

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        return _capture_decision(self._chat([self._image_part(image_path), {"type": "text", "text": FRAME_DECISION_PROMPT}]))

    def match_monitoring_tags(self, image_path: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        normalized = {field: [value for value in candidates.get(field, ()) if isinstance(value, str) and value] for field in ("organisms", "substrates", "geomorphologies")}
        raw = self._chat(
            [self._image_part(image_path), {"type": "text", "text": CONSTRAINED_TAG_MATCH_PROMPT + "\n" + json.dumps(normalized, ensure_ascii=False)}]
        )
        return _monitoring_tag_match(raw)

    def describe_monitoring_frame(self, image_path: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        reference = {
            "organisms": [item.name for item in tags.organisms],
            "substrates": [item.name for item in tags.substrates],
            "geomorphologies": [item.name for item in tags.geomorphologies],
            "label_descriptions": descriptions,
        }
        raw = self._chat(
            [self._image_part(image_path), {"type": "text", "text": MONITORING_DESCRIPTION_PROMPT + "\n" + json.dumps(reference, ensure_ascii=False)}]
        )
        return _monitoring_description(raw, tags.organisms, tags.substrates, tags.geomorphologies)

    def answer(self, question: str) -> Iterator[StreamEvent]:
        if not question.strip():
            raise ModelUnavailableError("question must not be empty")
        yield StreamEvent(StreamEventType.CHUNK, text=self._chat([{"type": "text", "text": question.strip()}]))
        yield StreamEvent(StreamEventType.FINAL)

    def summarize_report(self, material: dict[str, object]) -> str:
        compact = json.dumps(compact_report_material(material), ensure_ascii=False, separators=(",", ":"), default=str)
        prompt = "你是深海探测任务分析专家。请根据下面的 JSON 材料撰写一段简体中文智能任务总结。只输出连续正文，不要标题、列表、Markdown 或思考过程，控制在 300 至 500 个汉字。\n任务材料：" + compact
        return _clean_report_summary(self._chat([{"type": "text", "text": prompt}], max_tokens=512))

    def evaluate_survey_event(self, reference_image: Path | None, current_image: Path, metadata: dict[str, object]) -> SurveyEventEvaluation:
        content: list[dict[str, object]] = []
        if reference_image is not None and reference_image.is_file():
            content.extend([{"type": "text", "text": "图像 1：上一次已确认的场景参考图。"}, self._image_part(reference_image)])
        content.extend([{"type": "text", "text": "图像 2：当前候选图。"}, self._image_part(current_image)])
        content.append({"type": "text", "text": SURVEY_EVENT_PROMPT + "\n候选检测元数据（仅用于辅助比对）：" + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))})
        try:
            return _survey_event_evaluation(self._chat(content, max_tokens=384))
        except Exception:
            return SurveyEventEvaluation(False, "none", False, (), "未确认有效变化。", 0.0, ())
