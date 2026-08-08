from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, StreamEvent
from deep_sea_explorer.services.key_frame_detection import SurveyEventEvaluation


class FakeVisionGateway:
    def health(self) -> ModelHealth:
        return ModelHealth(True, "fake")

    def describe_video(self, video_path: Path) -> str:
        return "固定场景摘要"

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        return CaptureDecision(False, False, CaptureType.ENV, "")

    def answer(self, video_path: Path, question: str) -> Iterator[StreamEvent]:
        yield StreamEvent(StreamEventType.CHUNK, text="固定回答")
        yield StreamEvent(StreamEventType.FINAL, text="固定回答")

    def summarize_report(self, material: dict[str, object]) -> str:
        return "固定任务摘要。"

    def evaluate_survey_event(self, reference_image, current_image, metadata) -> SurveyEventEvaluation:
        changes = metadata.get("yolo_changes") or []
        return SurveyEventEvaluation(
            survey_value=bool(changes or metadata.get("scene_change_metrics", {}).get("changed")),
            event_type="new_element" if changes else "major_scene_change",
            scene_changed=bool(metadata.get("scene_change_metrics", {}).get("changed")),
            new_elements=tuple({"category": item.get("category", "other"), "name": item.get("category", "element"), "is_new": True} for item in changes),
            description="检测到关键画面变化。",
            confidence=0.9,
        )


class FakeImageGateway:
    def generate(self, prompt: str) -> bytes:
        return b"\xff\xd8\xff\xd9"


class FakeEmbeddingGateway:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]
