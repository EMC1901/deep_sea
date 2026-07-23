from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, StreamEvent


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


class FakeImageGateway:
    def generate(self, prompt: str) -> bytes:
        return b"\xff\xd8\xff\xd9"


class FakeEmbeddingGateway:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]
