"""Domain gateways backed by the shared in-process local model runtime."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, MonitoringCoordinates, MonitoringTagMatch, StreamEvent
from deep_sea_explorer.services.key_frame_detection import SurveyEventEvaluation

from .adapters import EmbeddingAdapter, ImageAdapter, QwenAdapter
from .runtime import LocalModelRuntime


class LocalVisionGateway:
    def __init__(self, runtime: LocalModelRuntime, adapter: QwenAdapter) -> None:
        self.runtime = runtime
        self.adapter = adapter

    def health(self) -> ModelHealth:
        return self.runtime.health(self.adapter)

    def describe_video(self, video_path: Path) -> str:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.describe_video(video_path))

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.evaluate_frame(image_path))

    def extract_monitoring_coordinates(self, image_path: Path) -> MonitoringCoordinates | None:
        return self.runtime.invoke(
            self.adapter, lambda: self.adapter.extract_monitoring_coordinates(image_path)
        )

    def select_monitoring_labels(
        self, image_path: Path, candidates: tuple[str, ...], *, stage: str, maximum: int
    ) -> tuple[str, ...]:
        return self.runtime.invoke(
            self.adapter,
            lambda: self.adapter.select_monitoring_labels(image_path, candidates, stage=stage, maximum=maximum),
        )

    def match_monitoring_tags(self, image_path: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        return self.runtime.invoke(
            self.adapter, lambda: self.adapter.match_monitoring_tags(image_path, candidates)
        )

    def describe_monitoring_frame(self, image_path: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        return self.runtime.invoke(
            self.adapter, lambda: self.adapter.describe_monitoring_frame(image_path, tags, descriptions)
        )

    def evaluate_survey_event(self, reference_image: Path | None, current_image: Path, metadata: dict[str, object]) -> SurveyEventEvaluation:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.evaluate_survey_event(reference_image, current_image, metadata))

    def answer(self, question: str) -> Iterator[StreamEvent]:
        for text in self.runtime.stream(self.adapter, lambda: self.adapter.answer_stream(question)):
            yield StreamEvent(StreamEventType.CHUNK, text=text)
        yield StreamEvent(StreamEventType.FINAL)

    def summarize_report(self, material: dict[str, object]) -> str:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.summarize_report(material))


class LocalImageGateway:
    def __init__(self, runtime: LocalModelRuntime, adapter: ImageAdapter) -> None:
        self.runtime = runtime
        self.adapter = adapter

    def generate(self, prompt: str) -> bytes:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.generate(prompt))

    def health(self) -> ModelHealth:
        return self.runtime.health(self.adapter)


class DisabledImageGateway:
    """Reject image requests without loading an image model."""

    def generate(self, prompt: str) -> bytes:
        raise ModelUnavailableError("image generation is disabled")

    def health(self) -> ModelHealth:
        return ModelHealth(False, "disabled")


class LocalEmbeddingGateway:
    def __init__(self, runtime: LocalModelRuntime, adapter: EmbeddingAdapter) -> None:
        self.runtime = runtime
        self.adapter = adapter

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.embed(texts))

    def health(self) -> ModelHealth:
        return self.runtime.health(self.adapter)
