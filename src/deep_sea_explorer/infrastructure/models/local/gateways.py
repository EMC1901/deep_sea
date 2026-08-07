"""Domain gateways backed by the shared in-process local model runtime."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, StreamEvent

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

    def answer(self, video_path: Path, question: str) -> Iterator[StreamEvent]:
        for text in self.runtime.stream(self.adapter, lambda: self.adapter.answer_stream(video_path, question)):
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


class LocalEmbeddingGateway:
    def __init__(self, runtime: LocalModelRuntime, adapter: EmbeddingAdapter) -> None:
        self.runtime = runtime
        self.adapter = adapter

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.runtime.invoke(self.adapter, lambda: self.adapter.embed(texts))

    def health(self) -> ModelHealth:
        return self.runtime.health(self.adapter)
