"""服务器 local 模式的延迟适配器；开发机导入此模块不加载 GPU 库。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, ModelHealth, StreamEvent


class LocalVisionGateway:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore[import-not-found]
        except ImportError as error:
            raise ModelUnavailableError("local model dependencies are not installed") from error
        self._torch, self._processor = (
            torch,
            AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True),
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto", trust_remote_code=True
        )
        self._loaded = True

    def health(self) -> ModelHealth:
        return ModelHealth(self._loaded, "loaded" if self._loaded else "lazy")

    def describe_video(self, video_path: Path) -> str:
        self._load()
        raise ModelUnavailableError("local vision inference migration is pending server validation")

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        self._load()
        raise ModelUnavailableError("local vision inference migration is pending server validation")

    def answer(self, video_path: Path, question: str) -> Iterator[StreamEvent]:
        self._load()
        raise ModelUnavailableError("local vision inference migration is pending server validation")
        yield  # pragma: no cover

    def summarize_report(self, material: dict[str, object]) -> str:
        self._load()
        raise ModelUnavailableError("local report summary migration is pending server validation")


class LocalUnavailableGateway:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str) -> bytes:
        raise ModelUnavailableError(f"local {self.name} adapter is pending server validation")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ModelUnavailableError(f"local {self.name} adapter is pending server validation")
