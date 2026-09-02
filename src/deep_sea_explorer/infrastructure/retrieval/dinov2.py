"""Frozen DINOv2 encoder used by the project's image-retrieval service."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from deep_sea_explorer.services.image_retrieval import l2_normalize

from .errors import ImageEmbeddingError


def resolve_device(requested: str = "auto") -> str:
    """Prefer CUDA, then MPS, while keeping imports optional for non-model tests."""

    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class DinoV2ImageEncoder:
    """Lazily load DINOv2 and return one L2-normalized CLS embedding per image."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        if not model_path:
            raise ValueError("a DINOv2 model path or identifier is required")
        self.model_path = model_path
        self.device = resolve_device(device)
        self._resource: tuple[Any, Any, Any] | None = None
        self._lock = threading.RLock()

    def embed_image(self, image_path: Path) -> np.ndarray:
        return self.embed_images((image_path,))[0]

    def embed_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Encode an ordered image batch in one DINOv2 forward pass."""

        if not image_paths:
            raise ValueError("at least one image path is required")
        missing = next((path for path in image_paths if not path.is_file()), None)
        if missing is not None:
            raise ImageEmbeddingError(f"image file does not exist: {missing}")
        torch, processor, model = self._load_resource()
        images: list[Any] = []
        try:
            from PIL import Image

            for image_path in image_paths:
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB").copy())
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            vectors = output.last_hidden_state[:, 0, :].detach().cpu().float().numpy()
        except ImageEmbeddingError:
            raise
        except Exception as error:
            raise ImageEmbeddingError(f"unable to encode image with frozen DINOv2: {error}") from error
        finally:
            for image in images:
                image.close()
        return l2_normalize(vectors)

    def _load_resource(self) -> tuple[Any, Any, Any]:
        if self._resource is not None:
            return self._resource
        with self._lock:
            if self._resource is not None:
                return self._resource
            try:
                from transformers import AutoImageProcessor, AutoModel
                import torch
            except ImportError as error:
                raise ImageEmbeddingError(
                    "DINOv2 retrieval requires torch, transformers, and Pillow"
                ) from error
            try:
                processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
                model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
                model.to(self.device).eval()
            except Exception as error:
                raise ImageEmbeddingError(f"unable to load frozen DINOv2: {error}") from error
            self._resource = (torch, processor, model)
            return self._resource
