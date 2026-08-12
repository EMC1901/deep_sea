"""Ports for frozen image encoding and labelled image retrieval."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from deep_sea_explorer.domain.retrieval import ImageRetrievalHealth, ImageRetrievalQuery, RetrievedImage


class ImageEmbeddingGateway(Protocol):
    """Encodes one image into a normalized visual feature vector."""

    def embed_image(self, image_path: Path) -> np.ndarray: ...


class ImageBatchEmbeddingGateway(ImageEmbeddingGateway, Protocol):
    """Encodes a batch of images in the order supplied by the index builder."""

    def embed_images(self, image_paths: Sequence[Path]) -> np.ndarray: ...


class ImageRetrievalGateway(Protocol):
    """Finds labelled visual examples for a candidate image."""

    def retrieve(self, query: ImageRetrievalQuery) -> tuple[RetrievedImage, ...]: ...

    def health(self) -> ImageRetrievalHealth: ...
