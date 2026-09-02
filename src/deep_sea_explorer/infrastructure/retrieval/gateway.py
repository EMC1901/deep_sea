"""Application-facing retrieval gateway composed from an encoder and local index."""

from __future__ import annotations

from pathlib import Path

from deep_sea_explorer.domain.retrieval import (
    ImageRetrievalHealth,
    ImageRetrievalQuery,
    RetrievedImage,
)
from deep_sea_explorer.infrastructure.models.local.runtime import InferenceCoordinator
from deep_sea_explorer.ports.image_retrieval import ImageEmbeddingGateway

from .errors import ImageRetrievalError
from .numpy_index import NumpyImageRetrievalIndex


class LocalImageRetrievalGateway:
    """Retrieve labelled examples without any network or reference-package dependency."""

    def __init__(self, encoder: ImageEmbeddingGateway, index: NumpyImageRetrievalIndex) -> None:
        self.encoder = encoder
        self.index = index

    def retrieve(self, query: ImageRetrievalQuery) -> tuple[RetrievedImage, ...]:
        vector = self.encoder.embed_image(query.image_path)
        return self.index.search(
            vector,
            k=query.k,
            exclude_image_id=query.image_id,
            query_site=query.site,
            exclude_same_site=query.exclude_same_site,
        )

    def health(self) -> ImageRetrievalHealth:
        gallery_root = self.index.gallery_root
        if gallery_root is None:
            return ImageRetrievalHealth(
                True,
                False,
                "index manifest has no gallery source root",
                self.index.size,
                self.index.dimension,
            )
        if not gallery_root.is_dir():
            return ImageRetrievalHealth(
                True,
                False,
                "configured gallery source root is unavailable",
                self.index.size,
                self.index.dimension,
            )
        model_path = getattr(self.encoder, "model_path", None)
        if not isinstance(model_path, str) or not model_path:
            return ImageRetrievalHealth(
                True,
                False,
                "configured DINOv2 model path is unavailable",
                self.index.size,
                self.index.dimension,
            )
        if not Path(model_path).is_dir():
            return ImageRetrievalHealth(
                True,
                False,
                "configured DINOv2 model path is unavailable",
                self.index.size,
                self.index.dimension,
            )
        return ImageRetrievalHealth(
            True,
            True,
            "index loaded; DINOv2 loads on the first retrieval",
            self.index.size,
            self.index.dimension,
        )


class CoordinatedImageRetrievalGateway:
    """Serialize DINOv2 and Qwen GPU work through the existing runtime coordinator."""

    def __init__(self, gateway: LocalImageRetrievalGateway, coordinator: InferenceCoordinator) -> None:
        self.gateway = gateway
        self.coordinator = coordinator

    def retrieve(self, query: ImageRetrievalQuery) -> tuple[RetrievedImage, ...]:
        return self.coordinator.execute(lambda: self.gateway.retrieve(query))

    def health(self) -> ImageRetrievalHealth:
        return self.gateway.health()


class UnavailableImageRetrievalGateway:
    """Report optional retrieval failures while callers continue with two-image VLM input."""

    def __init__(self, detail: str, *, enabled: bool) -> None:
        self.detail = detail
        self.enabled = enabled

    def retrieve(self, query: ImageRetrievalQuery) -> tuple[RetrievedImage, ...]:
        del query
        raise ImageRetrievalError(self.detail)

    def health(self) -> ImageRetrievalHealth:
        return ImageRetrievalHealth(self.enabled, False, self.detail)
