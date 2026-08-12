"""Portable NumPy index for labelled deep-sea image retrieval."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np

from deep_sea_explorer.domain.retrieval import RetrievalIndexRecord, RetrievedImage
from deep_sea_explorer.services.image_retrieval import (
    cosine_top_k,
    l2_normalize,
    normalize_image_id,
    normalize_label_mapping,
    normalize_labels,
)

from .errors import ImageIndexFormatError


_CATEGORY_FIELDS = ("biota", "substrate", "bedforms", "relief")


class NumpyImageRetrievalIndex:
    """Read-only cosine index with deterministic ranking and portable metadata."""

    def __init__(
        self,
        embeddings: np.ndarray,
        records: Sequence[RetrievalIndexRecord],
        *,
        embeddings_normalized: bool = False,
        gallery_root: Path | None = None,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ImageIndexFormatError("gallery embeddings must be a non-empty 2-D array")
        if len(matrix) != len(records):
            raise ImageIndexFormatError("gallery embeddings and metadata have different lengths")
        if not np.isfinite(matrix).all():
            raise ImageIndexFormatError("gallery embeddings contain non-finite values")
        if not embeddings_normalized:
            matrix = l2_normalize(matrix)
        self._embeddings = matrix
        self._records = tuple(records)
        self._gallery_root = gallery_root

    @property
    def dimension(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def gallery_root(self) -> Path | None:
        """The source gallery root recorded by an index build, when available."""

        return self._gallery_root

    @classmethod
    def from_directory(cls, index_dir: Path) -> "NumpyImageRetrievalIndex":
        """Load the project index layout without recording machine-specific paths."""

        embedding_path = index_dir / "pool_embeddings.npy"
        metadata_path = index_dir / "pool_meta.json"
        if not embedding_path.is_file() or not metadata_path.is_file():
            raise ImageIndexFormatError(
                "index requires pool_embeddings.npy and pool_meta.json in the configured directory"
            )
        try:
            embeddings = np.load(embedding_path, mmap_mode="r")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ImageIndexFormatError(f"unable to read image retrieval index: {error}") from error
        if not isinstance(metadata, list):
            raise ImageIndexFormatError("pool_meta.json must contain a list of image records")

        manifest = _read_manifest(index_dir)
        if manifest is not None and manifest.get("count") is not None and manifest.get("count") != len(metadata):
            raise ImageIndexFormatError("index manifest count does not match pool metadata")
        if (
            manifest is not None
            and manifest.get("embedding_shape") is not None
            and manifest.get("embedding_shape") != list(embeddings.shape)
        ):
            raise ImageIndexFormatError("index manifest embedding shape does not match pool embeddings")

        records = tuple(_record_from_metadata(value) for value in metadata)
        normalized = bool(manifest and manifest.get("l2_normalized") is True)
        return cls(
            embeddings,
            records,
            embeddings_normalized=normalized,
            gallery_root=_gallery_root_from_manifest(manifest),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        k: int,
        exclude_image_id: str | None = None,
        query_site: str | None = None,
        exclude_same_site: bool = False,
    ) -> tuple[RetrievedImage, ...]:
        """Find the most similar eligible labelled gallery images."""

        excluded: np.ndarray = np.zeros(self.size, dtype=bool)
        image_key = normalize_image_id(exclude_image_id) if exclude_image_id else None
        site_key = _site_key(query_site)
        for index, record in enumerate(self._records):
            if image_key and normalize_image_id(record.image_id) == image_key:
                excluded[index] = True
            if exclude_same_site and site_key and _site_key(record.site) == site_key:
                excluded[index] = True

        scores, indices = cosine_top_k(
            query_embedding,
            self._embeddings,
            k,
            excluded=excluded,
            gallery_normalized=True,
        )
        return tuple(
            RetrievedImage(
                image_id=self._records[int(index)].image_id,
                labels=self._records[int(index)].labels,
                similarity=float(scores[int(index)]),
                site=self._records[int(index)].site,
                image_path=self._gallery_path(self._records[int(index)].image_id),
            )
            for index in indices
        )

    def _gallery_path(self, image_id: str) -> Path | None:
        if self._gallery_root is None:
            return None
        candidate = (self._gallery_root / image_id).resolve()
        try:
            candidate.relative_to(self._gallery_root)
        except ValueError:
            return None
        return candidate


def _read_manifest(index_dir: Path) -> Mapping[str, object] | None:
    path = index_dir / "index_manifest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageIndexFormatError(f"unable to read index manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise ImageIndexFormatError("index manifest must contain an object")
    return value


def _gallery_root_from_manifest(manifest: Mapping[str, object] | None) -> Path | None:
    if manifest is None:
        return None
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        return None
    value = source.get("image_root")
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _record_from_metadata(value: object) -> RetrievalIndexRecord:
    if not isinstance(value, Mapping):
        raise ImageIndexFormatError("every pool metadata record must be an object")
    image = value.get("image", value.get("image_id"))
    if not isinstance(image, str) or not image.strip():
        raise ImageIndexFormatError("every pool metadata record requires a non-empty image")
    try:
        labels = _labels_from_metadata(value)
    except ValueError as error:
        raise ImageIndexFormatError(f"invalid labels for image {image!r}: {error}") from error
    site = value.get("site")
    if site is not None and not isinstance(site, str):
        raise ImageIndexFormatError(f"site for image {image!r} must be a string")
    return RetrievalIndexRecord(
        image_id=image.strip(),
        labels=MappingProxyType(labels),
        site=site.strip() if isinstance(site, str) and site.strip() else None,
    )


def _labels_from_metadata(value: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    labels = value.get("labels")
    if labels is not None:
        return normalize_label_mapping(labels)
    result: dict[str, tuple[str, ...]] = {}
    for category in _CATEGORY_FIELDS:
        if category in value:
            normalized = normalize_labels(value[category])
            if normalized:
                result[category] = normalized
    if "cls" in value:
        normalized = normalize_labels(value["cls"])
        if normalized:
            result.setdefault("substrate", normalized)
    return result


def _site_key(value: str | None) -> str | None:
    return value.strip().casefold() if isinstance(value, str) and value.strip() else None
