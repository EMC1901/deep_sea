"""Domain types for retrieval-augmented image understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ImageRetrievalQuery:
    """A candidate image and the constraints used to find visual examples."""

    image_path: Path
    image_id: str | None = None
    site: str | None = None
    k: int = 4
    exclude_same_site: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalIndexRecord:
    """Portable metadata stored alongside one gallery embedding."""

    image_id: str
    labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    site: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedImage:
    """One labelled gallery image selected for a retrieval-augmented prompt."""

    image_id: str
    labels: Mapping[str, tuple[str, ...]]
    similarity: float
    site: str | None = None
    image_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ImageRetrievalHealth:
    """Readiness state surfaced without forcing the DINOv2 encoder to load."""

    enabled: bool
    ready: bool
    detail: str
    index_size: int = 0
    embedding_dimension: int = 0
