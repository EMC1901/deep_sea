"""Pure, deterministic helpers for image-retrieval metadata and ranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


_SURVEY_CATEGORY_BY_LABEL_ROOT = {
    "biota": "organism",
    "no visible biota": "organism",
    "substrate": "seabed_substrate",
    "bedforms": "micro_topography",
    "no bedforms": "micro_topography",
    "bioturbation": "micro_topography",
    "flat": "micro_topography",
    "high": "micro_topography",
    "low / moderate": "micro_topography",
    "veneer": "micro_topography",
}


def normalize_image_id(value: str) -> str:
    """Return a stable comparison key without making image paths absolute."""

    return value.strip().replace("\\", "/").removeprefix("./").casefold()


def normalize_label_path(value: str) -> str:
    """Normalize CATAMI-style hierarchical paths while retaining their hierarchy."""

    # A slash can be part of an actual class name (for example "Sand / mud"),
    # so only the documented ``>`` separator defines hierarchy here.
    parts = [" ".join(part.split()) for part in value.split(">")]
    return " > ".join(part for part in parts if part)


def label_path_with_ancestors(value: str) -> tuple[str, ...]:
    """Return a normalized path followed by each progressively broader ancestor."""

    path = normalize_label_path(value)
    if not path:
        return ()
    parts = path.split(" > ")
    return tuple(" > ".join(parts[:depth]) for depth in range(len(parts), 0, -1))


def normalize_labels(values: object) -> tuple[str, ...]:
    """Canonicalize scalar or collection label values and remove duplicates."""

    if isinstance(values, str):
        candidates: Iterable[object] = (values,)
    elif isinstance(values, Iterable) and not isinstance(values, Mapping):
        candidates = values
    elif values is None:
        candidates = ()
    else:
        raise ValueError("labels must be a string, iterable of strings, or null")

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise ValueError("every label must be a string")
        label = normalize_label_path(candidate)
        key = label.casefold()
        if label and key not in seen:
            seen.add(key)
            normalized.append(label)
    return tuple(normalized)


def normalize_label_mapping(value: object) -> dict[str, tuple[str, ...]]:
    """Normalize category-to-label metadata from a portable JSON index."""

    if not isinstance(value, Mapping):
        raise ValueError("labels must be an object keyed by category")
    result: dict[str, tuple[str, ...]] = {}
    for category, labels in value.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("label categories must be non-empty strings")
        normalized = normalize_labels(labels)
        if normalized:
            result[category.strip().casefold()] = normalized
    return result


def map_labels_to_survey_categories(
    labels: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Map CATAMI labels into the existing event JSON categories without loss.

    The source index retains its original labels.  This mapping is only a prompt
    aid for Qwen and never changes the parser or event-acceptance semantics.
    """

    grouped: dict[str, list[str]] = {}
    for source_category, values in labels.items():
        normalized_category = source_category.strip().casefold()
        if normalized_category in {"biota", "substrate", "bedforms", "relief"}:
            target = {
                "biota": "organism",
                "substrate": "seabed_substrate",
                "bedforms": "micro_topography",
                "relief": "micro_topography",
            }[normalized_category]
            label_values = values
        else:
            target = "other"
            label_values = values
        for value in label_values:
            root = normalize_label_path(value).split(" > ", 1)[0].casefold()
            target_for_value = _SURVEY_CATEGORY_BY_LABEL_ROOT.get(root, target)
            bucket = grouped.setdefault(target_for_value, [])
            if value not in bucket:
                bucket.append(value)
    return {category: tuple(values) for category, values in grouped.items()}


def l2_normalize(vector: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Normalize floats without producing NaN for zero vectors."""

    values = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 1e-12)


def cosine_top_k(
    query: np.ndarray,
    gallery: np.ndarray,
    k: int,
    *,
    excluded: np.ndarray | None = None,
    gallery_normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine scores and deterministic gallery indices in descending order."""

    if k < 1:
        raise ValueError("k must be positive")
    matrix = np.asarray(gallery, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("gallery embeddings must be a 2-D array")
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError("query embedding dimension does not match the gallery")
    if excluded is None:
        excluded_mask: np.ndarray = np.zeros(len(matrix), dtype=bool)
    else:
        excluded_mask = np.asarray(excluded, dtype=bool)
        if excluded_mask.shape != (len(matrix),):
            raise ValueError("excluded mask must contain one value per gallery embedding")

    # Index construction normalizes gallery rows once. Reusing that invariant
    # avoids allocating a second large matrix for every real-time query.
    normalized_gallery = matrix if gallery_normalized else l2_normalize(matrix)
    scores = normalized_gallery @ l2_normalize(vector)
    eligible = np.flatnonzero(~excluded_mask)
    order = eligible[np.lexsort((eligible, -scores[eligible]))]
    return scores, order[:k]
