"""Build and verify the project's portable labelled-image retrieval index."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from deep_sea_explorer.ports.image_retrieval import ImageBatchEmbeddingGateway
from deep_sea_explorer.services.image_retrieval import l2_normalize, normalize_labels

from .errors import ImageIndexFormatError
from .numpy_index import NumpyImageRetrievalIndex


_EMBEDDING_FILENAME = "pool_embeddings.npy"
_METADATA_FILENAME = "pool_meta.json"
_MANIFEST_FILENAME = "index_manifest.json"
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


class ImageIndexBuildError(RuntimeError):
    """Raised when input data cannot safely produce a portable index."""


@dataclass(frozen=True, slots=True)
class GalleryImage:
    """One unambiguous labelled source image selected for embedding."""

    relative_path: str
    site: str
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GallerySelection:
    """The selected gallery together with auditable source-data counters."""

    images: tuple[GalleryImage, ...]
    counters: Mapping[str, int]
    annotation_files: tuple[str, ...]
    annotation_root: Path


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Facts about a successfully written and verified index directory."""

    index_dir: Path
    count: int
    dimension: int
    selection_counters: Mapping[str, int]


def discover_gallery(image_root: Path, annotation_root: Path) -> GallerySelection:
    """Match captions to unique image names without guessing across duplicate stems.

    The supplied caption shards identify files by basename only.  A basename that
    occurs in more than one gallery path is therefore deliberately excluded: using
    it would associate a correct label with an arbitrary, potentially wrong site.
    """

    images_root = image_root.resolve()
    labels_root = annotation_root.resolve()
    if not images_root.is_dir():
        raise ImageIndexBuildError(f"image root does not exist: {images_root}")
    if not labels_root.is_dir():
        raise ImageIndexBuildError(f"annotation root does not exist: {labels_root}")

    annotations, duplicate_annotation_stems, annotation_files = _load_caption_annotations(labels_root)
    paths_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
            paths_by_stem[path.stem.casefold()].append(path)

    selected: list[GalleryImage] = []
    ambiguous_image_stems = 0
    ambiguous_image_files = 0
    unlabelled_unique_image_files = 0
    labelled_ambiguous_stems = 0
    for stem, paths in paths_by_stem.items():
        if len(paths) != 1:
            ambiguous_image_stems += 1
            ambiguous_image_files += len(paths)
            if stem in annotations:
                labelled_ambiguous_stems += 1
            continue
        labels = annotations.get(stem)
        if labels is None:
            unlabelled_unique_image_files += 1
            continue
        path = paths[0]
        relative_path = path.relative_to(images_root).as_posix()
        site = relative_path.split("/", 1)[0]
        selected.append(GalleryImage(relative_path=relative_path, site=site, labels=labels))

    selected.sort(key=lambda item: item.relative_path.casefold())
    selected_stems = {Path(item.relative_path).stem.casefold() for item in selected}
    counters = {
        "annotation_records": len(annotations),
        "annotation_duplicate_stems_excluded": duplicate_annotation_stems,
        "annotation_stems_without_image": len(set(annotations) - set(paths_by_stem)),
        "image_files": sum(len(paths) for paths in paths_by_stem.values()),
        "image_unique_stems": len(paths_by_stem),
        "image_ambiguous_stems_excluded": ambiguous_image_stems,
        "image_ambiguous_files_excluded": ambiguous_image_files,
        "labelled_ambiguous_stems_excluded": labelled_ambiguous_stems,
        "unlabelled_unique_image_files_excluded": unlabelled_unique_image_files,
        "selected_images": len(selected),
        "selected_annotation_stems": len(selected_stems),
    }
    if not selected:
        raise ImageIndexBuildError("no unambiguous labelled images were found")
    return GallerySelection(tuple(selected), counters, annotation_files, labels_root)


def build_index(
    selection: GallerySelection,
    image_root: Path,
    output_dir: Path,
    encoder: ImageBatchEmbeddingGateway,
    *,
    model_name: str,
    batch_size: int = 32,
    overwrite: bool = False,
    limit: int | None = None,
) -> IndexBuildResult:
    """Encode selected images in batches and atomically publish a NumPy index."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    images = selection.images if limit is None else selection.images[:limit]
    if not images:
        raise ImageIndexBuildError("no images remain after applying the build limit")

    target = output_dir.resolve()
    if target.exists() and not overwrite:
        raise ImageIndexBuildError(f"output directory already exists: {target}; use --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{uuid4().hex}"
    staging.mkdir()
    embedding_file = staging / _EMBEDDING_FILENAME
    metadata_file = staging / _METADATA_FILENAME
    manifest_file = staging / _MANIFEST_FILENAME
    embeddings: np.memmap | None = None

    try:
        dimension = 0
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            batch_paths = [image_root / image.relative_path for image in batch]
            vectors = _validated_batch_vectors(encoder.embed_images(batch_paths), len(batch), dimension)
            if embeddings is None:
                dimension = int(vectors.shape[1])
                embeddings = np.lib.format.open_memmap(
                    embedding_file,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(images), dimension),
                )
            embeddings[start : start + len(batch)] = vectors
        if embeddings is None or dimension == 0:
            raise ImageIndexBuildError("encoder did not produce any embedding vectors")
        embeddings.flush()
        del embeddings
        embeddings = None

        metadata = [
            {"image": image.relative_path, "site": image.site, "labels": {"catami": list(image.labels)}}
            for image in images
        ]
        _write_json(metadata_file, metadata)
        manifest = {
            "schema_version": 1,
            "count": len(images),
            "embedding_shape": [len(images), dimension],
            "embedding_dtype": "float32",
            "l2_normalized": True,
            "encoder": {"type": "dinov2", "model": model_name},
            "source": {
                "image_root": str(image_root.resolve()),
                "image_paths": "relative_to_image_root",
                "site": "first_relative_path_component",
                "annotation_root": str(selection.annotation_root),
                "annotation_files": list(selection.annotation_files),
                "match_key": "casefolded_image_stem",
                "ambiguous_image_stems": "excluded",
            },
            "selection": dict(selection.counters) | {"indexed_images": len(images)},
        }
        _write_json(manifest_file, manifest)
        result = verify_index_directory(staging)
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        return IndexBuildResult(target, result.count, result.dimension, selection.counters)
    except Exception:
        if embeddings is not None:
            del embeddings
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_index_directory(index_dir: Path) -> IndexBuildResult:
    """Prove vectors, metadata and manifest are a finite one-to-one index."""

    directory = index_dir.resolve()
    embedding_file = directory / _EMBEDDING_FILENAME
    metadata_file = directory / _METADATA_FILENAME
    manifest_file = directory / _MANIFEST_FILENAME
    if not embedding_file.is_file() or not metadata_file.is_file() or not manifest_file.is_file():
        raise ImageIndexBuildError("index requires embeddings, metadata, and manifest files")
    try:
        embeddings = np.load(embedding_file, mmap_mode="r")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ImageIndexBuildError(f"unable to read index files: {error}") from error
    if not isinstance(metadata, list) or not isinstance(manifest, Mapping):
        raise ImageIndexBuildError("index metadata or manifest has an invalid JSON root")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ImageIndexBuildError("index embeddings must be a non-empty 2-D array")
    if len(metadata) != embeddings.shape[0]:
        raise ImageIndexBuildError("embedding rows and metadata records have different lengths")
    if manifest.get("count") != len(metadata):
        raise ImageIndexBuildError("manifest count does not match index metadata")
    if manifest.get("embedding_shape") != list(embeddings.shape):
        raise ImageIndexBuildError("manifest embedding shape does not match embeddings")
    if manifest.get("l2_normalized") is not True:
        raise ImageIndexBuildError("manifest must declare L2-normalized embeddings")

    image_ids: set[str] = set()
    for record in metadata:
        if not isinstance(record, Mapping) or not isinstance(record.get("image"), str):
            raise ImageIndexBuildError("every metadata record requires an image path")
        image_id = record["image"].strip().casefold()
        if not image_id or image_id in image_ids:
            raise ImageIndexBuildError("metadata image paths must be non-empty and unique")
        image_ids.add(image_id)
    _validate_embeddings(embeddings)
    try:
        loaded = NumpyImageRetrievalIndex.from_directory(directory)
    except ImageIndexFormatError as error:
        raise ImageIndexBuildError(f"index cannot be loaded by the runtime: {error}") from error
    return IndexBuildResult(directory, loaded.size, loaded.dimension, {})


def _load_caption_annotations(root: Path) -> tuple[dict[str, tuple[str, ...]], int, tuple[str, ...]]:
    files = tuple(sorted(root.glob("captioning_*.json")))
    if not files:
        raise ImageIndexBuildError("annotation root contains no captioning_*.json files")
    annotations: dict[str, tuple[str, ...]] = {}
    duplicate_stems: set[str] = set()
    for path in files:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImageIndexBuildError(f"unable to read annotation file {path.name}: {error}") from error
        if not isinstance(rows, list):
            raise ImageIndexBuildError(f"annotation file {path.name} must contain a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ImageIndexBuildError(f"annotation file {path.name} contains a non-object record")
            image = row.get("image")
            if not isinstance(image, str) or not image.strip():
                raise ImageIndexBuildError(f"annotation file {path.name} contains a record without image")
            try:
                labels = normalize_labels(row.get("labels"))
            except ValueError as error:
                raise ImageIndexBuildError(f"invalid labels for {image!r}: {error}") from error
            if not labels:
                raise ImageIndexBuildError(f"annotation record {image!r} contains no labels")
            stem = Path(image).stem.casefold()
            if stem in annotations:
                duplicate_stems.add(stem)
            else:
                annotations[stem] = labels
    for stem in duplicate_stems:
        annotations.pop(stem, None)
    return annotations, len(duplicate_stems), tuple(path.name for path in files)


def _validated_batch_vectors(
    values: np.ndarray,
    expected_rows: int,
    dimension: int,
) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
        raise ImageIndexBuildError("encoder returned vectors with an unexpected batch shape")
    if dimension and vectors.shape[1] != dimension:
        raise ImageIndexBuildError("encoder returned inconsistent embedding dimensions")
    if not np.isfinite(vectors).all():
        raise ImageIndexBuildError("encoder returned non-finite embedding values")
    vectors = l2_normalize(vectors)
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-6):
        raise ImageIndexBuildError("encoder returned a zero embedding vector")
    return vectors


def _validate_embeddings(embeddings: np.ndarray) -> None:
    for start in range(0, len(embeddings), 4096):
        block = np.asarray(embeddings[start : start + 4096], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ImageIndexBuildError("index embeddings contain non-finite values")
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise ImageIndexBuildError("index embeddings are not L2-normalized")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
