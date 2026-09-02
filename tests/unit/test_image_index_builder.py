from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_sea_explorer.infrastructure.retrieval.index_builder import (
    ImageIndexBuildError,
    build_index,
    discover_gallery,
    verify_index_directory,
)
from deep_sea_explorer.infrastructure.retrieval.dinov2 import DinoV2ImageEncoder


class StubBatchEncoder:
    def __init__(self) -> None:
        self.batches: list[tuple[Path, ...]] = []

    def embed_image(self, image_path: Path) -> np.ndarray:
        return self.embed_images((image_path,))[0]

    def embed_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        self.batches.append(tuple(image_paths))
        return np.asarray([[float(index + 1), 1.0] for index, _ in enumerate(image_paths)])


def _write_caption_shard(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    (root / name).write_text(json.dumps(rows), encoding="utf-8")


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def to(self, device: str) -> "FakeTensor":
        assert device == "cpu"
        return self

    def __getitem__(self, key: object) -> "FakeTensor":
        return FakeTensor(self.values[key])

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def float(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeTorch:
    @contextmanager
    def inference_mode(self) -> object:
        yield


class FakeProcessor:
    def __init__(self) -> None:
        self.image_counts: list[int] = []

    def __call__(self, *, images: Sequence[object], return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        self.image_counts.append(len(images))
        return {"pixels": FakeTensor(np.ones((len(images), 1), dtype=np.float32))}


class FakeModel:
    def __call__(self, **inputs: FakeTensor) -> SimpleNamespace:
        count = next(iter(inputs.values())).values.shape[0]
        values = np.arange(1, count * 2 + 1, dtype=np.float32).reshape(count, 1, 2)
        return SimpleNamespace(last_hidden_state=FakeTensor(values))


def test_dinov2_encoder_uses_one_forward_pass_for_an_ordered_image_batch(tmp_path: Path) -> None:
    from PIL import Image

    image_paths = (tmp_path / "one.jpg", tmp_path / "two.jpg")
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    processor = FakeProcessor()
    encoder = DinoV2ImageEncoder("local-dinov2", device="cpu")
    encoder._resource = (FakeTorch(), processor, FakeModel())

    vectors = encoder.embed_images(image_paths)

    assert processor.image_counts == [2]
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(encoder.embed_image(image_paths[0]), vectors[0])


def test_discovery_excludes_ambiguous_image_names_and_keeps_site_metadata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for relative in (
        "site-a/duplicate.jpg",
        "site-b/duplicate.jpg",
        "site-c/good.JPG",
        "site-d/unlabelled.jpg",
    ):
        path = images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    _write_caption_shard(
        annotations,
        "captioning_0_1.json",
        [
            {"image": "duplicate.jpg", "labels": ["Biota > Fish"]},
            {"image": "good.JPG", "labels": ["Substrate > Sand / mud"]},
            {"image": "missing.jpg", "labels": ["Biota > Sponge"]},
        ],
    )

    selection = discover_gallery(images, annotations)

    assert [item.relative_path for item in selection.images] == ["site-c/good.JPG"]
    assert selection.images[0].site == "site-c"
    assert selection.images[0].labels == ("Substrate > Sand / mud",)
    assert selection.counters["image_ambiguous_stems_excluded"] == 1
    assert selection.counters["image_ambiguous_files_excluded"] == 2
    assert selection.counters["annotation_stems_without_image"] == 1


def test_build_writes_batch_embeddings_and_verifies_one_to_one_metadata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for relative in ("dive-a/one.jpg", "dive-b/two.jpg"):
        path = images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    _write_caption_shard(
        annotations,
        "captioning_0_1.json",
        [
            {"image": "one.jpg", "labels": ["Biota > Fish"]},
            {"image": "two.jpg", "labels": ["Bedforms > Ripple"]},
        ],
    )
    selection = discover_gallery(images, annotations)
    encoder = StubBatchEncoder()
    index_dir = tmp_path / "runtime" / "image-index"

    result = build_index(
        selection,
        images,
        index_dir,
        encoder,
        model_name="local-dinov2",
        batch_size=1,
    )

    assert result.count == 2
    assert result.dimension == 2
    assert [len(batch) for batch in encoder.batches] == [1, 1]
    embeddings = np.load(index_dir / "pool_embeddings.npy", mmap_mode="r")
    assert embeddings.shape == (2, 2)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    metadata = json.loads((index_dir / "pool_meta.json").read_text(encoding="utf-8"))
    assert metadata == [
        {"image": "dive-a/one.jpg", "site": "dive-a", "labels": {"catami": ["Biota > Fish"]}},
        {
            "image": "dive-b/two.jpg",
            "site": "dive-b",
            "labels": {"catami": ["Bedforms > Ripple"]},
        },
    ]
    manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(metadata) == embeddings.shape[0]
    assert manifest["embedding_shape"] == [2, 2]
    assert manifest["source"]["annotation_root"] == str(annotations.resolve())
    verified = verify_index_directory(index_dir)
    assert (verified.count, verified.dimension) == (2, 2)


def test_verification_rejects_metadata_that_no_longer_matches_vectors(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    path = images / "dive/one.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"jpeg")
    _write_caption_shard(
        annotations,
        "captioning_0_1.json",
        [{"image": "one.jpg", "labels": ["Biota > Fish"]}],
    )
    selection = discover_gallery(images, annotations)
    index_dir = tmp_path / "index"
    build_index(selection, images, index_dir, StubBatchEncoder(), model_name="local-dinov2")
    (index_dir / "pool_meta.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ImageIndexBuildError, match="different lengths"):
        verify_index_directory(index_dir)
