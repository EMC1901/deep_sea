from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deep_sea_explorer.domain.retrieval import ImageRetrievalQuery, RetrievalIndexRecord
from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.infrastructure.retrieval.errors import ImageIndexFormatError
from deep_sea_explorer.infrastructure.models.local.runtime import InferenceCoordinator
from deep_sea_explorer.infrastructure.retrieval.gateway import (
    CoordinatedImageRetrievalGateway,
    LocalImageRetrievalGateway,
)
from deep_sea_explorer.infrastructure.retrieval.numpy_index import NumpyImageRetrievalIndex
from deep_sea_explorer.services.image_retrieval import (
    label_path_with_ancestors,
    map_labels_to_survey_categories,
    normalize_label_path,
    normalize_labels,
)


class StubImageEncoder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.seen: list[Path] = []

    def embed_image(self, image_path: Path) -> np.ndarray:
        self.seen.append(image_path)
        return self.vector


class HealthStubImageEncoder(StubImageEncoder):
    def __init__(self, vector: np.ndarray, model_path: str) -> None:
        super().__init__(vector)
        self.model_path = model_path


def test_label_normalization_preserves_class_name_slashes_and_hierarchy() -> None:
    assert normalize_label_path("  Biota >  Cnidaria  > Coral ") == "Biota > Cnidaria > Coral"
    assert normalize_labels(["Sand / mud (<2mm)", " sand / MUD (<2mm) "]) == (
        "Sand / mud (<2mm)",
    )
    assert label_path_with_ancestors("Biota > Cnidaria > Coral") == (
        "Biota > Cnidaria > Coral",
        "Biota > Cnidaria",
        "Biota",
    )
    assert map_labels_to_survey_categories(
        {
            "catami": (
                "Biota > Fish",
                "Substrate > Sand / mud",
                "No bedforms",
                "Anthropogenic > Tile",
            )
        }
    ) == {
        "organism": ("Biota > Fish",),
        "seabed_substrate": ("Substrate > Sand / mud",),
        "micro_topography": ("No bedforms",),
        "other": ("Anthropogenic > Tile",),
    }


def test_numpy_index_ranks_deterministically_and_excludes_query_image() -> None:
    index = NumpyImageRetrievalIndex(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        (
            RetrievalIndexRecord("gallery/a.jpg", {"biota": ("Fish",)}, "site-a"),
            RetrievalIndexRecord("gallery/b.jpg", {"biota": ("Sponge",)}, "site-b"),
            RetrievalIndexRecord("gallery/c.jpg", {"substrate": ("Sand",)}, "site-c"),
        ),
        embeddings_normalized=True,
    )

    results = index.search(np.asarray([1.0, 0.0]), k=2, exclude_image_id="gallery\\a.jpg")

    assert [item.image_id for item in results] == ["gallery/b.jpg", "gallery/c.jpg"]
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity == pytest.approx(0.0)


def test_numpy_index_can_exclude_all_images_from_the_query_site() -> None:
    index = NumpyImageRetrievalIndex(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        (
            RetrievalIndexRecord("a.jpg", {}, "Dive-1"),
            RetrievalIndexRecord("b.jpg", {}, " dive-1 "),
            RetrievalIndexRecord("c.jpg", {}, "Dive-2"),
        ),
    )

    results = index.search(
        np.asarray([1.0, 0.0]),
        k=3,
        query_site="DIVE-1",
        exclude_same_site=True,
    )

    assert [item.image_id for item in results] == ["c.jpg"]


def test_index_directory_loads_normalized_metadata_and_validates_manifest(tmp_path: Path) -> None:
    gallery_root = tmp_path / "gallery"
    gallery_root.mkdir()
    np.save(tmp_path / "pool_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    (tmp_path / "pool_meta.json").write_text(
        json.dumps(
            [
                {
                    "image": "gallery/a.jpg",
                    "site": "Dive-A",
                    "biota": [" Biota >  Fish "],
                    "substrate": "Sand / mud (<2mm)",
                },
                {"image": "gallery/b.jpg", "labels": {"relief": ["Rock > Ridge"]}},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "index_manifest.json").write_text(
        json.dumps(
            {
                "count": 2,
                "embedding_shape": [2, 2],
                "l2_normalized": True,
                "source": {"image_root": str(gallery_root)},
            }
        ),
        encoding="utf-8",
    )

    index = NumpyImageRetrievalIndex.from_directory(tmp_path)
    results = index.search(np.asarray([1.0, 0.0]), k=1)

    assert index.size == 2
    assert index.dimension == 2
    assert index.gallery_root == gallery_root.resolve()
    assert results[0].image_path == (gallery_root / "gallery/a.jpg").resolve()
    assert results[0].labels == {
        "biota": ("Biota > Fish",),
        "substrate": ("Sand / mud (<2mm)",),
    }

    (tmp_path / "index_manifest.json").write_text(json.dumps({"count": 3}), encoding="utf-8")
    with pytest.raises(ImageIndexFormatError, match="count"):
        NumpyImageRetrievalIndex.from_directory(tmp_path)


def test_gateway_encodes_the_query_then_applies_its_retrieval_constraints(tmp_path: Path) -> None:
    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"jpeg")
    encoder = StubImageEncoder(np.asarray([0.0, 1.0]))
    index = NumpyImageRetrievalIndex(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        (RetrievalIndexRecord("a.jpg", {}), RetrievalIndexRecord("b.jpg", {})),
    )
    gateway = LocalImageRetrievalGateway(encoder, index)

    results = gateway.retrieve(ImageRetrievalQuery(image, image_id="b.jpg", k=1))

    assert encoder.seen == [image]
    assert [item.image_id for item in results] == ["a.jpg"]


def test_retrieval_health_requires_readable_gallery_and_dinov2_model(tmp_path: Path) -> None:
    gallery_root = tmp_path / "gallery"
    dino_root = tmp_path / "dinov2"
    gallery_root.mkdir()
    dino_root.mkdir()
    encoder = HealthStubImageEncoder(np.asarray([1.0, 0.0]), str(dino_root))
    gateway = LocalImageRetrievalGateway(
        encoder,
        NumpyImageRetrievalIndex(
            np.asarray([[1.0, 0.0]]),
            (RetrievalIndexRecord("example.jpg", {}),),
            gallery_root=gallery_root,
        ),
    )

    health = gateway.health()

    assert health.enabled is True
    assert health.ready is True
    assert health.index_size == 1
    assert health.embedding_dimension == 2

    dino_root.rmdir()
    assert gateway.health().ready is False
    assert "DINOv2" in gateway.health().detail


def test_local_container_retrieval_uses_the_vision_runtime_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deep_sea_explorer import container

    index = NumpyImageRetrievalIndex(
        np.asarray([[1.0, 0.0]]), (RetrievalIndexRecord("example.jpg", {}),)
    )
    coordinator = InferenceCoordinator()
    settings = Settings(
        model_backend=ModelBackend.LOCAL,
        image_retrieval_enabled=True,
        image_retrieval_index_dir="/index",
        image_retrieval_dino_model_path="/dinov2",
    )

    monkeypatch.setattr(
        container.NumpyImageRetrievalIndex,
        "from_directory",
        classmethod(lambda cls, _: index),
    )
    monkeypatch.setattr(container, "DinoV2ImageEncoder", lambda *args, **kwargs: object())

    retrieval = container._build_local_image_retrieval(settings, coordinator)

    assert isinstance(retrieval, CoordinatedImageRetrievalGateway)
    assert retrieval.coordinator is coordinator
