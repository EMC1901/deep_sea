from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from deep_sea_explorer.domain.models import CountItem, MonitoringTagMatch
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService
from deep_sea_explorer.services.monitoring_knowledge import MonitoringKnowledgeBase


class Vision:
    def __init__(self, release: bool = True) -> None:
        self.release = release
        self.started: list[str] = []

    def match_monitoring_tags(self, image_path: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        return MonitoringTagMatch(
            (CountItem("海绵", 2),) if "海绵" in candidates["organisms"] else (),
            (CountItem("沙泥", 1),) if "沙泥" in candidates["substrates"] else (),
            (CountItem("平坦海床", 1),) if "平坦海床" in candidates["geomorphologies"] else (),
        )

    def describe_monitoring_frame(self, image_path: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        self.started.append(image_path.read_bytes().decode("ascii"))
        while not self.release:
            time.sleep(0.002)
        return f"画面 {self.started[-1]}"


class Dino:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = iter(values)

    def embed_image(self, _image_path: Path) -> np.ndarray:
        return np.asarray(next(self.values), dtype=np.float32)


def service(
    tmp_path: Path,
    vision: Vision,
    dino: Dino,
    capacity: int = 2,
    similarity_threshold: float = 0.5,
) -> MonitoringService:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    directory.joinpath("label_universe.json").write_text(json.dumps({"labels": {
        "bio": [{"canonical_label": "海绵", "description": "多孔状固着生物。"}],
        "substrate": [{"canonical_label": "沙泥", "description": "细颗粒沉积物。"}],
        "geomorphology": [{"canonical_label": "平坦海床", "description": "起伏较小的海底表面。"}],
    }}), encoding="utf-8")
    return MonitoringService(
        vision, object(), MemorySessionStore(60, 5), MemoryMemoBroker(),
        TempFileStore(tmp_path, 60), CaptureStatsService(), 0.85,
        dino_encoder=dino,
        knowledge_base=MonitoringKnowledgeBase(directory),
        queue_capacity=capacity,
        blur_threshold=1,
        similarity_threshold=similarity_threshold,
    )


@pytest.fixture
def valid_frame(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MonitoringService, "_decode_and_sharpness", staticmethod(lambda _path: 100.0))


def test_first_frame_is_queued_and_similar_frame_is_rejected(tmp_path: Path, valid_frame) -> None:
    vision = Vision()
    monitored = service(tmp_path, vision, Dino([[1, 0], [0.9, 0.1]]))

    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while not vision.started and time.time() < deadline:
        time.sleep(0.01)
    response = monitored.process_frame("s", b"two")

    assert response["status"] == "rejected_similar"
    assert response["similarity"] > 0.5
    assert response["metrics"]["rejected_similar"] == 1


def test_dino_similarity_boundary_of_half_is_accepted(tmp_path: Path, valid_frame) -> None:
    monitored = service(tmp_path, Vision(release=False), Dino([[1, 0], [0.5, 0]]))

    assert monitored.process_frame("s", b"one")["status"] == "queued"
    response = monitored.process_frame("s", b"two")

    assert response["status"] == "queued"
    assert response["similarity"] == 0.5


def test_dino_similarity_above_half_is_rejected(tmp_path: Path, valid_frame) -> None:
    monitored = service(tmp_path, Vision(release=False), Dino([[1, 0], [0.5001, 0]]))

    assert monitored.process_frame("s", b"one")["status"] == "queued"
    response = monitored.process_frame("s", b"two")

    assert response["status"] == "rejected_similar"
    assert response["similarity"] > 0.5


def test_undecodable_and_blurry_frames_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monitored = service(tmp_path, Vision(), Dino([[1, 0]]))
    monkeypatch.setattr(MonitoringService, "_decode_and_sharpness", staticmethod(lambda _path: (_ for _ in ()).throw(ValueError("bad jpeg"))))
    assert monitored.process_frame("s", b"bad")["status"] == "rejected_undecodable"
    monkeypatch.setattr(MonitoringService, "_decode_and_sharpness", staticmethod(lambda _path: 0.1))
    assert monitored.process_frame("s", b"blur")["status"] == "rejected_blurry"


def test_queue_discards_oldest_waiting_frame_and_preserves_fifo(tmp_path: Path, valid_frame) -> None:
    vision = Vision(release=False)
    monitored = service(tmp_path, vision, Dino([[1, 0], [0, 1], [-1, 0], [0, -1]]), capacity=2)

    assert monitored.process_frame("s", b"one")["status"] == "queued"
    assert monitored.process_frame("s", b"two")["status"] == "queued"
    assert monitored.process_frame("s", b"three")["status"] == "queued"
    fourth = monitored.process_frame("s", b"four")
    assert fourth["queue2_dropped_oldest"] is True

    vision.release = True
    deadline = time.time() + 2
    while len(vision.started) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert vision.started == ["one", "three", "four"]
    assert monitored.metrics("s")["queue2_dropped_oldest"] == 1


def test_monitoring_result_is_published_as_a_memo(tmp_path: Path, valid_frame) -> None:
    vision = Vision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)
    assert monitored.broker.drain("s")[0].content == "画面 one"


def test_monitoring_result_keeps_three_tag_categories_separate(tmp_path: Path, valid_frame) -> None:
    vision = Vision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)
    memo = monitored.broker.drain("s")[0]
    captures = {capture.type.value: capture for capture in memo.captures}
    assert set(captures) == {"bio", "substrate", "geomorphology"}
    assert [item.name for item in captures["bio"].organisms] == ["海绵"]
    assert not captures["bio"].substrates and not captures["bio"].geomorphologies
    assert [item.name for item in captures["substrate"].substrates] == ["沙泥"]
    assert not captures["substrate"].organisms and not captures["substrate"].geomorphologies
    assert [item.name for item in captures["geomorphology"].geomorphologies] == ["平坦海床"]
    assert not captures["geomorphology"].organisms and not captures["geomorphology"].substrates


def test_monitoring_statistics_count_a_matching_label_again_only_after_three_frames(
    tmp_path: Path, valid_frame
) -> None:
    vision = Vision()
    monitored = service(
        tmp_path,
        vision,
        Dino(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
        capacity=5,
    )

    for index in range(1, 6):
        assert monitored.process_frame("s", f"frame-{index}".encode())["status"] == "queued"
    deadline = time.time() + 2
    while monitored.metrics("s")["qwen_completed"] != 5 and time.time() < deadline:
        time.sleep(0.01)

    memos = monitored.broker.drain("s")
    assert len(memos) == 5
    for capture_type, field in (
        ("bio", "organisms"),
        ("substrate", "substrates"),
        ("geomorphology", "geomorphologies"),
    ):
        counts = [
            getattr({capture.type.value: capture for capture in memo.captures}[capture_type], field)[0].count
            for memo in memos
        ]
        # Frames 2–4 are within three queue-2 frames of counted frame 1;
        # frame 5 is four frames later and therefore increments once.
        assert counts == [1, 1, 1, 1, 2]


def test_monitoring_rejects_free_labels_and_uses_unknown_only_with_evidence(tmp_path: Path, valid_frame) -> None:
    class ConstrainedVision(Vision):
        def match_monitoring_tags(self, image_path, candidates):
            return MonitoringTagMatch(
                (CountItem("自由生成标签", 9),),
                (),
                (CountItem("海绵", 3),),
                ("bio",),
            )

        def describe_monitoring_frame(self, image_path, tags, descriptions):
            assert [item.name for item in tags.organisms] == ["未知生物"]
            assert not tags.substrates and not tags.geomorphologies
            assert descriptions == {}
            return "画面中可见未能匹配标签的生物特征。"

    vision = ConstrainedVision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)
    memo = monitored.broker.drain("s")[0]
    assert [item.name for item in memo.captures[0].organisms] == ["未知生物"]
