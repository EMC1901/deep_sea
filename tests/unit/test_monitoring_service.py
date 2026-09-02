from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from deep_sea_explorer.domain.models import CountItem, MonitoringCoordinates, MonitoringTagMatch
from deep_sea_explorer.infrastructure.models.local.adapters import _monitoring_coordinates
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService
from deep_sea_explorer.services.hierarchical_monitoring_knowledge import HierarchicalMonitoringKnowledgeBase


class Vision:
    def __init__(self, release: bool = True, coordinates: list[MonitoringCoordinates | None] | None = None) -> None:
        self.release = release
        self.started: list[str] = []
        self.calls: list[str] = []
        self.coordinates = iter(coordinates or [MonitoringCoordinates.from_text("-13.472523", "-27.161464")])

    def extract_monitoring_coordinates(self, image_path: Path) -> MonitoringCoordinates | None:
        self.calls.append("coordinates")
        try:
            return next(self.coordinates)
        except StopIteration:
            return MonitoringCoordinates.from_text("-13.472523", "-27.161464")

    def select_monitoring_labels(self, image_path: Path, candidates: tuple[str, ...], *, stage: str, maximum: int) -> tuple[str, ...]:
        self.calls.append(stage)
        if stage == "biotic Class" and "海绵类" in candidates:
            return ("海绵类",)
        return candidates[:maximum]

    def match_monitoring_tags(self, image_path: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        self.calls.append("tags")
        return MonitoringTagMatch(
            (CountItem("海绵", 2),) if "海绵" in candidates["organisms"] else (),
            (CountItem("沙泥", 1),) if "沙泥" in candidates["substrates"] else (),
            (CountItem("平坦海床", 1),) if "平坦海床" in candidates["geomorphologies"] else (),
        )

    def describe_monitoring_frame(self, image_path: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        self.calls.append("description")
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
    directory.joinpath("hierarchical_label_knowledge.json").write_text(json.dumps({
        "substrate": [{
            "component": "底质组件", "class": "未固结矿物基质", "subclass": "沙泥基底",
            "group": "细沙", "definition": "以细颗粒沉积物为主的海底基质。",
        }],
        "biotic": [
            {"component": "生物组件", "class": "海绵类", "subclass": "海绵", "definition": "多孔状固着生物。"},
            {"component": "生物组件", "class": "新增类", "subclass": "新生物", "definition": "新增的测试生物。"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return MonitoringService(
        vision, object(), MemorySessionStore(60, 5), MemoryMemoBroker(),
        TempFileStore(tmp_path, 60), CaptureStatsService(), 0.85,
        dino_encoder=dino,
        knowledge_base=HierarchicalMonitoringKnowledgeBase(directory),
        queue_capacity=capacity,
        blur_threshold=1,
        similarity_threshold=similarity_threshold,
    )


@pytest.fixture
def valid_frame(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MonitoringService, "_decode_and_sharpness", staticmethod(lambda _path: 100.0))


def test_monitoring_coordinates_require_signed_in_range_decimals() -> None:
    assert _monitoring_coordinates('{"LO":"-13.472480","LA":"-27.161434"}') == MonitoringCoordinates.from_text(
        "-13.472480", "-27.161434"
    )
    assert _monitoring_coordinates('{"LO":"13.472480","LA":"-27.161434"}') is None
    assert _monitoring_coordinates('{"LO":"-181.0","LA":"-27.161434"}') is None
    assert _monitoring_coordinates('{"LO":null,"LA":"-27.161434"}') is None


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


def test_coordinates_are_extracted_before_tag_matching(tmp_path: Path, valid_frame) -> None:
    vision = Vision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))

    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)

    assert vision.calls == [
        "coordinates",
        "substrate Class",
        "substrate Subclass",
        "substrate Group",
        "biotic Class",
        "biotic Subclass",
        "description",
    ]
    memo = monitored.broker.drain("s")[0]
    assert memo.coordinates == MonitoringCoordinates.from_text("-13.472523", "-27.161464")


def test_monitoring_result_keeps_three_tag_categories_separate(tmp_path: Path, valid_frame) -> None:
    vision = Vision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)
    memo = monitored.broker.drain("s")[0]
    captures = {capture.type.value: capture for capture in memo.captures}
    assert set(captures) == {"bio", "substrate"}
    assert [item.name for item in captures["bio"].organisms] == ["海绵"]
    assert not captures["bio"].substrates and not captures["bio"].geomorphologies
    assert [item.name for item in captures["substrate"].substrates] == ["细沙"]
    assert not captures["substrate"].organisms and not captures["substrate"].geomorphologies


def test_monitoring_statistics_compare_only_adjacent_queue2_coordinates(tmp_path: Path, valid_frame) -> None:
    same = MonitoringCoordinates.from_text("-13.472523", "-27.161464")
    changed = MonitoringCoordinates.from_text("-13.472480", "-27.161434")
    vision = Vision(coordinates=[same, same, same, changed])
    monitored = service(
        tmp_path,
        vision,
        Dino(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ),
        capacity=5,
    )

    for index in range(1, 5):
        assert monitored.process_frame("s", f"frame-{index}".encode())["status"] == "queued"
    deadline = time.time() + 2
    while monitored.metrics("s")["qwen_completed"] != 4 and time.time() < deadline:
        time.sleep(0.01)

    memos = monitored.broker.drain("s")
    assert len(memos) == 4
    assert [memo.statistics["bio"][0].count for memo in memos] == [1, 1, 1, 2]
    assert [memo.statistics["substrate"][0].count for memo in memos] == [1, 1, 1, 2]
    assert [memo.statistics["geomorphology"] for memo in memos] == [(), (), (), ()]
    # Repeated-position frames still preserve their raw recognised tags for the gallery.
    assert [memo.captures[0].organisms[0].count for memo in memos] == [1, 1, 1, 1]


def test_monitoring_statistics_count_new_label_at_the_same_coordinates(tmp_path: Path, valid_frame) -> None:
    class ChangedTagsVision(Vision):
        def __init__(self) -> None:
            same = MonitoringCoordinates.from_text("-13.472523", "-27.161464")
            super().__init__(coordinates=[same, same])
            self.frame = 0

        def select_monitoring_labels(self, image_path, candidates, *, stage, maximum):
            if stage != "biotic Class":
                return super().select_monitoring_labels(image_path, candidates, stage=stage, maximum=maximum)
            self.frame += 1
            return ("海绵类",) if self.frame == 1 else ("海绵类", "新增类")

    vision = ChangedTagsVision()
    monitored = service(tmp_path, vision, Dino([[1, 0], [0, 1]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    assert monitored.process_frame("s", b"two")["status"] == "queued"
    deadline = time.time() + 2
    while monitored.metrics("s")["qwen_completed"] != 2 and time.time() < deadline:
        time.sleep(0.01)

    memos = monitored.broker.drain("s")
    assert memos[-1].statistics["bio"] == (CountItem("海绵", 1), CountItem("新生物", 1))


def test_unreadable_coordinates_do_not_suppress_monitoring_statistics(tmp_path: Path, valid_frame) -> None:
    vision = Vision(coordinates=[None, None])
    monitored = service(tmp_path, vision, Dino([[1, 0], [0, 1]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    assert monitored.process_frame("s", b"two")["status"] == "queued"
    deadline = time.time() + 2
    while monitored.metrics("s")["qwen_completed"] != 2 and time.time() < deadline:
        time.sleep(0.01)

    assert monitored.broker.drain("s")[-1].statistics["bio"][0].count == 2


def test_monitoring_rejects_free_labels_and_uses_unknown_only_with_evidence(tmp_path: Path, valid_frame) -> None:
    class ConstrainedVision(Vision):
        def select_monitoring_labels(self, image_path, candidates, *, stage, maximum):
            return ("自由生成标签",)

        def describe_monitoring_frame(self, image_path, tags, descriptions):
            assert not tags.organisms and not tags.substrates and not tags.geomorphologies
            assert descriptions == {"substrate": [], "biotic": []}
            return "画面中未确认可归类的特征。"

    vision = ConstrainedVision()
    monitored = service(tmp_path, vision, Dino([[1, 0]]))
    assert monitored.process_frame("s", b"one")["status"] == "queued"
    deadline = time.time() + 1
    while monitored.metrics("s")["qwen_completed"] != 1 and time.time() < deadline:
        time.sleep(0.01)
    memo = monitored.broker.drain("s")[0]
    assert memo.captures == ()
