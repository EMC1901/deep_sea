from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from deep_sea_explorer.domain.models import MonitoringAnalysis
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService


class Vision:
    def __init__(self, release: bool = True) -> None:
        self.release = release
        self.started: list[str] = []

    def analyze_monitoring_frame(self, image_path: Path) -> MonitoringAnalysis:
        self.started.append(image_path.read_bytes().decode("ascii"))
        while not self.release:
            time.sleep(0.002)
        return MonitoringAnalysis(f"画面 {self.started[-1]}")


class Dino:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = iter(values)

    def embed_image(self, _image_path: Path) -> np.ndarray:
        return np.asarray(next(self.values), dtype=np.float32)


def service(tmp_path: Path, vision: Vision, dino: Dino, capacity: int = 2) -> MonitoringService:
    return MonitoringService(
        vision, object(), MemorySessionStore(60, 5), MemoryMemoBroker(),
        TempFileStore(tmp_path, 60), CaptureStatsService(), 0.85,
        dino_encoder=dino, queue_capacity=capacity, blur_threshold=1, similarity_threshold=0.7,
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
    assert response["similarity"] >= 0.7
    assert response["metrics"]["rejected_similar"] == 1


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
