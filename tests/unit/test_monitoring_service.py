from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, CountItem
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService


@pytest.fixture
def monitoring_temp() -> Iterator[Path]:
    root = Path.cwd() / ".deep-sea-explorer-tmp" / f"monitoring-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["new deep-sea scene"]
        return [[1.0, 0.0]]


def monitoring_service(
    tmp_path: Path,
    vision: object,
) -> tuple[MonitoringService, MemorySessionStore, MemoryMemoBroker, TempFileStore]:
    sessions = MemorySessionStore(ttl_seconds=60, max_sessions=2)
    broker = MemoryMemoBroker()
    files = TempFileStore(tmp_path / "monitoring", ttl_seconds=60)
    service = MonitoringService(
        vision,
        FakeEmbedding(),
        sessions,
        broker,
        files,
        CaptureStatsService(),
        threshold=0.95,
    )
    return service, sessions, broker, files


def prepare_video(tmp_path: Path, sessions: MemorySessionStore) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    sessions.get("session").latest_video = str(video)
    return video


def write_fake_frame(_: Path, output: Path) -> None:
    output.write_bytes(b"\xff\xd8frame")


def test_monitoring_publishes_description_when_capture_classification_fails(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    monitoring_temp: Path,
) -> None:
    class FailingVision:
        def describe_video(self, _: Path) -> str:
            return "new deep-sea scene"

        def evaluate_frame(self, _: Path) -> CaptureDecision:
            raise ModelUnavailableError("invalid frame decision")

    service, sessions, broker, files = monitoring_service(monitoring_temp, FailingVision())
    video = prepare_video(monitoring_temp, sessions)
    monkeypatch.setattr(MonitoringService, "_last_frame", staticmethod(write_fake_frame))

    with caplog.at_level(logging.ERROR):
        memo = service.process_session("session")

    assert memo is not None
    assert memo.content == "new deep-sea scene"
    assert memo.capture is None
    assert broker.drain("session") == [memo]
    assert sessions.get("session").last_analyzed_video == str(video)
    assert "capture_degraded session_id=session stage=evaluate_frame" in caplog.text
    assert not list((files.root / "sessions" / "session").iterdir())


def test_monitoring_keeps_capture_and_statistics_when_classification_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    monitoring_temp: Path,
) -> None:
    class SuccessfulVision:
        def describe_video(self, _: Path) -> str:
            return "new deep-sea scene"

        def evaluate_frame(self, _: Path) -> CaptureDecision:
            return CaptureDecision(
                True,
                True,
                CaptureType.ENV,
                "rocky seafloor",
                env_features=(CountItem("rock", 1),),
            )

    service, sessions, broker, _ = monitoring_service(monitoring_temp, SuccessfulVision())
    prepare_video(monitoring_temp, sessions)
    monkeypatch.setattr(MonitoringService, "_last_frame", staticmethod(write_fake_frame))

    memo = service.process_session("session")

    assert memo is not None
    assert memo.capture is not None
    assert memo.capture.type is CaptureType.ENV
    assert memo.capture.env_features == (CountItem("rock", 1),)
    assert memo.capture.image_data_uri.startswith("data:image/jpeg;base64,")
    assert sessions.get("session").cumulative_stats["env"] == {"rock": 1}
    assert broker.drain("session") == [memo]
