from __future__ import annotations

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CountItem, Memo, SessionState
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.rag_service import TextChunker


def test_settings_validate_modes_without_loading_models() -> None:
    assert Settings(model_backend=ModelBackend.FAKE).validate_for_runtime() == []
    assert (
        Settings(
            model_backend=ModelBackend.REMOTE, model_service_enabled=False
        ).validate_for_runtime()
        == []
    )
    assert (
        "remote mode requires a real MODEL_SERVICE_BASE_URL"
        in Settings(
            model_backend=ModelBackend.REMOTE, model_service_enabled=True
        ).validate_for_runtime()
    )
    assert len(Settings(model_backend=ModelBackend.LOCAL).validate_for_runtime()) == 4


def test_disabled_remote_client_never_attempts_network() -> None:
    client = RemoteModelClient(
        Settings(model_backend=ModelBackend.REMOTE, model_service_enabled=False)
    )
    try:
        client.request("GET", "/health")
    except Exception as error:
        assert "disabled" in str(error)
    else:  # pragma: no cover
        raise AssertionError("disabled client must not issue a request")


def test_memos_and_session_stats_are_isolated() -> None:
    broker = MemoryMemoBroker()
    broker.publish(Memo("12:00", "A", "a"))
    broker.publish(Memo("12:00", "B", "b"))
    assert [memo.content for memo in broker.drain("a")] == ["A"]
    assert [memo.content for memo in broker.drain("b")] == ["B"]

    sessions = MemorySessionStore(60, 2)
    stats = CaptureStatsService()
    stats.update(sessions.get("a"), CaptureType.BIO, (CountItem("虾", 2),))
    stats.update(sessions.get("b"), CaptureType.BIO, (CountItem("虾", 1),))
    assert sessions.get("a").cumulative_stats["bio"] == {"虾": 2}
    assert sessions.get("b").cumulative_stats["bio"] == {"虾": 1}


def test_capture_stats_and_chunker_keep_documented_rules() -> None:
    state = SessionState()
    stats = CaptureStatsService()
    result = stats.update(
        state, CaptureType.ENV, (CountItem("海底沉积物", 1), CountItem("第二项", 1))
    )
    assert result == (CountItem("海底沉积物", 1),)
    assert stats.update(state, CaptureType.BIO, (CountItem("未知", 9), CountItem("鱼", -1))) == ()
    assert TextChunker(10, 2).split("短文本") == []
    assert TextChunker(25, 5).split("a" * 30)
