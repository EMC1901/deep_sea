from __future__ import annotations

import logging

import httpx
import pytest

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CountItem, Memo, SessionState
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService
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
    assert (
        "MODEL_MAX_CONCURRENT_REQUESTS must be positive"
        in Settings(model_max_concurrent_requests=0).validate_for_runtime()
    )
    assert (
        "MODEL_MAX_EMBEDDING_TEXTS must be positive"
        in Settings(model_max_embedding_texts=0).validate_for_runtime()
    )


def test_settings_accepts_the_server_model_temp_directory_name() -> None:
    settings = Settings.from_env({"MODEL_TEMP_DIR": "/server/tmp"})

    assert settings.temp_dir.as_posix() == "/server/tmp"


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


def test_remote_client_logs_safe_server_failure_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_id = "c1eabacb-d1ab-4eac-bcff-9fef56e2980f"
    token = "token-that-must-not-be-logged"

    def model_service_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            headers={"X-Request-ID": request_id},
            json={
                "request_id": request_id,
                "error": {"code": "MODEL_NOT_READY", "message": "requested model is not ready"},
            },
        )

    client = RemoteModelClient(
        Settings(
            model_backend=ModelBackend.REMOTE,
            model_service_enabled=True,
            model_service_base_url="http://model-service.test",
            model_service_auth_token=token,
        ),
        transport=httpx.MockTransport(model_service_failure),
    )
    try:
        with caplog.at_level(logging.ERROR), pytest.raises(ModelUnavailableError) as captured:
            client.request("POST", "/vision/evaluate-frame")
    finally:
        client.close()

    assert "status=503" in str(captured.value)
    assert "code=MODEL_NOT_READY" in str(captured.value)
    assert request_id in str(captured.value)
    assert "endpoint=/vision/evaluate-frame" in caplog.text
    assert token not in caplog.text


def test_monitoring_logs_the_failed_pipeline_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        MonitoringService._stage(
            "session-for-logging",
            "evaluate_frame",
            lambda: (_ for _ in ()).throw(RuntimeError("classification failed")),
        )

    assert "session_id=session-for-logging" in caplog.text
    assert "stage=evaluate_frame" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


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
