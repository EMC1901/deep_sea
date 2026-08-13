from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from flask import Response

from deep_sea_explorer.api.app_factory import create_app
from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_fake_container
from deep_sea_explorer.domain.models import Memo


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(model_backend=ModelBackend.FAKE)
    container = build_fake_container(settings)
    monkeypatch.setattr(
        container.reports, "generate", lambda material: Path("contract_report_never_created.pdf")
    )
    import deep_sea_explorer.api.routes.reports as reports_route

    monkeypatch.setattr(
        reports_route,
        "send_file",
        lambda *args, **kwargs: Response(
            b"%PDF-1.4\n",
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Report.pdf"},
        ),
    )
    app = create_app(settings, container)
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_keeps_current_top_level_fields(client) -> None:
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["model"] == "loaded"
    assert body["rag"] == "no_documents"
    assert body["documents"] == 0
    assert body["image_retrieval"] == {
        "enabled": False,
        "ready": False,
        "detail": "image retrieval is disabled",
        "index_size": 0,
        "embedding_dimension": 0,
    }


def test_video_analyze_streams_text_only_questions(client) -> None:
    response = client.post(
        "/videoanalyze",
        json={"question": "你好，介绍一下你自己。"},
        headers={"X-Session-ID": "contract"},
    )
    events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
    assert response.mimetype == "application/x-ndjson"
    assert events[-1] == {"type": "final", "text": "固定回答"}


def test_video_analyze_rejects_question_frame_uploads(client) -> None:
    response = client.post(
        "/videoanalyze",
        data={"question": "你好", "video": (io.BytesIO(b"frame"), "frame.jpg")},
        headers={"X-Session-ID": "contract"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "questions accept text only"


def test_memos_are_session_isolated(client) -> None:
    container = client.application.extensions["container"]
    container.memos.publish(Memo("12:00:00", "A", "a"))
    container.memos.publish(Memo("12:00:00", "B", "b"))
    assert (
        client.get("/memos", headers={"X-Session-ID": "a"}).get_json()["memos"][0]["content"] == "A"
    )
    assert (
        client.get("/memos", headers={"X-Session-ID": "b"}).get_json()["memos"][0]["content"] == "B"
    )


def test_rag_validation_and_report_download_contract(client) -> None:
    assert (
        client.post("/rag/upload", data={}, content_type="multipart/form-data").status_code == 400
    )
    assert (
        client.post(
            "/rag/upload",
            data={"file": (io.BytesIO(b"x"), "notes.txt")},
            content_type="multipart/form-data",
        ).status_code
        == 400
    )
    report = client.post("/generate_report", json={"memos": [], "chats": []})
    assert report.status_code == 200
    assert report.mimetype == "application/pdf"
    assert "attachment" in report.headers["Content-Disposition"]


def test_report_file_is_removed_only_after_the_response_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Mock()
    settings = Settings(model_backend=ModelBackend.FAKE)
    container = build_fake_container(settings)
    monkeypatch.setattr(container.reports, "generate", lambda _: target)
    import deep_sea_explorer.api.routes.reports as reports_route

    monkeypatch.setattr(
        reports_route,
        "send_file",
        lambda *args, **kwargs: Response(b"%PDF-1.4\n", mimetype="application/pdf"),
    )
    app = create_app(settings, container)
    response = app.test_client().post("/generate_report", json={})

    target.unlink.assert_not_called()
    response.close()
    target.unlink.assert_called_once_with(missing_ok=True)
