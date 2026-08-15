from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, CountItem, ModelHealth, StreamEvent
from deep_sea_explorer.infrastructure.models.local.errors import (
    InferenceQueueFull,
    InferenceTimeout,
    ModelOutputInvalid,
)
from deep_sea_explorer.model_service import app_factory as service_api
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.models.remote.embedding import RemoteEmbeddingGateway
from deep_sea_explorer.infrastructure.models.remote.image import RemoteImageGateway
from deep_sea_explorer.infrastructure.models.remote.vision import RemoteVisionGateway


TEST_TEMP = Path(
    os.getenv("MODEL_TEST_TEMP_DIR")
    or os.getenv("TEMP_DIR")
    or Path(__file__).resolve().parents[2] / ".deep-sea-explorer-tmp"
)


@pytest.fixture(autouse=True)
def isolate_test_temp_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give each API-contract test an empty, private temporary directory."""
    isolated = TEST_TEMP / f"pytest-model-api-{uuid.uuid4().hex}"
    isolated.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(sys.modules[__name__], "TEST_TEMP", isolated)
    try:
        yield isolated
    finally:
        shutil.rmtree(isolated, ignore_errors=True)


class FakeVision:
    def __init__(self) -> None:
        self.uploads: list[Path] = []

    def health(self) -> ModelHealth:
        return ModelHealth(True, "ready")

    def describe_video(self, path: Path) -> str:
        assert path.is_file()
        self.uploads.append(path)
        return "video description"

    def evaluate_frame(self, path: Path) -> CaptureDecision:
        assert path.is_file()
        self.uploads.append(path)
        return CaptureDecision(
            True,
            True,
            CaptureType.BIO,
            "deep sea fish",
            organisms=(CountItem("fish", 1),),
        )

    def answer(self, question: str):
        assert question == "what happened?"
        yield StreamEvent(StreamEventType.CHUNK, text="answer")
        yield StreamEvent(StreamEventType.FINAL, text="answer")

    def summarize_report(self, material: dict[str, object]) -> str:
        assert material == {"memos": []}
        return "report summary"


class FakeImage:
    def health(self) -> ModelHealth:
        return ModelHealth(False, "not_loaded")

    def generate(self, prompt: str) -> bytes:
        assert prompt == "deep sea illustration"
        return b"\xff\xd8\xff\xd9"


class FakeEmbedding:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def health(self) -> ModelHealth:
        return ModelHealth(True, "ready")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def service_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_backend": ModelBackend.LOCAL,
        "model_service_auth_token": "test-token-not-a-secret",
        "qwen_model_path": "/models/qwen",
        "image_model_path": "/models/image",
        "memo_embedding_model_path": "/models/gte",
        "rag_embedding_model_path": "/models/minilm",
        "temp_dir": TEST_TEMP,
    }
    return Settings(
        **(values | overrides)  # type: ignore[arg-type]
    )


def headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer test-token-not-a-secret",
        "X-Model-API-Version": "1",
        "X-Request-ID": str(uuid.uuid4()),
    }


def write_media() -> tuple[Path, Path]:
    image_path = TEST_TEMP / f"image-{uuid.uuid4().hex}.jpg"
    video_path = TEST_TEMP / f"video-{uuid.uuid4().hex}.mp4"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 32))
    assert writer.isOpened()
    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return image_path, video_path


@pytest.fixture
def fake_container() -> SimpleNamespace:
    return SimpleNamespace(
        vision=FakeVision(), image=FakeImage(), memo_embedding=FakeEmbedding(2), rag_embedding=FakeEmbedding(3)
    )


def test_model_service_v1_routes_follow_contract_and_cleanup_uploads(
    monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    app = service_api.create_app(service_settings(), fake_container)
    client = app.test_client()
    request_headers = headers()

    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json["models"] == {
        "qwen": "ready",
        "image": "not_loaded",
        "memo": "ready",
        "rag": "ready",
    }

    describe = client.post(
        "/v1/vision/describe-video",
        headers=request_headers,
        data={"video": (io.BytesIO(b"video"), "clip.mp4")},
    )
    assert describe.status_code == 200
    assert describe.json["text"] == "video description"

    frame = client.post(
        "/v1/vision/evaluate-frame",
        headers=headers(),
        data={"image": (io.BytesIO(b"image"), "frame.jpg")},
    )
    assert frame.status_code == 200
    assert frame.json["decision"]["category"] == "bio"

    answer = client.post(
        "/v1/vision/answer",
        headers=headers(),
        json={"question": "what happened?"},
    )
    assert answer.content_type.startswith("application/x-ndjson")
    assert answer.data.splitlines() == [
        b'{"type":"delta","text":"answer"}',
        b'{"type":"done","usage":{"output_chars":6}}',
    ]

    summary = client.post("/v1/vision/summarize-report", headers=headers(), json={"material": {"memos": []}})
    assert summary.json["text"] == "report summary"
    image = client.post("/v1/images/generate", headers=headers(), json={"prompt": "deep sea illustration"})
    assert image.content_type.startswith("image/jpeg") and image.data.startswith(b"\xff\xd8")
    memo = client.post("/v1/embeddings", headers=headers(), json={"model": "memo", "texts": ["one"]})
    rag = client.post("/v1/embeddings", headers=headers(), json={"model": "rag", "texts": ["one"]})
    assert (memo.json["dimension"], rag.json["dimension"]) == (2, 3)

    assert all(not path.exists() for path in fake_container.vision.uploads)
    assert not list(TEST_TEMP.iterdir())
    assert describe.headers["X-Request-ID"] == request_headers["X-Request-ID"]


def test_model_service_logs_request_id_without_request_secrets(
    caplog: pytest.LogCaptureFixture, fake_container: SimpleNamespace
) -> None:
    app = service_api.create_app(service_settings(), fake_container)
    request_headers = headers()
    with caplog.at_level(logging.INFO):
        response = app.test_client().get("/v1/health", headers=request_headers)

    assert response.status_code == 200
    assert request_headers["X-Request-ID"] in caplog.text
    assert "test-token-not-a-secret" not in caplog.text


def test_model_service_logs_handled_model_failure_without_request_secrets(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fake_container: SimpleNamespace,
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    fake_container.vision.evaluate_frame = lambda _: (_ for _ in ()).throw(
        ModelOutputInvalid("sensitive raw model output")
    )
    app = service_api.create_app(service_settings(), fake_container)
    request_headers = headers()

    with caplog.at_level(logging.ERROR):
        response = app.test_client().post(
            "/v1/vision/evaluate-frame",
            headers=request_headers,
            data={"image": (io.BytesIO(b"image"), "frame.jpg")},
        )

    assert response.status_code == 503
    assert response.json["error"]["code"] == "MODEL_NOT_READY"
    assert request_headers["X-Request-ID"] in caplog.text
    assert "error_type=ModelOutputInvalid" in caplog.text
    assert "test-token-not-a-secret" not in caplog.text


def test_model_service_logs_streaming_model_failure_without_request_secrets(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fake_container: SimpleNamespace,
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    fake_container.vision.answer = lambda *_: (_ for _ in ()).throw(
        ModelOutputInvalid("internal streaming failure")
    )
    app = service_api.create_app(service_settings(), fake_container)
    request_headers = headers()

    with caplog.at_level(logging.ERROR):
        response = app.test_client().post(
            "/v1/vision/answer",
            headers=request_headers,
            json={"question": "what happened?"},
        )

    assert response.status_code == 200
    assert b'"type":"error"' in response.data
    assert request_headers["X-Request-ID"] in caplog.text
    assert "error_type=ModelOutputInvalid" in caplog.text
    assert "test-token-not-a-secret" not in caplog.text


def test_remote_client_works_against_the_fake_model_service_contract(
    fake_container: SimpleNamespace,
) -> None:
    app = service_api.create_app(service_settings(), fake_container)
    remote_settings = Settings(
        model_backend=ModelBackend.REMOTE,
        model_service_enabled=True,
        model_service_base_url="http://model-service.test",
        model_service_auth_token="test-token-not-a-secret",
    )
    client = RemoteModelClient(remote_settings, transport=httpx.WSGITransport(app=app))
    vision = RemoteVisionGateway(client)
    image = RemoteImageGateway(client)
    memo = RemoteEmbeddingGateway(client, "memo")
    rag = RemoteEmbeddingGateway(client, "rag")
    image_path, video_path = write_media()
    try:
        assert vision.health().ready is True
        assert vision.describe_video(video_path) == "video description"
        assert vision.evaluate_frame(image_path).category is CaptureType.BIO
        assert [event.type for event in vision.answer("what happened?")] == [
            StreamEventType.CHUNK,
            StreamEventType.FINAL,
        ]
        assert vision.summarize_report({"memos": []}) == "report summary"
        assert image.generate("deep sea illustration").startswith(b"\xff\xd8")
        assert len(memo.embed(["one"])[0]) == 2
        assert len(rag.embed(["one"])[0]) == 3
    finally:
        client.close()
        image_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)


def test_model_service_rejects_unauthorized_and_invalid_requests(fake_container: SimpleNamespace) -> None:
    app = service_api.create_app(service_settings(), fake_container)
    client = app.test_client()

    unauthorized = client.post("/v1/images/generate", json={"prompt": "deep sea"})
    assert unauthorized.status_code == 401
    assert unauthorized.json["error"]["code"] == "UNAUTHORIZED"
    assert "test-token" not in str(unauthorized.json)

    wrong_token = client.post(
        "/v1/images/generate",
        headers={"Authorization": "Bearer wrong", "X-Model-API-Version": "1"},
        json={"prompt": "deep sea"},
    )
    assert wrong_token.status_code == 401

    invalid = client.post("/v1/images/generate", headers=headers(), json={"prompt": "x", "seed": 1})
    assert invalid.status_code == 400
    assert invalid.json["error"]["code"] == "INVALID_INPUT"

    wrong_media = client.post(
        "/v1/vision/evaluate-frame",
        headers=headers(),
        data={"image": (io.BytesIO(b"not a jpeg"), "frame.png")},
    )
    assert wrong_media.status_code == 415
    assert wrong_media.json["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    missing_video = client.post("/v1/vision/describe-video", headers=headers())
    empty_question = client.post(
        "/v1/vision/answer",
        headers=headers(),
        json={"question": ""},
    )
    multipart_question = client.post(
        "/v1/vision/answer",
        headers=headers(),
        data={"question": "hello", "video": (io.BytesIO(b"video"), "clip.mp4")},
    )
    assert (missing_video.status_code, empty_question.status_code, multipart_question.status_code) == (
        400,
        400,
        400,
    )


def test_model_service_rejects_undecodable_upload_and_removes_it(
    fake_container: SimpleNamespace,
) -> None:
    app = service_api.create_app(service_settings(), fake_container)
    response = app.test_client().post(
        "/v1/vision/evaluate-frame",
        headers=headers(),
        data={"image": (io.BytesIO(b"not a jpeg"), "frame.jpg")},
    )

    assert response.status_code == 422
    assert response.json["error"]["code"] == "UNPROCESSABLE_INPUT"
    assert not list(TEST_TEMP.iterdir())


def test_model_service_maps_queue_full_without_leaking_exception_details(
    monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    fake_container.vision.describe_video = lambda _: (_ for _ in ()).throw(InferenceQueueFull("internal"))
    app = service_api.create_app(service_settings(), fake_container)
    response = app.test_client().post(
        "/v1/vision/describe-video",
        headers=headers(),
        data={"video": (io.BytesIO(b"video"), "clip.mp4")},
    )

    assert response.status_code == 429
    assert response.json["error"] == {"code": "QUEUE_FULL", "message": "model inference queue is full"}


def test_model_service_maps_timeout_and_unavailable_failures(
    monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    app = service_api.create_app(service_settings(), fake_container)
    client = app.test_client()
    data = {"video": (io.BytesIO(b"video"), "clip.mp4")}

    fake_container.vision.describe_video = lambda _: (_ for _ in ()).throw(InferenceTimeout("internal"))
    timeout = client.post("/v1/vision/describe-video", headers=headers(), data=data)
    assert timeout.status_code == 504
    assert timeout.json["error"]["code"] == "MODEL_TIMEOUT"

    fake_container.vision.describe_video = lambda _: (_ for _ in ()).throw(ModelUnavailableError("internal"))
    unavailable = client.post(
        "/v1/vision/describe-video",
        headers=headers(),
        data={"video": (io.BytesIO(b"video"), "clip.mp4")},
    )
    assert unavailable.status_code == 503
    assert unavailable.json["error"]["code"] == "MODEL_NOT_READY"


def test_model_service_rejects_bad_embedding_and_image_model_outputs(fake_container: SimpleNamespace) -> None:
    app = service_api.create_app(service_settings(model_max_embedding_texts=1), fake_container)
    client = app.test_client()

    too_many = client.post(
        "/v1/embeddings", headers=headers(), json={"model": "memo", "texts": ["one", "two"]}
    )
    assert too_many.status_code == 400

    fake_container.memo_embedding.embed = lambda _: [[float("nan")]]
    invalid_vectors = client.post(
        "/v1/embeddings", headers=headers(), json={"model": "memo", "texts": ["one"]}
    )
    assert invalid_vectors.status_code == 500
    assert invalid_vectors.json["error"]["code"] == "INTERNAL_ERROR"

    fake_container.image.generate = lambda _: b"not-a-jpeg"
    invalid_image = client.post(
        "/v1/images/generate", headers=headers(), json={"prompt": "deep sea illustration"}
    )
    assert invalid_image.status_code == 500
    assert invalid_image.json["error"]["code"] == "INTERNAL_ERROR"


def test_model_service_removes_upload_when_streaming_client_disconnects(
    monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace
) -> None:
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    app = service_api.create_app(service_settings(), fake_container)
    response = app.test_client().open(
        "/v1/vision/answer",
        method="POST",
        headers=headers(),
        json={"question": "what happened?"},
        buffered=False,
    )
    response.close()

    assert not list(TEST_TEMP.iterdir())


def test_model_service_actual_video_validation_accepts_a_readable_file() -> None:
    path = TEST_TEMP / f"video-{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 32))
    assert writer.isOpened()
    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    try:
        service_api._verify_upload(path, "video")
    finally:
        path.unlink(missing_ok=True)


def test_model_service_monitoring_frame_returns_three_categories(monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace) -> None:
    from deep_sea_explorer.domain.models import MonitoringAnalysis
    fake_container.vision.analyze_monitoring_frame = lambda _: MonitoringAnalysis(
        "沙泥底质表面可见海绵，底面较为平坦。",
        organisms=(CountItem("海绵", 1),),
        substrates=(CountItem("沙泥", 1),),
        geomorphologies=(CountItem("平坦海床", 1),),
    )
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    app = service_api.create_app(service_settings(), fake_container)
    response = app.test_client().post("/v1/vision/analyze-monitoring-frame", headers=headers(), data={"image": (io.BytesIO(b"image"), "frame.jpg")})
    assert response.status_code == 200
    assert {"description", "organisms", "substrates", "geomorphologies"}.issubset(response.json)
    assert response.json["substrates"][0]["name"] == "沙泥"
    assert response.json["geomorphologies"][0]["name"] == "平坦海床"


def test_model_service_two_pass_monitoring_endpoints(monkeypatch: pytest.MonkeyPatch, fake_container: SimpleNamespace) -> None:
    from deep_sea_explorer.domain.models import MonitoringTagMatch

    seen: dict[str, object] = {}

    def match(_: Path, candidates: dict[str, tuple[str, ...]]) -> MonitoringTagMatch:
        seen["candidates"] = candidates
        return MonitoringTagMatch(organisms=(CountItem("海绵", 1),), unknown_categories=("substrate",))

    def describe(_: Path, tags: MonitoringTagMatch, descriptions: dict[str, str]) -> str:
        seen["tags"] = tags
        seen["descriptions"] = descriptions
        return "沙泥底质上可见海绵。"

    fake_container.vision.match_monitoring_tags = match
    fake_container.vision.describe_monitoring_frame = describe
    monkeypatch.setattr(service_api, "_verify_upload", lambda path, kind: None)
    app = service_api.create_app(service_settings(), fake_container)
    client = app.test_client()
    candidates = {"organisms": ["海绵"], "substrates": [], "geomorphologies": []}
    matched = client.post(
        "/v1/vision/match-monitoring-tags",
        headers=headers(),
        data={"image": (io.BytesIO(b"image"), "frame.jpg"), "candidates": json.dumps(candidates)},
    )
    assert matched.status_code == 200
    assert matched.json["match"]["organisms"] == [{"name": "海绵", "count": 1}]
    assert seen["candidates"] == {"organisms": ("海绵",), "substrates": (), "geomorphologies": ()}

    tags = {"organisms": [{"name": "海绵", "count": 1}], "substrates": [], "geomorphologies": [], "unknown_categories": []}
    described = client.post(
        "/v1/vision/describe-monitoring-frame",
        headers=headers(),
        data={
            "image": (io.BytesIO(b"image"), "frame.jpg"),
            "tags": json.dumps(tags),
            "descriptions": json.dumps({"海绵": "附着于底质表面的多孔生物。"}),
        },
    )
    assert described.status_code == 200
    assert described.json["description"] == "沙泥底质上可见海绵。"
    assert seen["descriptions"] == {"海绵": "附着于底质表面的多孔生物。"}
