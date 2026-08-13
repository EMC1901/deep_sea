from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import httpx

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.models.remote.embedding import RemoteEmbeddingGateway
from deep_sea_explorer.infrastructure.models.remote.image import RemoteImageGateway
from deep_sea_explorer.infrastructure.models.remote.vision import RemoteVisionGateway


class FakeServerModelApi:
    """S2 契约假服务：只验证 HTTP 边界，绝不加载模型或访问网络。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        request_id = request.headers["x-request-id"]
        headers = {"X-Request-ID": request_id}
        path = request.url.path

        if path == "/v1/health":
            return httpx.Response(
                200,
                headers=headers,
                json={
                    "request_id": request_id,
                    "status": "ok",
                    "models": {"qwen": "ready", "image": "ready", "memo": "ready", "rag": "ready"},
                },
                request=request,
            )
        if path == "/v1/vision/describe-video":
            return self._json(request, {"request_id": request_id, "text": "视频描述"}, headers)
        if path == "/v1/vision/evaluate-frame":
            return self._json(
                request,
                {
                    "request_id": request_id,
                    "decision": {
                        "is_deepsea": True,
                        "is_typical": True,
                        "category": "bio",
                        "description": "测试虾",
                        "organisms": [{"name": "虾", "count": 1}],
                        "env_features": [],
                    },
                },
                headers,
            )
        if path == "/v1/vision/answer":
            return httpx.Response(
                200,
                headers={**headers, "Content-Type": "application/x-ndjson"},
                content='{"type":"delta","text":"答"}\n{"type":"done","usage":{"output_chars":1}}\n'.encode(),
                request=request,
            )
        if path == "/v1/vision/summarize-report":
            return self._json(request, {"request_id": request_id, "text": "报告摘要"}, headers)
        if path == "/v1/images/generate":
            return httpx.Response(
                200,
                headers={**headers, "Content-Type": "image/jpeg"},
                content=b"\xff\xd8\xff\xd9",
                request=request,
            )
        if path == "/v1/embeddings":
            body = json.loads(request.content)
            texts = body["texts"]
            return self._json(
                request,
                {
                    "request_id": request_id,
                    "model": body["model"],
                    "normalized": True,
                    "dimension": 2,
                    "embeddings": [[0.6, 0.8] for _ in texts],
                },
                headers,
            )
        return self._json(request, {"error": {"code": "NOT_FOUND"}}, headers, status_code=404)

    @staticmethod
    def _json(
        request: httpx.Request,
        body: dict[str, object],
        headers: dict[str, str],
        *,
        status_code: int = 200,
    ) -> httpx.Response:
        return httpx.Response(status_code, headers=headers, json=body, request=request)


def enabled_settings() -> Settings:
    return Settings(
        model_backend=ModelBackend.REMOTE,
        model_service_enabled=True,
        model_service_base_url="https://models.test",
        model_service_auth_token="test-token-not-a-secret",
    )


def test_remote_clients_follow_frozen_v1_contract() -> None:
    fake_api = FakeServerModelApi()
    client = RemoteModelClient(enabled_settings(), transport=httpx.MockTransport(fake_api))
    vision = RemoteVisionGateway(client)
    image = RemoteImageGateway(client)
    memo = RemoteEmbeddingGateway(client, "memo")
    rag = RemoteEmbeddingGateway(client, "rag")
    frame = Path(__file__)

    assert vision.health().ready is True
    assert vision.describe_video(Path(__file__)) == "视频描述"
    decision = vision.evaluate_frame(frame)
    assert decision.category is CaptureType.BIO
    assert decision.organisms[0].name == "虾"
    assert [(event.type, event.text) for event in vision.answer("有什么？")] == [
        (StreamEventType.CHUNK, "答"),
        (StreamEventType.FINAL, ""),
    ]
    assert vision.summarize_report({"memos": [], "chats": []}) == "报告摘要"
    assert image.generate("深海科研插画").startswith(b"\xff\xd8")
    assert memo.embed(["一条记录"]) == [[0.6, 0.8]]
    assert rag.embed(["一条检索文本"]) == [[0.6, 0.8]]
    client.close()

    assert [request.url.path for request in fake_api.requests] == [
        "/v1/health",
        "/v1/vision/describe-video",
        "/v1/vision/evaluate-frame",
        "/v1/vision/answer",
        "/v1/vision/summarize-report",
        "/v1/images/generate",
        "/v1/embeddings",
        "/v1/embeddings",
    ]
    assert all(request.headers["authorization"] == "Bearer test-token-not-a-secret" for request in fake_api.requests)
    assert all(request.headers["x-model-api-version"] == "1" for request in fake_api.requests)
    request_ids = [request.headers["x-request-id"] for request in fake_api.requests]
    assert len(request_ids) == len(set(request_ids))
    for request_id in request_ids:
        UUID(request_id)

    multipart_requests = fake_api.requests[1:3]
    assert b'name="video"' in multipart_requests[0].content
    assert b'name="image"' in multipart_requests[1].content
    assert json.loads(fake_api.requests[3].content) == {"question": "有什么？"}
    assert json.loads(fake_api.requests[4].content) == {"material": {"memos": [], "chats": []}}
    assert json.loads(fake_api.requests[6].content)["model"] == "memo"
    assert json.loads(fake_api.requests[7].content)["model"] == "rag"


def test_embedding_client_rejects_contract_mismatch() -> None:
    def invalid_embeddings(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": request.headers["x-request-id"],
                "model": "memo",
                "normalized": False,
                "dimension": 2,
                "embeddings": [[0.6, 0.8]],
            },
            request=request,
        )

    client = RemoteModelClient(enabled_settings(), transport=httpx.MockTransport(invalid_embeddings))
    try:
        RemoteEmbeddingGateway(client, "memo").embed(["测试"])
    except ModelUnavailableError as error:
        assert "API contract" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid embedding response must be rejected")
    finally:
        client.close()
