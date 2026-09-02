from __future__ import annotations

import json
from pathlib import Path

import httpx

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.infrastructure.models.gguf.vision import LlamaCppVisionGateway


def settings(tmp_path: Path) -> Settings:
    model = tmp_path / "qwen3.6-27b.gguf"
    projector = tmp_path / "mmproj-BF16.gguf"
    model.write_bytes(b"GGUF")
    projector.write_bytes(b"GGUF")
    return Settings(
        model_backend=ModelBackend.GGUF,
        gguf_server_url="http://127.0.0.1:19001",
        gguf_model_path=str(model),
        gguf_mmproj_path=str(projector),
        memo_embedding_model_path="/models/gte",
        rag_embedding_model_path="/models/minilm",
    )


def test_gguf_health_requires_server_reported_vision(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/props"
        return httpx.Response(200, json={"modalities": {"vision": True}})

    gateway = LlamaCppVisionGateway(settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert gateway.health().ready is True
    assert gateway.health().text_ready is True
    assert gateway.health().vision_ready is True


def test_gguf_tag_match_uses_multimodal_direct_response(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"organisms":[{"name":"Fish","count":1}],"substrates":[],"geomorphologies":[],"unknown_categories":[]}'}}]},
        )

    gateway = LlamaCppVisionGateway(settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = gateway.match_monitoring_tags(image, {"organisms": ("Fish",), "substrates": (), "geomorphologies": ()})

    assert result.organisms[0].name == "Fish"
    assert observed["temperature"] == 0
    assert observed["chat_template_kwargs"] == {"enable_thinking": False}
    content = observed["messages"][0]["content"]  # type: ignore[index]
    assert content[0]["type"] == "image_url"  # type: ignore[index]
