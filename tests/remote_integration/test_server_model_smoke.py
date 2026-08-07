"""Development-machine tests that use an explicitly enabled SSH tunnel."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.models.remote.embedding import RemoteEmbeddingGateway
from deep_sea_explorer.infrastructure.models.remote.image import RemoteImageGateway
from deep_sea_explorer.infrastructure.models.remote.vision import RemoteVisionGateway


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REMOTE_MODEL_TESTS") != "1",
    reason="requires an explicitly enabled SSH tunnel and remote model authorization",
)
pytestmark = [
    pytestmark,
    pytest.mark.allow_loopback_network,
]


def _settings() -> Settings:
    source = Settings.from_env()
    if not source.model_service_auth_token:
        pytest.fail("MODEL_SERVICE_AUTH_TOKEN is required when RUN_REMOTE_MODEL_TESTS=1")
    return replace(source, model_backend=ModelBackend.REMOTE, model_service_enabled=True)


def _media(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    image = root / f"remote-model-{uuid.uuid4().hex}.jpg"
    video = root / f"remote-model-{uuid.uuid4().hex}.mp4"
    import numpy as np

    assert cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 32))
    assert writer.isOpened()
    # Qwen3-VL requires more than its temporal factor of two video frames.
    for _ in range(4):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return image, video


def test_development_machine_reaches_all_server_model_capabilities() -> None:
    settings = _settings()
    client = RemoteModelClient(settings)
    vision = RemoteVisionGateway(client)
    image_gateway = RemoteImageGateway(client)
    memo = RemoteEmbeddingGateway(client, "memo")
    rag = RemoteEmbeddingGateway(client, "rag")
    image_path, video_path = _media(Path.cwd() / ".deep-sea-explorer-tmp")
    try:
        assert vision.health().ready is True
        assert vision.describe_video(video_path).strip()
        assert vision.evaluate_frame(image_path).category in {CaptureType.BIO, CaptureType.ENV}
        events = list(vision.answer(video_path, "What changes occur in this video?"))
        assert any(event.type is StreamEventType.CHUNK and event.text for event in events)
        assert events[-1].type is StreamEventType.FINAL
        assert vision.summarize_report({"memos": [], "chats": []}).strip()
        assert image_gateway.generate("scientific illustration of a deep sea vent").startswith(b"\xff\xd8")
        assert len(memo.embed(["deep sea observation"])[0]) == 768
        assert len(rag.embed(["deep sea observation"])[0]) == 384
    finally:
        client.close()
        image_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
