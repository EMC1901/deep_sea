from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, CountItem, ModelHealth, StreamEvent

from .client import RemoteModelClient


class RemoteVisionGateway:
    def __init__(self, client: RemoteModelClient) -> None:
        self.client = client

    def health(self) -> ModelHealth:
        if not self.client.settings.model_service_enabled:
            return ModelHealth(False, "remote service disabled")
        try:
            data = self.client.request("GET", "/health").json()
            return ModelHealth(bool(data.get("ready", data.get("status") == "ok")), "remote")
        except ModelUnavailableError as error:
            return ModelHealth(False, type(error).__name__)

    @staticmethod
    def _file(path: Path) -> dict[str, tuple[str, bytes, str]]:
        return {"file": (path.name, path.read_bytes(), "application/octet-stream")}

    def describe_video(self, video_path: Path) -> str:
        return str(
            self.client.request(
                "POST", "/vision/describe-video", files=self._file(video_path)
            ).json()["text"]
        )

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        data = self.client.request(
            "POST", "/vision/evaluate-frame", files=self._file(image_path)
        ).json()
        category = CaptureType(data.get("category", "env"))

        def items(values):
            return tuple(
                CountItem(str(item.get("name", "")), int(item.get("count", 1)))
                for item in values or []
            )

        return CaptureDecision(
            bool(data.get("is_deepsea")),
            bool(data.get("is_typical")),
            category,
            str(data.get("description", "")),
            items(data.get("organisms")),
            items(data.get("env_features")),
        )

    def answer(self, video_path: Path, question: str) -> Iterator[StreamEvent]:
        response = self.client.request(
            "POST",
            "/vision/answer",
            files={
                **self._file(video_path),
                "question": ("question.txt", question.encode(), "text/plain"),
            },
        )
        for raw_line in response.text.splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
                yield StreamEvent(
                    StreamEventType(event["type"]),
                    text=str(event.get("text", "")),
                    content=str(event.get("content", "")),
                    prompt=str(event.get("prompt", "")),
                )
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                raise ModelUnavailableError("remote answer stream is invalid") from error

    def summarize_report(self, material: dict[str, object]) -> str:
        return str(
            self.client.request("POST", "/vision/summarize-report", json=material).json()["text"]
        )
