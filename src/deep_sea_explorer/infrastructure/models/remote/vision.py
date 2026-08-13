from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from deep_sea_explorer.domain.enums import CaptureType, StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError
from deep_sea_explorer.domain.models import CaptureDecision, CountItem, ModelHealth, MonitoringAnalysis, StreamEvent
from deep_sea_explorer.services.key_frame_detection import SurveyEventEvaluation

from .client import RemoteModelClient


class RemoteVisionGateway:
    def __init__(self, client: RemoteModelClient) -> None:
        self.client = client

    def health(self) -> ModelHealth:
        if not self.client.settings.model_service_enabled:
            return ModelHealth(False, "remote service disabled")
        try:
            data = self.client.json_body(self.client.request("GET", "/health"))
            return ModelHealth(data.get("status") == "ok", "remote")
        except ModelUnavailableError as error:
            return ModelHealth(False, type(error).__name__)

    @staticmethod
    def _file(path: Path, field: str, content_type: str) -> dict[str, tuple[str, bytes, str]]:
        return {field: (path.name, path.read_bytes(), content_type)}

    def describe_video(self, video_path: Path) -> str:
        data = self.client.json_body(
            self.client.request(
                "POST",
                "/vision/describe-video",
                files=self._file(video_path, "video", "video/mp4"),
            )
        )
        text = data.get("text")
        if not isinstance(text, str):
            raise ModelUnavailableError("remote video description is invalid")
        return text

    def evaluate_frame(self, image_path: Path) -> CaptureDecision:
        body = self.client.json_body(
            self.client.request(
                "POST",
                "/vision/evaluate-frame",
                files=self._file(image_path, "image", "image/jpeg"),
            )
        )
        data = body.get("decision")
        if not isinstance(data, dict):
            raise ModelUnavailableError("remote frame decision is invalid")
        try:
            category = CaptureType(data["category"])
        except (KeyError, ValueError) as error:
            raise ModelUnavailableError("remote frame decision category is invalid") from error

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

    def analyze_monitoring_frame(self, image_path: Path) -> MonitoringAnalysis:
        body = self.client.json_body(
            self.client.request(
                "POST",
                "/vision/analyze-monitoring-frame",
                files=self._file(image_path, "image", "image/jpeg"),
            )
        )
        description = body.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ModelUnavailableError("remote monitoring analysis is invalid")
        def items(values):
            return tuple(
                CountItem(str(item.get("name", "")), int(item.get("count", 1)))
                for item in values or []
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            )
        return MonitoringAnalysis(
            description.strip(),
            items(body.get("organisms")),
            items(body.get("env_features")),
            items(body.get("substrates")),
            items(body.get("geomorphologies")),
        )

    def answer(self, question: str) -> Iterator[StreamEvent]:
        with self.client.stream(
            "POST",
            "/vision/answer",
            json={"question": question},
        ) as response:
            for raw_line in response.iter_lines():
                if not raw_line.strip():
                    continue
                try:
                    event = json.loads(raw_line)
                    event_type_name = event["type"]
                    event_type = {"delta": StreamEventType.CHUNK, "done": StreamEventType.FINAL}.get(
                        event_type_name
                    )
                    if event_type is None:
                        event_type = StreamEventType(event_type_name)
                    yield StreamEvent(
                        event_type,
                        text=str(event.get("text", event.get("message", ""))),
                    )
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    raise ModelUnavailableError("remote answer stream is invalid") from error

    def summarize_report(self, material: dict[str, object]) -> str:
        data = self.client.json_body(
            self.client.request("POST", "/vision/summarize-report", json={"material": material})
        )
        text = data.get("text")
        if not isinstance(text, str):
            raise ModelUnavailableError("remote report summary is invalid")
        return text

    def evaluate_survey_event(self, reference_image: Path | None, current_image: Path, metadata: dict[str, object]) -> SurveyEventEvaluation:
        files = self._file(current_image, "current_image", "image/jpeg")
        if reference_image is not None and reference_image.is_file():
            files.update(self._file(reference_image, "reference_image", "image/jpeg"))
        body = self.client.json_body(self.client.request("POST", "/vision/evaluate-survey-event", files=files, data={"metadata": json.dumps(metadata, ensure_ascii=False)}))
        try:
            return SurveyEventEvaluation(bool(body["survey_value"]), str(body["event_type"]), bool(body.get("scene_changed")), tuple(body.get("new_elements") or ()), str(body.get("description", "")), float(body.get("confidence", 0.0)), tuple(body.get("observed_elements") or ()))
        except (KeyError, TypeError, ValueError) as error:
            raise ModelUnavailableError("remote survey event response is invalid") from error
