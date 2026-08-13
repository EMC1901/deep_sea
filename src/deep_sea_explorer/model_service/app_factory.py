"""Flask application factory for the frozen server-model API v1."""

from __future__ import annotations

import hmac
import json
import math
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from flask import Flask, Response, g, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.wrappers.response import Response as WerkzeugResponse

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_local_container
from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.exceptions import ModelUnavailableError, ValidationError
from deep_sea_explorer.infrastructure.models.local.errors import (
    InferenceQueueFull,
    InferenceTimeout,
)


API_PREFIX = "/v1"


class ApiProblem(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def create_app(settings: Settings | None = None, container: object | None = None) -> Flask:
    settings = settings or Settings.from_env()
    _validate_service_settings(settings)
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=settings.max_content_length_mb * 1024 * 1024)
    app.extensions["settings"] = settings
    app.extensions["container"] = container or build_local_container(settings)

    @app.before_request
    def request_context_and_authentication() -> None:
        g.request_id = _request_id()
        g.request_started = time.monotonic()
        if request.method == "OPTIONS" or request.path == f"{API_PREFIX}/health":
            return
        value = request.headers.get("Authorization", "")
        expected = f"Bearer {settings.model_service_auth_token}"
        if not hmac.compare_digest(value, expected):
            raise ApiProblem(401, "UNAUTHORIZED", "valid bearer token is required")
        if request.headers.get("X-Model-API-Version") != "1":
            raise ApiProblem(400, "INVALID_INPUT", "X-Model-API-Version must be 1")

    @app.after_request
    def response_request_id(response: WerkzeugResponse) -> WerkzeugResponse:
        response.headers["X-Request-ID"] = g.get("request_id", uuid.uuid4().hex)
        response.headers.setdefault("Cache-Control", "no-store")
        app.logger.info(
            "model_api request_id=%s endpoint=%s status=%s duration_ms=%.1f",
            response.headers["X-Request-ID"],
            request.path,
            response.status_code,
            (time.monotonic() - g.get("request_started", time.monotonic())) * 1000,
        )
        return response

    @app.errorhandler(ApiProblem)
    def api_problem(error: ApiProblem) -> tuple[Response, int]:
        if error.status >= 500:
            _log_model_failure(app, error, error.status, include_traceback=False)
        return _error_response(error.status, error.code, error.message)

    @app.errorhandler(ValidationError)
    def invalid_model_input(error: ValidationError) -> tuple[Response, int]:
        return _error_response(400, "INVALID_INPUT", str(error))

    @app.errorhandler(InferenceQueueFull)
    def queue_full(error: InferenceQueueFull) -> tuple[Response, int]:
        _log_model_failure(app, error, 429, include_traceback=False)
        return _error_response(429, "QUEUE_FULL", "model inference queue is full")

    @app.errorhandler(InferenceTimeout)
    def timeout(error: InferenceTimeout) -> tuple[Response, int]:
        _log_model_failure(app, error, 504, include_traceback=True)
        return _error_response(504, "MODEL_TIMEOUT", "model inference timed out")

    @app.errorhandler(ModelUnavailableError)
    def model_not_ready(error: ModelUnavailableError) -> tuple[Response, int]:
        _log_model_failure(app, error, 503, include_traceback=True)
        return _error_response(503, "MODEL_NOT_READY", "requested model is not ready")

    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_: RequestEntityTooLarge) -> tuple[Response, int]:
        return _error_response(413, "PAYLOAD_TOO_LARGE", "request body exceeds the configured limit")

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception) -> tuple[Response, int] | HTTPException:
        if isinstance(error, HTTPException):
            return error
        app.logger.exception(
            "model_api unexpected_error request_id=%s endpoint=%s error_type=%s",
            g.get("request_id", "unknown"),
            request.path,
            type(error).__name__,
        )
        return _error_response(500, "INTERNAL_ERROR", "model service request failed")

    _register_routes(app)
    return app


def _log_model_failure(
    app: Flask,
    error: Exception,
    status: int,
    *,
    include_traceback: bool,
) -> None:
    cause = error.__cause__
    app.logger.error(
        "model_api failure request_id=%s endpoint=%s status=%s error_type=%s cause_type=%s",
        g.get("request_id", "unknown"),
        request.path,
        status,
        type(error).__name__,
        type(cause).__name__ if cause is not None else "none",
        exc_info=(type(error), error, error.__traceback__) if include_traceback else None,
    )


def _validate_service_settings(settings: Settings) -> None:
    if settings.model_backend is not ModelBackend.LOCAL:
        raise RuntimeError("model service requires MODEL_BACKEND=local")
    if settings.model_service_auth_type != "bearer" or not settings.model_service_auth_token:
        raise RuntimeError("model service requires a configured bearer token")
    if settings.model_service_api_prefix != API_PREFIX:
        raise RuntimeError("model service API prefix must be /v1")
    if settings.model_service_host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("model service must bind to a loopback address")
    errors = settings.validate_for_runtime()
    if errors:
        raise RuntimeError("invalid model service configuration: " + "; ".join(errors))


def _register_routes(app: Flask) -> None:
    @app.get(f"{API_PREFIX}/health")
    def health() -> Response:
        container = _container()
        return _json_response(
            {
                "status": "ok",
                "gpu": _gpu_status(),
                "models": {
                    "qwen": _model_state(container.vision),
                    "image": _model_state(container.image),
                    "memo": _model_state(container.memo_embedding),
                    "rag": _model_state(container.rag_embedding),
                },
            }
        )

    @app.post(f"{API_PREFIX}/vision/describe-video")
    def describe_video() -> Response:
        with _uploaded_file("video") as video_path:
            text = _container().vision.describe_video(video_path)
        return _json_response({"text": text})

    @app.post(f"{API_PREFIX}/vision/evaluate-frame")
    def evaluate_frame() -> Response:
        with _uploaded_file("image") as image_path:
            decision = _container().vision.evaluate_frame(image_path)
        return _json_response(
            {
                "decision": {
                    "is_deepsea": decision.is_deepsea,
                    "is_typical": decision.is_typical,
                    "category": decision.category.value,
                    "description": decision.description,
                    "organisms": [{"name": item.name, "count": item.count} for item in decision.organisms],
                    "env_features": [
                        {"name": item.name, "count": item.count} for item in decision.env_features
                    ],
                }
            }
        )

    @app.post(f"{API_PREFIX}/vision/analyze-monitoring-frame")
    def analyze_monitoring_frame() -> Response:
        with _uploaded_file("image") as image_path:
            analysis = _container().vision.analyze_monitoring_frame(image_path)
        return _json_response(
            {
                "description": analysis.description,
                "organisms": [
                    {"name": item.name, "count": item.count} for item in analysis.organisms
                ],
                "env_features": [
                    {"name": item.name, "count": item.count}
                    for item in analysis.env_features
                ],
            }
        )

    @app.post(f"{API_PREFIX}/vision/evaluate-survey-event")
    def evaluate_survey_event() -> Response:
        """Evaluate two JPEGs plus detector metadata; no video or embedding is involved."""
        current = _save_uploaded_file("current_image")
        reference = None
        try:
            if request.files.get("reference_image") is not None:
                reference = _save_uploaded_file("reference_image")
            raw_metadata = request.form.get("metadata", "{}")
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as error:
                raise ApiProblem(400, "INVALID_INPUT", "metadata must be valid JSON") from error
            if not isinstance(metadata, dict):
                raise ApiProblem(400, "INVALID_INPUT", "metadata must be an object")
            evaluation = _container().vision.evaluate_survey_event(reference, current, metadata)
            return _json_response({
                "survey_value": evaluation.survey_value,
                "event_type": evaluation.event_type,
                "scene_changed": evaluation.scene_changed,
                "new_elements": list(evaluation.new_elements),
                "observed_elements": list(evaluation.observed_elements),
                "description": evaluation.description,
                "confidence": evaluation.confidence,
            })
        finally:
            current.unlink(missing_ok=True)
            if reference is not None:
                reference.unlink(missing_ok=True)

    @app.post(f"{API_PREFIX}/vision/answer")
    def answer() -> Response:
        body = _json_object({"question"})
        question = body.get("question")
        if not isinstance(question, str):
            raise ApiProblem(400, "INVALID_INPUT", "question must be a string")
        question = question.strip()
        if not question or len(question) > _settings().max_question_length:
            raise ApiProblem(400, "INVALID_INPUT", "question is required and exceeds the allowed length")

        def events() -> Iterator[str]:
            try:
                output_chars = 0
                for event in _container().vision.answer(question):
                    if event.type is StreamEventType.CHUNK:
                        output_chars += len(event.text)
                        yield _ndjson({"type": "delta", "text": event.text})
                    elif event.type is StreamEventType.FINAL:
                        yield _ndjson({"type": "done", "usage": {"output_chars": output_chars}})
                        return
                yield _ndjson({"type": "done", "usage": {"output_chars": output_chars}})
            except ApiProblem as error:
                yield _ndjson({"type": "error", "code": error.code, "message": error.message})
            except Exception as error:
                app.logger.error(
                    "model_api stream_failure request_id=%s endpoint=%s error_type=%s cause_type=%s",
                    g.get("request_id", "unknown"),
                    request.path,
                    type(error).__name__,
                    type(error.__cause__).__name__ if error.__cause__ is not None else "none",
                    exc_info=(type(error), error, error.__traceback__),
                )
                yield _ndjson(
                    {"type": "error", "code": "MODEL_FAILURE", "message": "model inference failed"}
                )

        response = Response(stream_with_context(events()), content_type="application/x-ndjson")
        return response

    @app.post(f"{API_PREFIX}/vision/summarize-report")
    def summarize_report() -> Response:
        body = _json_object({"material"})
        material = body.get("material")
        if not isinstance(material, dict):
            raise ApiProblem(400, "INVALID_INPUT", "material must be an object")
        return _json_response({"text": _container().vision.summarize_report(material)})

    @app.post(f"{API_PREFIX}/images/generate")
    def generate_image() -> Response:
        body = _json_object({"prompt"})
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiProblem(400, "INVALID_INPUT", "prompt is required")
        image = _container().image.generate(prompt)
        if not image.startswith(b"\xff\xd8"):
            raise ApiProblem(500, "INTERNAL_ERROR", "image model returned invalid output")
        return Response(image, content_type="image/jpeg")

    @app.post(f"{API_PREFIX}/embeddings")
    def embeddings() -> Response:
        body = _json_object({"model", "texts"})
        model, texts = body.get("model"), body.get("texts")
        if model not in {"memo", "rag"}:
            raise ApiProblem(400, "INVALID_INPUT", "model must be memo or rag")
        if not isinstance(texts, list) or not texts or len(texts) > _settings().model_max_embedding_texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ApiProblem(400, "INVALID_INPUT", "texts must be a non-empty string array")
        gateway = _container().memo_embedding if model == "memo" else _container().rag_embedding
        vectors = gateway.embed(texts)
        if not vectors or len(vectors) != len(texts) or any(
            len(vector) != len(vectors[0]) or any(not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ApiProblem(500, "INTERNAL_ERROR", "embedding model returned invalid output")
        return _json_response(
            {"model": model, "normalized": True, "dimension": len(vectors[0]), "embeddings": vectors}
        )


def _request_id() -> str:
    value = request.headers.get("X-Request-ID", "")
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid4())


def _container() -> Any:
    return app_extension("container")


def _settings() -> Settings:
    return app_extension("settings")


def app_extension(name: str) -> Any:
    from flask import current_app

    return current_app.extensions[name]


def _json_response(body: dict[str, object]) -> Response:
    return jsonify({"request_id": g.request_id, **body})


def _error_response(status: int, code: str, message: str) -> tuple[Response, int]:
    return jsonify({"request_id": g.get("request_id", str(uuid.uuid4())), "error": {"code": code, "message": message}}), status


def _json_object(allowed: set[str]) -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) - allowed:
        raise ApiProblem(400, "INVALID_INPUT", "request body is invalid")
    return body


def _model_state(gateway: object) -> str:
    health = getattr(gateway, "health", None)
    if health is None:
        return "ready"
    value = health()
    detail = getattr(value, "detail", "")
    return detail if detail in {"not_loaded", "loading", "ready"} else "unavailable"


def _gpu_status() -> str:
    try:
        import torch  # type: ignore[import-not-found]

        return "available" if torch.cuda.is_available() else "unavailable"
    except ImportError:
        return "unavailable"


def _ndjson(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


@contextmanager
def _uploaded_file(kind: str) -> Iterator[Path]:
    target = _save_uploaded_file(kind)
    try:
        yield target
    finally:
        target.unlink(missing_ok=True)


def _save_uploaded_file(kind: str) -> Path:
    upload = request.files.get(kind)
    if upload is None or not upload.filename:
        raise ApiProblem(400, "INVALID_INPUT", f"{kind} file is required")
    content_type = upload.mimetype.lower()
    allowed = {
        "image": {"image/jpeg"},
        "current_image": {"image/jpeg"},
        "reference_image": {"image/jpeg"},
        "video": {"video/mp4", "video/x-msvideo", "video/quicktime"},
    }
    if content_type not in allowed[kind]:
        raise ApiProblem(415, "UNSUPPORTED_MEDIA_TYPE", f"{kind} media type is not supported")
    root = _settings().temp_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg" if kind in {"image", "current_image", "reference_image"} else ".mp4"
    target = (root / f"model-api-{uuid.uuid4().hex}{suffix}").resolve()
    if root not in target.parents:
        raise ApiProblem(500, "INTERNAL_ERROR", "temporary upload path is invalid")
    try:
        upload.save(target)
        if target.stat().st_size == 0:
            raise ApiProblem(422, "UNPROCESSABLE_INPUT", f"{kind} file is empty")
        _verify_upload(target, kind)
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _verify_upload(path: Path, kind: str) -> None:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np

        if kind in {"image", "current_image", "reference_image"}:
            data = path.read_bytes()
            if not data.startswith(b"\xff\xd8") or cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
            ) is None:
                raise ValueError("invalid image")
        else:
            capture = cv2.VideoCapture(str(path))
            ok, _ = capture.read()
            capture.release()
            if not ok:
                raise ValueError("invalid video")
    except (ImportError, ValueError, OSError) as error:
        raise ApiProblem(422, "UNPROCESSABLE_INPUT", f"{kind} content cannot be decoded") from error
