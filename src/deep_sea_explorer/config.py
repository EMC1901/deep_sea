"""集中且无副作用的运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse


class ModelBackend(StrEnum):
    REMOTE = "remote"
    LOCAL = "local"
    FAKE = "fake"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError as error:
        raise ValueError(f"Expected an integer, got {value!r}") from error


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 9001
    speech_host: str = "0.0.0.0"
    speech_port: int = 9009
    model_backend: ModelBackend = ModelBackend.REMOTE
    model_service_enabled: bool = False
    model_service_base_url: str = "https://model-server.example.invalid"
    model_service_api_prefix: str = "/v1"
    model_service_host: str = "127.0.0.1"
    model_service_port: int = 19000
    model_service_auth_type: str = "bearer"
    model_service_auth_token: str = ""
    model_service_connect_timeout_seconds: int = 5
    model_service_read_timeout_seconds: int = 120
    model_service_verify_tls: bool = True
    qwen_model_path: str = ""
    image_model_path: str = ""
    memo_embedding_model_path: str = ""
    rag_embedding_model_path: str = ""
    temp_dir: Path = Path.cwd() / ".deep-sea-explorer-tmp"
    report_font_path: str = ""
    cors_origins: tuple[str, ...] = ("http://localhost:8000",)
    max_content_length_mb: int = 50
    max_frames_per_request: int = 150
    max_question_length: int = 4_000
    max_active_sessions: int = 100
    session_ttl_seconds: int = 3_600
    file_ttl_seconds: int = 86_400
    memo_similarity_threshold: float = 0.85
    model_max_concurrent_requests: int = 1
    model_max_queue_size: int = 4
    model_queue_timeout_seconds: int = 180
    model_max_embedding_texts: int = 32

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        backend = ModelBackend(env.get("MODEL_BACKEND", ModelBackend.REMOTE))
        origins = tuple(
            value.strip()
            for value in env.get("CORS_ORIGINS", "http://localhost:8000").split(",")
            if value.strip()
        )
        return cls(
            app_env=env.get("APP_ENV", "development"),
            api_host=env.get("API_HOST", "0.0.0.0"),
            api_port=_as_int(env.get("API_PORT"), 9001),
            speech_host=env.get("SPEECH_HOST", "0.0.0.0"),
            speech_port=_as_int(env.get("SPEECH_PORT"), 9009),
            model_backend=backend,
            model_service_enabled=_as_bool(env.get("MODEL_SERVICE_ENABLED")),
            model_service_base_url=env.get(
                "MODEL_SERVICE_BASE_URL", "https://model-server.example.invalid"
            ),
            model_service_api_prefix=env.get("MODEL_SERVICE_API_PREFIX", "/v1"),
            model_service_host=env.get("MODEL_SERVICE_HOST", "127.0.0.1"),
            model_service_port=_as_int(env.get("MODEL_SERVICE_PORT"), 19000),
            model_service_auth_type=env.get("MODEL_SERVICE_AUTH_TYPE", "bearer"),
            model_service_auth_token=env.get("MODEL_SERVICE_AUTH_TOKEN", ""),
            model_service_connect_timeout_seconds=_as_int(
                env.get("MODEL_SERVICE_CONNECT_TIMEOUT_SECONDS"), 5
            ),
            model_service_read_timeout_seconds=_as_int(
                env.get("MODEL_SERVICE_READ_TIMEOUT_SECONDS"), 120
            ),
            model_service_verify_tls=_as_bool(env.get("MODEL_SERVICE_VERIFY_TLS"), True),
            qwen_model_path=env.get("QWEN_MODEL_PATH", ""),
            image_model_path=env.get("IMAGE_MODEL_PATH", ""),
            memo_embedding_model_path=env.get("MEMO_EMBEDDING_MODEL_PATH", ""),
            rag_embedding_model_path=env.get("RAG_EMBEDDING_MODEL_PATH", ""),
            temp_dir=Path(
                env.get("TEMP_DIR")
                or env.get("MODEL_TEMP_DIR")
                or Path.cwd() / ".deep-sea-explorer-tmp"
            ),
            report_font_path=env.get("REPORT_FONT_PATH", ""),
            cors_origins=origins,
            max_content_length_mb=_as_int(env.get("MAX_CONTENT_LENGTH_MB"), 50),
            memo_similarity_threshold=float(env.get("MEMO_SIMILARITY_THRESHOLD", "0.85")),
            model_max_concurrent_requests=_as_int(env.get("MODEL_MAX_CONCURRENT_REQUESTS"), 1),
            model_max_queue_size=_as_int(env.get("MODEL_MAX_QUEUE_SIZE"), 4),
            model_queue_timeout_seconds=_as_int(env.get("MODEL_QUEUE_TIMEOUT_SECONDS"), 180),
            model_max_embedding_texts=_as_int(env.get("MODEL_MAX_EMBEDDING_TEXTS"), 32),
        )

    def validate_for_runtime(self) -> list[str]:
        errors: list[str] = []
        if self.model_backend is ModelBackend.REMOTE and self.model_service_enabled:
            parsed = urlparse(self.model_service_base_url)
            if (
                not parsed.scheme
                or not parsed.netloc
                or parsed.hostname == "model-server.example.invalid"
            ):
                errors.append("remote mode requires a real MODEL_SERVICE_BASE_URL")
            if self.model_service_auth_type == "bearer" and not self.model_service_auth_token:
                errors.append("remote bearer mode requires MODEL_SERVICE_AUTH_TOKEN")
        if self.model_backend is ModelBackend.LOCAL:
            for name, path in (
                ("QWEN_MODEL_PATH", self.qwen_model_path),
                ("IMAGE_MODEL_PATH", self.image_model_path),
                ("MEMO_EMBEDDING_MODEL_PATH", self.memo_embedding_model_path),
                ("RAG_EMBEDDING_MODEL_PATH", self.rag_embedding_model_path),
            ):
                if not path:
                    errors.append(f"local mode requires {name}")
        if self.max_content_length_mb <= 0:
            errors.append("MAX_CONTENT_LENGTH_MB must be positive")
        if self.model_max_concurrent_requests <= 0:
            errors.append("MODEL_MAX_CONCURRENT_REQUESTS must be positive")
        if self.model_max_queue_size < 0:
            errors.append("MODEL_MAX_QUEUE_SIZE must not be negative")
        if self.model_queue_timeout_seconds <= 0:
            errors.append("MODEL_QUEUE_TIMEOUT_SECONDS must be positive")
        if self.model_max_embedding_texts <= 0:
            errors.append("MODEL_MAX_EMBEDDING_TEXTS must be positive")
        return errors
