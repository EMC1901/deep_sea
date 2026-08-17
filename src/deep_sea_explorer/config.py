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
    GGUF = "gguf"
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
    qwen_adapter_path: str = ""
    gguf_server_url: str = "http://127.0.0.1:19001"
    gguf_model_path: str = ""
    gguf_mmproj_path: str = ""
    gguf_context_size: int = 4096
    gguf_gpu_layers: int = 0
    gguf_connect_timeout_seconds: int = 5
    gguf_read_timeout_seconds: int = 180
    image_retrieval_enabled: bool = False
    image_retrieval_index_dir: str = ""
    image_retrieval_dino_model_path: str = ""
    image_retrieval_top_k: int = 4
    image_retrieval_device: str = "auto"
    image_retrieval_exclude_same_site: bool = False
    monitoring_dino_model_path: str = ""
    monitoring_dino_device: str = "auto"
    monitoring_blur_threshold: float = 35.0
    monitoring_similarity_threshold: float = 0.7
    monitoring_queue_capacity: int = 10
    label_knowledge_base_dir: str = "runtime/label-knowledge-base"
    monitoring_label_batch_size: int = 64
    image_generation_enabled: bool = True
    image_model_path: str = ""
    yolo_model_path: str = ""
    yolo_confidence: float = 0.35
    scene_change_threshold: float = 0.22
    scene_confirm_frames: int = 3
    track_confirm_frames: int = 3
    memo_embedding_model_path: str = ""
    rag_embedding_model_path: str = ""
    temp_dir: Path = Path.cwd() / ".deep-sea-explorer-tmp"
    data_dir: Path = Path.cwd() / "data"
    report_font_path: str = ""
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:19100",
        "http://localhost:19100",
        "http://localhost:8000",
    )
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
            for value in env.get(
                "CORS_ORIGINS", "http://127.0.0.1:19100,http://localhost:19100,http://localhost:8000"
            ).split(",")
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
            qwen_adapter_path=env.get("QWEN_ADAPTER_PATH", ""),
            gguf_server_url=env.get("GGUF_SERVER_URL", "http://127.0.0.1:19001"),
            gguf_model_path=env.get("GGUF_MODEL_PATH", ""),
            gguf_mmproj_path=env.get("GGUF_MMPROJ_PATH", ""),
            gguf_context_size=_as_int(env.get("GGUF_CONTEXT_SIZE"), 4096),
            gguf_gpu_layers=_as_int(env.get("GGUF_GPU_LAYERS"), 0),
            gguf_connect_timeout_seconds=_as_int(env.get("GGUF_CONNECT_TIMEOUT_SECONDS"), 5),
            gguf_read_timeout_seconds=_as_int(env.get("GGUF_READ_TIMEOUT_SECONDS"), 180),
            image_retrieval_enabled=_as_bool(env.get("IMAGE_RETRIEVAL_ENABLED")),
            image_retrieval_index_dir=env.get("IMAGE_RETRIEVAL_INDEX_DIR", ""),
            image_retrieval_dino_model_path=env.get("IMAGE_RETRIEVAL_DINO_MODEL_PATH", ""),
            image_retrieval_top_k=_as_int(env.get("IMAGE_RETRIEVAL_TOP_K"), 4),
            image_retrieval_device=env.get("IMAGE_RETRIEVAL_DEVICE", "auto"),
            image_retrieval_exclude_same_site=_as_bool(
                env.get("IMAGE_RETRIEVAL_EXCLUDE_SAME_SITE")
            ),
            monitoring_dino_model_path=env.get(
                "MONITORING_DINO_MODEL_PATH", env.get("IMAGE_RETRIEVAL_DINO_MODEL_PATH", "")
            ),
            monitoring_dino_device=env.get(
                "MONITORING_DINO_DEVICE", env.get("IMAGE_RETRIEVAL_DEVICE", "auto")
            ),
            monitoring_blur_threshold=float(env.get("MONITORING_BLUR_THRESHOLD", "35")),
            monitoring_similarity_threshold=float(
                env.get("MONITORING_SIMILARITY_THRESHOLD", "0.7")
            ),
            monitoring_queue_capacity=_as_int(env.get("MONITORING_QUEUE_CAPACITY"), 10),
            label_knowledge_base_dir=env.get("LABEL_KNOWLEDGE_BASE_DIR", "runtime/label-knowledge-base"),
            monitoring_label_batch_size=_as_int(env.get("MONITORING_LABEL_BATCH_SIZE"), 64),
            image_generation_enabled=_as_bool(env.get("IMAGE_GENERATION_ENABLED"), True),
            image_model_path=env.get("IMAGE_MODEL_PATH", ""),
            yolo_model_path=env.get("YOLO_MODEL_PATH", ""),
            yolo_confidence=float(env.get("YOLO_CONFIDENCE", "0.35")),
            scene_change_threshold=float(env.get("SCENE_CHANGE_THRESHOLD", "0.22")),
            scene_confirm_frames=_as_int(env.get("SCENE_CONFIRM_FRAMES"), 3),
            track_confirm_frames=_as_int(env.get("TRACK_CONFIRM_FRAMES"), 3),
            memo_embedding_model_path=env.get("MEMO_EMBEDDING_MODEL_PATH", ""),
            rag_embedding_model_path=env.get("RAG_EMBEDDING_MODEL_PATH", ""),
            temp_dir=Path(
                env.get("TEMP_DIR")
                or env.get("MODEL_TEMP_DIR")
                or Path.cwd() / ".deep-sea-explorer-tmp"
            ),
            data_dir=Path(env.get("DATA_DIR") or Path.cwd() / "data"),
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
                ("MEMO_EMBEDDING_MODEL_PATH", self.memo_embedding_model_path),
                ("RAG_EMBEDDING_MODEL_PATH", self.rag_embedding_model_path),
            ):
                if not path:
                    errors.append(f"local mode requires {name}")
            if self.image_generation_enabled and not self.image_model_path:
                errors.append("local mode requires IMAGE_MODEL_PATH when image generation is enabled")
        if self.model_backend is ModelBackend.GGUF:
            parsed = urlparse(self.gguf_server_url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
                errors.append("gguf mode requires a localhost HTTP GGUF_SERVER_URL")
            for name, path in (("GGUF_MODEL_PATH", self.gguf_model_path), ("GGUF_MMPROJ_PATH", self.gguf_mmproj_path)):
                if not path or not Path(path).is_file() or Path(path).suffix.lower() != ".gguf":
                    errors.append(f"gguf mode requires a readable GGUF {name}")
            if self.gguf_context_size < 1024:
                errors.append("GGUF_CONTEXT_SIZE must be at least 1024 for vision")
            if self.gguf_gpu_layers < 0:
                errors.append("GGUF_GPU_LAYERS must be non-negative")
            for name, path in (("MEMO_EMBEDDING_MODEL_PATH", self.memo_embedding_model_path), ("RAG_EMBEDDING_MODEL_PATH", self.rag_embedding_model_path)):
                if not path:
                    errors.append(f"gguf mode requires {name}")
        if self.image_retrieval_enabled:
            if self.model_backend not in {ModelBackend.LOCAL, ModelBackend.GGUF}:
                errors.append("image retrieval requires MODEL_BACKEND=local or gguf")
            if not self.image_retrieval_index_dir:
                errors.append("enabled image retrieval requires IMAGE_RETRIEVAL_INDEX_DIR")
            if not self.image_retrieval_dino_model_path:
                errors.append("enabled image retrieval requires IMAGE_RETRIEVAL_DINO_MODEL_PATH")
        if not 0 <= self.image_retrieval_top_k <= 8:
            errors.append("IMAGE_RETRIEVAL_TOP_K must be between 0 and 8")
        if not 0 <= self.monitoring_similarity_threshold <= 1:
            errors.append("MONITORING_SIMILARITY_THRESHOLD must be between 0 and 1")
        if self.monitoring_blur_threshold < 0:
            errors.append("MONITORING_BLUR_THRESHOLD must not be negative")
        if self.monitoring_queue_capacity <= 0:
            errors.append("MONITORING_QUEUE_CAPACITY must be positive")
        if self.monitoring_label_batch_size <= 0:
            errors.append("MONITORING_LABEL_BATCH_SIZE must be positive")
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
