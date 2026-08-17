from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.infrastructure.models.fake import (
    FakeEmbeddingGateway,
    FakeImageGateway,
    FakeVisionGateway,
)
from deep_sea_explorer.infrastructure.models.local.adapters import (
    EmbeddingAdapter,
    ImageAdapter,
    QwenAdapter,
)
from deep_sea_explorer.infrastructure.models.local.gateways import (
    DisabledImageGateway,
    LocalEmbeddingGateway,
    LocalImageGateway,
    LocalVisionGateway,
)
from deep_sea_explorer.infrastructure.models.local.runtime import (
    InferenceCoordinator,
    LocalModelRuntime,
)
from deep_sea_explorer.infrastructure.models.gguf.vision import LlamaCppVisionGateway
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.models.remote.embedding import RemoteEmbeddingGateway
from deep_sea_explorer.infrastructure.models.remote.image import RemoteImageGateway
from deep_sea_explorer.infrastructure.models.remote.vision import RemoteVisionGateway
from deep_sea_explorer.infrastructure.retrieval.dinov2 import DinoV2ImageEncoder
from deep_sea_explorer.infrastructure.retrieval.errors import ImageRetrievalError
from deep_sea_explorer.infrastructure.retrieval.gateway import (
    CoordinatedImageRetrievalGateway,
    LocalImageRetrievalGateway,
    UnavailableImageRetrievalGateway,
)
from deep_sea_explorer.infrastructure.retrieval.numpy_index import NumpyImageRetrievalIndex
from deep_sea_explorer.infrastructure.reports.reportlab_renderer import ReportLabRenderer
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.infrastructure.storage.event_store import EventStore
from deep_sea_explorer.ports.embedding_gateway import EmbeddingGateway
from deep_sea_explorer.ports.image_retrieval import ImageRetrievalGateway
from deep_sea_explorer.ports.model_gateway import ImageGenerationGateway, VisionModelGateway
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService
from deep_sea_explorer.services.monitoring_knowledge import MonitoringKnowledgeBase
from deep_sea_explorer.services.question_answering import QuestionAnsweringService
from deep_sea_explorer.services.rag_service import RagService
from deep_sea_explorer.services.report_service import ReportService
from deep_sea_explorer.services.video_ingestion import VideoIngestionService
from deep_sea_explorer.workers.memo_worker import MemoWorker


LOGGER = logging.getLogger(__name__)


class _UnavailableMonitoringDinoEncoder:
    """Defer a missing optional model-path error until monitoring receives a frame."""

    def embed_image(self, _path: object) -> object:
        raise RuntimeError("MONITORING_DINO_MODEL_PATH is not configured")


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    vision: VisionModelGateway
    image: ImageGenerationGateway
    memo_embedding: EmbeddingGateway
    rag_embedding: EmbeddingGateway
    image_retrieval: ImageRetrievalGateway
    sessions: MemorySessionStore
    memos: MemoryMemoBroker
    files: TempFileStore
    ingestion: VideoIngestionService
    monitoring: MonitoringService
    questions: QuestionAnsweringService
    rag: RagService
    reports: ReportService
    worker: MemoWorker


def _build(
    settings: Settings,
    vision: VisionModelGateway,
    image: ImageGenerationGateway,
    memo_embedding: EmbeddingGateway,
    rag_embedding: EmbeddingGateway,
    image_retrieval: ImageRetrievalGateway | None = None,
    monitoring_dino_encoder: object | None = None,
) -> ApplicationContainer:
    retrieval = image_retrieval or UnavailableImageRetrievalGateway(
        "image retrieval is disabled", enabled=False
    )
    sessions = MemorySessionStore(settings.session_ttl_seconds, settings.max_active_sessions)
    memos = MemoryMemoBroker()
    files = TempFileStore(settings.temp_dir, settings.file_ttl_seconds)
    stats = CaptureStatsService()
    ingestion = VideoIngestionService(files, sessions, settings.max_frames_per_request)
    rag = RagService(rag_embedding)
    dino_encoder = monitoring_dino_encoder
    if dino_encoder is None:
        dino_encoder = (
            DinoV2ImageEncoder(settings.monitoring_dino_model_path, device=settings.monitoring_dino_device)
            if settings.monitoring_dino_model_path
            else _UnavailableMonitoringDinoEncoder()
        )
    monitoring = MonitoringService(
        vision, memo_embedding, sessions, memos, files, stats, settings.memo_similarity_threshold,
        dino_encoder=dino_encoder,
        knowledge_base=MonitoringKnowledgeBase(
            Path(settings.label_knowledge_base_dir), batch_size=settings.monitoring_label_batch_size
        ),
        queue_capacity=settings.monitoring_queue_capacity,
        blur_threshold=settings.monitoring_blur_threshold,
        similarity_threshold=settings.monitoring_similarity_threshold,
    )
    questions = QuestionAnsweringService(vision, sessions)
    reports = ReportService(vision, ReportLabRenderer(settings.report_font_path), files)
    return ApplicationContainer(
        settings,
        vision,
        image,
        memo_embedding,
        rag_embedding,
        retrieval,
        sessions,
        memos,
        files,
        ingestion,
        monitoring,
        questions,
        rag,
        reports,
        MemoWorker(monitoring, sessions),
    )


def build_fake_container(settings: Settings) -> ApplicationContainer:
    class FakeDinoEncoder:
        def embed_image(self, _path):
            import numpy as np
            return np.array([1.0, 0.0], dtype=np.float32)

    return _build(
        settings,
        FakeVisionGateway(),
        FakeImageGateway(),
        FakeEmbeddingGateway(),
        FakeEmbeddingGateway(),
        monitoring_dino_encoder=FakeDinoEncoder(),
    )


def build_remote_container(settings: Settings) -> ApplicationContainer:
    client = RemoteModelClient(settings)
    return _build(
        settings,
        RemoteVisionGateway(client),
        RemoteImageGateway(client),
        RemoteEmbeddingGateway(client, "memo"),
        RemoteEmbeddingGateway(client, "rag"),
    )


def build_local_container(settings: Settings) -> ApplicationContainer:
    runtime = LocalModelRuntime(
        InferenceCoordinator(
            settings.model_max_concurrent_requests,
            settings.model_max_queue_size,
            settings.model_queue_timeout_seconds,
        )
    )
    image = (
        LocalImageGateway(runtime, ImageAdapter(settings.image_model_path))
        if settings.image_generation_enabled
        else DisabledImageGateway()
    )
    return _build(
        settings,
        LocalVisionGateway(
            runtime,
            QwenAdapter(settings.qwen_model_path, settings.qwen_adapter_path),
        ),
        image,
        LocalEmbeddingGateway(
            runtime,
            EmbeddingAdapter("gte", settings.memo_embedding_model_path, 768, trust_remote_code=True),
        ),
        LocalEmbeddingGateway(
            runtime,
            EmbeddingAdapter("minilm", settings.rag_embedding_model_path, 384, trust_remote_code=False),
        ),
        _build_local_image_retrieval(settings, runtime.coordinator),
        (
            DinoV2ImageEncoder(settings.monitoring_dino_model_path, device=settings.monitoring_dino_device)
            if settings.monitoring_dino_model_path
            else _UnavailableMonitoringDinoEncoder()
        ),
    )


def build_gguf_container(settings: Settings) -> ApplicationContainer:
    """Keep embeddings/DINO local while vision inference stays in llama-server."""
    runtime = LocalModelRuntime(
        InferenceCoordinator(
            settings.model_max_concurrent_requests,
            settings.model_max_queue_size,
            settings.model_queue_timeout_seconds,
        )
    )
    return _build(
        settings,
        LlamaCppVisionGateway(settings),
        DisabledImageGateway(),
        LocalEmbeddingGateway(
            runtime,
            EmbeddingAdapter("gte", settings.memo_embedding_model_path, 768, trust_remote_code=True),
        ),
        LocalEmbeddingGateway(
            runtime,
            EmbeddingAdapter("minilm", settings.rag_embedding_model_path, 384, trust_remote_code=False),
        ),
        _build_local_image_retrieval(settings, runtime.coordinator),
        (
            DinoV2ImageEncoder(settings.monitoring_dino_model_path, device=settings.monitoring_dino_device)
            if settings.monitoring_dino_model_path
            else _UnavailableMonitoringDinoEncoder()
        ),
    )


def _build_local_image_retrieval(
    settings: Settings,
    coordinator: InferenceCoordinator,
) -> ImageRetrievalGateway:
    if not settings.image_retrieval_enabled:
        return UnavailableImageRetrievalGateway("image retrieval is disabled", enabled=False)
    try:
        index = NumpyImageRetrievalIndex.from_directory(Path(settings.image_retrieval_index_dir))
        gateway = LocalImageRetrievalGateway(
            DinoV2ImageEncoder(
                settings.image_retrieval_dino_model_path,
                device=settings.image_retrieval_device,
            ),
            index,
        )
        return CoordinatedImageRetrievalGateway(gateway, coordinator)
    except (ImageRetrievalError, OSError, ValueError) as error:
        LOGGER.warning(
            "image retrieval unavailable at startup error_type=%s detail=%s",
            type(error).__name__,
            str(error),
        )
        return UnavailableImageRetrievalGateway(
            f"image retrieval initialization failed: {type(error).__name__}",
            enabled=True,
        )


def build_container(settings: Settings) -> ApplicationContainer:
    if settings.model_backend is ModelBackend.FAKE:
        return build_fake_container(settings)
    if settings.model_backend is ModelBackend.LOCAL:
        return build_local_container(settings)
    if settings.model_backend is ModelBackend.GGUF:
        return build_gguf_container(settings)
    return build_remote_container(settings)
