from __future__ import annotations

from dataclasses import dataclass

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.infrastructure.models.fake import (
    FakeEmbeddingGateway,
    FakeImageGateway,
    FakeVisionGateway,
)
from deep_sea_explorer.infrastructure.models.local.lazy import (
    LocalUnavailableGateway,
    LocalVisionGateway,
)
from deep_sea_explorer.infrastructure.models.remote.client import RemoteModelClient
from deep_sea_explorer.infrastructure.models.remote.embedding import RemoteEmbeddingGateway
from deep_sea_explorer.infrastructure.models.remote.image import RemoteImageGateway
from deep_sea_explorer.infrastructure.models.remote.vision import RemoteVisionGateway
from deep_sea_explorer.infrastructure.reports.reportlab_renderer import ReportLabRenderer
from deep_sea_explorer.infrastructure.storage.memory_memo import MemoryMemoBroker
from deep_sea_explorer.infrastructure.storage.memory_session import MemorySessionStore
from deep_sea_explorer.infrastructure.storage.temp_file_store import TempFileStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring import MonitoringService
from deep_sea_explorer.services.question_answering import QuestionAnsweringService
from deep_sea_explorer.services.rag_service import RagService
from deep_sea_explorer.services.report_service import ReportService
from deep_sea_explorer.services.video_ingestion import VideoIngestionService
from deep_sea_explorer.workers.memo_worker import MemoWorker


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    vision: object
    image: object
    memo_embedding: object
    rag_embedding: object
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
    settings: Settings, vision: object, image: object, memo_embedding: object, rag_embedding: object
) -> ApplicationContainer:
    sessions = MemorySessionStore(settings.session_ttl_seconds, settings.max_active_sessions)
    memos = MemoryMemoBroker()
    files = TempFileStore(settings.temp_dir, settings.file_ttl_seconds)
    stats = CaptureStatsService()
    ingestion = VideoIngestionService(files, sessions, settings.max_frames_per_request)
    rag = RagService(rag_embedding)
    monitoring = MonitoringService(
        vision, memo_embedding, sessions, memos, files, stats, settings.memo_similarity_threshold
    )
    questions = QuestionAnsweringService(vision, image, rag, sessions)
    reports = ReportService(vision, ReportLabRenderer(), files)
    return ApplicationContainer(
        settings,
        vision,
        image,
        memo_embedding,
        rag_embedding,
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
    return _build(
        settings,
        FakeVisionGateway(),
        FakeImageGateway(),
        FakeEmbeddingGateway(),
        FakeEmbeddingGateway(),
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
    return _build(
        settings,
        LocalVisionGateway(settings.qwen_model_path),
        LocalUnavailableGateway("image"),
        LocalUnavailableGateway("memo embedding"),
        LocalUnavailableGateway("rag embedding"),
    )


def build_container(settings: Settings) -> ApplicationContainer:
    if settings.model_backend is ModelBackend.FAKE:
        return build_fake_container(settings)
    if settings.model_backend is ModelBackend.LOCAL:
        return build_local_container(settings)
    return build_remote_container(settings)
