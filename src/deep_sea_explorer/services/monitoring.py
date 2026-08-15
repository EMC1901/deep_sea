"""Simplified real-time pipeline: validate -> DINOv2 deduplicate -> Qwen."""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import Capture, CountItem, Memo, MonitoringAnalysis, MonitoringTagMatch
from deep_sea_explorer.ports.embedding_gateway import EmbeddingGateway
from deep_sea_explorer.ports.file_store import FileStore
from deep_sea_explorer.ports.memo_broker import MemoBroker
from deep_sea_explorer.ports.model_gateway import VisionModelGateway
from deep_sea_explorer.ports.session_store import SessionStore
from deep_sea_explorer.services.capture_stats import CaptureStatsService
from deep_sea_explorer.services.monitoring_queue import MonitoringFrame, PerSessionFrameQueue
from deep_sea_explorer.services.monitoring_knowledge import MonitoringKnowledgeBase


LOGGER = logging.getLogger(__name__)


class MonitoringService:
    """Per-session real-time monitoring without detection, retrieval, or event comparison."""

    def __init__(
        self,
        vision: VisionModelGateway,
        embedding: EmbeddingGateway,
        sessions: SessionStore,
        broker: MemoBroker,
        files: FileStore,
        stats: CaptureStatsService,
        threshold: float,
        *,
        dino_encoder: object,
        knowledge_base: MonitoringKnowledgeBase,
        queue_capacity: int = 10,
        blur_threshold: float = 35.0,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.vision = vision
        self.embedding = embedding
        self.sessions = sessions
        self.broker = broker
        self.files = files
        self.stats = stats
        self.threshold = threshold
        self.dino_encoder = dino_encoder
        self.knowledge_base = knowledge_base
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self._queue = PerSessionFrameQueue(queue_capacity, self._analyze, self._complete)
        self._tail_embeddings: dict[str, np.ndarray] = {}
        self._metrics: dict[str, dict[str, float | int]] = {}
        self._lock = threading.RLock()

    def process_frame(self, session_id: str, frame: bytes | Path) -> dict[str, object]:
        """Apply both filters and submit an accepted JPEG to the bounded Qwen FIFO."""
        received = time.monotonic()
        self.sessions.get(session_id)
        frame_path = self.files.frame_path(session_id)
        try:
            frame_path.write_bytes(frame.read_bytes() if isinstance(frame, Path) else frame)
            sharpness = self._decode_and_sharpness(frame_path)
        except ValueError as error:
            frame_path.unlink(missing_ok=True)
            self._increment(session_id, "rejected_undecodable")
            return self._response(session_id, "rejected_undecodable", detail=str(error))
        if sharpness < self.blur_threshold:
            frame_path.unlink(missing_ok=True)
            self._increment(session_id, "rejected_blurry")
            return self._response(session_id, "rejected_blurry", sharpness=round(sharpness, 3))

        try:
            embedding = np.asarray(self.dino_encoder.embed_image(frame_path), dtype=np.float32).reshape(-1)
            if embedding.size == 0 or not np.isfinite(embedding).all():
                raise ValueError("DINOv2 embedding is invalid")
            with self._lock:
                previous = self._tail_embeddings.get(session_id)
            similarity = float(np.dot(embedding, previous)) if previous is not None else None
        except Exception as error:
            frame_path.unlink(missing_ok=True)
            self._increment(session_id, "rejected_dino_error")
            LOGGER.warning("monitoring DINO filter failed session_id=%s error_type=%s", session_id, type(error).__name__)
            return self._response(session_id, "rejected_dino_error")
        if similarity is not None and similarity >= self.similarity_threshold:
            frame_path.unlink(missing_ok=True)
            self._increment(session_id, "rejected_similar")
            return self._response(session_id, "rejected_similar", similarity=round(similarity, 4))

        submission = self._queue.submit(MonitoringFrame(session_id, frame_path, received))
        with self._lock:
            self._tail_embeddings[session_id] = embedding
        self._increment(session_id, "accepted")
        if submission.dropped_oldest:
            self._increment(session_id, "queue2_dropped_oldest")
        return self._response(
            session_id,
            "queued",
            sharpness=round(sharpness, 3),
            similarity=None if similarity is None else round(similarity, 4),
            queue2_waiting=submission.waiting,
            queue2_dropped_oldest=submission.dropped_oldest,
        )

    def _analyze(self, frame: MonitoringFrame) -> MonitoringAnalysis:
        matches: list[MonitoringTagMatch] = []
        for batch in self.knowledge_base.batches():
            candidates = {"organisms": (), "substrates": (), "geomorphologies": ()}
            field = {"bio": "organisms", "substrate": "substrates", "geomorphology": "geomorphologies"}[batch.category]
            candidates[field] = batch.labels
            matches.append(self.vision.match_monitoring_tags(frame.image_path, candidates))
        tags = self._validated_tags(matches)
        selections = {
            "bio": tuple(item.name for item in tags.organisms if item.name != "未知生物"),
            "substrate": tuple(item.name for item in tags.substrates if item.name != "未知底质"),
            "geomorphology": tuple(item.name for item in tags.geomorphologies if item.name != "未知地貌"),
        }
        description = self.vision.describe_monitoring_frame(
            frame.image_path, tags, self.knowledge_base.descriptions_for(selections)
        )
        return MonitoringAnalysis(description, tags.organisms, (), tags.substrates, tags.geomorphologies)

    def _validated_tags(self, matches: list[MonitoringTagMatch]) -> MonitoringTagMatch:
        grouped = {
            "bio": ("organisms", "未知生物"),
            "substrate": ("substrates", "未知底质"),
            "geomorphology": ("geomorphologies", "未知地貌"),
        }
        accepted: dict[str, tuple[CountItem, ...]] = {}
        unknown: set[str] = set()
        for category, (field, unknown_label) in grouped.items():
            values: dict[str, int] = {}
            evidence = False
            allowed = self.knowledge_base.allowed(category)
            for match in matches:
                evidence = evidence or category in match.unknown_categories
                for item in getattr(match, field):
                    if item.name in allowed:
                        values[item.name] = max(values.get(item.name, 0), max(1, item.count))
            if not values and evidence:
                values[unknown_label] = 1
                unknown.add(category)
            accepted[category] = tuple(CountItem(name, count) for name, count in sorted(values.items()))
        return MonitoringTagMatch(
            accepted["bio"], accepted["substrate"], accepted["geomorphology"], tuple(sorted(unknown))
        )

    def _complete(self, frame: MonitoringFrame, analysis: MonitoringAnalysis, elapsed: float) -> None:
        state = self.sessions.get(frame.session_id)
        state.model_task_in_flight = False
        state.last_model_call_time = datetime.now().timestamp()
        captures = self._captures(state, analysis, frame.image_path)
        self.broker.publish(
            Memo(
                datetime.now().strftime("%H:%M:%S"),
                analysis.description,
                frame.session_id,
                captures[0] if captures else None,
                tuple(captures),
            )
        )
        with self._lock:
            metrics = self._session_metrics(frame.session_id)
            metrics["qwen_completed"] += 1
            metrics["qwen_total_seconds"] += elapsed
            metrics["end_to_end_total_seconds"] += time.monotonic() - frame.captured_monotonic
        LOGGER.info(
            "monitoring frame completed session_id=%s qwen_seconds=%.3f end_to_end_seconds=%.3f",
            frame.session_id,
            elapsed,
            time.monotonic() - frame.captured_monotonic,
        )

    def _captures(self, state: object, analysis: MonitoringAnalysis, image_path: Path) -> list[Capture]:
        image = self._data_uri(image_path)
        captures: list[Capture] = []
        if analysis.organisms:
            organisms = self.stats.update(state, CaptureType.BIO, analysis.organisms)
            captures.append(Capture(CaptureType.BIO, image, analysis.description, organisms=organisms))
        if analysis.substrates:
            substrates = self.stats.update(state, CaptureType.SUBSTRATE, analysis.substrates)
            captures.append(Capture(CaptureType.SUBSTRATE, image, analysis.description, substrates=substrates))
        if analysis.geomorphologies:
            geomorphologies = self.stats.update(state, CaptureType.GEOMORPHOLOGY, analysis.geomorphologies)
            captures.append(Capture(CaptureType.GEOMORPHOLOGY, image, analysis.description, geomorphologies=geomorphologies))
        return captures

    @staticmethod
    def _decode_and_sharpness(path: Path) -> float:
        try:
            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        except ImportError as error:  # pragma: no cover - common dependency
            raise ValueError("OpenCV is unavailable") from error
        if decoded is None or decoded.size == 0:
            raise ValueError("JPEG cannot be decoded")
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _response(self, session_id: str, status: str, **extra: object) -> dict[str, object]:
        in_flight, waiting = self._queue.state(session_id)
        state = self.sessions.get(session_id)
        state.model_task_in_flight = in_flight
        return {"status": status, "queue2_in_flight": in_flight, "queue2_waiting": waiting, "metrics": self.metrics(session_id), **extra}

    def _increment(self, session_id: str, key: str) -> None:
        with self._lock:
            self._session_metrics(session_id)[key] += 1

    def _session_metrics(self, session_id: str) -> dict[str, float | int]:
        return self._metrics.setdefault(
            session_id,
            {
                "accepted": 0,
                "rejected_undecodable": 0,
                "rejected_blurry": 0,
                "rejected_similar": 0,
                "rejected_dino_error": 0,
                "queue2_dropped_oldest": 0,
                "qwen_completed": 0,
                "qwen_total_seconds": 0.0,
                "end_to_end_total_seconds": 0.0,
            },
        )

    def metrics(self, session_id: str) -> dict[str, float | int]:
        with self._lock:
            metrics = dict(self._session_metrics(session_id))
        in_flight, waiting = self._queue.state(session_id)
        metrics["queue2_in_flight"] = int(in_flight)
        metrics["queue2_waiting"] = waiting
        completed = int(metrics["qwen_completed"])
        metrics["qwen_average_seconds"] = round(float(metrics["qwen_total_seconds"]) / completed, 4) if completed else 0.0
        metrics["end_to_end_average_seconds"] = round(float(metrics["end_to_end_total_seconds"]) / completed, 4) if completed else 0.0
        return metrics

    def process_session(self, session_id: str) -> None:
        """The old video polling path is intentionally inactive for real-time monitoring."""
        return None

    @staticmethod
    def _stage(session_id: str, stage: str, operation: Callable[[], object]) -> object:
        """Retained for legacy worker diagnostics outside the real-time path."""
        try:
            return operation()
        except Exception as error:
            LOGGER.exception(
                "monitoring stage_failed session_id=%s stage=%s error_type=%s",
                session_id,
                stage,
                type(error).__name__,
            )
            raise

    @staticmethod
    def _data_uri(path: Path) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
