from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from dataclasses import asdict
from typing import TypeVar

from deep_sea_explorer.domain.models import Capture, Memo, SessionState
from deep_sea_explorer.infrastructure.storage.event_store import EventStore
from deep_sea_explorer.services.candidate_queue import PerSessionEventQueue
from deep_sea_explorer.services.key_frame_detection import (
    ByteTrackState,
    NullObjectDetector,
    SceneChangeDetector,
    make_candidate,
)


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class MonitoringService:
    def __init__(
        self,
        vision: object,
        embedding: object,
        sessions: object,
        broker: object,
        files: object,
        stats: object,
        threshold: float,
        detector: object | None = None,
        scene_detector: SceneChangeDetector | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        (
            self.vision,
            self.embedding,
            self.sessions,
            self.broker,
            self.files,
            self.stats,
            self.threshold,
        ) = vision, embedding, sessions, broker, files, stats, threshold
        self.detector = detector or NullObjectDetector()
        self.scene_detector = scene_detector or SceneChangeDetector()
        self.event_store = event_store or EventStore(files.root / "data")
        self._trackers: dict[str, ByteTrackState] = {}
        self._confirmed_tracks: dict[str, set[int]] = {}
        self._scene_detectors: dict[str, SceneChangeDetector] = {}
        self._queues = PerSessionEventQueue(self._evaluate_candidate, self._complete_candidate)

    def process_frame(self, session_id: str, frame: bytes | Path) -> dict[str, object]:
        """Ingest one JPEG and emit at most one replaceable candidate per session."""
        state = self.sessions.get(session_id)
        frame_path = self.files.frame_path(session_id)
        try:
            if isinstance(frame, Path):
                frame_path.write_bytes(frame.read_bytes())
            else:
                frame_path.write_bytes(frame)
            if not frame_path.read_bytes().startswith(b"\xff\xd8"):
                raise ValueError("frame must be a JPEG")
            tracker = self._trackers.setdefault(session_id, ByteTrackState())
            detections = self.detector.detect(frame_path)
            tracks, _ = tracker.update(detections)
            confirmed = self._confirmed_tracks.setdefault(session_id, set())
            new_tracks = [track for track in tracks if track.continuous_frames >= tracker.confirm_frames and track.track_id not in confirmed]
            confirmed.update(track.track_id for track in new_tracks)
            scene_detector = self._scene_detectors.setdefault(session_id, SceneChangeDetector(self.scene_detector.threshold, self.scene_detector.confirm_frames))
            reference = Path(state.last_scene_reference) if state.last_scene_reference else scene_detector.reference
            had_reference = reference is not None and reference.is_file()
            metrics = scene_detector.compare(frame_path, reference)
            if not had_reference:
                reference = self.event_store.save_reference(session_id, frame_path)
                scene_detector.accept(reference)
            trigger_parts = []
            if new_tracks:
                trigger_parts.append("yolo")
            if metrics.changed:
                trigger_parts.append("scene")
            if not trigger_parts:
                return {"status": "monitoring", "candidate": False, "scene_change_metrics": asdict(metrics)}
            candidate = make_candidate(session_id, frame_path, reference, new_tracks, metrics, "+".join(trigger_parts))
            candidate_path = self.event_store.save_candidate(candidate)
            candidate = candidate.__class__(candidate.candidate_id, candidate.session_id, candidate.captured_at, candidate_path, candidate.reference_image_path, candidate.yolo_changes, candidate.scene_change_metrics, candidate.trigger_type, candidate.signature)
            state.pending_candidate = candidate
            state.model_task_in_flight = self._queues.state(session_id)[0]
            self._queues.submit(candidate)
            state.model_task_in_flight = True
            return {"status": "candidate_pending", "candidate": True, "candidate_id": candidate.candidate_id, "scene_change_metrics": asdict(metrics)}
        finally:
            frame_path.unlink(missing_ok=True)

    def _evaluate_candidate(self, candidate):
        metadata = {"yolo_changes": list(candidate.yolo_changes), "scene_change_metrics": candidate.scene_change_metrics, "trigger_type": candidate.trigger_type}
        return self.vision.evaluate_survey_event(candidate.reference_image_path, candidate.current_image_path, metadata)

    def _complete_candidate(self, candidate, evaluation) -> None:
        state = self.sessions.get(candidate.session_id)
        state.model_task_in_flight = False
        state.last_model_call_time = datetime.now().timestamp()
        state.pending_candidate = None
        accepted = evaluation.survey_value and evaluation.event_type in {"new_element", "major_scene_change"} and bool(evaluation.new_elements or evaluation.scene_changed)
        if not accepted or state.active_event_signature == candidate.signature:
            return
        image_path = self.event_store.accept(candidate, evaluation)
        state.active_event_signature = candidate.signature
        state.last_accepted_frame = str(image_path)
        state.last_scene_reference = str(candidate.current_image_path)
        self._scene_detectors.setdefault(candidate.session_id, self.scene_detector).accept(candidate.current_image_path)
        self.broker.publish(Memo(datetime.now().strftime("%H:%M:%S"), evaluation.description, candidate.session_id))

    def process_session(self, session_id: str) -> Memo | None:
        state = self.sessions.get(session_id)
        if (
            state.is_answering
            or not state.latest_video
            or state.latest_video == state.last_analyzed_video
        ):
            return None
        video_path = Path(state.latest_video)
        content = self._stage(
            session_id, "describe_video", lambda: self.vision.describe_video(video_path)
        )
        vector = tuple(
            self._stage(session_id, "memo_embedding", lambda: self.embedding.embed([content]))[0]
        )
        if state.last_memo_embedding:
            similarity = sum(left * right for left, right in zip(vector, state.last_memo_embedding))
            if similarity >= self.threshold:
                state.last_analyzed_video = str(video_path)
                return None
        state.last_memo_embedding = vector
        capture = self._capture_or_none(session_id, state, video_path)
        memo = Memo(datetime.now().strftime("%H:%M:%S"), content, session_id, capture)
        self.broker.publish(memo)
        state.last_analyzed_video = str(video_path)
        return memo

    def _capture_or_none(
        self,
        session_id: str,
        state: SessionState,
        video_path: Path,
    ) -> Capture | None:
        frame_path: Path | None = None
        stage = "frame_path"
        try:
            frame_path = self.files.frame_path(session_id)
            stage = "extract_last_frame"
            self._last_frame(video_path, frame_path)
            stage = "evaluate_frame"
            decision = self.vision.evaluate_frame(frame_path)
            if decision.is_deepsea and decision.is_typical:
                if decision.category.value == "bio":
                    stage = "update_bio_stats"
                    organisms = self.stats.update(state, decision.category, decision.organisms)
                    stage = "encode_capture"
                    return Capture(
                        decision.category,
                        self._data_uri(frame_path),
                        decision.description,
                        organisms=organisms,
                    )
                else:
                    stage = "update_env_stats"
                    features = self.stats.update(state, decision.category, decision.env_features)
                    stage = "encode_capture"
                    return Capture(
                        decision.category,
                        self._data_uri(frame_path),
                        decision.description,
                        env_features=features,
                    )
            return None
        except Exception as error:
            LOGGER.exception(
                "monitoring capture_degraded session_id=%s stage=%s error_type=%s",
                session_id,
                stage,
                type(error).__name__,
            )
            return None
        finally:
            if frame_path is not None:
                frame_path.unlink(missing_ok=True)

    @staticmethod
    def _stage(session_id: str, stage: str, operation: Callable[[], T]) -> T:
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
    def _last_frame(video_path: Path, output: Path) -> None:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError("video has no readable frame")
        cv2.imwrite(str(output), frame)

    @staticmethod
    def _data_uri(path: Path) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
