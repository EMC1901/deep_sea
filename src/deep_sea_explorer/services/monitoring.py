from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import TypeVar

from deep_sea_explorer.domain.models import Capture, Memo, SessionState


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
