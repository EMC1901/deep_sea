from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from deep_sea_explorer.domain.models import Capture, Memo


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
        state.last_analyzed_video = state.latest_video
        content = self.vision.describe_video(video_path)
        vector = tuple(self.embedding.embed([content])[0])
        if state.last_memo_embedding:
            similarity = sum(left * right for left, right in zip(vector, state.last_memo_embedding))
            if similarity >= self.threshold:
                return None
        state.last_memo_embedding = vector
        capture = None
        try:
            frame_path = self.files.frame_path(session_id)
            self._last_frame(video_path, frame_path)
            decision = self.vision.evaluate_frame(frame_path)
            if decision.is_deepsea and decision.is_typical:
                if decision.category.value == "bio":
                    organisms = self.stats.update(state, decision.category, decision.organisms)
                    capture = Capture(
                        decision.category,
                        self._data_uri(frame_path),
                        decision.description,
                        organisms=organisms,
                    )
                else:
                    features = self.stats.update(state, decision.category, decision.env_features)
                    capture = Capture(
                        decision.category,
                        self._data_uri(frame_path),
                        decision.description,
                        env_features=features,
                    )
        finally:
            if "frame_path" in locals():
                frame_path.unlink(missing_ok=True)
        memo = Memo(datetime.now().strftime("%H:%M:%S"), content, session_id, capture)
        self.broker.publish(memo)
        return memo

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
