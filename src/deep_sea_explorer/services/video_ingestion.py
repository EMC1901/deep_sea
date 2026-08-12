from __future__ import annotations

from pathlib import Path
from typing import Protocol

from deep_sea_explorer.domain.exceptions import ValidationError
from deep_sea_explorer.ports.file_store import FileStore
from deep_sea_explorer.ports.session_store import SessionStore


class FrameMonitoringGateway(Protocol):
    def process_frame(self, session_id: str, frame: bytes) -> dict[str, object]: ...


class VideoIngestionService:
    def __init__(self, store: FileStore, sessions: SessionStore, max_frames: int) -> None:
        self.store, self.sessions, self.max_frames = store, sessions, max_frames

    def ingest(self, session_id: str, frames: list[bytes]) -> Path:
        if not frames or len(frames) > self.max_frames:
            raise ValidationError("frame count is invalid")
        try:
            import cv2
            import numpy as np
        except ImportError as error:  # pragma: no cover - declared common dependency
            raise ValidationError("OpenCV is unavailable") from error
        decoded = [
            cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames
        ]
        if decoded[0] is None:
            raise ValidationError("first frame is not a valid JPEG")
        height, width = decoded[0].shape[:2]
        target = self.store.session_video_path(session_id)
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
        writer = cv2.VideoWriter(str(target), fourcc, 10, (width, height))
        if not writer.isOpened():
            raise ValidationError("video writer cannot be opened")
        try:
            for frame in decoded:
                if frame is not None:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            writer.release()
        self.sessions.get(session_id).latest_video = str(target)
        return target

    def ingest_frame(
        self,
        session_id: str,
        frame: bytes,
        monitoring: FrameMonitoringGateway,
    ) -> dict[str, object]:
        """Forward one browser JPEG to event-driven monitoring without creating a video."""
        if not frame or len(frame) > 10 * 1024 * 1024:
            raise ValidationError("frame is empty or too large")
        try:
            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        except ImportError as error:  # pragma: no cover
            raise ValidationError("OpenCV is unavailable") from error
        if decoded is None:
            raise ValidationError("frame is not a valid JPEG")
        return monitoring.process_frame(session_id, frame)
