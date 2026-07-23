from __future__ import annotations

import time
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename


class TempFileStore:
    def __init__(self, root: Path, ttl_seconds: int) -> None:
        self.root = root.resolve()
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _inside_root(self, candidate: Path) -> Path:
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("file path escapes configured temporary root")
        return resolved

    def _session_dir(self, session_id: str) -> Path:
        safe = secure_filename(session_id)
        if not safe:
            raise ValueError("invalid session id")
        directory = self._inside_root(self.root / "sessions" / safe)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def session_video_path(self, session_id: str) -> Path:
        return self._inside_root(self._session_dir(session_id) / f"video_{uuid.uuid4().hex}.mp4")

    def frame_path(self, session_id: str) -> Path:
        return self._inside_root(self._session_dir(session_id) / f"frame_{uuid.uuid4().hex}.jpg")

    def upload_path(self, filename: str) -> Path:
        safe = secure_filename(filename) or "upload.pdf"
        directory = self._inside_root(self.root / "uploads")
        directory.mkdir(parents=True, exist_ok=True)
        return self._inside_root(directory / f"{uuid.uuid4().hex}_{safe}")

    def report_path(self) -> Path:
        directory = self._inside_root(self.root / "reports")
        directory.mkdir(parents=True, exist_ok=True)
        return self._inside_root(directory / f"report_{uuid.uuid4().hex}.pdf")

    def cleanup(self, dry_run: bool = True) -> list[Path]:
        cutoff = time.time() - self.ttl_seconds
        candidates = [
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.stat().st_mtime < cutoff
        ]
        if not dry_run:
            for path in candidates:
                self._inside_root(path).unlink(missing_ok=True)
            for directory in sorted(
                (path for path in self.root.rglob("*") if path.is_dir()), reverse=True
            ):
                if not any(directory.iterdir()):
                    directory.rmdir()
        return candidates
