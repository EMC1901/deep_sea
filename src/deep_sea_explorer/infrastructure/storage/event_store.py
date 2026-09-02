from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path

from deep_sea_explorer.services.key_frame_detection import CandidateEvent, SurveyEventEvaluation


class EventStore:
    """Durable candidate/formal capture storage rooted under the application data directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.candidate_root = self.root / "candidates"
        self.capture_root = self.root / "captures"
        self.candidate_root.mkdir(parents=True, exist_ok=True)
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "events.sqlite3"
        self._lock = threading.RLock()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_time REAL NOT NULL,
                    event_type TEXT NOT NULL, trigger_type TEXT NOT NULL, element_category TEXT,
                    element_name TEXT, description TEXT NOT NULL, confidence REAL NOT NULL,
                    image_path TEXT NOT NULL, yolo_track_ids TEXT NOT NULL, visual_fingerprint TEXT NOT NULL
                )"""
            )
            connection.commit()

    def save_candidate(self, candidate: CandidateEvent) -> Path:
        target_dir = self.candidate_root / candidate.session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{candidate.captured_at:.6f}_{candidate.candidate_id}.jpg"
        shutil.copy2(candidate.current_image_path, target)
        return target

    def save_reference(self, session_id: str, source: Path) -> Path:
        target_dir = self.candidate_root / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "scene-reference.jpg"
        shutil.copy2(source, target)
        return target

    def accept(self, candidate: CandidateEvent, evaluation: SurveyEventEvaluation, source: Path | None = None) -> Path:
        source = source or candidate.current_image_path
        target_dir = self.capture_root / candidate.session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{candidate.captured_at:.6f}_{uuid.uuid4().hex}.jpg"
        shutil.copy2(source, target)
        elements = evaluation.new_elements or evaluation.observed_elements or (
            {"category": "other", "name": "", "is_new": True},
        )
        with self._lock, sqlite3.connect(self.database) as connection:
            for element in elements:
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), candidate.session_id, candidate.captured_at, evaluation.event_type,
                     candidate.trigger_type, str(element.get("category", "other")), str(element.get("name", "")),
                     evaluation.description, evaluation.confidence, str(target),
                     json.dumps([item.get("track_id") for item in candidate.yolo_changes]), candidate.signature),
                )
            connection.commit()
        return target

    def list_events(self, session_id: str) -> list[dict[str, object]]:
        with self._lock, sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM events WHERE session_id=? ORDER BY event_time", (session_id,)).fetchall()
        return [dict(row) for row in rows]
