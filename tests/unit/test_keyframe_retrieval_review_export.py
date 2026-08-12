from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

from deep_sea_explorer.domain.retrieval import RetrievedImage


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_keyframe_retrieval_review.py"
SPEC = importlib.util.spec_from_file_location("retrieval_review_export", SCRIPT_PATH)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class StubRetrievalGateway:
    def __init__(self, image: Path) -> None:
        self.image = image
        self.queries: list[Path] = []

    def retrieve(self, query):
        self.queries.append(query.image_path)
        return (
            RetrievedImage(
                image_id="gallery/example.jpg",
                labels={"biota": ("Biota > Fish",)},
                similarity=0.9,
                site="site-a",
                image_path=self.image,
            ),
        )


def test_export_review_copies_unique_key_frames_and_retrieval_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    capture = data_dir / "captures" / "session-a" / "frame.jpg"
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"key-frame")
    gallery = tmp_path / "gallery.jpg"
    gallery.write_bytes(b"retrieval-match")
    database = data_dir / "events.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE events (
                event_id TEXT, session_id TEXT, event_time REAL, event_type TEXT,
                trigger_type TEXT, element_category TEXT, element_name TEXT,
                description TEXT, confidence REAL, image_path TEXT,
                yolo_track_ids TEXT, visual_fingerprint TEXT
            )"""
        )
        for event_id in ("one", "two"):
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, "session-a", 1.0, "new", "scene", "organism", "fish", "desc", 1.0,
                 str(capture), "[]", "signature"),
            )
        connection.commit()

    retrieval = StubRetrievalGateway(gallery)
    output = tmp_path / "review"
    manifest = EXPORTER.export_review(data_dir, output, retrieval, top_k=1, session_id="session-a")

    assert len(manifest["frames"]) == 1
    assert len(manifest["frames"][0]["events"]) == 2
    assert retrieval.queries == [capture]
    assert (output / "key_frames" / "0001_key_frame.jpg").read_bytes() == b"key-frame"
    assert (output / "similar_images" / "0001" / "01_similar.jpg").read_bytes() == b"retrieval-match"
    assert (output / "review_manifest.json").is_file()
