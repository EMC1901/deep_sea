"""Export persisted key frames with reproducible Top-K image-retrieval matches."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_sea_explorer.domain.retrieval import ImageRetrievalQuery, RetrievedImage  # noqa: E402
from deep_sea_explorer.infrastructure.retrieval.dinov2 import DinoV2ImageEncoder  # noqa: E402
from deep_sea_explorer.infrastructure.retrieval.gateway import LocalImageRetrievalGateway  # noqa: E402
from deep_sea_explorer.infrastructure.retrieval.numpy_index import NumpyImageRetrievalIndex  # noqa: E402


class RetrievalGateway(Protocol):
    def retrieve(self, query: ImageRetrievalQuery) -> tuple[RetrievedImage, ...]: ...


def _contained_path(value: str, root: Path) -> Path | None:
    path = Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def load_key_frames(data_dir: Path, session_id: str | None) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Load unique accepted frames from the event store without changing it."""

    database = data_dir / "events.sqlite3"
    capture_root = data_dir / "captures"
    if not database.is_file():
        raise RuntimeError("no event database exists under the configured data directory")

    query = "SELECT event_id, session_id, event_time, event_type, element_category, element_name, description, confidence, image_path FROM events"
    arguments: tuple[str, ...] = ()
    if session_id:
        query += " WHERE session_id=?"
        arguments = (session_id,)
    query += " ORDER BY event_time, event_id"

    events_by_image: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query, arguments):
            image_path = _contained_path(str(row["image_path"]), capture_root)
            if image_path is None:
                continue
            events_by_image[image_path].append(dict(row))
    return sorted(events_by_image.items(), key=lambda item: item[0].name)


def _copy_image(source: Path, destination: Path) -> str:
    suffix = source.suffix.lower() if source.suffix else ".jpg"
    target = destination.with_suffix(suffix)
    shutil.copy2(source, target)
    return target.name


def export_review(
    data_dir: Path,
    output_dir: Path,
    retrieval: RetrievalGateway,
    *,
    top_k: int,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Copy frames and their fresh retrieval matches to an audit-friendly directory."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("review output directory already exists and is not empty")

    key_frames = load_key_frames(data_dir, session_id)
    if not key_frames:
        raise RuntimeError("no accepted key frames are available for the requested session")

    frames_dir = output_dir / "key_frames"
    matches_dir = output_dir / "similar_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir()
    manifest: dict[str, Any] = {
        "format": 1,
        "session_id": session_id,
        "top_k": top_k,
        "frames": [],
    }

    for sequence, (frame_path, events) in enumerate(key_frames, start=1):
        frame_id = f"{sequence:04d}"
        frame_file = _copy_image(frame_path, frames_dir / f"{frame_id}_key_frame")
        frame_matches = retrieval.retrieve(ImageRetrievalQuery(image_path=frame_path, k=top_k))
        match_directory = matches_dir / frame_id
        match_directory.mkdir()
        results: list[dict[str, Any]] = []
        for rank, result in enumerate(frame_matches, start=1):
            copied_name: str | None = None
            if result.image_path is not None and result.image_path.is_file():
                copied_name = _copy_image(result.image_path, match_directory / f"{rank:02d}_similar")
            results.append(
                {
                    "rank": rank,
                    "image_file": copied_name,
                    "image_id": result.image_id,
                    "similarity": result.similarity,
                    "site": result.site,
                    "labels": {key: list(value) for key, value in result.labels.items()},
                }
            )
        manifest["frames"].append(
            {
                "sequence": sequence,
                "key_frame_file": f"key_frames/{frame_file}",
                "events": events,
                "matches": results,
            }
        )

    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--dino-model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--session-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = NumpyImageRetrievalIndex.from_directory(args.index_dir)
    retrieval = LocalImageRetrievalGateway(
        DinoV2ImageEncoder(args.dino_model_path, args.device), index
    )
    manifest = export_review(
        args.data_dir,
        args.output_dir,
        retrieval,
        top_k=args.top_k,
        session_id=args.session_id,
    )
    print(f"exported_frames={len(manifest['frames'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
