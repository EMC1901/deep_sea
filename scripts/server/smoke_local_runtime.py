"""Server-only S8 verification: call all four models through local gateways."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("unable to create S8 test video")
    try:
        for value in (24, 72, 120, 168):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def main() -> int:
    os.environ["MODEL_BACKEND"] = "local"
    os.environ["MODEL_SERVICE_ENABLED"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from deep_sea_explorer.config import Settings
    from deep_sea_explorer.container import build_local_container

    result: dict[str, object] = {"status": "failed"}
    video_path: Path | None = None
    try:
        settings = Settings.from_env()
        errors = settings.validate_for_runtime()
        if errors:
            raise RuntimeError("invalid S8 local configuration: " + "; ".join(errors))
        temp_dir = settings.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / "s8-local-runtime-smoke.avi"
        write_video(video_path)
        container = build_local_container(settings)
        description = container.vision.describe_video(video_path)
        memo = container.memo_embedding.embed(["deep sea observation"])
        rag = container.rag_embedding.embed(["deep sea observation"])
        image = container.image.generate("scientific illustration of a deep sea hydrothermal vent")
        if not description.strip() or len(memo[0]) != 768 or len(rag[0]) != 384:
            raise RuntimeError("S8 local model output validation failed")
        if not image.startswith(b"\xff\xd8"):
            raise RuntimeError("S8 local image output is not JPEG")
        result = {
            "status": "passed",
            "qwen_text": True,
            "gte_dimension": len(memo[0]),
            "minilm_dimension": len(rag[0]),
            "image_jpeg": True,
        }
        return 0
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__}
        return 1
    finally:
        if video_path is not None:
            video_path.unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    raise SystemExit(main())
