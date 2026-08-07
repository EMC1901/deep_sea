"""Create disposable, non-sensitive media files for S9 loopback API checks."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    import cv2
    import numpy as np

    root = Path(os.environ["S9_API_TEST_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "s9-api-check.jpg"
    video_path = root / "s9-api-check.mp4"

    image = np.full((64, 64, 3), (90, 50, 20), dtype=np.uint8)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError("unable to create S9 test image")
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("unable to create S9 test video")
    try:
        for value in (20, 60, 100, 140):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    print(json.dumps({"image": image_path.name, "video": video_path.name}))


if __name__ == "__main__":
    main()
