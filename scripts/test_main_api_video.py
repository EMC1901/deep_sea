"""Submit sampled frames from a local video through the development-machine API."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import cv2
import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="local MP4 video to sample")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:9001")
    parser.add_argument("--question", default="请描述这段视频中的主要内容。")
    return parser.parse_args()


def _sample_jpeg_frames(video_path: Path, *, limit: int = 8) -> list[bytes]:
    if not video_path.is_file():
        raise ValueError(f"video file does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 4:
            raise ValueError("video must contain at least four readable frames")
        positions = {round(index * (total - 1) / (min(total, limit) - 1)) for index in range(min(total, limit))}
        frames: list[bytes] = []
        position = 0
        while position <= max(positions):
            ok, frame = capture.read()
            if not ok:
                break
            if position in positions:
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encoded:
                    raise ValueError(f"cannot encode video frame {position}")
                frames.append(jpeg.tobytes())
            position += 1
    finally:
        capture.release()
    if len(frames) < 4:
        raise ValueError("video must yield at least four JPEG frames")
    return frames


def main() -> int:
    args = _parse_args()
    frames = _sample_jpeg_frames(args.video)
    files = [
        ("video", (f"frame-{index:02d}.jpg", frame, "image/jpeg"))
        for index, frame in enumerate(frames)
    ]
    headers = {"X-Session-ID": str(uuid.uuid4())}
    url = args.api_base_url.rstrip("/") + "/videoanalyze"
    print(f"Submitting {len(frames)} sampled frames to the development API.")
    with httpx.Client(timeout=httpx.Timeout(240.0, connect=5.0)) as client:
        with client.stream("POST", url, headers=headers, data={"question": args.question}, files=files) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                print(json.dumps(event, ensure_ascii=False))
                if event.get("type") == "error":
                    return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, ValueError) as error:
        print(f"Video API verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
