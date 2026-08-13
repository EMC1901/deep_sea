"""Exercise the real monitoring API with frames sampled from a video file."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path


def post_frame(api_url: str, session_id: str, image: bytes) -> dict[str, object]:
    boundary = f"----monitoring-{uuid.uuid4().hex}"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="video"; filename="frame.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{api_url}/videoanalyze",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Session-ID": session_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read())


def get_memos(api_url: str, session_id: str) -> list[dict[str, object]]:
    request = urllib.request.Request(f"{api_url}/memos", headers={"X-Session-ID": session_id})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read()).get("memos", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:9001")
    parser.add_argument("--frame-step", type=int, default=150)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--wait-seconds", type=int, default=180)
    args = parser.parse_args()

    import cv2

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    session_id = f"video-acceptance-{uuid.uuid4().hex[:10]}"
    try:
        for offset in range(args.frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, offset * args.frame_step)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"cannot decode source frame {offset * args.frame_step}")
            sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise SystemExit("cannot JPEG-encode video frame")
            result = post_frame(args.api_url, session_id, encoded.tobytes())
            print(json.dumps({"source_frame": offset * args.frame_step, "sharpness": round(sharpness, 3), "result": result}, ensure_ascii=False))
            time.sleep(0.2)
    finally:
        capture.release()

    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        memos = get_memos(args.api_url, session_id)
        if memos:
            print(json.dumps({"session_id": session_id, "memos": memos}, ensure_ascii=False))
            return
        time.sleep(3)
    raise SystemExit("timed out waiting for a Qwen monitoring memo")


if __name__ == "__main__":
    main()
