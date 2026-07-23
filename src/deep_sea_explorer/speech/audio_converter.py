from __future__ import annotations

import subprocess
from pathlib import Path


class AudioConverter:
    def __init__(self, ffmpeg_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path

    def convert_webm_to_wav(self, source: Path, target: Path) -> None:
        result = subprocess.run(
            [
                self.ffmpeg_path,
                "-y",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("audio conversion failed")
