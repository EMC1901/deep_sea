"""Deprecated compatibility wrapper; the sole implementation is in src/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_sea_explorer.speech_main import main


if __name__ == "__main__":
    main()
