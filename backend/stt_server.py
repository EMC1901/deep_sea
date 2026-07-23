"""迁移期语音兼容入口；唯一实现位于 deep_sea_explorer.speech。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_sea_explorer.speech.app_factory import create_speech_app
from deep_sea_explorer.speech_main import main

app = create_speech_app()


if __name__ == "__main__":
    main()
