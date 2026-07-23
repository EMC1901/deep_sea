"""迁移期兼容入口；实际应用位于 deep_sea_explorer.main。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_sea_explorer.api.app_factory import create_app
from deep_sea_explorer.config import Settings
from deep_sea_explorer.main import main

app = create_app(Settings.from_env())


if __name__ == "__main__":
    main()
