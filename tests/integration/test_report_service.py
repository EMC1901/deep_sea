from __future__ import annotations

from pathlib import Path

from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_fake_container


def test_fake_report_service_generates_a_readable_pdf() -> None:
    root = Path(".deep-sea-explorer-tmp-test")
    container = build_fake_container(Settings(model_backend=ModelBackend.FAKE, temp_dir=root))
    target = container.reports.generate({"memos": [], "chats": []})
    try:
        assert target.read_bytes().startswith(b"%PDF")
    finally:
        target.unlink(missing_ok=True)
        for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            directory.rmdir()
        root.rmdir()
