"""Protocol for report renderers used by the application service."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ReportRenderer(Protocol):
    def render(self, target: Path, material: dict[str, object], summary: str) -> Path: ...
