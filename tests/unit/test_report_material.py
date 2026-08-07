from __future__ import annotations

import json
from pathlib import Path

from deep_sea_explorer.domain.report_material import (
    MAX_COLLECTION_ITEMS,
    SUMMARY_TEXT_BUDGET,
    compact_report_material,
)
from deep_sea_explorer.services.report_service import ReportService


def test_report_summary_material_removes_images_and_bounds_text() -> None:
    material = {
        "memos": [{"time": "12:00:00", "text": "observation " * 5_000}],
        "chats": [
            {
                "role": "assistant",
                "text": "answer",
                "image": "data:image/jpeg;base64," + "A" * 500_000,
            }
        ],
        "bio_samples": [
            {
                "name": "fish",
                "description": "sample",
                "image_data_uri": "data:image/jpeg;base64," + "B" * 500_000,
            }
        ],
        "env_stats": [{"name": f"feature-{index}", "count": 1} for index in range(500)],
    }

    compacted = compact_report_material(material)
    serialized = json.dumps(compacted)

    assert "data:image" not in serialized
    assert "image_data_uri" not in serialized
    assert '"image"' not in serialized
    assert len(compacted["env_stats"]) == MAX_COLLECTION_ITEMS
    assert len(serialized) < SUMMARY_TEXT_BUDGET + 20_000


def test_report_service_summarizes_compact_material_but_renders_original() -> None:
    class RecordingVision:
        summary_material: dict[str, object] | None = None

        def summarize_report(self, material: dict[str, object]) -> str:
            self.summary_material = material
            return "summary"

    class RecordingRenderer:
        rendered_material: dict[str, object] | None = None

        def render(
            self,
            target: Path,
            material: dict[str, object],
            summary: str,
        ) -> Path:
            assert summary == "summary"
            self.rendered_material = material
            return target

    class Files:
        def report_path(self) -> Path:
            return Path("report.pdf")

    vision = RecordingVision()
    renderer = RecordingRenderer()
    material = {
        "chats": [{"text": "answer", "image": "data:image/jpeg;base64,AAA"}],
    }

    target = ReportService(vision, renderer, Files()).generate(material)

    assert target == Path("report.pdf")
    assert vision.summary_material == {"chats": [{"text": "answer"}]}
    assert renderer.rendered_material is material
