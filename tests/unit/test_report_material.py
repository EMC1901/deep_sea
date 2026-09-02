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


def test_report_summary_material_reserves_space_for_each_report_section() -> None:
    material = {
        "memos": [{"time": "12:00:00", "text": "监测记录" * 100} for _ in range(20)],
        "chats": [{"role": "AI助手", "text": "问答记录" * 100} for _ in range(20)],
        "bio_samples": [{"name": "深海鱼", "description": "生物样本" * 100} for _ in range(20)],
        "env_samples": [{"name": "软底质", "description": "环境样本" * 100} for _ in range(20)],
        "bio_stats": [{"name": "深海鱼", "count": 1} for _ in range(20)],
        "env_stats": [{"name": "软底质", "count": 1} for _ in range(20)],
        "meta": {"session_id": "mission-001", "time_range": "12:00:00 ~ 12:30:00"},
    }

    compacted = compact_report_material(material)
    serialized = json.dumps(compacted, ensure_ascii=False)

    assert all(compacted[name] for name in ("memos", "chats", "bio_samples", "env_samples"))
    assert len(compacted["memos"]) == MAX_COLLECTION_ITEMS
    assert len(compacted["bio_stats"]) == MAX_COLLECTION_ITEMS
    # Even if every retained character tokenized independently, prompt and
    # generation still fit comfortably within the deployed 4,096-token limit.
    assert len(serialized) < 2_500


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
