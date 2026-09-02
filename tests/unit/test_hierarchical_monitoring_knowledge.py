from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from deep_sea_explorer.services.hierarchical_monitoring_knowledge import HierarchicalMonitoringKnowledgeBase


def _record(**fields: object) -> dict[str, object]:
    return fields


def test_builder_discards_incomplete_records_deduplicates_and_builds_parent_indexes(tmp_path: Path) -> None:
    substrate = {
        "records": [
            _record(Component=["Component", "底质组件"], Class=["Class", "软底"], Subclass=["Subclass", "沙质基底"], Group=["Group", "细沙"], Definition=["Definition", "细颗粒沉积物。"]),
            _record(Component=["Component", "底质组件"], Class=["Class", "软底"], Subclass=["Subclass", "沙质基底"], Group=["Group", "细沙"], Definition=["Definition", "细颗粒沉积物。"]),
            _record(Component=["Component", "底质组件"], Class=["", ""], Subclass=["Subclass", "应丢弃"], Group=["Group", "应丢弃"], Definition=["Definition", "应丢弃。"]),
        ]
    }
    biotic = {
        "records": [
            _record(Component=["Component", "生物组件"], Class=["Class", "刺胞动物"], Subclass=["Subclass", "珊瑚"], Definition=["Definition", "固着海洋动物。"]),
            _record(Component=["Component", "生物组件"], Class=["Class", "刺胞动物"], Subclass=["Subclass", "珊瑚"], Definition=["Definition", "固着海洋动物。"]),
            _record(Component=["Component", "生物组件"], Class=["Class", "应丢弃"], Subclass=["", ""], Definition=["Definition", "应丢弃。"]),
        ]
    }
    substrate_path = tmp_path / "substrate.json"
    biotic_path = tmp_path / "biotic.json"
    output = tmp_path / "hierarchical_label_knowledge.json"
    substrate_path.write_text(json.dumps(substrate, ensure_ascii=False), encoding="utf-8")
    biotic_path.write_text(json.dumps(biotic, ensure_ascii=False), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_hierarchical_monitoring_knowledge.py"
    subprocess.run(
        [sys.executable, str(script), "--substrate", str(substrate_path), "--biotic", str(biotic_path), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["substrate"]) == 1
    assert len(payload["biotic"]) == 1
    knowledge = HierarchicalMonitoringKnowledgeBase(tmp_path)
    assert knowledge.substrate_classes() == ("软底",)
    assert knowledge.substrate_subclasses("软底") == ("沙质基底",)
    assert knowledge.substrate_groups("软底", "沙质基底") == ("细沙",)
    assert knowledge.biotic_classes() == ("刺胞动物",)
    assert knowledge.biotic_subclasses(("刺胞动物",)) == ("珊瑚",)
    path = knowledge.substrate_path("软底", "沙质基底", "细沙")
    assert path is not None
    assert knowledge.reference_for(path, ("刺胞动物",), ("珊瑚",)) == {
        "substrate": [{"label": "细沙", "definitions": ("细颗粒沉积物。",)}],
        "biotic": [{"label": "珊瑚", "definitions": ("固着海洋动物。",)}],
    }
