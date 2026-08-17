"""Read the compact offline label knowledge base for real-time monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


MODEL_FIELDS = {"bio": "organisms", "substrate": "substrates", "geomorphology": "geomorphologies"}
DISPLAY_NAMES_PATH = Path(__file__).resolve().parents[1] / "resources" / "label_chinese_names.json"


@dataclass(frozen=True, slots=True)
class LabelBatch:
    category: str
    labels: tuple[str, ...]


class MonitoringKnowledgeBase:
    """Immutable whitelist and description lookup loaded once during startup."""

    def __init__(self, directory: Path, *, batch_size: int = 64) -> None:
        if batch_size < 1:
            raise ValueError("label batch size must be positive")
        path = directory / "label_universe.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            groups = payload["labels"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("label knowledge base is unavailable") from error
        descriptions: dict[str, dict[str, str]] = {}
        for category in MODEL_FIELDS:
            values = groups.get(category)
            if not isinstance(values, list):
                raise ValueError(f"label knowledge base misses {category}")
            current: dict[str, str] = {}
            for value in values:
                if not isinstance(value, dict):
                    continue
                label, description = value.get("canonical_label"), value.get("description")
                if isinstance(label, str) and label.strip() and isinstance(description, str) and description.strip():
                    current[label.strip()] = description.strip()
            descriptions[category] = current
        if not any(descriptions.values()):
            raise ValueError("label knowledge base has no completed descriptions")
        self._descriptions = descriptions
        self._display_names = self._load_display_names()
        self.batch_size = batch_size

    @staticmethod
    def _load_display_names() -> dict[str, str]:
        try:
            payload = json.loads(DISPLAY_NAMES_PATH.read_text(encoding="utf-8"))
            entries = payload["labels"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("label Chinese-name dictionary is unavailable") from error
        if not isinstance(entries, dict):
            raise ValueError("label Chinese-name dictionary is invalid")
        result: dict[str, str] = {}
        for label, entry in entries.items():
            name = entry.get("chinese_name") if isinstance(entry, dict) else None
            if isinstance(label, str) and label.strip() and isinstance(name, str) and name.strip():
                result[label.strip()] = name.strip()
        if not result:
            raise ValueError("label Chinese-name dictionary has no entries")
        return result

    def batches(self) -> tuple[LabelBatch, ...]:
        result: list[LabelBatch] = []
        for category in MODEL_FIELDS:
            labels = tuple(sorted(self._descriptions[category]))
            for start in range(0, len(labels), self.batch_size):
                result.append(LabelBatch(category, labels[start : start + self.batch_size]))
        return tuple(result)

    def allowed(self, category: str) -> frozenset[str]:
        return frozenset(self._descriptions.get(category, {}))

    def descriptions_for(self, selections: dict[str, tuple[str, ...]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for category, labels in selections.items():
            known = self._descriptions.get(category, {})
            result.update({label: known[label] for label in labels if label in known})
        return result

    def display_name(self, category: str, label: str) -> str:
        """Return the reviewed Chinese name, retaining reserved unknown labels as-is."""
        if label.startswith("未知"):
            return label
        if label not in self._descriptions.get(category, {}):
            return label
        return self._display_names.get(label, label)

    def missing_display_names(self) -> dict[str, tuple[str, ...]]:
        """Expose incomplete catalog coverage so deployment validation can fail closed."""
        return {
            category: tuple(sorted(set(labels) - set(self._display_names)))
            for category, labels in self._descriptions.items()
            if set(labels) - set(self._display_names)
        }
