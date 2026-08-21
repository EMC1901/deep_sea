"""Read the curated CMECS hierarchy used by real-time image monitoring."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
PACKAGED_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "resources" / "hierarchical_label_knowledge.json"


@dataclass(frozen=True, slots=True)
class SubstratePath:
    component: str
    label_class: str
    subclass: str
    group: str


@dataclass(frozen=True, slots=True)
class BioticPath:
    component: str
    label_class: str
    subclass: str


class HierarchicalMonitoringKnowledgeBase:
    """Immutable Chinese CMECS labels, indexed by the matching hierarchy."""

    def __init__(self, directory: Path) -> None:
        path = directory / "hierarchical_label_knowledge.json"
        if not path.is_file():
            path = PACKAGED_KNOWLEDGE_PATH
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            substrate_records = payload["substrate"]
            biotic_records = payload["biotic"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("hierarchical monitoring knowledge base is unavailable") from error
        self._substrate = tuple(self._substrate_path(record) for record in substrate_records)
        self._biotic = tuple(self._biotic_path(record) for record in biotic_records)
        if not self._substrate or not self._biotic:
            raise ValueError("hierarchical monitoring knowledge base has no completed records")
        self._substrate_definitions = self._definition_index(substrate_records, ("component", "class", "subclass", "group"))
        self._biotic_definitions = self._definition_index(biotic_records, ("component", "class", "subclass"))
        self._substrate_subclasses = self._child_index(self._substrate, "label_class", "subclass")
        self._substrate_groups = self._child_index(self._substrate, ("label_class", "subclass"), "group")
        self._biotic_subclasses = self._child_index(self._biotic, "label_class", "subclass")

    @staticmethod
    def _required(record: object, keys: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(record, dict):
            raise ValueError("hierarchical record is invalid")
        values = tuple(record.get(key) for key in keys)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("hierarchical record is incomplete")
        return tuple(value.strip() for value in values)

    @classmethod
    def _substrate_path(cls, record: object) -> SubstratePath:
        return SubstratePath(*cls._required(record, ("component", "class", "subclass", "group")))

    @classmethod
    def _biotic_path(cls, record: object) -> BioticPath:
        return BioticPath(*cls._required(record, ("component", "class", "subclass")))

    @classmethod
    def _definition_index(cls, records: object, path_keys: tuple[str, ...]) -> dict[tuple[str, ...], tuple[str, ...]]:
        result: dict[tuple[str, ...], set[str]] = defaultdict(set)
        if not isinstance(records, list):
            raise ValueError("hierarchical records are invalid")
        for record in records:
            path = cls._required(record, path_keys)
            definition = cls._required(record, ("definition",))[0]
            result[path].add(definition)
        return {path: tuple(sorted(definitions)) for path, definitions in result.items()}

    @staticmethod
    def _child_index(records: tuple[object, ...], parent: str | tuple[str, ...], child: str) -> dict[tuple[str, ...], tuple[str, ...]]:
        result: dict[tuple[str, ...], set[str]] = defaultdict(set)
        parent_fields = (parent,) if isinstance(parent, str) else parent
        for record in records:
            key = tuple(getattr(record, field) for field in parent_fields)
            result[key].add(getattr(record, child))
        return {key: tuple(sorted(values)) for key, values in result.items()}

    def substrate_classes(self) -> tuple[str, ...]:
        return tuple(sorted({record.label_class for record in self._substrate}))

    def substrate_subclasses(self, label_class: str) -> tuple[str, ...]:
        return self._substrate_subclasses.get((label_class,), ())

    def substrate_groups(self, label_class: str, subclass: str) -> tuple[str, ...]:
        return self._substrate_groups.get((label_class, subclass), ())

    def substrate_path(self, label_class: str, subclass: str, group: str) -> SubstratePath | None:
        for record in self._substrate:
            if (record.label_class, record.subclass, record.group) == (label_class, subclass, group):
                return record
        return None

    def biotic_classes(self) -> tuple[str, ...]:
        return tuple(sorted({record.label_class for record in self._biotic}))

    def biotic_subclasses(self, label_classes: tuple[str, ...]) -> tuple[str, ...]:
        result: set[str] = set()
        for label_class in label_classes:
            result.update(self._biotic_subclasses.get((label_class,), ()))
        return tuple(sorted(result))

    def reference_for(self, substrate: SubstratePath | None, biotic_classes: tuple[str, ...], biotic_subclasses: tuple[str, ...]) -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {"substrate": [], "biotic": []}
        if substrate is not None:
            key = (substrate.component, substrate.label_class, substrate.subclass, substrate.group)
            result["substrate"].append({"label": substrate.group, "definitions": self._substrate_definitions.get(key, ())})
        for subclass in biotic_subclasses:
            definitions: set[str] = set()
            for record in self._biotic:
                if record.label_class in biotic_classes and record.subclass == subclass:
                    definitions.update(self._biotic_definitions.get((record.component, record.label_class, record.subclass), ()))
            result["biotic"].append({"label": subclass, "definitions": tuple(sorted(definitions))})
        return result

    def display_name(self, _category: str, label: str) -> str:
        return label
