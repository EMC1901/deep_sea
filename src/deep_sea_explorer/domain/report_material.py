"""Bound report-summary input without changing the material used by the renderer."""

from __future__ import annotations


SUMMARY_TEXT_BUDGET = 24_000
MAX_STRING_CHARS = 2_000
MAX_COLLECTION_ITEMS = 200
MAX_OBJECT_FIELDS = 100
MAX_DEPTH = 6
IMAGE_FIELDS = frozenset(
    {
        "image",
        "img",
        "image_data",
        "image_data_uri",
        "image_url",
        "thumbnail",
    }
)

_SKIP = object()


class _TextBudget:
    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def take(self, value: str) -> str:
        allowed = min(len(value), MAX_STRING_CHARS, self.remaining)
        self.remaining -= allowed
        return value[:allowed]


def compact_report_material(material: dict[str, object]) -> dict[str, object]:
    """Remove image payloads and bound text before sending report material to a model."""
    value = _compact(material, _TextBudget(SUMMARY_TEXT_BUDGET), depth=0)
    return value if isinstance(value, dict) else {}


def _compact(value: object, budget: _TextBudget, *, depth: int) -> object:
    if depth > MAX_DEPTH:
        return _SKIP
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= MAX_OBJECT_FIELDS:
                break
            key = str(raw_key)[:100]
            if key.strip().lower() in IMAGE_FIELDS:
                continue
            compacted = _compact(item, budget, depth=depth + 1)
            if compacted is not _SKIP:
                result[key] = compacted
        return result
    if isinstance(value, list):
        result: list[object] = []
        for item in value[:MAX_COLLECTION_ITEMS]:
            compacted = _compact(item, budget, depth=depth + 1)
            if compacted is not _SKIP:
                result.append(compacted)
        return result
    if isinstance(value, str):
        if value.lstrip().lower().startswith("data:image/"):
            return _SKIP
        return budget.take(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return budget.take(str(value))
