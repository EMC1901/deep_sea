"""Bound report-summary input without changing the material used by the renderer."""

from __future__ import annotations


# The deployed GGUF service has a 4,096-token context.  Reserve enough space
# for the prompt template and a 300–500 character conclusion; the source
# material therefore needs to stay deliberately small even when a monitoring
# session has accumulated many memo cards and samples.
SUMMARY_TEXT_BUDGET = 1_000
MAX_STRING_CHARS = 240
MAX_COLLECTION_ITEMS = 6
MAX_OBJECT_FIELDS = 12
MAX_DEPTH = 4
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

_FIELD_TEXT_BUDGETS = {
    "memos": 220,
    "chats": 200,
    "bio_samples": 160,
    "env_samples": 160,
    "bio_stats": 80,
    "env_stats": 80,
    "meta": 100,
}
_FIELD_COLLECTION_LIMITS = {
    "memos": 6,
    "chats": 6,
    "bio_samples": 6,
    "env_samples": 6,
    "bio_stats": 6,
    "env_stats": 6,
}


class _TextBudget:
    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def take(self, value: str) -> str:
        allowed = min(len(value), MAX_STRING_CHARS, self.remaining)
        self.remaining -= allowed
        return value[:allowed]


def compact_report_material(material: dict[str, object]) -> dict[str, object]:
    """Create a balanced, context-safe summary input without changing the PDF material."""
    if not isinstance(material, dict):
        return {}

    remaining = _TextBudget(SUMMARY_TEXT_BUDGET)
    result: dict[str, object] = {}
    for index, (raw_key, item) in enumerate(material.items()):
        if index >= MAX_OBJECT_FIELDS or remaining.remaining <= 0:
            break
        key = str(raw_key)[:100]
        if key.strip().lower() in IMAGE_FIELDS:
            continue
        field_budget = _TextBudget(
            min(_FIELD_TEXT_BUDGETS.get(key, 100), remaining.remaining)
        )
        compacted = _compact(
            item,
            field_budget,
            depth=1,
            collection_limit=_FIELD_COLLECTION_LIMITS.get(key, MAX_COLLECTION_ITEMS),
        )
        remaining.remaining -= field_budget.remaining
        if compacted is not _SKIP:
            result[key] = compacted
    return result


def _compact(
    value: object,
    budget: _TextBudget,
    *,
    depth: int,
    collection_limit: int = MAX_COLLECTION_ITEMS,
) -> object:
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
            compacted = _compact(
                item, budget, depth=depth + 1, collection_limit=collection_limit
            )
            if compacted is not _SKIP:
                result[key] = compacted
        return result
    if isinstance(value, list):
        items: list[object] = []
        for item in value[:collection_limit]:
            compacted = _compact(
                item, budget, depth=depth + 1, collection_limit=collection_limit
            )
            if compacted is not _SKIP:
                items.append(compacted)
        return items
    if isinstance(value, str):
        if value.lstrip().lower().startswith("data:image/"):
            return _SKIP
        return budget.take(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return budget.take(str(value))
