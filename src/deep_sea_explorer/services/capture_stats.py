from __future__ import annotations

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CountItem, SessionState


class CaptureStatsService:
    INVALID_NAMES = {"", "未知", "某种生物", "生物体"}

    def update(
        self, state: SessionState, category: CaptureType, items: tuple[CountItem, ...]
    ) -> tuple[CountItem, ...]:
        accepted = items[:1] if category is CaptureType.ENV else items
        bucket = state.cumulative_stats[category.value]
        for item in accepted:
            name = item.name.strip()
            if name in self.INVALID_NAMES or item.count < 0 or item.count > 1_000_000:
                continue
            bucket[name] = bucket.get(name, 0) + (item.count if category is CaptureType.BIO else 1)
        return tuple(CountItem(name, count) for name, count in bucket.items())
