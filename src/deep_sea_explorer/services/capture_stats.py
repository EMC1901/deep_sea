from __future__ import annotations

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CountItem, SessionState


class CaptureStatsService:
    INVALID_NAMES = {"", "未知", "某种生物", "生物体"}
    ADJACENT_QUEUE2_FRAME_WINDOW = 3

    def update(
        self,
        state: SessionState,
        category: CaptureType,
        items: tuple[CountItem, ...],
        *,
        monitoring_frame_index: int | None = None,
    ) -> tuple[CountItem, ...]:
        accepted = items[:1] if category is CaptureType.ENV else items
        bucket = state.cumulative_stats[category.value]
        counted_frames = state.last_counted_label_frame.get(category.value)
        for item in accepted:
            name = item.name.strip()
            if name in self.INVALID_NAMES or item.count < 0 or item.count > 1_000_000:
                continue
            if monitoring_frame_index is not None:
                if counted_frames is None:
                    counted_frames = state.last_counted_label_frame.setdefault(category.value, {})
                previous = counted_frames.get(name)
                if (
                    previous is not None
                    and monitoring_frame_index - previous <= self.ADJACENT_QUEUE2_FRAME_WINDOW
                ):
                    continue
                counted_frames[name] = monitoring_frame_index
                # Monitoring statistics count a label occurrence per eligible
                # queue-2 frame, not the model's estimated object count.
                increment = 1
            else:
                increment = item.count if category is CaptureType.BIO else 1
            bucket[name] = bucket.get(name, 0) + increment
        return tuple(CountItem(name, count) for name, count in bucket.items())
