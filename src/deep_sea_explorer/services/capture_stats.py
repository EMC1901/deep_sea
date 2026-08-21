from __future__ import annotations

from deep_sea_explorer.domain.enums import CaptureType
from deep_sea_explorer.domain.models import CountItem, MonitoringCoordinates, SessionState


class CaptureStatsService:
    INVALID_NAMES = {"", "未知", "某种生物", "生物体"}
    MONITORING_CATEGORIES = (CaptureType.BIO, CaptureType.SUBSTRATE, CaptureType.GEOMORPHOLOGY)

    def update(
        self,
        state: SessionState,
        category: CaptureType,
        items: tuple[CountItem, ...],
    ) -> tuple[CountItem, ...]:
        accepted = items[:1] if category is CaptureType.ENV else items
        bucket = state.cumulative_stats[category.value]
        for item in accepted:
            name = item.name.strip()
            if name in self.INVALID_NAMES or item.count < 0 or item.count > 1_000_000:
                continue
            increment = item.count if category is CaptureType.BIO else 1
            bucket[name] = bucket.get(name, 0) + increment
        return tuple(CountItem(name, count) for name, count in bucket.items())

    def update_monitoring_frame(
        self,
        state: SessionState,
        coordinates: MonitoringCoordinates | None,
        items_by_category: dict[CaptureType, tuple[CountItem, ...]],
    ) -> dict[str, tuple[CountItem, ...]]:
        """Count a queue-2 image once, comparing it only with its predecessor."""
        previous_coordinates = state.previous_monitoring_coordinates
        same_location = (
            coordinates is not None
            and previous_coordinates is not None
            and coordinates == previous_coordinates
        )
        current_labels = {
            category.value: self._valid_names(items_by_category.get(category, ()))
            for category in self.MONITORING_CATEGORIES
        }
        snapshots: dict[str, tuple[CountItem, ...]] = {}
        for category in self.MONITORING_CATEGORIES:
            category_name = category.value
            previous_labels = state.previous_monitoring_labels.get(category_name, frozenset())
            bucket = state.cumulative_stats[category_name]
            for label in current_labels[category_name]:
                if not same_location or label not in previous_labels:
                    bucket[label] = bucket.get(label, 0) + 1
            snapshots[category_name] = tuple(CountItem(name, count) for name, count in bucket.items())

        # Always advance the snapshot. This makes equal-coordinate comparisons
        # chain frame1 -> frame2 -> frame3 instead of comparing to an older count.
        state.previous_monitoring_coordinates = coordinates
        state.previous_monitoring_labels = {
            category_name: frozenset(labels)
            for category_name, labels in current_labels.items()
        }
        return snapshots

    def _valid_names(self, items: tuple[CountItem, ...]) -> frozenset[str]:
        return frozenset(
            item.name.strip()
            for item in items
            if item.name.strip() not in self.INVALID_NAMES and 0 <= item.count <= 1_000_000
        )
