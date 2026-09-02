from __future__ import annotations

import threading
from collections import defaultdict, deque

from deep_sea_explorer.domain.models import Memo


class MemoryMemoBroker:
    def __init__(self, max_per_session: int = 100) -> None:
        self._max_per_session = max_per_session
        self._queues: dict[str, deque[Memo]] = defaultdict(deque)
        self._lock = threading.RLock()

    def publish(self, memo: Memo) -> None:
        with self._lock:
            queue = self._queues[memo.session_id]
            if len(queue) >= self._max_per_session:
                queue.popleft()
            queue.append(memo)

    def drain(self, session_id: str | None = None) -> list[Memo]:
        with self._lock:
            if session_id is not None:
                session_items = list(self._queues.pop(session_id, ()))
                return session_items
            all_items: list[Memo] = []
            for sid in list(self._queues):
                all_items.extend(self._queues.pop(sid))
            return all_items
