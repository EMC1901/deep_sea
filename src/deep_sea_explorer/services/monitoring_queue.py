"""Bounded per-session FIFO queues for real-time monitoring frames."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from deep_sea_explorer.domain.models import MonitoringAnalysis


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MonitoringFrame:
    session_id: str
    image_path: Path
    captured_monotonic: float


@dataclass(frozen=True, slots=True)
class QueueSubmission:
    waiting: int
    dropped_oldest: bool


@dataclass(slots=True)
class _SessionQueue:
    in_flight: Future[MonitoringAnalysis] | None = None
    pending: deque[MonitoringFrame] = field(default_factory=deque)


class PerSessionFrameQueue:
    """Keep a bounded FIFO backlog while never cancelling the active analysis."""

    def __init__(
        self,
        max_pending: int,
        evaluator: Callable[[MonitoringFrame], MonitoringAnalysis],
        on_result: Callable[[MonitoringFrame, MonitoringAnalysis, float], None],
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self.evaluator = evaluator
        self.on_result = on_result
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="monitoring-qwen")
        self._queues: dict[str, _SessionQueue] = {}
        self._lock = threading.RLock()

    def submit(self, frame: MonitoringFrame) -> QueueSubmission:
        with self._lock:
            queue = self._queues.setdefault(frame.session_id, _SessionQueue())
            if queue.in_flight is None or queue.in_flight.done():
                self._start_locked(queue, frame)
                return QueueSubmission(len(queue.pending), False)
            dropped = len(queue.pending) >= self.max_pending
            if dropped:
                evicted = queue.pending.popleft()
                evicted.image_path.unlink(missing_ok=True)
            queue.pending.append(frame)
            return QueueSubmission(len(queue.pending), dropped)

    def _start_locked(self, queue: _SessionQueue, frame: MonitoringFrame) -> None:
        started = time.monotonic()
        future = self._executor.submit(self.evaluator, frame)
        queue.in_flight = future
        future.add_done_callback(lambda completed: self._completed(frame, started, completed))

    def _completed(
        self,
        frame: MonitoringFrame,
        started: float,
        future: Future[MonitoringAnalysis],
    ) -> None:
        try:
            self.on_result(frame, future.result(), time.monotonic() - started)
        except Exception as error:
            LOGGER.warning(
                "monitoring frame analysis failed session_id=%s error_type=%s",
                frame.session_id,
                type(error).__name__,
            )
        finally:
            frame.image_path.unlink(missing_ok=True)
            with self._lock:
                queue = self._queues.get(frame.session_id)
                if queue is None:
                    return
                queue.in_flight = None
                if queue.pending:
                    self._start_locked(queue, queue.pending.popleft())

    def state(self, session_id: str) -> tuple[bool, int]:
        with self._lock:
            queue = self._queues.get(session_id)
            if queue is None:
                return False, 0
            return bool(queue.in_flight and not queue.in_flight.done()), len(queue.pending)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
