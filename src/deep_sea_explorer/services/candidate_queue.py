from __future__ import annotations

import threading
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from deep_sea_explorer.services.key_frame_detection import CandidateEvent, SurveyEventEvaluation

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SessionQueue:
    in_flight: Future[SurveyEventEvaluation] | None = None
    pending: CandidateEvent | None = None


class PerSessionEventQueue:
    """One in-flight model task and one replaceable pending candidate per session."""

    def __init__(self, evaluator: Callable[[CandidateEvent], SurveyEventEvaluation], on_result: Callable[[CandidateEvent, SurveyEventEvaluation], None]) -> None:
        self.evaluator, self.on_result = evaluator, on_result
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="survey-event")
        self._queues: dict[str, _SessionQueue] = {}
        self._lock = threading.RLock()

    def submit(self, candidate: CandidateEvent) -> str:
        with self._lock:
            queue = self._queues.setdefault(candidate.session_id, _SessionQueue())
            if queue.in_flight is not None and not queue.in_flight.done():
                if queue.pending is None or candidate.signature != queue.pending.signature:
                    queue.pending = candidate
                return "pending"
            self._start_locked(queue, candidate)
            return "started"

    def _start_locked(self, queue: _SessionQueue, candidate: CandidateEvent) -> None:
        future = self._executor.submit(self.evaluator, candidate)
        queue.in_flight = future
        future.add_done_callback(lambda completed, c=candidate: self._completed(c, completed))

    def _completed(self, candidate: CandidateEvent, future: Future[SurveyEventEvaluation]) -> None:
        try:
            evaluation = future.result()
            self.on_result(candidate, evaluation)
        except Exception as error:
            LOGGER.warning(
                "survey event evaluation failed session_id=%s error_type=%s detail=%s",
                candidate.session_id,
                type(error).__name__,
                str(error),
            )
        finally:
            with self._lock:
                queue = self._queues.get(candidate.session_id)
                if queue is None:
                    return
                queue.in_flight = None
                pending = queue.pending
                queue.pending = None
                if pending is not None:
                    self._start_locked(queue, pending)

    def state(self, session_id: str) -> tuple[bool, bool]:
        with self._lock:
            queue = self._queues.get(session_id)
            return bool(queue and queue.in_flight and not queue.in_flight.done()), bool(queue and queue.pending)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
