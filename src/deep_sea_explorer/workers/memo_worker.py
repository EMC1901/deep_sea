from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from deep_sea_explorer.ports.session_store import SessionStore


LOGGER = logging.getLogger(__name__)


class MonitoringGateway(Protocol):
    def process_session(self, session_id: str) -> object: ...


class MemoWorker:
    def __init__(
        self,
        monitoring: MonitoringGateway,
        sessions: SessionStore,
        interval_seconds: float = 1.0,
    ) -> None:
        self.monitoring, self.sessions, self.interval_seconds = (
            monitoring,
            sessions,
            interval_seconds,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_success_monotonic: float | None = None

    def run_once(self) -> None:
        for session_id in self.sessions.session_ids():
            try:
                self.monitoring.process_session(session_id)
                self.last_success_monotonic = time.monotonic()
            except Exception as error:
                LOGGER.warning(
                    "memo worker retry_scheduled session_id=%s error_type=%s",
                    session_id,
                    type(error).__name__,
                )
                continue

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="memo-worker")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval_seconds)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
