from __future__ import annotations

import threading
import time

from deep_sea_explorer.domain.models import SessionState


class MemorySessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._states: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                if len(self._states) >= self._max_sessions:
                    raise RuntimeError("maximum active sessions reached")
                state = SessionState()
                self._states[session_id] = state
            state.last_active_monotonic = time.monotonic()
            return state

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def remove_expired(self) -> list[str]:
        cutoff = time.monotonic() - self._ttl_seconds
        with self._lock:
            expired = [
                sid
                for sid, state in self._states.items()
                if state.last_active_monotonic < cutoff and not state.is_answering
            ]
            for sid in expired:
                self._states.pop(sid, None)
            return expired

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._states)
