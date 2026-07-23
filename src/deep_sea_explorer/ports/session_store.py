from deep_sea_explorer.domain.models import SessionState
from typing import Protocol


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState: ...
    def remove_expired(self) -> list[str]: ...
