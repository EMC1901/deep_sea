from typing import Protocol

from deep_sea_explorer.domain.models import Memo


class MemoBroker(Protocol):
    def publish(self, memo: Memo) -> None: ...
    def drain(self, session_id: str | None = None) -> list[Memo]: ...
