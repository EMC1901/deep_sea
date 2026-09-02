from collections.abc import Iterator

from deep_sea_explorer.domain.models import StreamEvent
from deep_sea_explorer.ports.model_gateway import VisionModelGateway
from deep_sea_explorer.ports.session_store import SessionStore


class QuestionAnsweringService:
    def __init__(
        self,
        vision: VisionModelGateway,
        sessions: SessionStore,
    ) -> None:
        self.vision, self.sessions = vision, sessions

    def answer(self, session_id: str, question: str) -> Iterator[StreamEvent]:
        state = self.sessions.get(session_id)
        state.is_answering = True
        try:
            yield from self.vision.answer(question)
        finally:
            state.is_answering = False
