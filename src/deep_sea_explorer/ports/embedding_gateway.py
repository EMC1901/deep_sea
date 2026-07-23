from typing import Protocol


class EmbeddingGateway(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
