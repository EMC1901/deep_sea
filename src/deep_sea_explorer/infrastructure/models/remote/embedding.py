import math

from deep_sea_explorer.domain.exceptions import ModelUnavailableError

from .client import RemoteModelClient


class RemoteEmbeddingGateway:
    def __init__(self, client: RemoteModelClient, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self.client.json_body(
            self.client.request("POST", "/embeddings", json={"model": self.model, "texts": texts})
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ModelUnavailableError("remote embeddings have invalid count")
        try:
            result = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as error:
            raise ModelUnavailableError("remote embeddings are invalid") from error
        if not result or any(
            not vector or any(not math.isfinite(value) for value in vector) for vector in result
        ):
            raise ModelUnavailableError("remote embeddings are non-finite")
        if (
            data.get("model") != self.model
            or data.get("normalized") is not True
            or data.get("dimension") != len(result[0])
            or any(len(vector) != len(result[0]) for vector in result)
        ):
            raise ModelUnavailableError("remote embeddings do not match the API contract")
        return result
