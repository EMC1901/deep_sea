from deep_sea_explorer.domain.exceptions import ModelUnavailableError

from .client import RemoteModelClient


class RemoteImageGateway:
    def __init__(self, client: RemoteModelClient) -> None:
        self.client = client

    def generate(self, prompt: str) -> bytes:
        response = self.client.request("POST", "/images/generate", json={"prompt": prompt})
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/jpeg") or not response.content.startswith(
            b"\xff\xd8"
        ):
            raise ModelUnavailableError("remote image response is not a JPEG")
        return response.content
