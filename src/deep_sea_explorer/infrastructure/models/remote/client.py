from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import httpx

from deep_sea_explorer.config import Settings
from deep_sea_explorer.domain.exceptions import ModelUnavailableError


class RemoteModelClient:
    """开发机远程模型 HTTP 客户端；服务禁用时绝不创建网络请求。"""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.base_url = (
            settings.model_service_base_url.rstrip("/")
            + "/"
            + settings.model_service_api_prefix.strip("/")
        )
        self._client = httpx.Client(
            verify=settings.model_service_verify_tls,
            timeout=httpx.Timeout(
                settings.model_service_read_timeout_seconds,
                connect=settings.model_service_connect_timeout_seconds,
            ),
            transport=transport,
        )

    def _ensure_enabled(self) -> None:
        if not self.settings.model_service_enabled:
            raise ModelUnavailableError("remote model service is disabled by configuration")

    def _headers(self) -> dict[str, str]:
        headers = {"X-Model-API-Version": "1", "X-Request-ID": str(uuid4())}
        if (
            self.settings.model_service_auth_type == "bearer"
            and self.settings.model_service_auth_token
        ):
            headers["Authorization"] = f"Bearer {self.settings.model_service_auth_token}"
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        self._ensure_enabled()
        try:
            response = self._client.request(
                method,
                self.base_url + path,
                headers=self._headers(),
                json=json,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            raise ModelUnavailableError(
                f"remote model request failed: {type(error).__name__}"
            ) from error

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        data: dict[str, str] | None = None,
    ) -> Iterator[httpx.Response]:
        """Yield a response without buffering its body, then close it promptly."""
        self._ensure_enabled()
        request = self._client.build_request(
            method,
            self.base_url + path,
            headers=self._headers(),
            json=json,
            files=files,
            data=data,
        )
        response: httpx.Response | None = None
        try:
            response = self._client.send(request, stream=True)
            response.raise_for_status()
            yield response
        except httpx.HTTPError as error:
            raise ModelUnavailableError(
                f"remote model stream failed: {type(error).__name__}"
            ) from error
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def json_body(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as error:
            raise ModelUnavailableError("remote model response is not valid JSON") from error
        if not isinstance(body, dict):
            raise ModelUnavailableError("remote model response must be a JSON object")
        return body

    def close(self) -> None:
        self._client.close()
