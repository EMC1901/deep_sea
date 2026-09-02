from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import httpx

from deep_sea_explorer.config import Settings
from deep_sea_explorer.domain.exceptions import ModelUnavailableError


LOGGER = logging.getLogger(__name__)


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
            raise _remote_failure(error, method, path, stream=False) from error

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
            raise _remote_failure(error, method, path, stream=True) from error
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


def _remote_failure(
    error: httpx.HTTPError,
    method: str,
    path: str,
    *,
    stream: bool,
) -> ModelUnavailableError:
    status: int | str = "unavailable"
    code = "NETWORK_ERROR"
    request_id = "unknown"
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        status = response.status_code
        request_id = response.headers.get("X-Request-ID", "unknown")
        try:
            body = response.json()
            if isinstance(body, dict):
                request_id = str(body.get("request_id") or request_id)
                problem = body.get("error")
                if isinstance(problem, dict) and isinstance(problem.get("code"), str):
                    code = problem["code"]
        except ValueError:
            code = "INVALID_ERROR_RESPONSE"
    operation = "stream" if stream else "request"
    LOGGER.error(
        "remote_model failure operation=%s method=%s endpoint=%s status=%s code=%s request_id=%s error_type=%s",
        operation,
        method,
        path,
        status,
        code,
        request_id,
        type(error).__name__,
    )
    return ModelUnavailableError(
        f"remote model {operation} failed: status={status} code={code} request_id={request_id}"
    )
