from __future__ import annotations

import httpx

from deep_sea_explorer.config import Settings
from deep_sea_explorer.domain.exceptions import ModelUnavailableError


class RemoteModelClient:
    """开发机远程模型 HTTP 客户端；服务禁用时绝不创建网络请求。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (
            settings.model_service_base_url.rstrip("/")
            + "/"
            + settings.model_service_api_prefix.strip("/")
        )

    def _ensure_enabled(self) -> None:
        if not self.settings.model_service_enabled:
            raise ModelUnavailableError("remote model service is disabled by configuration")

    def _headers(self) -> dict[str, str]:
        headers = {"X-Model-API-Version": "v1"}
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
    ) -> httpx.Response:
        self._ensure_enabled()
        timeout = httpx.Timeout(
            self.settings.model_service_read_timeout_seconds,
            connect=self.settings.model_service_connect_timeout_seconds,
        )
        try:
            with httpx.Client(
                verify=self.settings.model_service_verify_tls, timeout=timeout
            ) as client:
                response = client.request(
                    method, self.base_url + path, headers=self._headers(), json=json, files=files
                )
                response.raise_for_status()
                return response
        except httpx.HTTPError as error:
            raise ModelUnavailableError(
                f"remote model request failed: {type(error).__name__}"
            ) from error
