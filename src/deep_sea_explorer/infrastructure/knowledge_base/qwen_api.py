"""DashScope MaaS client used for offline label-description generation."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


class QwenApiError(RuntimeError):
    """Raised when the remote Qwen endpoint cannot produce a text response."""


class QwenApiGenerator:
    """Generate one description at a time without persisting credentials."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 180.0,
        session: Any | None = None,
    ) -> None:
        api_key = api_key.strip()
        if not base_url.strip() or not model.strip() or not api_key:
            raise ValueError("base_url, model, and api_key are required")
        self.endpoint = base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation"
        self.model = model
        # stdin-based secret handoff can carry a platform line ending; never send it in a header.
        self._api_key = api_key
        self.timeout = timeout
        self._session = session

    def __call__(self, image_path: Path, prompt: str) -> str:
        return self.generate(image_path, prompt)

    def generate(self, image_path: Path, prompt: str) -> str:
        try:
            import requests
        except ImportError as error:  # pragma: no cover - deployment dependency
            raise QwenApiError("requests is required for the Qwen API generator") from error
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as error:
            raise QwenApiError(f"cannot read representative image: {image_path}") from error
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": f"data:image/jpeg;base64,{encoded}"},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }
        requester = self._session.post if self._session is not None else requests.post
        response: Any | None = None
        try:
            response = requester(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception as error:
            # Never include response text or request headers: either can contain secrets.
            status = getattr(response, "status_code", None)
            detail = f" HTTP status {status}" if isinstance(status, int) else ""
            raise QwenApiError(f"Qwen API request failed: {type(error).__name__}.{detail}") from error
        try:
            body = response.json()
            content = body["output"]["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise QwenApiError("Qwen API response has no message content") from error
        if isinstance(content, list):
            content = "".join(item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str))
        if not isinstance(content, str) or not content.strip():
            raise QwenApiError("Qwen API message content is empty")
        return content.strip()
