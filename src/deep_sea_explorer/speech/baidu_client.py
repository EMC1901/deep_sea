from __future__ import annotations

from deep_sea_explorer.domain.exceptions import ModelUnavailableError


class BaiduSpeechClient:
    def __init__(self, app_id: str, api_key: str, secret_key: str) -> None:
        self._credentials = (app_id, api_key, secret_key)
        self._client = None

    def _sdk(self):
        if self._client is None:
            if not all(self._credentials):
                raise ModelUnavailableError("Baidu speech credentials are not configured")
            from aip import AipSpeech  # type: ignore[import-untyped]

            self._client = AipSpeech(*self._credentials)
        return self._client

    def recognize(self, wav: bytes) -> str | None:
        result = self._sdk().asr(wav, "wav", 16000, {"dev_pid": 1537})
        return result.get("result", [None])[0] if result.get("err_no") == 0 else None

    def synthesize(self, text: str) -> bytes | None:
        result = self._sdk().synthesis(
            text, "zh", 1, {"vol": 6, "spd": 6, "pit": 5, "per": 4, "aue": 6}
        )
        return result if isinstance(result, bytes) else None
