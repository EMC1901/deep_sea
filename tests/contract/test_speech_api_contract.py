from __future__ import annotations

import io

import pytest
from werkzeug.datastructures import FileStorage

import deep_sea_explorer.speech.app_factory as speech_factory


class FakeSpeechClient:
    def recognize(self, wav: bytes) -> str:
        return "固定语音识别文本"

    def synthesize(self, text: str) -> bytes:
        return b"RIFFfakeWAVE"


class FakeConverter:
    def convert_webm_to_wav(self, source, target) -> None:
        return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    class FakeTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return "fake-temp"

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(FileStorage, "save", lambda self, dst, buffer_size=16384: None)
    monkeypatch.setattr(speech_factory.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(speech_factory.Path, "read_bytes", lambda self: b"wav")
    app = speech_factory.create_speech_app(FakeSpeechClient(), FakeConverter())
    app.config.update(TESTING=True)
    return app.test_client()


def test_stt_validates_missing_audio(client) -> None:
    assert client.post("/stt", data={}, content_type="multipart/form-data").get_json() == {
        "error": "No audio file"
    }


def test_stt_and_tts_keep_response_contract(client) -> None:
    stt = client.post(
        "/stt",
        data={"audio": (io.BytesIO(b"webm"), "question.webm")},
        content_type="multipart/form-data",
    )
    tts = client.post("/tts", json={"text": "你好"})
    assert stt.get_json() == {"text": "固定语音识别文本"}
    assert tts.mimetype == "audio/wav"
    assert tts.data.startswith(b"RIFF")


def test_tts_validates_missing_text(client) -> None:
    assert client.post("/tts", json={}).get_json() == {"error": "Missing text"}
