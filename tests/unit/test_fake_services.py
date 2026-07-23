from __future__ import annotations

from tests.fakes.fake_services import (
    FakeBaiduSpeechClient,
    FakeRag,
    TINY_JPEG_BYTES,
    TINY_WAV_BYTES,
)


def test_binary_fixtures_have_expected_file_signatures() -> None:
    assert TINY_JPEG_BYTES.startswith(b"\xff\xd8")
    assert TINY_JPEG_BYTES.endswith(b"\xff\xd9")
    assert TINY_WAV_BYTES.startswith(b"RIFF")
    assert TINY_WAV_BYTES[8:12] == b"WAVE"


def test_fake_rag_builds_and_searches() -> None:
    rag = FakeRag()
    assert rag.add_pdf("fixture.pdf", "fixture.pdf") is True
    rag.build_index()

    assert rag.index is not None
    assert rag.search("深海") == [
        {
            "content": "深海测试文档内容",
            "doc_id": "fixture.pdf",
            "chunk_id": 0,
            "score": 0.99,
        }
    ]


def test_fake_speech_client_never_needs_network() -> None:
    client = FakeBaiduSpeechClient("unused", "unused", "unused")
    assert client.asr(b"audio", "wav", 16000, {})["result"] == ["固定语音识别文本"]
    assert client.synthesis("你好", "zh", 1, {}).startswith(b"RIFF")
