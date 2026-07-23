"""P0 契约测试使用的最小可预测替身。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterator


# 1×1 JPEG 和空 PCM WAV。它们只作为 API/报告测试夹具，不代表模型输出质量。
TINY_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/a"
    "AAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAA"
    "AAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9k="
)
TINY_WAV_BYTES = base64.b64decode("UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=")
TINY_PDF_BYTES = b"%PDF-1.4\n% fake contract fixture\n%%EOF\n"


class FakeRag:
    """只记录文档数量并返回固定搜索结果的内存替身。"""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.index: object | None = None

    def add_pdf(self, pdf_path: str, doc_id: str | None = None) -> bool:
        self.documents.append(
            {
                "content": "深海测试文档内容",
                "doc_id": doc_id or Path(pdf_path).name,
                "chunk_id": 0,
            }
        )
        return True

    def build_index(self) -> None:
        self.index = object()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return [
            {
                "content": "深海测试文档内容",
                "doc_id": "fixture.pdf",
                "chunk_id": 0,
                "score": 0.99,
            }
        ][:top_k]


class FakeAnalyzer:
    """替代 VideoAnalyzer，避免导入 Torch、Transformers、FAISS 和模型权重。"""

    def __init__(self) -> None:
        self.temp_dir = "fake-temp"
        self.model = object()
        self.rag = FakeRag()
        self.is_processing_user_question = False
        self._latest_videos: dict[str, str] = {}
        self._memos: list[dict[str, Any]] = []

    def process_frames(self, session_id: str, frames_data: list[bytes]) -> str:
        video_path = f"fake://{session_id}/fake_video.mp4"
        self._latest_videos[session_id] = video_path
        return video_path

    def get_latest_video(self, session_id: str) -> str | None:
        return self._latest_videos.get(session_id)

    def stream_process_question(
        self, session_id: str, question: str, video_path: str
    ) -> Iterator[str]:
        yield json.dumps({"type": "chunk", "text": "固定回答"}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "final", "text": "固定回答"}, ensure_ascii=False) + "\n"

    def get_memos(self) -> list[dict[str, Any]]:
        memos = self._memos
        self._memos = []
        return memos

    def queue_memo(self, session_id: str = "test-session") -> None:
        self._memos.append(
            {
                "timestamp": "12:00:00",
                "content": "固定场景摘要",
                "session_id": session_id,
                "capture": {
                    "type": "bio",
                    "image": "data:image/jpeg;base64,"
                    + base64.b64encode(TINY_JPEG_BYTES).decode("ascii"),
                    "description": "固定生物样本",
                    "organisms": [{"name": "测试虾", "count": 1}],
                    "env_features": [],
                },
            }
        )


class FakeReportGenerator:
    """替代 ReportGenerator，生成最小 PDF 头供下载契约验证。"""

    def generate_summary_with_llm(
        self, report_payload: dict[str, Any], analyzer: FakeAnalyzer
    ) -> str:
        return "固定任务摘要。"

    def create_pdf(self, filename: str, report_payload: dict[str, Any], summary: str) -> str:
        return filename


class FakeBaiduSpeechClient:
    """替代百度 SDK 客户端。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args

    def asr(
        self, data: bytes, audio_format: str, sample_rate: int, options: dict[str, Any]
    ) -> dict[str, Any]:
        return {"err_no": 0, "result": ["固定语音识别文本"]}

    def synthesis(
        self, text: str, language: str, options_number: int, options: dict[str, Any]
    ) -> bytes:
        return TINY_WAV_BYTES
