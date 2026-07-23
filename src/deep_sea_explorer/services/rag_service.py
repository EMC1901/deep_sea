from __future__ import annotations

from pathlib import Path

from deep_sea_explorer.domain.models import RagDocumentChunk


class TextChunker:
    def __init__(self, size: int = 500, overlap: int = 50) -> None:
        self.size, self.overlap = size, overlap

    def split(self, text: str) -> list[str]:
        chunks = [
            text[index : index + self.size].strip()
            for index in range(0, len(text), self.size - self.overlap)
        ]
        return [chunk for chunk in chunks if len(chunk) > 20]


class RagService:
    """当前明确采用内存索引；服务重启后文档和索引会丢失。"""

    def __init__(self, embedding: object, chunker: TextChunker | None = None) -> None:
        self.embedding, self.chunker = embedding, chunker or TextChunker()
        self.documents: list[RagDocumentChunk] = []
        self._vectors: list[list[float]] = []
        self.index: object | None = None

    def add_pdf(self, path: Path, doc_id: str) -> bool:
        try:
            import fitz

            document = fitz.open(path)
            text = "".join(page.get_text() for page in document)
            document.close()
        except Exception:
            return False
        chunks = self.chunker.split(text)
        if not chunks:
            return False
        self.documents.extend(
            RagDocumentChunk(chunk, doc_id, index) for index, chunk in enumerate(chunks)
        )
        return True

    def build_index(self) -> None:
        if not self.documents:
            self.index, self._vectors = None, []
            return
        self._vectors = self.embedding.embed([document.content for document in self.documents])
        self.index = object()

    def search(self, query: str, top_k: int = 5) -> list[RagDocumentChunk]:
        if not self.index or not query:
            return []
        query_vector = self.embedding.embed([query])[0]

        def score(vector: list[float]) -> float:
            return sum(left * right for left, right in zip(vector, query_vector))

        ranked = sorted(
            zip(self.documents, self._vectors), key=lambda pair: score(pair[1]), reverse=True
        )[:top_k]
        return [
            RagDocumentChunk(document.content, document.doc_id, document.chunk_id, score(vector))
            for document, vector in ranked
        ]

    def context(self, query: str) -> str:
        return "\n\n".join(result.content for result in self.search(query))
