from __future__ import annotations

from smoke_embedding import run


if __name__ == "__main__":
    run("minilm", "RAG_EMBEDDING_MODEL_PATH", expected_dimension=384, trust_remote_code=False)

