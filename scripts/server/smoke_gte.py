from __future__ import annotations

from smoke_embedding import run


if __name__ == "__main__":
    run("gte", "MEMO_EMBEDDING_MODEL_PATH", expected_dimension=768, trust_remote_code=True)

