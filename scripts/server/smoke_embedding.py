"""Shared implementation for the two S7 embedding-model smoke tests."""

from __future__ import annotations

import math
import time

from smoke_common import main, required_model_path


def embedding_smoke(
    result: dict[str, object], *, environment_name: str, expected_dimension: int, trust_remote_code: bool
) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_path = required_model_path(environment_name)
    started = time.perf_counter()
    model = SentenceTransformer(
        str(model_path),
        device="cuda",
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    result["load_seconds"] = round(time.perf_counter() - started, 3)
    result["precision"] = str(next(model.parameters()).dtype)

    texts = ["深海热液喷口附近发现发光生物。", "水下机器人正在记录海底沉积物。"]
    started = time.perf_counter()
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    result["inference_seconds"] = round(time.perf_counter() - started, 3)
    repeated = model.encode(
        [texts[0]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    if embeddings.shape != (2, expected_dimension):
        raise RuntimeError(f"unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("embedding contains non-finite values")
    if not np.allclose(embeddings[0], repeated[0], rtol=1e-4, atol=1e-5):
        raise RuntimeError("embedding is not stable across repeated inference")
    norms = np.linalg.norm(embeddings, axis=1)
    if any(not math.isclose(float(norm), 1.0, rel_tol=1e-3, abs_tol=1e-3) for norm in norms):
        raise RuntimeError("embedding normalization check failed")
    result["dimension"] = expected_dimension
    result["normalized"] = True
    result["input_count"] = len(texts)
    del model


def run(name: str, environment_name: str, expected_dimension: int, trust_remote_code: bool) -> None:
    main(
        lambda result: embedding_smoke(
            result,
            environment_name=environment_name,
            expected_dimension=expected_dimension,
            trust_remote_code=trust_remote_code,
        ),
        name,
    )

