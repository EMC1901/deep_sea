from __future__ import annotations

from smoke_common import main


def check(result: dict[str, object]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    matrix = torch.ones((1024, 1024), device="cuda:0")
    result["cuda_version"] = torch.version.cuda
    result["matrix_sum"] = float(matrix.sum().item())
    del matrix


if __name__ == "__main__":
    main(check, "gpu")

