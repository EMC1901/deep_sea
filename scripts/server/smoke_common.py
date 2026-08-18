"""Shared, offline-only utilities for S7 single-model smoke tests."""

from __future__ import annotations

import gc
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "logs" / "s7"


def required_model_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"{name} does not point to a readable model directory")
    return path


def gpu_snapshot() -> dict[str, float | int | str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    return {
        "device": torch.cuda.get_device_name(device),
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
    }


def clear_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - cleanup must not hide the original error
        pass


def safe_error(error: Exception) -> str:
    """Keep error summaries useful without logging server filesystem paths."""
    text = str(error).replace("\n", " ")
    text = re.sub(r"/(?:[^\s'\"]+/)+[^\s'\"]+", "<path>", text)
    return text[:500]


def write_result(name: str, result: dict[str, Any]) -> Path:
    output_dir = Path(os.environ.get("S7_RESULTS_DIR", DEFAULT_RESULTS_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_smoke(name: str, action: Callable[[dict[str, Any]], None]) -> int:
    """Run one model only, save a compact result, and return a shell-friendly status."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    clear_gpu_memory()
    result: dict[str, Any] = {"model": name, "status": "failed", "started_at": time.time()}
    try:
        result["before"] = gpu_snapshot()
        action(result)
        result["status"] = "passed"
        return_code = 0
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = safe_error(error)
        return_code = 1
    finally:
        clear_gpu_memory()
        try:
            result["after"] = gpu_snapshot()
        except Exception as error:  # pragma: no cover - reported in result when CUDA fails
            result["after_error"] = safe_error(error)
        result["duration_seconds"] = round(time.time() - result["started_at"], 3)
        report = write_result(name, result)
        print(json.dumps({"model": name, "status": result["status"], "report": report.name}))
    return return_code


def main(action: Callable[[dict[str, Any]], None], name: str) -> None:
    raise SystemExit(run_smoke(name, action))
