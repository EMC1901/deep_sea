from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "server"


def test_s7_smoke_scripts_exist_and_use_configured_model_paths() -> None:
    expected = {
        "check_gpu.py": (),
        "smoke_qwen.py": ("QWEN_MODEL_PATH", "local_files_only=True"),
        "smoke_image_generation.py": ("IMAGE_MODEL_PATH", "local_files_only=True"),
        "smoke_embedding.py": ("local_files_only=True",),
        "smoke_gte.py": ("MEMO_EMBEDDING_MODEL_PATH",),
        "smoke_minilm.py": ("RAG_EMBEDDING_MODEL_PATH",),
    }

    for filename, markers in expected.items():
        content = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in content


def test_s7_smoke_scripts_keep_results_in_the_controlled_log_directory() -> None:
    common = (SCRIPT_DIR / "smoke_common.py").read_text(encoding="utf-8")

    assert "/projects/deep-sea-explorer/logs/s7" in common
    assert "HF_HUB_OFFLINE" in common
    assert "TRANSFORMERS_OFFLINE" in common


def test_s8_local_runtime_smoke_uses_gateways_and_removes_its_test_video() -> None:
    script = (SCRIPT_DIR / "smoke_local_runtime.py").read_text(encoding="utf-8")

    for value in (
        "MODEL_BACKEND",
        "build_local_container",
        "describe_video",
        "memo_embedding.embed",
        "rag_embedding.embed",
        "image.generate",
        "video_path.unlink",
    ):
        assert value in script


def test_s9_media_helper_creates_disposable_jpeg_and_video() -> None:
    script = (SCRIPT_DIR / "create_s9_api_test_media.py").read_text(encoding="utf-8")

    for value in ("S9_API_TEST_DIR", "s9-api-check.jpg", "s9-api-check.mp4", "cv2.VideoWriter"):
        assert value in script
