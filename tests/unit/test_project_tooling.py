from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_dotenv_template(filename: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_pyproject_separates_remote_and_local_model_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    extras = project["project"]["optional-dependencies"]
    assert extras["remote"] == []
    assert {
        "torch",
        "torchvision==0.21.0",
        "Pillow==12.1.1",
        "transformers==4.57.1",
        "modelscope",
        "sentence-transformers",
        "accelerate",
        "diffusers",
    } == set(extras["local-model"])
    assert project["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_environment_templates_keep_remote_development_disabled_by_default() -> None:
    shared = read_dotenv_template(".env.example")
    development = read_dotenv_template(".env.development.example")
    server = read_dotenv_template(".env.server.example")

    assert shared["MODEL_BACKEND"] == "remote"
    assert shared["MODEL_SERVICE_ENABLED"] == "false"
    assert shared["MODEL_SERVICE_BASE_URL"].endswith(".invalid")
    assert development["MODEL_BACKEND"] == "remote"
    assert development["QWEN_MODEL_PATH"] == ""
    assert shared["IMAGE_RETRIEVAL_ENABLED"] == "false"
    assert development["IMAGE_RETRIEVAL_ENABLED"] == "false"
    assert server["IMAGE_RETRIEVAL_ENABLED"] == "false"
    assert server["IMAGE_RETRIEVAL_TOP_K"] == "4"
    assert server["MODEL_BACKEND"] == "local"
    assert server["MODEL_SERVICE_ENABLED"] == "false"
    assert server["MODEL_SERVICE_HOST"] == "127.0.0.1"
    assert server["MODEL_SERVICE_PORT"] == "19000"
    assert server["IMAGE_GENERATION_ENABLED"] == "false"
    assert server["API_HOST"] == "127.0.0.1"
    assert server["SPEECH_HOST"] == "127.0.0.1"
    assert "http://127.0.0.1:19100" in server["CORS_ORIGINS"]
    assert server["MODEL_MAX_CONCURRENT_REQUESTS"] == "1"
    assert server["MODEL_MAX_QUEUE_SIZE"] == "4"
    assert server["MODEL_MAX_EMBEDDING_TEXTS"] == "32"


def test_environment_templates_do_not_contain_baidu_credentials() -> None:
    for filename in (".env.example", ".env.development.example", ".env.server.example"):
        values = read_dotenv_template(filename)
        assert values["BAIDU_APP_ID"] == ""
        assert values["BAIDU_API_KEY"] == ""
        assert values["BAIDU_SECRET_KEY"] == ""


def test_development_script_exposes_expected_actions_and_remote_api_entry() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")

    for command in (
        "test",
        "lint",
        "api",
        "speech",
        "web",
        "video-test",
        "remote-model-check",
        "remote-model-test",
        "remote-api-video-test",
    ):
        assert f'"{command}"' in script
    assert "MODEL_BACKEND=remote" in script
    assert "deep_sea_explorer.main" in script
    assert "RUN_REMOTE_MODEL_TESTS" in script
    assert "test_main_api_video.py" in script


def test_fake_camera_launcher_forces_an_isolated_chrome_profile() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "start-video-camera-test.ps1"
    ).read_text(encoding="utf-8")

    for value in (
        "YUV4MPEG2",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--use-file-for-fake-video-capture",
        "--user-data-dir",
        "DeepSeaExplorerChromeProfiles",
        "ValidateOnly",
        "PassThru",
    ):
        assert value in script


def test_system_lifecycle_scripts_keep_process_ownership_scoped() -> None:
    local_script = (PROJECT_ROOT / "scripts" / "manage-system.ps1").read_text(encoding="utf-8")
    server_script = (
        PROJECT_ROOT / "scripts" / "server" / "manage-development-system.sh"
    ).read_text(encoding="utf-8")

    for value in (
        'ValidateSet("start", "stop", "status")',
        'ValidateSet("real", "simulated")',
        "start-video-camera-test.ps1",
        "system-tunnel.pid",
        "Invoke-RemoteSystemAction",
    ):
        assert value in local_script
    for value in (
        "APP_ENV_FILE",
        "refusing to replace an unmanaged process",
        "adopted existing",
        "does not match this project",
        "not force-killed",
        "deep_sea_explorer.production_wsgi:app",
    ):
        assert value.lower() in server_script.lower()


def test_double_click_launchers_offer_real_simulated_and_stop_actions() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "launch-system.ps1").read_text(encoding="utf-8")

    for value in (
        "System.Windows.Forms.OpenFileDialog",
        "Y4M",
        "manage-system.ps1",
        "DEEP_SEA_AUTOMATION_TEST",
        'ValidateSet("start", "stop")',
        'ValidateSet("real", "simulated")',
    ):
        assert value in launcher
    for filename, action in (
        ("启动系统（真实摄像头）.vbs", "-Action start -Mode real"),
        ("启动系统（视频模拟）.vbs", "-Action start -Mode simulated"),
        ("关闭系统.vbs", "-Action stop"),
    ):
        script = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "-WindowStyle Hidden" in script
        assert action in script
    simulated = (PROJECT_ROOT / "启动系统（视频模拟）.vbs").read_text(encoding="utf-8")
    assert "DEEP_SEA_SIMULATION_VIDEO" in simulated


def test_retrieval_review_export_scripts_do_not_require_the_reference_directory() -> None:
    local_script = (
        PROJECT_ROOT / "scripts" / "export-keyframe-retrieval-review.ps1"
    ).read_text(encoding="utf-8")
    server_script = (
        PROJECT_ROOT / "scripts" / "server" / "export-keyframe-retrieval-review.sh"
    ).read_text(encoding="utf-8")
    exporter = (PROJECT_ROOT / "scripts" / "export_keyframe_retrieval_review.py").read_text(
        encoding="utf-8"
    )

    assert "C:\\Users\\emc20\\Downloads" in local_script
    assert "REVIEW_EXPORT_PATH=" in server_script
    assert "events.sqlite3" in exporter
    assert "NumpyImageRetrievalIndex" in exporter
    assert "no_training_vlm_retrieval" not in local_script + server_script + exporter


def test_main_api_video_helper_samples_frames_and_uses_the_local_api() -> None:
    script = (PROJECT_ROOT / "scripts" / "test_main_api_video.py").read_text(encoding="utf-8")

    for value in ("cv2.VideoCapture", "at least four", "/videoanalyze", "X-Session-ID"):
        assert value in script


def test_model_service_tunnel_is_local_only_and_does_not_embed_credentials() -> None:
    script = (PROJECT_ROOT / "scripts" / "start-model-service-tunnel.ps1").read_text(
        encoding="utf-8"
    )

    assert "-L \"${LocalPort}:127.0.0.1:${RemotePort}\"" in script
    assert "ExitOnForwardFailure=yes" in script
    assert "ServerAliveInterval=30" in script
    assert "Authorization" not in script
    assert "MODEL_SERVICE_AUTH_TOKEN" not in script


def test_private_server_tunnel_uses_a_user_supplied_identity_file() -> None:
    script = (PROJECT_ROOT / "scripts" / "start-private-server-tunnel.ps1").read_text(
        encoding="utf-8"
    )

    assert '"-i", $IdentityFile' in script
    assert '"127.0.0.1:$LocalPort`:127.0.0.1:$RemotePort"' in script
    assert "ExitOnForwardFailure=yes" in script


def test_runtime_assets_do_not_reference_the_read_only_retrieval_reference_directory() -> None:
    reference_directory = "no" + "_training_vlm_retrieval"
    runtime_roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "deploy",
        PROJECT_ROOT / "frontend",
    )
    templates = (
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / ".env.development.example",
        PROJECT_ROOT / ".env.server.example",
    )
    files = [
        path
        for root in runtime_roots
        for path in root.rglob("*")
        if path.is_file()
    ] + list(templates)

    offenders = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in files
        if reference_directory in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == []
