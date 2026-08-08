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
    ):
        assert value in script


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
