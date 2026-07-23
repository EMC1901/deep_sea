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
    assert {"torch", "transformers", "modelscope", "sentence-transformers"} == set(
        extras["local-model"]
    )
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


def test_environment_templates_do_not_contain_baidu_credentials() -> None:
    for filename in (".env.example", ".env.development.example", ".env.server.example"):
        values = read_dotenv_template(filename)
        assert values["BAIDU_APP_ID"] == ""
        assert values["BAIDU_API_KEY"] == ""
        assert values["BAIDU_SECRET_KEY"] == ""


def test_development_script_exposes_expected_actions_and_remote_api_entry() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")

    for command in ("test", "lint", "api", "speech", "web", "remote-model-check"):
        assert f'"{command}"' in script
    assert "MODEL_BACKEND=remote" in script
    assert "deep_sea_explorer.main" in script
