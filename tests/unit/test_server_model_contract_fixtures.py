from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_FILE = PROJECT_ROOT / "docs" / "server-model-api-contract.md"
PLACEHOLDER_ENV = PROJECT_ROOT / "docs" / "server-model-placeholder.env"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "server_model_api"


def read_env_template(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def test_server_model_contract_is_explicitly_draft_and_not_implemented() -> None:
    contract = CONTRACT_FILE.read_text(encoding="utf-8")

    assert "DRAFT / NOT IMPLEMENTED" in contract
    assert "不能被视为可用接口" in contract
    for endpoint in (
        "GET /v1/health",
        "POST /v1/vision/describe-video",
        "POST /v1/vision/evaluate-frame",
        "POST /v1/vision/answer",
        "POST /v1/vision/summarize-report",
        "POST /v1/images/generate",
        "POST /v1/embeddings",
    ):
        assert endpoint in contract


def test_placeholder_environment_disables_remote_connection() -> None:
    values = read_env_template(PLACEHOLDER_ENV)

    assert values["MODEL_BACKEND"] == "remote"
    assert values["MODEL_SERVICE_ENABLED"] == "false"
    assert values["MODEL_SERVICE_BASE_URL"].endswith(".invalid")
    assert values["MODEL_SERVICE_AUTH_TOKEN"] == ""
    assert values["MODEL_SERVICE_VERIFY_TLS"] == "true"


def test_mock_fixtures_match_the_draft_contract() -> None:
    health = json.loads((FIXTURE_DIR / "health.json").read_text(encoding="utf-8"))
    decision = json.loads((FIXTURE_DIR / "capture_decision.json").read_text(encoding="utf-8"))
    embedding = json.loads((FIXTURE_DIR / "embedding.json").read_text(encoding="utf-8"))
    error = json.loads((FIXTURE_DIR / "error_not_configured.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (FIXTURE_DIR / "answer.ndjson").read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert set(health["models"]) == {"qwen", "image", "gte", "minilm"}
    assert decision["decision"]["category"] == "bio"
    assert decision["decision"]["organisms"][0]["count"] == 1
    assert embedding["model"] == "memo"
    assert embedding["dimension"] == len(embedding["vectors"][0])
    assert embedding["normalized"] is True
    assert error["error"]["code"] == "MODEL_SERVICE_NOT_CONFIGURED"
    assert [event["type"] for event in events] == ["chunk", "chunk", "final"]
