from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "server" / "collect_environment.sh"
BASELINE_PATH = PROJECT_ROOT / "docs" / "acceptance" / "environment-baseline.md"


def test_p1_01_collector_contains_every_required_read_only_query() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    required_commands = (
        "date -Is",
        "hostname",
        "uname -a",
        "cat /etc/os-release",
        "id",
        "python3 --version",
        "nvidia-smi",
        "nvcc --version",
        "free -h",
        "df -hT",
        "lsblk",
        "docker version",
        "docker compose version",
        "nvidia-ctk --version",
        "docker info --format '{{.DockerRootDir}}'",
        "dpkg-query -W nvidia-container-toolkit",
        "ffmpeg -version",
        "pro status",
        "apt-mark showhold",
        "ss -lnt",
    )

    for command in required_commands:
        assert command in script


def test_p1_01_collector_enforces_safety_and_expected_outputs() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "/sevenH/deep-sea-realtime/artifacts/phase1/" in script
    assert "docs/acceptance" in script
    assert "environment-baseline.md" in script
    assert "umask 077" in script
    assert "[exit_code] 127" in script
    assert "Ubuntu Pro 客户端不可用" in script
    assert "已拒绝生成证据文件" in script
    assert "<redacted-hostname>" in script
    assert "<redacted-user-and-groups>" in script
    assert "<redacted-ip>" in script
    assert "environment.before-${COLLECTION_TIMESTAMP}.txt" in script
    assert "docs/server-model-deployment-record.md" in script
    assert "gcc --version" not in script
    assert "cmake --version" not in script
    assert "str | None" not in script
    assert "re.Match[" not in script

    assert not re.search(r"(?m)^\s*sudo(?:\s|$)", script)
    assert not re.search(r"(?m)^\s*(?:apt|apt-get)\s+(?:install|remove|upgrade)", script)
    assert not re.search(r"(?m)^\s*rm\s+-[^\n]*r", script)
    assert not re.search(r"(?m)^\s*(?:find|ls)\s+/sevenH", script)
    assert not re.search(r"(?m)^\s*chmod(?:\s|$)", script)


def test_p1_01_baseline_requires_human_confirmation() -> None:
    baseline = BASELINE_PATH.read_text(encoding="utf-8")

    for marker in (
        "GPU 确实为 RTX 3090",
        "Ubuntu 确实为 20.04.6 LTS",
        "`/projects`",
        "`/sevenH`",
        "GPU 进程",
        "Docker data-root",
        "`19000`",
        "`19100`",
        "Ubuntu Pro/ESM",
        "docs/server-model-deployment-record.md",
        "与历史记录的差异",
        "确认人：EMC1901",
        "是否允许进入 P1-02：否",
    ):
        assert marker in baseline
