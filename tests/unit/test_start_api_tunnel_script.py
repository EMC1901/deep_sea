from __future__ import annotations

import re
import shutil
import socket
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev" / "start-api-tunnel.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is required for the Windows SSH tunnel script tests")

    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_tunnel_script_has_fixed_loopback_security_boundary() -> None:
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"127.0.0.1:19100:127.0.0.1:19100"' in content
    assert '"-N"' in content
    assert '"-T"' in content
    assert '"ExitOnForwardFailure=yes"' in content
    assert "19000" not in content

    ipv4_values = set(re.findall(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])", content))
    assert ipv4_values == {"127.0.0.1"}


def test_tunnel_script_keeps_connection_values_as_parameters() -> None:
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    lowered = content.lower()

    for parameter in ("$ServerAddress", "$SshUser", "$SshPort"):
        assert parameter in content

    for forbidden in (
        "password=",
        "token=",
        "identityfile=",
        "begin openssh private key",
        "begin rsa private key",
    ):
        assert forbidden not in lowered


def test_tunnel_script_displays_help_when_parameters_are_missing() -> None:
    result = _run_script()
    output = result.stdout + result.stderr

    assert result.returncode == 2
    assert "Usage:" in output
    assert "-ServerAddress" in output
    assert "-SshUser" in output
    assert "-SshPort" in output


def test_tunnel_script_refuses_an_occupied_local_port() -> None:
    listener: socket.socket | None = None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 19100))
        listener.listen(1)
    except OSError:
        if listener is not None:
            listener.close()
        listener = None

    try:
        result = _run_script(
            "-ServerAddress",
            "example.invalid",
            "-SshUser",
            "tester",
            "-SshPort",
            "22",
        )
    finally:
        if listener is not None:
            listener.close()

    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "127.0.0.1:19100 is already in use" in output
