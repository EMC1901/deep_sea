"""P0 测试公共保护措施。"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "allow_loopback_network: permit TCP connections to localhost for an explicit integration test",
    )


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """禁止测试意外创建真实网络连接。

    Flask 的 test_client 不经过 TCP socket；语音、模型和其他外部依赖在
    契约测试中均由 fake 注入。若后续测试确实需要网络，应显式覆盖本 fixture，
    并且不得加入默认测试集。
    """

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("测试禁止访问真实网络")

    if request.node.get_closest_marker("allow_loopback_network") is None:
        monkeypatch.setattr(socket, "create_connection", deny_network)
        monkeypatch.setattr(socket.socket, "connect", deny_network)
        return

    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect

    def is_loopback(address: object) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        return address[0] in {"127.0.0.1", "::1", "localhost"}

    def create_loopback_connection(address: object, *args: object, **kwargs: object) -> socket.socket:
        if not is_loopback(address):
            deny_network()
        return original_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]

    def connect_loopback(sock: socket.socket, address: object) -> None:
        if not is_loopback(address):
            deny_network()
        original_connect(sock, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "create_connection", create_loopback_connection)
    monkeypatch.setattr(socket.socket, "connect", connect_loopback)
