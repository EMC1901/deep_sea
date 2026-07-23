"""P0 测试公共保护措施。"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁止测试意外创建真实网络连接。

    Flask 的 test_client 不经过 TCP socket；语音、模型和其他外部依赖在
    契约测试中均由 fake 注入。若后续测试确实需要网络，应显式覆盖本 fixture，
    并且不得加入默认测试集。
    """

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("测试禁止访问真实网络")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
