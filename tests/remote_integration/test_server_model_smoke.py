"""真实服务器模型测试仅由人工显式开启；普通 pytest 不会执行。"""

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REMOTE_MODEL_TESTS") != "1",
    reason="requires explicit remote model authorization",
)


def test_remote_model_placeholder() -> None:
    pytest.fail("服务器地址、鉴权和协议尚未确认；不得在此占位测试中访问服务器。")
