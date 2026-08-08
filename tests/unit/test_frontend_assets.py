from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_uses_runtime_configuration_and_safe_dom_helper() -> None:
    page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    config = (PROJECT_ROOT / "frontend" / "js" / "runtime-config.js").read_text(encoding="utf-8")
    helper = (PROJECT_ROOT / "frontend" / "js" / "safe-dom.js").read_text(encoding="utf-8")
    assert "js/runtime-config.js" in page
    assert "window.APP_CONFIG.apiBaseUrl" in page
    assert "apiBaseUrl" in config
    assert "textContent" in helper
    assert "isSafeImageSource" in helper
    assert 'headers: { "X-Session-ID": state.sessionId }' in page


def test_login_page_is_explicitly_marked_as_a_demo() -> None:
    page = (PROJECT_ROOT / "frontend" / "login.html").read_text(encoding="utf-8")
    assert 'data-page-purpose="demo-only"' in page
