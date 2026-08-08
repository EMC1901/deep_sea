from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


def test_single_server_assets_keep_internal_services_loopback_only() -> None:
    nginx = (DEPLOY_ROOT / "nginx" / "deep-sea-explorer.conf").read_text(encoding="utf-8")
    runtime = (DEPLOY_ROOT / "runtime-config.server.js").read_text(encoding="utf-8")

    assert "proxy_pass http://127.0.0.1:9001/" in nginx
    assert "proxy_pass http://127.0.0.1:9009/" in nginx
    assert 'apiBaseUrl: "/api"' in runtime
    assert 'speechBaseUrl: "/speech"' in runtime


def test_apache_proxy_is_available_for_servers_where_port_80_is_already_in_use() -> None:
    apache = (DEPLOY_ROOT / "apache" / "deep-sea-explorer.conf").read_text(encoding="utf-8")

    assert "ServerName __SERVER_NAME__" in apache
    assert "ProxyPass /api/ http://127.0.0.1:9001/" in apache
    assert "ProxyPass /speech/ http://127.0.0.1:9009/" in apache
    assert "DocumentRoot /opt/deep-sea-explorer/frontend" in apache


def test_private_apache_tunnel_does_not_bind_a_public_web_port() -> None:
    tunnel = (DEPLOY_ROOT / "apache" / "deep-sea-explorer-tunnel.conf").read_text(
        encoding="utf-8"
    )

    assert "Listen 127.0.0.1:19100" in tunnel
    assert "<VirtualHost 127.0.0.1:19100>" in tunnel
    assert "ProxyPass /api/ http://127.0.0.1:9001/" in tunnel


def test_main_api_gunicorn_configuration_uses_one_worker_and_starts_monitoring() -> None:
    config = (DEPLOY_ROOT / "gunicorn_api.conf.py").read_text(encoding="utf-8")

    assert "workers = 1" in config
    assert "start_background_services(worker.wsgi)" in config
    assert "stop_background_services(worker.wsgi)" in config


def test_systemd_services_use_the_unprivileged_service_account_and_wrappers() -> None:
    for name, wrapper in (
        ("deep-sea-explorer-api.service", "run-api.sh"),
        ("deep-sea-explorer-speech.service", "run-speech.sh"),
    ):
        unit = (DEPLOY_ROOT / "systemd" / name).read_text(encoding="utf-8")
        assert "User=__SERVICE_USER__" in unit
        assert f"ExecStart=/opt/deep-sea-explorer/deploy/{wrapper}" in unit
        assert "ProtectSystem=full" in unit


def test_installer_renders_the_default_paths_for_an_explicit_app_root() -> None:
    installer = (DEPLOY_ROOT / "install-production.sh").read_text(encoding="utf-8")

    assert "--app-root" in installer
    assert "render_template" in installer
    assert "s|/opt/deep-sea-explorer|${escaped_root}|g" in installer
    assert "--web-server" in installer
    assert "--server-name" in installer
    assert "apache-tunnel" in installer
