from __future__ import annotations

from deep_sea_explorer.api.app_factory import (
    create_app,
    start_background_services,
    stop_background_services,
)
from deep_sea_explorer.config import ModelBackend, Settings
from deep_sea_explorer.container import build_fake_container


def test_app_factory_creates_without_model_import_or_worker_start() -> None:
    settings = Settings(model_backend=ModelBackend.FAKE)
    container = build_fake_container(settings)
    app = create_app(settings, container)
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_backend"] == "fake"
    assert container.worker.last_success_monotonic is None


def test_background_services_start_only_when_the_serving_process_requests_them(
    monkeypatch,
) -> None:
    container = build_fake_container(Settings(model_backend=ModelBackend.FAKE))
    calls: list[str] = []
    monkeypatch.setattr(container.worker, "start", lambda: calls.append("start"))
    monkeypatch.setattr(container.worker, "stop", lambda: calls.append("stop"))
    app = create_app(container.settings, container)

    start_background_services(app)
    stop_background_services(app)

    assert calls == ["start", "stop"]
