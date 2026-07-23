from __future__ import annotations

from deep_sea_explorer.api.app_factory import create_app
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
