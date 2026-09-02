"""Temporary S9 launcher; S12 will replace it with a managed Gunicorn service."""

from deep_sea_explorer.config import Settings
from deep_sea_explorer.model_service.app_factory import create_app


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    app.run(host=settings.model_service_host, port=settings.model_service_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
