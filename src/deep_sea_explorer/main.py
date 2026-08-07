from deep_sea_explorer.api.app_factory import (
    create_app,
    start_background_services,
    stop_background_services,
)
from deep_sea_explorer.config import Settings


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    start_background_services(app)
    try:
        app.run(host=settings.api_host, port=settings.api_port, debug=False, threaded=True)
    finally:
        stop_background_services(app)


if __name__ == "__main__":
    main()
