from deep_sea_explorer.api.app_factory import create_app
from deep_sea_explorer.config import Settings


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    app.run(host=settings.api_host, port=settings.api_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
