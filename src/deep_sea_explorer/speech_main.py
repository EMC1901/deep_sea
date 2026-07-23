from deep_sea_explorer.config import Settings
from deep_sea_explorer.speech.app_factory import create_speech_app


def main() -> None:
    settings = Settings.from_env()
    create_speech_app(temp_dir=settings.temp_dir).run(
        host=settings.speech_host, port=settings.speech_port, debug=False, threaded=True
    )


if __name__ == "__main__":
    main()
