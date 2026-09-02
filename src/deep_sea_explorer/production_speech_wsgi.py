"""WSGI entry point for the production speech service."""

from deep_sea_explorer.config import Settings
from deep_sea_explorer.speech.app_factory import create_speech_app


settings = Settings.from_env()
app = create_speech_app(temp_dir=settings.temp_dir)
