"""WSGI entry point for the single-process production API service."""

from deep_sea_explorer.api.app_factory import create_app
from deep_sea_explorer.config import Settings


app = create_app(Settings.from_env())
