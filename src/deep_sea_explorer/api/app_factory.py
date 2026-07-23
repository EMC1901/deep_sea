from __future__ import annotations

import time
import uuid

from flask import Flask, g
from flask_cors import CORS

from deep_sea_explorer.config import Settings
from deep_sea_explorer.container import build_container

from .errors import register_error_handlers
from .routes import health, memos, rag, reports, video


def create_app(settings: Settings | None = None, container: object | None = None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=settings.max_content_length_mb * 1024 * 1024)
    CORS(app, origins=list(settings.cors_origins))
    app.extensions["container"] = container or build_container(settings)

    @app.before_request
    def request_context() -> None:
        g.request_id = uuid.uuid4().hex
        g.request_started = time.monotonic()

    register_error_handlers(app)
    for blueprint in (health.bp, video.bp, memos.bp, rag.bp, reports.bp):
        app.register_blueprint(blueprint)
    return app
