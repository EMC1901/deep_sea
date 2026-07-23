from __future__ import annotations

from flask import jsonify, g

from deep_sea_explorer.domain.exceptions import DomainError, ModelUnavailableError, ValidationError


def error_response(code: str, message: str, status: int):
    return jsonify(
        {
            "error": message,
            "error_detail": {
                "code": code,
                "message": message,
                "request_id": getattr(g, "request_id", ""),
            },
        }
    ), status


def register_error_handlers(app) -> None:
    @app.errorhandler(ValidationError)
    def validation(error):
        return error_response(error.code, str(error), 400)

    @app.errorhandler(ModelUnavailableError)
    def unavailable(error):
        return error_response(error.code, str(error), 503)

    @app.errorhandler(DomainError)
    def domain(error):
        return error_response(error.code, str(error), 400)
