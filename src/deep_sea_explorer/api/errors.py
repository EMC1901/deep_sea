from __future__ import annotations

from flask import current_app, jsonify, g, request
from werkzeug.exceptions import HTTPException

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
        current_app.logger.error(
            "api model_failure request_id=%s endpoint=%s error_type=%s cause_type=%s",
            getattr(g, "request_id", "unknown"),
            request.path,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ is not None else "none",
            exc_info=(type(error), error, error.__traceback__),
        )
        return error_response(error.code, str(error), 503)

    @app.errorhandler(DomainError)
    def domain(error):
        return error_response(error.code, str(error), 400)

    @app.errorhandler(Exception)
    def unexpected(error):
        if isinstance(error, HTTPException):
            return error
        current_app.logger.error(
            "api unexpected_failure request_id=%s endpoint=%s error_type=%s",
            getattr(g, "request_id", "unknown"),
            request.path,
            type(error).__name__,
            exc_info=(type(error), error, error.__traceback__),
        )
        return error_response("INTERNAL_ERROR", "request failed", 500)
