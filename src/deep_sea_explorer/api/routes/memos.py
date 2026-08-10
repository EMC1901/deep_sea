from dataclasses import asdict

from flask import Blueprint, current_app, request

from deep_sea_explorer.api.schemas import session_id

bp = Blueprint("memos", __name__)


def serialize(memo):
    def serialize_capture(capture):
        return {
            "type": capture.type.value,
            "image": capture.image_data_uri,
            "description": capture.description,
            "organisms": [asdict(item) for item in capture.organisms],
            "env_features": [asdict(item) for item in capture.env_features],
        }

    captures = memo.captures or ((memo.capture,) if memo.capture else ())
    return {
        "timestamp": memo.timestamp,
        "content": memo.content,
        "session_id": memo.session_id,
        # Retain the original field for clients that only understand one capture.
        "capture": serialize_capture(memo.capture) if memo.capture else None,
        "captures": [serialize_capture(capture) for capture in captures],
    }


@bp.get("/memos")
def get_memos():
    container = current_app.extensions["container"]
    sid = request.headers.get("X-Session-ID") or request.args.get("session_id")
    return {
        "memos": [
            serialize(memo) for memo in container.memos.drain(session_id(sid) if sid else None)
        ]
    }
