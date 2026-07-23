from dataclasses import asdict

from flask import Blueprint, current_app, request

from deep_sea_explorer.api.schemas import session_id

bp = Blueprint("memos", __name__)


def serialize(memo):
    capture = None
    if memo.capture:
        capture = {
            "type": memo.capture.type.value,
            "image": memo.capture.image_data_uri,
            "description": memo.capture.description,
            "organisms": [asdict(item) for item in memo.capture.organisms],
            "env_features": [asdict(item) for item in memo.capture.env_features],
        }
    return {
        "timestamp": memo.timestamp,
        "content": memo.content,
        "session_id": memo.session_id,
        "capture": capture,
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
