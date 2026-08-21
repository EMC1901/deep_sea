from dataclasses import asdict

from flask import Blueprint, current_app, request

from deep_sea_explorer.api.schemas import session_id

bp = Blueprint("memos", __name__)


def serialize(memo, knowledge_base):
    def serialize_items(items, category):
        return [
            {**asdict(item), "display_name": knowledge_base.display_name(category, item.name)}
            for item in items
        ]

    def serialize_capture(capture):
        return {
            "type": capture.type.value,
            "image": capture.image_data_uri,
            "description": capture.description,
            "organisms": serialize_items(capture.organisms, "bio"),
            "env_features": serialize_items(capture.env_features, "env"),
            "substrates": serialize_items(capture.substrates, "substrate"),
            "geomorphologies": serialize_items(capture.geomorphologies, "geomorphology"),
        }

    captures = memo.captures or ((memo.capture,) if memo.capture else ())
    statistics = {
        category: serialize_items(memo.statistics.get(category, ()), category)
        for category in ("bio", "substrate", "geomorphology")
    }
    return {
        "timestamp": memo.timestamp,
        "content": memo.content,
        "session_id": memo.session_id,
        # Retain the original field for clients that only understand one capture.
        "capture": serialize_capture(memo.capture) if memo.capture else None,
        "captures": [serialize_capture(capture) for capture in captures],
        "coordinates": memo.coordinates.as_payload() if memo.coordinates else None,
        "statistics": statistics,
    }


@bp.get("/memos")
def get_memos():
    container = current_app.extensions["container"]
    sid = request.headers.get("X-Session-ID") or request.args.get("session_id")
    return {
        "memos": [
            serialize(memo, container.monitoring.knowledge_base)
            for memo in container.memos.drain(session_id(sid) if sid else None)
        ]
    }
