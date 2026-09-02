from __future__ import annotations

import json

from flask import Blueprint, Response, current_app, request, stream_with_context

from deep_sea_explorer.api.schemas import question, session_id
from deep_sea_explorer.domain.exceptions import ValidationError

bp = Blueprint("video", __name__)


@bp.post("/videoanalyze")
def video_analyze():
    container = current_app.extensions["container"]
    sid = session_id(request.headers.get("X-Session-ID"))
    payload = request.get_json(silent=True)
    raw_question = payload.get("question") if isinstance(payload, dict) else request.form.get("question")
    prompt = question(raw_question, container.settings.max_question_length)
    frames = request.files.getlist("video")
    frame_results = []
    if frames and not prompt:
        # Monitoring receives independent JPEGs; it never assembles a short MP4.
        for frame in frames:
            frame_results.append(container.ingestion.ingest_frame(sid, frame.read(), container.monitoring))
        return {"status": "frames_processed", "frames": len(frame_results), "results": frame_results}
    if not prompt:
        return {"status": "no_input"}
    if frames:
        raise ValidationError("questions accept text only")

    def stream():
        try:
            for event in container.questions.answer(sid, prompt):
                body = {"type": event.type.value}
                body["text"] = event.text
                yield json.dumps(body, ensure_ascii=False) + "\n"
        except Exception as error:
            yield json.dumps({"type": "error", "text": str(error)}, ensure_ascii=False) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")
