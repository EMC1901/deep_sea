from __future__ import annotations

import json

from flask import Blueprint, Response, current_app, request, stream_with_context

from deep_sea_explorer.api.schemas import question, session_id
from deep_sea_explorer.domain.enums import StreamEventType
from deep_sea_explorer.domain.exceptions import ValidationError

bp = Blueprint("video", __name__)


@bp.post("/videoanalyze")
def video_analyze():
    container = current_app.extensions["container"]
    sid = session_id(request.headers.get("X-Session-ID"))
    prompt = question(request.form.get("question"), container.settings.max_question_length)
    frames = request.files.getlist("video")
    video_path = None
    frame_results = []
    if frames and not prompt:
        # Monitoring receives independent JPEGs; it never assembles a short MP4.
        for frame in frames:
            frame_results.append(container.ingestion.ingest_frame(sid, frame.read(), container.monitoring))
        return {"status": "frames_processed", "frames": len(frame_results), "results": frame_results}
    if frames:
        video_path = container.ingestion.ingest(sid, [frame.read() for frame in frames])
    if not prompt:
        return {"status": "video_saved", "path": str(video_path) if video_path else None}
    if video_path is None:
        latest = container.sessions.get(sid).latest_video
        if not latest:
            raise ValidationError("No video data available")
        from pathlib import Path

        video_path = Path(latest)

    def stream():
        try:
            for event in container.questions.answer(sid, prompt, video_path):
                body = {"type": event.type.value}
                if event.type is StreamEventType.IMAGE:
                    body.update(content=event.content, prompt=event.prompt)
                else:
                    body["text"] = event.text
                yield json.dumps(body, ensure_ascii=False) + "\n"
        except Exception as error:
            yield json.dumps({"type": "error", "text": str(error)}, ensure_ascii=False) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")
