from __future__ import annotations

import os
import tempfile
from pathlib import Path

import imageio_ffmpeg  # type: ignore[import-untyped]
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from .audio_converter import AudioConverter
from .baidu_client import BaiduSpeechClient


def create_speech_app(
    client: object | None = None,
    converter: object | None = None,
    temp_dir: Path | None = None,
) -> Flask:
    app = Flask(__name__)
    CORS(app)
    client = client or BaiduSpeechClient(
        os.getenv("BAIDU_APP_ID", ""),
        os.getenv("BAIDU_API_KEY", ""),
        os.getenv("BAIDU_SECRET_KEY", ""),
    )
    converter = converter or AudioConverter(imageio_ffmpeg.get_ffmpeg_exe())

    @app.post("/stt")
    def stt():
        audio = request.files.get("audio")
        if audio is None:
            return jsonify({"error": "No audio file"}), 400
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as directory:
            source, target = Path(directory) / "input.webm", Path(directory) / "output.wav"
            audio.save(source)
            try:
                converter.convert_webm_to_wav(source, target)
                text = client.recognize(target.read_bytes())
            except Exception:
                return jsonify({"error": "语音处理服务内部错误"}), 500
        return (
            jsonify({"text": text})
            if text
            else jsonify({"error": "未能识别出语音，请更清晰地说话"})
        )

    @app.post("/tts")
    def tts():
        text = (request.get_json(silent=True) or {}).get("text")
        if not text:
            return jsonify({"error": "Missing text"}), 400
        try:
            audio = client.synthesize(str(text))
        except Exception:
            audio = None
        return Response(audio, mimetype="audio/wav") if audio else jsonify(
            {"error": "TTS failed"}
        ), 500

    return app
