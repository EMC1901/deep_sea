"""Gunicorn lifecycle hooks for the stateful main API.

Keep exactly one worker: sessions, memo queues, and the monitoring worker are
in-memory objects and cannot be shared safely between Gunicorn workers.
"""

import os
from pathlib import Path


bind = os.getenv("DEEP_SEA_API_BIND", "127.0.0.1:9001")
workers = 1
threads = int(os.getenv("DEEP_SEA_API_THREADS", "4"))
worker_class = "gthread"
timeout = int(os.getenv("DEEP_SEA_GUNICORN_TIMEOUT", "240"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
preload_app = False


def _qwen_runtime_override() -> list[str]:
    """Read the non-secret adapter override, if an operator supplied one."""
    runtime_env = Path(__file__).resolve().parents[1] / "runtime" / "qwen-runtime.env"
    try:
        lines = runtime_env.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "QWEN_ADAPTER_PATH":
            # An empty value deliberately selects the Qwen base model.
            return [f"QWEN_ADAPTER_PATH={value.strip()}"]
    return []


# Gunicorn reapplies this when HUP reloads the configuration and starts workers.
raw_env = _qwen_runtime_override()


def post_worker_init(worker):
    from deep_sea_explorer.api.app_factory import start_background_services

    start_background_services(worker.wsgi)


def worker_exit(server, worker):
    from deep_sea_explorer.api.app_factory import stop_background_services

    stop_background_services(worker.wsgi)
