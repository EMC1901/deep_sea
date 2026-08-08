"""Gunicorn lifecycle hooks for the stateful main API.

Keep exactly one worker: sessions, memo queues, and the monitoring worker are
in-memory objects and cannot be shared safely between Gunicorn workers.
"""

import os


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


def post_worker_init(worker):
    from deep_sea_explorer.api.app_factory import start_background_services

    start_background_services(worker.wsgi)


def worker_exit(server, worker):
    from deep_sea_explorer.api.app_factory import stop_background_services

    stop_background_services(worker.wsgi)
