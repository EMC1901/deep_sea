"""Gunicorn configuration for the independent speech API."""

import os


bind = os.getenv("DEEP_SEA_SPEECH_BIND", "127.0.0.1:9009")
workers = 1
threads = 2
worker_class = "gthread"
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
