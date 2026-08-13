#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="${DEEP_SEA_VENV:-${PROJECT_ROOT}/.venv}"
GUNICORN="${VENV_DIR}/bin/gunicorn"
RETRIEVAL_ENV="${PROJECT_ROOT}/runtime/image-retrieval.env"

if [[ ! -x "${GUNICORN}" ]]; then
    printf '%s\n' "Gunicorn is unavailable at ${GUNICORN}; set DEEP_SEA_VENV in app.env." >&2
    exit 1
fi

cd -- "${PROJECT_ROOT}"
if [[ -f "${RETRIEVAL_ENV}" ]]; then
    # The frozen DINOv2 weights are shared by retrieval (when enabled) and
    # real-time frame filtering. This file contains no credentials.
    set -a
    . "${RETRIEVAL_ENV}"
    set +a
fi
if [[ -z "${MONITORING_DINO_MODEL_PATH:-}" && -n "${IMAGE_RETRIEVAL_DINO_MODEL_PATH:-}" ]]; then
    export MONITORING_DINO_MODEL_PATH="${IMAGE_RETRIEVAL_DINO_MODEL_PATH}"
fi
if [[ -z "${MONITORING_DINO_DEVICE:-}" && -n "${IMAGE_RETRIEVAL_DEVICE:-}" ]]; then
    export MONITORING_DINO_DEVICE="${IMAGE_RETRIEVAL_DEVICE}"
fi
exec "${GUNICORN}" --config "${PROJECT_ROOT}/deploy/gunicorn_api.conf.py" \
    deep_sea_explorer.production_wsgi:app
