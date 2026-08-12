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
    # This optional file holds only IMAGE_RETRIEVAL_* values, never secrets.
    set -a
    . "${RETRIEVAL_ENV}"
    set +a
fi
exec "${GUNICORN}" --config "${PROJECT_ROOT}/deploy/gunicorn_api.conf.py" \
    deep_sea_explorer.production_wsgi:app
