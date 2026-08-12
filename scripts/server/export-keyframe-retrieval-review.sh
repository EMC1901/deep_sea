#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
VENV_DIR="${DEEP_SEA_VENV:-${PROJECT_ROOT}/.venv}"
APP_ENV_FILE="${DEEP_SEA_RUNTIME_ENV:-${PROJECT_ROOT}/runtime/app.env}"
RETRIEVAL_ENV_FILE="${PROJECT_ROOT}/runtime/image-retrieval.env"
SESSION_ID=""

usage() {
    printf '%s\n' "Usage: $0 [--session-id SESSION_ID]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session-id)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            SESSION_ID="$2"
            shift 2
            ;;
        *) usage >&2; exit 2 ;;
    esac
done

[[ -r "${APP_ENV_FILE}" ]] || { printf '%s\n' "Runtime configuration is unavailable." >&2; exit 2; }
[[ -r "${RETRIEVAL_ENV_FILE}" ]] || { printf '%s\n' "Image retrieval configuration is unavailable." >&2; exit 2; }
[[ -x "${VENV_DIR}/bin/python" ]] || { printf '%s\n' "Project Python is unavailable." >&2; exit 2; }

# Configuration is consumed but never displayed, so no sensitive values are logged.
set -a
# shellcheck disable=SC1090
. "${APP_ENV_FILE}"
# shellcheck disable=SC1090
. "${RETRIEVAL_ENV_FILE}"
set +a

: "${DATA_DIR:?DATA_DIR must be configured}"
: "${IMAGE_RETRIEVAL_INDEX_DIR:?IMAGE_RETRIEVAL_INDEX_DIR must be configured}"
: "${IMAGE_RETRIEVAL_DINO_MODEL_PATH:?IMAGE_RETRIEVAL_DINO_MODEL_PATH must be configured}"

EXPORT_ROOT="${PROJECT_ROOT}/runtime/retrieval-review-exports"
mkdir -p -- "${EXPORT_ROOT}"
OUTPUT_DIR="$(mktemp -d "${EXPORT_ROOT}/review.XXXXXX")"
ARGS=(
    "${VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/export_keyframe_retrieval_review.py"
    --data-dir "${DATA_DIR}"
    --index-dir "${IMAGE_RETRIEVAL_INDEX_DIR}"
    --dino-model-path "${IMAGE_RETRIEVAL_DINO_MODEL_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --top-k "${IMAGE_RETRIEVAL_TOP_K:-4}"
    --device "${IMAGE_RETRIEVAL_DEVICE:-auto}"
)
if [[ -n "${SESSION_ID}" ]]; then
    ARGS+=(--session-id "${SESSION_ID}")
fi

"${ARGS[@]}"
printf '%s\n' "REVIEW_EXPORT_PATH=${OUTPUT_DIR}"
