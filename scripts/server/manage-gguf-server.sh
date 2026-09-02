#!/usr/bin/env bash
set -euo pipefail

# Manage only the llama-server process started by this script.  The service is
# intentionally loopback-only; Flask communicates with it over local HTTP.
ACTION="${1:-}"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime"
LOG_DIR="${PROJECT_ROOT}/logs"
APP_ENV_FILE="${DEEP_SEA_RUNTIME_ENV:-${RUNTIME_DIR}/app.env}"
MODEL_BACKEND_OVERRIDE_ENV="${DEEP_SEA_MODEL_BACKEND_OVERRIDE_ENV:-${RUNTIME_DIR}/model-backend.env}"
PID_FILE="${RUNTIME_DIR}/gguf-llama-server.pid"
LOG_FILE="${LOG_DIR}/gguf-llama-server.log"

usage() {
    printf '%s\n' "Usage: $0 {start|stop|status}"
}

load_config() {
    if [[ ! -r "${APP_ENV_FILE}" ]]; then
        printf '%s\n' "Missing runtime configuration: ${APP_ENV_FILE}" >&2
        exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    . "${APP_ENV_FILE}"
    set +a
    if [[ -r "${MODEL_BACKEND_OVERRIDE_ENV}" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "${MODEL_BACKEND_OVERRIDE_ENV}"
        set +a
    fi
    : "${GGUF_LLAMA_SERVER_PATH:?GGUF_LLAMA_SERVER_PATH is required}"
    : "${GGUF_MODEL_PATH:?GGUF_MODEL_PATH is required}"
    : "${GGUF_MMPROJ_PATH:?GGUF_MMPROJ_PATH is required}"
    : "${GGUF_CONTEXT_SIZE:?GGUF_CONTEXT_SIZE is required}"
    : "${GGUF_GPU_LAYERS:?GGUF_GPU_LAYERS is required}"
    : "${GGUF_SERVER_HOST:=127.0.0.1}"
    : "${GGUF_SERVER_PORT:=19001}"
    [[ "${GGUF_SERVER_HOST}" == "127.0.0.1" ]] || {
        printf '%s\n' "GGUF_SERVER_HOST must be 127.0.0.1" >&2
        exit 2
    }
    [[ -x "${GGUF_LLAMA_SERVER_PATH}" && -r "${GGUF_MODEL_PATH}" && -r "${GGUF_MMPROJ_PATH}" ]] || {
        printf '%s\n' "GGUF llama-server, model, or vision projector is unavailable" >&2
        exit 2
    }
}

pid_is_running() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] && kill -0 "$1" 2>/dev/null
}

matches_managed_server() {
    local pid="$1" args
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${args}" == *"${GGUF_LLAMA_SERVER_PATH}"* ]] &&
        [[ "${args}" == *"--model ${GGUF_MODEL_PATH}"* ]] &&
        [[ "${args}" == *"--mmproj ${GGUF_MMPROJ_PATH}"* ]] &&
        [[ "${args}" == *"--port ${GGUF_SERVER_PORT}"* ]]
}

server_healthy() {
    curl -fsS --max-time 5 "http://${GGUF_SERVER_HOST}:${GGUF_SERVER_PORT}/health" 2>/dev/null |
        grep -q '"status":"ok"'
}

start() {
    load_config
    mkdir -p -- "${RUNTIME_DIR}" "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(<"${PID_FILE}")"
        if pid_is_running "${pid}" && matches_managed_server "${pid}"; then
            printf '%s\n' "gguf llama-server is already running (PID ${pid})."
            return 0
        fi
        rm -f -- "${PID_FILE}"
    fi
    if ss -ltnH "sport = :${GGUF_SERVER_PORT}" 2>/dev/null | grep -q .; then
        printf '%s\n' "Port ${GGUF_SERVER_PORT} is already in use; refusing to replace it." >&2
        return 1
    fi
    nohup "${GGUF_LLAMA_SERVER_PATH}" \
        --model "${GGUF_MODEL_PATH}" \
        --mmproj "${GGUF_MMPROJ_PATH}" \
        --host "${GGUF_SERVER_HOST}" \
        --port "${GGUF_SERVER_PORT}" \
        --ctx-size "${GGUF_CONTEXT_SIZE}" \
        --gpu-layers "${GGUF_GPU_LAYERS}" \
        --parallel 1 \
        --reasoning off \
        --jinja \
        --mmproj-offload \
        --image-min-tokens 1024 \
        --no-warmup >>"${LOG_FILE}" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" >"${PID_FILE}"
    for _ in {1..30}; do
        if server_healthy; then
            printf '%s\n' "Started gguf llama-server (PID ${pid})."
            return 0
        fi
        if ! pid_is_running "${pid}"; then
            rm -f -- "${PID_FILE}"
            printf '%s\n' "gguf llama-server exited during startup; see ${LOG_FILE}" >&2
            return 1
        fi
        sleep 1
    done
    printf '%s\n' "gguf llama-server did not become healthy; see ${LOG_FILE}" >&2
    return 1
}

stop() {
    load_config
    if [[ ! -f "${PID_FILE}" ]]; then
        printf '%s\n' "gguf llama-server is not managed by this script."
        return 0
    fi
    local pid
    pid="$(<"${PID_FILE}")"
    if ! pid_is_running "${pid}"; then
        rm -f -- "${PID_FILE}"
        printf '%s\n' "Removed stale gguf llama-server PID record."
        return 0
    fi
    if ! matches_managed_server "${pid}"; then
        printf '%s\n' "Refusing to stop non-matching gguf llama-server process." >&2
        return 1
    fi
    kill -TERM "${pid}"
    for _ in {1..30}; do
        if ! pid_is_running "${pid}"; then
            rm -f -- "${PID_FILE}"
            printf '%s\n' "Stopped gguf llama-server."
            return 0
        fi
        sleep 1
    done
    printf '%s\n' "gguf llama-server did not stop within 30 seconds; it was not force-killed." >&2
    return 1
}

status() {
    load_config
    if [[ -f "${PID_FILE}" ]] && pid_is_running "$(<"${PID_FILE}")" && matches_managed_server "$(<"${PID_FILE}")" && server_healthy; then
        printf '%s\n' "gguf llama-server: running (PID $(<"${PID_FILE}"))"
    else
        printf '%s\n' "gguf llama-server: stopped or unmanaged"
        return 1
    fi
}

case "${ACTION}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) usage >&2; exit 2 ;;
esac
