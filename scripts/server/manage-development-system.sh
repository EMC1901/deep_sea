#!/usr/bin/env bash
set -euo pipefail

# Manage only processes whose PIDs were created by this script. It never
# searches for or terminates unrelated web, Gunicorn, or system services.
ACTION="${1:-}"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime"
LOG_DIR="${PROJECT_ROOT}/logs"
APP_ENV_FILE="${DEEP_SEA_RUNTIME_ENV:-${RUNTIME_DIR}/app.env}"
MODEL_BACKEND_OVERRIDE_ENV="${DEEP_SEA_MODEL_BACKEND_OVERRIDE_ENV:-${RUNTIME_DIR}/model-backend.env}"
VENV_DIR="${DEEP_SEA_VENV:-${PROJECT_ROOT}/.venv}"
PYTHON="${VENV_DIR}/bin/python"

usage() {
    printf '%s\n' "Usage: $0 {start|stop|status}"
}

gguf_backend_enabled() {
    # The optional override selects the deployed model backend without changing
    # the preserved base runtime configuration.  Do not print sourced values.
    [[ -r "${APP_ENV_FILE}" ]] || return 1
    (
        set -a
        # shellcheck disable=SC1090
        . "${APP_ENV_FILE}"
        if [[ -r "${MODEL_BACKEND_OVERRIDE_ENV}" ]]; then
            # shellcheck disable=SC1090
            . "${MODEL_BACKEND_OVERRIDE_ENV}"
        fi
        set +a
        [[ "${MODEL_BACKEND:-local}" == "gguf" ]]
    )
}

manage_gguf_server() {
    bash "${PROJECT_ROOT}/scripts/server/manage-gguf-server.sh" "$1"
}

assert_runtime_environment() {
    if [[ ! -r "${APP_ENV_FILE}" ]]; then
        printf '%s\n' "Missing runtime configuration: ${APP_ENV_FILE}" >&2
        printf '%s\n' "Create it from .env.server.example with server-local values; do not commit it." >&2
        exit 2
    fi
    if [[ ! -x "${PYTHON}" ]]; then
        printf '%s\n' "Python is unavailable at ${PYTHON}" >&2
        exit 2
    fi

    # This imports configuration without printing its values.
    set -a
    # shellcheck disable=SC1090
    . "${APP_ENV_FILE}"
    if [[ -r "${MODEL_BACKEND_OVERRIDE_ENV}" ]]; then
        # shellcheck disable=SC1090
        . "${MODEL_BACKEND_OVERRIDE_ENV}"
    fi
    set +a
    cd -- "${PROJECT_ROOT}"
    "${PYTHON}" -c 'from deep_sea_explorer.config import Settings; errors = Settings.from_env().validate_for_runtime(); raise SystemExit("; ".join(errors) if errors else 0)'
}

pid_file() {
    printf '%s/%s.pid\n' "${RUNTIME_DIR}" "$1"
}

log_file() {
    printf '%s/development-%s.log\n' "${LOG_DIR}" "$1"
}

pid_is_running() {
    local pid="$1"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null
}

port_is_in_use() {
    local port="$1"
    ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .
}

managed_process_matches() {
    local name="$1"
    local pid="$2"
    local args
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    case "${name}" in
        api)
            [[ "${args}" == *"${PROJECT_ROOT}/deploy/gunicorn_api.conf.py"* ]] &&
                [[ "${args}" == *"deep_sea_explorer.production_wsgi:app"* ]]
            ;;
        speech)
            ([[ "${args}" == *"${PROJECT_ROOT}/deploy/gunicorn_speech.conf.py"* ]] ||
                [[ "${args}" == *"--config deploy/gunicorn_speech.conf.py"* ]]) &&
                [[ "${args}" == *"deep_sea_explorer.production_speech_wsgi:app"* ]]
            ;;
        web)
            [[ "${args}" == *"http.server 19100"* ]] &&
                ([[ "${args}" == *"${PROJECT_ROOT}/frontend"* ]] ||
                    [[ "${args}" == *"--directory frontend"* ]])
            ;;
        *) return 1 ;;
    esac
}

find_project_process() {
    local name="$1"
    local pid args
    local matches=()
    while read -r pid args; do
        if managed_process_matches "${name}" "${pid}"; then
            matches+=("${pid}")
        fi
    done < <(ps -eo pid=,args=)

    local candidate parent other
    local roots=()
    for candidate in "${matches[@]}"; do
        parent="$(ps -p "${candidate}" -o ppid= 2>/dev/null | tr -d ' ')"
        for other in "${matches[@]}"; do
            if [[ "${parent}" == "${other}" ]]; then
                break
            fi
        done
        if [[ "${parent}" != "${other}" ]]; then
            roots+=("${candidate}")
        fi
    done
    if [[ ${#roots[@]} -eq 1 ]]; then
        printf '%s\n' "${roots[0]}"
        return 0
    fi
    return 1
}

ensure_or_adopt() {
    local name="$1"
    local port="$2"
    local file
    file="$(pid_file "${name}")"

    if [[ -f "${file}" ]]; then
        local existing_pid
        existing_pid="$(<"${file}")"
        if pid_is_running "${existing_pid}" && managed_process_matches "${name}" "${existing_pid}"; then
            return 0
        fi
        rm -f -- "${file}"
    fi
    if ! port_is_in_use "${port}"; then
        return 2
    fi

    local project_pid
    if ! project_pid="$(find_project_process "${name}")"; then
        printf '%s\n' "Port ${port} is occupied by an unmanaged process; refusing to replace it." >&2
        return 1
    fi
    printf '%s\n' "${project_pid}" >"${file}"
    printf '%s\n' "Adopted existing ${name} process (PID ${project_pid})."
}

start_managed() {
    local name="$1"
    local port="$2"
    shift 2
    local file
    file="$(pid_file "${name}")"

    if [[ -f "${file}" ]]; then
        local existing_pid
        existing_pid="$(<"${file}")"
        if pid_is_running "${existing_pid}" && managed_process_matches "${name}" "${existing_pid}"; then
            printf '%s\n' "${name} is already running (PID ${existing_pid})."
            return 0
        fi
        rm -f -- "${file}"
    fi
    if port_is_in_use "${port}"; then
        printf '%s\n' "Port ${port} is already in use; refusing to replace an unmanaged process." >&2
        return 1
    fi

    nohup "$@" >>"$(log_file "${name}")" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" >"${file}"
    printf '%s\n' "Started ${name} (PID ${pid})."
}

stop_managed() {
    local name="$1"
    local file
    file="$(pid_file "${name}")"
    if [[ ! -f "${file}" ]]; then
        printf '%s\n' "${name} is not managed by this script."
        return 0
    fi

    local pid
    pid="$(<"${file}")"
    if ! pid_is_running "${pid}"; then
        rm -f -- "${file}"
        printf '%s\n' "Removed stale ${name} PID record."
        return 0
    fi
    if ! managed_process_matches "${name}" "${pid}"; then
        printf '%s\n' "Refusing to stop ${name}: PID ${pid} does not match this project." >&2
        return 1
    fi

    kill -TERM "${pid}"
    for _ in {1..30}; do
        if ! pid_is_running "${pid}"; then
            rm -f -- "${file}"
            printf '%s\n' "Stopped ${name}."
            return 0
        fi
        sleep 1
    done
    printf '%s\n' "${name} did not stop within 30 seconds; it was not force-killed." >&2
    return 1
}

show_status() {
    local name
    for name in api speech web; do
        local file
        file="$(pid_file "${name}")"
        if [[ -f "${file}" ]] && pid_is_running "$(<"${file}")" && managed_process_matches "${name}" "$(<"${file}")"; then
            printf '%s\n' "${name}: running (PID $(<"${file}"))"
        else
            printf '%s\n' "${name}: stopped or unmanaged"
        fi
    done
}

case "${ACTION}" in
    start)
        mkdir -p -- "${RUNTIME_DIR}" "${LOG_DIR}"
        if gguf_backend_enabled; then
            manage_gguf_server start
        fi
        missing=()
        for component in api speech web; do
            case "${component}" in
                api) port=9001 ;;
                speech) port=9009 ;;
                web) port=19100 ;;
            esac
            if ensure_or_adopt "${component}" "${port}"; then
                continue
            else
                result=$?
            fi
            if [[ ${result} -eq 2 ]]; then
                missing+=("${component}")
            else
                exit "${result}"
            fi
        done
        if [[ ${#missing[@]} -gt 0 ]]; then
            assert_runtime_environment
        fi
        for component in "${missing[@]}"; do
            case "${component}" in
                api) start_managed api 9001 bash "${PROJECT_ROOT}/deploy/run-api.sh" ;;
                speech) start_managed speech 9009 bash "${PROJECT_ROOT}/deploy/run-speech.sh" ;;
                web) start_managed web 19100 "${PYTHON}" -m http.server 19100 --bind 127.0.0.1 --directory "${PROJECT_ROOT}/frontend" ;;
            esac
        done
        ;;
    stop)
        stop_managed web
        stop_managed speech
        stop_managed api
        if gguf_backend_enabled; then
            manage_gguf_server stop
        fi
        ;;
    status)
        show_status
        if gguf_backend_enabled; then
            manage_gguf_server status
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac
