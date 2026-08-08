#!/usr/bin/env bash
# Installs only the supplied service and Nginx configuration. It never reads
# application secrets; create /etc/deep-sea-explorer/app.env with sudoedit first.
set -euo pipefail

APP_ROOT="/opt/deep-sea-explorer"
APP_USER="deepsea"
APP_GROUP=""
ENV_DIR="/etc/deep-sea-explorer"
ENV_FILE="${ENV_DIR}/app.env"
VENV_DIR=""
NGINX_TARGET="/etc/nginx/sites-available/deep-sea-explorer"
NGINX_LINK="/etc/nginx/sites-enabled/deep-sea-explorer"
APACHE_TARGET="/etc/apache2/sites-available/deep-sea-explorer.conf"
WEB_SERVER=""
SERVER_NAME=""
APPLY=false

usage() {
    cat <<'EOF'
Usage: sudo deploy/install-production.sh --apply --web-server nginx|apache|apache-tunnel [options]

The project must already exist at --app-root, the service account must already
exist, and app.env must be created manually from .env.server.example. The
script does not install OS packages, models, or Python dependencies.

Apache mode also requires --server-name <approved-company-dns-name>. It creates
a named vhost and never replaces Apache's existing default site. apache-tunnel
adds a listener only on 127.0.0.1:19100 for a local SSH tunnel.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=true ;;
        --app-root)
            APP_ROOT="${2:?--app-root requires a path}"
            shift
            ;;
        --venv)
            VENV_DIR="${2:?--venv requires a path}"
            shift
            ;;
        --service-user)
            APP_USER="${2:?--service-user requires an existing Linux account}"
            shift
            ;;
        --web-server)
            WEB_SERVER="${2:?--web-server requires nginx or apache}"
            shift
            ;;
        --server-name)
            SERVER_NAME="${2:?--server-name requires a DNS name}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${EUID}" -ne 0 ]]; then
    printf '%s\n' 'Run this installer with sudo.' >&2
    exit 1
fi
if [[ "${WEB_SERVER}" != "nginx" && "${WEB_SERVER}" != "apache" && "${WEB_SERVER}" != "apache-tunnel" ]]; then
    printf '%s\n' 'Select --web-server nginx, apache, or apache-tunnel.' >&2
    exit 2
fi
if [[ "${WEB_SERVER}" == "apache" && ! "${SERVER_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
    printf '%s\n' 'Apache mode requires --server-name with an approved DNS name.' >&2
    exit 2
fi
if [[ ! -f "${APP_ROOT}/pyproject.toml" || ! -d "${APP_ROOT}/deploy" ]]; then
    printf '%s\n' "Invalid app root: ${APP_ROOT}" >&2
    exit 1
fi
if [[ ! "${APP_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    printf '%s\n' "App root must be an absolute path containing only letters, digits, ., _, /, and -: ${APP_ROOT}" >&2
    exit 1
fi
if ! id "${APP_USER}" >/dev/null 2>&1; then
    printf '%s\n' "Service user does not exist: ${APP_USER}" >&2
    exit 1
fi
APP_GROUP="$(id -gn "${APP_USER}")"
VENV_DIR="${VENV_DIR:-${APP_ROOT}/.venv}"
if [[ ! -x "${VENV_DIR}/bin/gunicorn" ]]; then
    printf '%s\n' "Gunicorn is unavailable at ${VENV_DIR}/bin/gunicorn." >&2
    printf '%s\n' 'Install the project and its local-model dependencies in the selected virtual environment first.' >&2
    exit 1
fi
WEB_SERVER_BINARY="${WEB_SERVER}"
WEB_SERVER_SERVICE="${WEB_SERVER}"
if [[ "${WEB_SERVER}" == "apache" || "${WEB_SERVER}" == "apache-tunnel" ]]; then
    WEB_SERVER_BINARY="apache2ctl"
    WEB_SERVER_SERVICE="apache2"
fi
for binary in "${WEB_SERVER_BINARY}" systemctl; do
    if ! command -v "${binary}" >/dev/null 2>&1; then
        printf '%s\n' "Required command is unavailable: ${binary}" >&2
        exit 1
    fi
done

if [[ "${APPLY}" != true ]]; then
    printf '%s\n' 'Preflight passed. Re-run with --apply after the maintenance window is approved.'
    exit 0
fi

install -d -m 0750 -o "${APP_USER}" -g "${APP_GROUP}" "${ENV_DIR}"
install -d -m 0750 -o "${APP_USER}" -g "${APP_GROUP}" /var/lib/deep-sea-explorer
if [[ ! -f "${ENV_FILE}" ]]; then
    printf '%s\n' "Missing ${ENV_FILE}; create it with sudoedit before applying service configuration." >&2
    exit 1
fi
chmod 0640 "${ENV_FILE}"
chown root:"${APP_GROUP}" "${ENV_FILE}"

backup_if_present() {
    local target="$1"
    if [[ -e "${target}" && ! -L "${target}" ]]; then
        cp -a -- "${target}" "${target}.bak.$(date +%Y%m%d%H%M%S)"
    fi
}

render_template() {
    local source="$1"
    local target="$2"
    local mode="$3"
    local temporary
    local escaped_root

    temporary="$(mktemp)"
    escaped_root="$(printf '%s' "${APP_ROOT}" | sed -e 's/[\\&|]/\\&/g')"
    sed -e "s|/opt/deep-sea-explorer|${escaped_root}|g" \
        -e "s|__SERVER_NAME__|${SERVER_NAME}|g" \
        -e "s|__SERVICE_USER__|${APP_USER}|g" \
        -e "s|__SERVICE_GROUP__|${APP_GROUP}|g" "${source}" >"${temporary}"
    install -m "${mode}" "${temporary}" "${target}"
    rm -f -- "${temporary}"
}

if [[ "${WEB_SERVER}" == "nginx" ]]; then
    backup_if_present "${NGINX_TARGET}"
    render_template "${APP_ROOT}/deploy/nginx/deep-sea-explorer.conf" "${NGINX_TARGET}" 0644
    ln -sfn ../sites-available/deep-sea-explorer "${NGINX_LINK}"
elif [[ "${WEB_SERVER}" == "apache" ]]; then
    backup_if_present "${APACHE_TARGET}"
    render_template "${APP_ROOT}/deploy/apache/deep-sea-explorer.conf" "${APACHE_TARGET}" 0644
    a2enmod proxy proxy_http headers
    a2ensite deep-sea-explorer.conf
else
    backup_if_present "${APACHE_TARGET%.conf}-tunnel.conf"
    render_template "${APP_ROOT}/deploy/apache/deep-sea-explorer-tunnel.conf" "${APACHE_TARGET%.conf}-tunnel.conf" 0644
    a2enmod proxy proxy_http headers
    a2ensite deep-sea-explorer-tunnel.conf
fi
install -m 0644 "${APP_ROOT}/deploy/runtime-config.server.js" "${ENV_DIR}/runtime-config.js"
render_template "${APP_ROOT}/deploy/systemd/deep-sea-explorer-api.service" /etc/systemd/system/deep-sea-explorer-api.service 0644
render_template "${APP_ROOT}/deploy/systemd/deep-sea-explorer-speech.service" /etc/systemd/system/deep-sea-explorer-speech.service 0644
chmod 0755 "${APP_ROOT}/deploy/run-api.sh" "${APP_ROOT}/deploy/run-speech.sh"

if [[ "${WEB_SERVER}" == "nginx" ]]; then
    nginx -t
else
    apache2ctl configtest
fi
systemctl daemon-reload
systemctl enable --now deep-sea-explorer-api.service deep-sea-explorer-speech.service
systemctl reload "${WEB_SERVER_SERVICE}"
printf '%s\n' 'Production services installed. Run the verification commands in deploy/README.md.'
