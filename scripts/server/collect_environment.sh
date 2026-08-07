#!/usr/bin/env bash

# Lightweight phase-one P1-01 read-only server inventory.
# The only writes are the required evidence file and its sanitized report.

set -u
umask 077
export LC_ALL=C

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
COLLECTION_DATE="$(date +%F)"
COLLECTION_TIMESTAMP="$(date +%Y%m%dT%H%M%S%z)"
ARTIFACT_DIR="/sevenH/deep-sea-realtime/artifacts/phase1/${COLLECTION_DATE}"
RAW_OUTPUT="${ARTIFACT_DIR}/environment.txt"
REPORT_DIR="${PROJECT_ROOT}/docs/acceptance"
REPORT_OUTPUT="${REPORT_DIR}/environment-baseline.md"

mkdir -p -- "${ARTIFACT_DIR}" || {
    printf '%s\n' "错误：无法创建证据目录 ${ARTIFACT_DIR}；未使用 sudo，也未修改权限。" >&2
    exit 1
}
mkdir -p -- "${REPORT_DIR}" || {
    printf '%s\n' "错误：无法创建脱敏报告目录 ${REPORT_DIR}；未使用 sudo，也未修改权限。" >&2
    exit 1
}

RAW_TEMP="$(mktemp "${ARTIFACT_DIR}/.environment.XXXXXX")" || exit 1
REPORT_TEMP="$(mktemp "${REPORT_DIR}/.environment-baseline.XXXXXX")" || exit 1
SANITIZED_TEMP="$(mktemp "${REPORT_DIR}/.environment-sanitized.XXXXXX")" || exit 1

cleanup() {
    rm -f -- "${RAW_TEMP}" "${REPORT_TEMP}" "${SANITIZED_TEMP}"
}
trap cleanup EXIT HUP INT TERM

run_command() {
    section="$1"
    missing_message="$2"
    failure_message="$3"
    shift 3

    printf '\n===== %s =====\n' "${section}"
    executable="$1"
    if ! command -v "${executable}" >/dev/null 2>&1; then
        printf '%s\n' "${missing_message}"
        printf '%s\n' "[exit_code] 127"
        return 0
    fi

    "$@" 2>&1
    command_status=$?
    printf '%s\n' "[exit_code] ${command_status}"
    if [ "${command_status}" -ne 0 ]; then
        printf '%s\n' "${failure_message}"
    fi
    return 0
}

{
    printf '%s\n' "# Lightweight phase-one P1-01 server environment evidence"
    printf '%s\n' "# This file intentionally excludes .env files, SSH keys, tokens, and model directory listings."

    run_command "date -Is" \
        "date 未安装" \
        "date -Is 执行失败" \
        date -Is
    run_command "hostname" \
        "hostname 未安装" \
        "hostname 执行失败" \
        hostname
    run_command "uname -a" \
        "uname 未安装" \
        "uname -a 执行失败" \
        uname -a
    run_command "cat /etc/os-release" \
        "cat 未安装" \
        "/etc/os-release 不存在或不可读" \
        cat /etc/os-release
    run_command "id" \
        "id 未安装" \
        "id 执行失败" \
        id
    run_command "python3 --version" \
        "python3 未安装" \
        "python3 --version 执行失败" \
        python3 --version
    run_command "nvidia-smi" \
        "nvidia-smi 未安装" \
        "nvidia-smi 执行失败；未尝试安装或修复驱动" \
        nvidia-smi
    run_command "nvcc --version" \
        "nvcc 未安装" \
        "nvcc --version 执行失败；未尝试安装或修改 CUDA" \
        nvcc --version
    run_command "free -h" \
        "free 未安装" \
        "free -h 执行失败" \
        free -h
    run_command "df -hT" \
        "df 未安装" \
        "df -hT 执行失败" \
        df -hT
    run_command "lsblk" \
        "lsblk 未安装" \
        "lsblk 执行失败" \
        lsblk
    run_command "docker version" \
        "docker 未安装" \
        "docker version 执行失败；可能未安装、无权限或服务未运行" \
        docker version
    run_command "docker compose version" \
        "docker 未安装，Docker Compose 不可用" \
        "Docker Compose 未安装或不可用" \
        docker compose version
    run_command "nvidia-ctk --version" \
        "nvidia-ctk 未安装" \
        "nvidia-ctk --version 执行失败" \
        nvidia-ctk --version
    run_command "docker info --format '{{.DockerRootDir}}'" \
        "docker 未安装" \
        "docker info 执行失败；可能无权限或服务未运行" \
        docker info --format '{{.DockerRootDir}}'
    run_command "dpkg-query -W nvidia-container-toolkit" \
        "dpkg-query 未安装，无法查询 nvidia-container-toolkit" \
        "nvidia-container-toolkit 未安装或 dpkg-query 查询失败" \
        dpkg-query -W nvidia-container-toolkit
    run_command "ffmpeg -version" \
        "ffmpeg 未安装" \
        "ffmpeg -version 执行失败" \
        ffmpeg -version
    run_command "ss -lnt" \
        "ss 未安装" \
        "ss -lnt 执行失败" \
        ss -lnt
    run_command "pro status" \
        "Ubuntu Pro 客户端不可用；未自动安装或启用" \
        "Ubuntu Pro 状态查询失败；未自动安装或启用" \
        pro status
    run_command "apt-mark showhold" \
        "apt-mark 未安装" \
        "apt-mark showhold 执行失败" \
        apt-mark showhold
} >"${RAW_TEMP}"

if grep -Eiq \
    '(authorization[[:space:]]*:|bearer[[:space:]]+[[:alnum:]_.-]+|token[[:space:]]*[=:]|password[[:space:]]*[=:]|secret[[:space:]]*[=:]|BEGIN[[:space:]].*PRIVATE[[:space:]]KEY)' \
    "${RAW_TEMP}"; then
    printf '%s\n' "错误：采集结果疑似包含密钥或凭据，已拒绝生成证据文件。" >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    python3 - "${RAW_TEMP}" <<'PY'
import ipaddress
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
candidates = set(re.findall(r"(?<![\w])(?:[0-9A-Fa-f]*:){2,}[0-9A-Fa-f:.%]*(?![\w])", text))
candidates.update(re.findall(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", text))

for candidate in candidates:
    value = candidate.strip("[](),")
    value = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        continue
    if address.is_global:
        raise SystemExit(3)
PY
    address_scan_status=$?
    if [ "${address_scan_status}" -eq 3 ]; then
        printf '%s\n' "错误：采集结果疑似包含公网地址，已拒绝生成证据文件；请人工脱敏后重试。" >&2
        exit 1
    fi
    if [ "${address_scan_status}" -ne 0 ]; then
        printf '%s\n' "错误：公网地址检查失败，已按安全原则停止。" >&2
        exit 1
    fi
else
    printf '%s\n' "警告：python3 未安装，无法完成公网 IPv6 检查；请先由管理员人工审查终端输出。" >&2
fi

if command -v python3 >/dev/null 2>&1; then
    python3 - "${RAW_TEMP}" >"${SANITIZED_TEMP}" <<'PY'
import ipaddress
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)

def section_value(name):
    pattern = rf"===== {re.escape(name)} =====\n(.*?)\n\[exit_code\]"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else None

hostname = section_value("hostname")
if hostname and "\n" not in hostname:
    text = text.replace(hostname, "<redacted-hostname>")

text = re.sub(
    r"(===== id =====\n).*?(\n\[exit_code\])",
    r"\1<redacted-user-and-groups>\2",
    text,
    flags=re.DOTALL,
)

candidate_pattern = re.compile(
    r"(?<![\w])(?:[0-9A-Fa-f]*:){2,}[0-9A-Fa-f:.%]*(?![\w])"
    r"|(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"
)

def redact_address(match):
    candidate = match.group(0)
    value = candidate.strip("[](),").split("%", 1)[0]
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return candidate
    if address.is_loopback or address.is_unspecified:
        return candidate
    return "<redacted-ip>"

print(candidate_pattern.sub(redact_address, text), end="")
PY
    sanitize_status=$?
    if [ "${sanitize_status}" -ne 0 ]; then
        printf '%s\n' "错误：脱敏报告生成失败，未发布任何输出。" >&2
        exit 1
    fi
else
    sed -E \
        -e 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<redacted-ip>/g' \
        -e '/^===== hostname =====$/,/^\[exit_code\]/{/^[^=\[].*$/c\<redacted-hostname>}' \
        -e '/^===== id =====$/,/^\[exit_code\]/{/^[^=\[].*$/c\<redacted-user-and-groups>}' \
        "${RAW_TEMP}" >"${SANITIZED_TEMP}"
fi

{
    cat <<EOF
# 轻量级阶段一 P1-01 服务器环境基线

- 状态：等待项目负责人完成人工确认
- 采集日期：${COLLECTION_DATE}
- 原始证据：\`${RAW_OUTPUT}\`（受控文件，不提交 Git）
- 有效方案：\`new_docs/阶段一详细实施方案-轻量级.md\` V1.0
- 历史对照：\`docs/server-model-deployment-record.md\`
- 采集方式：\`scripts/server/collect_environment.sh\`
- 安全说明：未使用 \`sudo\`，未读取密钥、\`.env\` 或模型目录；以下命令输出已自动脱敏。

## 人工确认清单

- [ ] Ubuntu 确实为 20.04.6 LTS。
- [ ] GPU 确实为 RTX 3090，显存容量和当前占用正常。
- [ ] NVIDIA 驱动仍为预期版本。
- [ ] Docker、Compose 和 NVIDIA Container Toolkit 的安装状态已确认。
- [ ] Docker data-root 位置和系统盘风险已确认。
- [ ] 系统盘、\`/projects\` 和 \`/sevenH\` 剩余空间已确认。
- [ ] 当前没有不可中断的 GPU 进程。
- [ ] 保留端口 \`19000\` 和计划端口 \`19100\` 的占用已确认。
- [ ] Ubuntu Pro/ESM 状态及客户采用的安全维护方式已有书面结论。
- [ ] 拉取镜像、构建容器和重启 Docker 的维护窗口已明确；本任务未执行这些操作。

## 与历史记录的差异

待项目负责人将本次结果与 \`docs/server-model-deployment-record.md\` 逐项比较后填写。

- GPU：
- 显存：
- 驱动：
- Ubuntu：
- Python/FFmpeg：
- CUDA：
- Docker/Compose/NVIDIA Container Toolkit：
- Docker data-root：
- 磁盘：
- GPU 任务：
- 端口 \`19000\`/\`19100\`：
- Ubuntu Pro/ESM：
- 维护窗口：

## 项目负责人确认

- 确认人：EMC1901
- 结论：待确认
- 确认日期：待确认
- 是否允许进入 P1-02：否
- 备注：

## 脱敏后的命令输出

EOF
    sed 's/^/    /' "${SANITIZED_TEMP}"
} >"${REPORT_TEMP}"

if [ -e "${RAW_OUTPUT}" ]; then
    PREVIOUS_OUTPUT="${ARTIFACT_DIR}/environment.before-${COLLECTION_TIMESTAMP}.txt"
    cp -p -- "${RAW_OUTPUT}" "${PREVIOUS_OUTPUT}" || {
        printf '%s\n' "错误：无法保留已有环境证据，已拒绝覆盖 ${RAW_OUTPUT}。" >&2
        exit 1
    }
fi
mv -- "${RAW_TEMP}" "${RAW_OUTPUT}" || exit 1
mv -- "${REPORT_TEMP}" "${REPORT_OUTPUT}" || exit 1

printf '%s\n' "P1-01 环境采集完成。"
printf '%s\n' "原始证据：${RAW_OUTPUT}"
printf '%s\n' "脱敏报告：${REPORT_OUTPUT}"
printf '%s\n' "下一步：由项目负责人完成报告中的人工确认清单；本脚本未修改服务器系统软件。"
