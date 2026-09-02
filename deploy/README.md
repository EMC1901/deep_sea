# Linux 单机部署

此目录将浏览器、主 API 和语音服务部署在同一台 Linux GPU 服务器上。浏览器只需访问服务器 URL；Nginx 或现有 Apache 提供静态页面，并把同源的 `/api/*` 和 `/speech/*` 分别代理到仅监听 `127.0.0.1` 的 Gunicorn 服务。

## 架构与约束

```text
Browser -> http(s)://server/ -> Nginx or Apache -> /api    -> Gunicorn main API -> local GPU models
                                                 -> /speech -> Gunicorn speech API -> Baidu speech API
```

- API 必须保持 `workers = 1`。当前 session、备忘录队列和监测线程都位于进程内存，多 worker 会导致会话不一致，并可能重复加载 GPU 模型。
- 模型服务端口 `19000` 不参与这套单机方案：主 API 使用 `MODEL_BACKEND=local` 直接加载本地模型。不要同时启动旧的 `model_service_main`，以免重复占用显存。
- Web 服务器是唯一对外监听的应用入口；端口 `9001`、`9009` 只绑定回环地址。
- 语音识别/合成依赖百度凭据；缺少凭据时，视频监测、样本捕获和文本问答仍能运行，但语音功能不可用。

### 私有 SSH 隧道模式

若只需在开发机浏览器访问服务器，选择 `apache-tunnel`：Apache 仅监听服务器 `127.0.0.1:19100`，Windows 使用 SSH 映射为本地 `127.0.0.1:19100`。此模式不占用服务器 80/443、不要求公司 DNS/证书、不对公司网络开放端口；浏览器的实际来源是 localhost，可使用摄像头。

## 部署前必须确认

需要服务器管理员在维护窗口内确认：Ubuntu 版本、GPU/驱动/CUDA 与现有 Torch 的兼容性、磁盘容量、80/443 端口与公司防火墙策略、部署域名或内网地址、模型目录、服务账户及是否允许配置当前 Web 服务器。不要把真实 Token、密码、私钥或 `.env` 内容发到聊天中。

在服务器上以具有读取项目权限的账户执行以下只读检查，并只回传不含 IP、用户名和凭据的结果：

```bash
cd /opt/deep-sea-explorer
bash scripts/server/collect_environment.sh
python3 scripts/server/check_gpu.py
```

若项目尚未位于 `/opt/deep-sea-explorer`，先由管理员通过 Git 或受控文件传输放置源码；不要复制本机 `.env`。

## 一次性准备

以下步骤需要 `sudo` 和经确认的维护窗口。示例将服务账户命名为 `deepsea`；如账户已存在，不要重复创建。仅在服务器未安装任何 Web 服务器时安装 Nginx；若 80 端口已由 Apache 占用，请使用后文的 Apache 模式，不能同时让两者监听同一端口。

```bash
sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin deepsea
sudo apt-get update
sudo apt-get install -y python3.11-venv ffmpeg libgl1 libglib2.0-0
sudo install -d -m 0750 -o deepsea -g deepsea /etc/deep-sea-explorer /var/lib/deep-sea-explorer
```

使用与服务器 GPU、CUDA 和已验证模型兼容的 Python 环境。在项目根目录安装应用及本地模型依赖；Torch 的安装方式和版本必须以服务器管理员的 CUDA 验证结果为准，不能照搬下列命令替换为不兼容的 GPU 包：

```bash
cd /opt/deep-sea-explorer
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[local-model]'
```

如果已经有通过 GPU 验证的虚拟环境，可在 `app.env` 设置 `DEEP_SEA_VENV=/实际/虚拟环境`，并确保该环境安装了本项目（`pip install -e /opt/deep-sea-explorer`）和 Gunicorn。

## 配置与启动

创建仅服务器可读的配置，不要在终端打印它：

```bash
cd /opt/deep-sea-explorer
sudoedit /etc/deep-sea-explorer/app.env
sudo chown root:deepsea /etc/deep-sea-explorer/app.env
sudo chmod 0640 /etc/deep-sea-explorer/app.env
```

以 [`.env.server.example`](../.env.server.example) 为模板填写模型目录与可选的百度语音凭据。必须保持：

```dotenv
MODEL_BACKEND=local
MODEL_SERVICE_ENABLED=false
API_HOST=127.0.0.1
SPEECH_HOST=127.0.0.1
TEMP_DIR=/var/lib/deep-sea-explorer
```

执行安装器。它会先进行无修改预检；`--apply` 会备份同名站点配置、安装 systemd/Web 服务器配置并启动服务。

若服务器使用 Nginx：

```bash
cd /opt/deep-sea-explorer
sudo bash deploy/install-production.sh --web-server nginx
sudo bash deploy/install-production.sh --apply --web-server nginx
```

使用既有虚拟环境时，以该环境进行预检和安装（并在 `app.env` 配置相同的 `DEEP_SEA_VENV`）：

```bash
sudo bash deploy/install-production.sh --web-server nginx --venv /实际/虚拟环境
sudo bash deploy/install-production.sh --apply --web-server nginx --venv /实际/虚拟环境
```

若 80 端口由 Apache 占用，使用已批准的公司 DNS 名创建一个独立 Apache 虚拟主机；此方式不会替换 Apache 默认站点：

```bash
sudo bash deploy/install-production.sh --web-server apache --server-name deepsea.example.company
sudo bash deploy/install-production.sh --apply --web-server apache --server-name deepsea.example.company
```

仅供开发机浏览器通过 SSH 访问时，使用隔离的 loopback Apache 入口。下面的示例复用项目中的 Conda 环境，并让服务以该项目拥有者身份运行；需先确认该环境中安装了本项目与 Gunicorn：

```bash
sudo bash deploy/install-production.sh --web-server apache-tunnel \
  --app-root /projects/deepsea_vlm --venv /projects/deepsea_vlm/.conda --service-user linglong
sudo bash deploy/install-production.sh --apply --web-server apache-tunnel \
  --app-root /projects/deepsea_vlm --venv /projects/deepsea_vlm/.conda --service-user linglong
```

安装成功后，在 Windows PowerShell 保持隧道运行：

```powershell
.\scripts\start-private-server-tunnel.ps1 `
  -ServerHost 172.17.10.20 `
  -SshUser linglong `
  -IdentityFile "$env:USERPROFILE\.ssh\deepsea_deploy"
```

然后在本机浏览器访问 `http://127.0.0.1:19100/`。

若公司已有反向代理或 TLS 网关，应由网络管理员将其上游指向本机 Web 服务器的 80 端口，并在网关处终止 HTTPS。若由本机直接提供 HTTPS，需要另行配置公司证书；不要用自签名证书替代受信任的公司证书。

## 验收

在服务器上运行：

```bash
sudo systemctl status deep-sea-explorer-api deep-sea-explorer-speech --no-pager
curl --fail --silent http://127.0.0.1/api/health
curl --fail --silent http://127.0.0.1/js/runtime-config.js
sudo journalctl -u deep-sea-explorer-api -n 100 --no-pager
```

从公司网络中的另一台机器访问 `http://<服务器域名或内网地址>/`。允许浏览器摄像头权限后，点击“开始监测”，确认实时备忘录出现；再提问一个关于当前视频画面的文本问题，确认流式回答。最后生成一次报告，确认 PDF 可下载。若使用 HTTP 而非 HTTPS，现代浏览器可能拒绝非 localhost 页面使用摄像头；正式使用必须经公司 HTTPS 域名访问。

## 常见故障

- API 启动后立即退出：执行 `sudo journalctl -u deep-sea-explorer-api -e --no-pager`，通常是模型路径、Python 环境或 GPU 依赖不匹配。
- 页面可打开但模型不可用：确认 `MODEL_BACKEND=local`，且未把模型目录、`TEMP_DIR` 或 `DEEP_SEA_VENV` 填成服务账户不可读/不可写的路径。
- 摄像头无法打开：使用 HTTPS 公司域名；HTTP 仅适用于 localhost 或受浏览器策略允许的受控环境。
- 语音返回错误：视频主链路不受影响；由凭据持有者在服务器的 `app.env` 中核对百度配置。
