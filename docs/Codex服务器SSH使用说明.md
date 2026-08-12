# Codex 服务器 SSH 使用说明

## 服务器入口

Codex 统一使用 SSH 别名：

```bash
ssh deepsea-codex
```

对应：

```text
Host: 172.17.10.20
User: linglong
Key: deepsea_deploy_v3
```

`deepsea_deploy_v3` 是 Codex 自动化专用密钥：

```text
- 无 passphrase
- 不依赖 Windows ssh-agent
- 支持 BatchMode 无交互连接
- 服务器端 authorized_keys 使用 restrict
```

不要让 Codex 使用 `deepsea_deploy_v2`。

`deepsea_deploy_v2` 保留给人工管理使用，并带 passphrase。

---

## 执行普通命令

统一使用：

```bash
ssh deepsea-codex "命令"
```

例如：

```bash
ssh deepsea-codex "whoami"
ssh deepsea-codex "pwd"
ssh deepsea-codex "git status"
```

已验证：

```text
ssh deepsea-codex "echo CODEX_ALIAS_OK; whoami"

CODEX_ALIAS_OK
linglong
```

---

## 管理员操作

Codex 不得直接尝试：

```bash
sudo docker ...
sudo systemctl ...
sudo bash
sudo -i
```

管理员操作统一通过受控入口：

```bash
sudo -n /usr/local/sbin/codex-admin <动作>
```

当前允许：

```bash
sudo -n /usr/local/sbin/codex-admin apache-test
sudo -n /usr/local/sbin/codex-admin apache-reload
sudo -n /usr/local/sbin/codex-admin apache-status
```

远程执行：

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-test"
```

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-reload"
```

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-status"
```

---

## 自动化要求

所有 Codex SSH 操作必须遵守：

```text
- 使用 ssh deepsea-codex
- 不依赖 Windows ssh-agent
- 不要求人工输入 SSH 密码
- 不要求人工输入私钥 passphrase
- 不要求人工输入 sudo 密码
- sudo 必须使用 sudo -n
- 管理员操作必须通过 /usr/local/sbin/codex-admin
- 权限不足时立即失败并报告
- 不得尝试绕过 sudo 白名单
```

---

## Apache

Apache 当前已运行。

配置检查：

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-test"
```

重载：

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-reload"
```

状态：

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin apache-status"
```

如果出现：

```text
AH00558: apache2: Could not reliably determine the server's fully qualified domain name
```

这是当前 `ServerName` 未设置导致的警告，不代表配置失败。

只要出现：

```text
Syntax OK
```

即表示 Apache 配置检查通过。

已验证：

```text
Syntax OK
apache_reload=PASS
```

---

## SSH 密钥分工

### Codex 自动化

```text
deepsea_deploy_v3
```

用途：

```text
Codex 自动 SSH
无 passphrase
不依赖 ssh-agent
服务器端使用 restrict
```

### 人工管理

```text
deepsea_deploy_v2
```

用途：

```text
人工 SSH 管理
带 passphrase
可通过 ssh-agent 使用
作为人工管理/恢复通道
```

---

## 项目隔离约束

所有部署与运维操作必须限于已明确授权的本项目目录和服务。不得读取、引用、修改或依赖任何非本项目的目录、服务、脚本、配置、数据或历史部署方案。

---

## 服务器当前 SSH 状态

当前主要登录用户：

```text
linglong
```

旧的：

```text
deepsea-rt
```

账号和用户组已经删除。

root SSH 直接登录已经关闭：

```text
PermitRootLogin no
```

Codex 应始终以：

```text
linglong
```

登录服务器。

---

## 当前链路状态

```text
Codex
→ ssh deepsea-codex
→ deepsea_deploy_v3
→ linglong
→ 普通开发命令
→ 必要时 sudo -n
→ /usr/local/sbin/codex-admin
→ 受控管理员操作
```

当前已验证：

```text
Codex SSH 无交互登录        ✅
不依赖 ssh-agent            ✅
deepsea_deploy_v3           ✅
linglong 登录               ✅
sudo -n                     ✅
Apache configtest           ✅
Apache reload               ✅
通用免密 sudo               ❌
root SSH 登录               ❌
```

## Codex 核心使用规则

普通操作：

```bash
ssh deepsea-codex "<command>"
```

管理员操作：

```bash
ssh deepsea-codex "sudo -n /usr/local/sbin/codex-admin <允许的动作>"
```

除非用户明确授权修改权限设计，否则不要使用其他 sudo 管理方式。
