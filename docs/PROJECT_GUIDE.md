# Deep Sea Explorer 项目指南

## 项目定位

Deep Sea Explorer 是一个服务器辅助的深海观测系统。浏览器采集视频画面，项目 API
负责编排关键画面检测、图像检索增强、视觉文本推理、样本统计与报告生成。

## 当前组成

| 部分 | 位置 | 作用 |
| --- | --- | --- |
| 前端 | `frontend/` | 视频接入、动态监测、问答、样本统计和报告下载 |
| 正式后端 | `src/deep_sea_explorer/` | API、服务编排、模型网关、存储和报告 |
| 开发脚本 | `scripts/` | 启动、测试、SSH 隧道、服务器冒烟和媒体转换 |
| 自动化测试 | `tests/` | 单元、契约、集成和显式远程测试 |
| 演示素材 | `assets/demo/` | 本机 MP4 与 Y4M 虚拟摄像头素材 |
| 历史实现 | `legacy/backend/` | 仅供迁移对照，不参与运行 |
| 文档 | `docs/` | 架构、部署、接口、操作手册及历史阶段记录 |

## 运行模式

- `fake`：普通自动化测试使用，不访问网络或真实模型；
- `remote`：保留的兼容模式，默认关闭实际远程调用；
- `local`：本项目服务器加载本地 GPU 模型并完成推理。

运行配置来自 `.env`；该文件可能含 Token 和语音凭据，不得提交 Git。可提交的安全
模板为 `.env.example`、`.env.development.example` 和 `.env.server.example`。

## 启动

完整启动和停止步骤见
[深海海底智能探测系统使用手册](深海海底智能探测系统的使用手册.md)。

开发机常用命令：

```powershell
.\scripts\dev.ps1 remote-model-check
.\scripts\dev.ps1 api
.\scripts\dev.ps1 speech
.\scripts\dev.ps1 web
.\scripts\dev.ps1 video-test
```

## 测试

普通测试：

```powershell
.\scripts\dev.ps1 test
```

代码检查：

```powershell
.\scripts\dev.ps1 lint
```

远程模型测试必须显式执行，不属于普通测试：

```powershell
.\scripts\dev.ps1 remote-model-test
```

## 关键文档

- [开发与服务器部署参考](开发与服务器部署参考.md)
- [关键画面检测与 LoRA 接入参考](关键画面检测重构与LoRA模型接入开发参考.md)
- [图像检索增强架构决策](图像检索增强架构决策.md)
- [系统使用手册](深海海底智能探测系统的使用手册.md)
