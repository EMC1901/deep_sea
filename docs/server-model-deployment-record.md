# 服务器模型部署记录

> 项目：deep_sea_explorer  
> 建立日期：2026-07-23  
> 当前阶段：服务器环境盘点完成，真实模型服务尚未部署

## 1. 服务器基本信息

| 项目 | 实际值 | 状态 |
|---|---|---|
| SSH 用户 | linglong | 已确认 |
| 主机名 | linglong-Z390-UD | 已确认 |
| 操作系统 | Ubuntu 20.04.6 LTS | 已确认 |
| Linux 内核 | 5.15.0-139-generic | 已确认 |
| 系统架构 | x86_64 | 已确认 |
| Python | 3.11.3 | 已确认 |
| Conda | 23.5.0 | 已确认 |

## 2. GPU 与 CUDA

| 项目 | 实际值 | 状态 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3090 | 已确认 |
| 显存 | 24576 MiB | 已确认 |
| NVIDIA 驱动 | 555.42.06 | 已确认 |
| 驱动支持 CUDA | 12.5 | 已确认 |
| CUDA Toolkit | 12.4 | 已确认 |
| nvcc | 12.4.131 | 已确认 |

## 3. 内存与磁盘

| 项目 | 实际值 | 状态 |
|---|---|---|
| 内存 | 62 GiB | 已确认 |
| Swap | 2 GiB | 已确认 |
| 系统盘剩余空间 | 12 GiB | 不用于模型 |
| /projects 剩余空间 | 288 GiB | 可用于代码和环境 |
| /sevenH 剩余空间 | 556 GiB | 可用于模型、缓存和临时文件 |

## 4. 网络与软件

| 项目 | 状态 | 说明 |
|---|---|---|
| PyPI | 可访问 | 可安装 Python 包 |
| ModelScope | 可访问 | 可作为模型下载来源 |
| Hugging Face | 无法访问 | 不允许启动时自动下载 |
| FFmpeg | 未安装 | 视频处理前需要安装 |
| nvcc | 已安装 | CUDA Toolkit 可用 |

## 5. 已有模型

| 用途 | 模型 | 路径 | 状态 |
|---|---|---|---|
| 视觉语言模型 | Qwen3-VL-2B-Instruct | /sevenH/deepsea_vlm/models/Qwen3-VL-2B-Instruct | 已确认 |
| 文本大模型 | Qwen3.6-27B-Q4_K_M.gguf | /sevenH/models/qwen3.6-27b-gguf | 已有，暂不纳入视觉服务 |
| 向量模型 | Qwen3-Embedding-0.6B-Q8_0.gguf | /sevenH/models/qwen3-embedding-0.6b-gguf | 已有，来自原 RAG 项目 |
| 文生图模型 | 待确认 | 待确认 | 未准备 |
| GTE | 待确认 | 待确认 | 未准备 |
| MiniLM | 待确认 | 待确认 | 未准备 |

## 6. 已有视觉模型环境

| 项目 | 实际值 |
|---|---|
| 环境路径 | /projects/deepsea_vlm/.conda |
| Python | 3.10.0 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.1 |
| GPU 可用 | 是 |
| 环境大小 | 5.6 GiB |

该环境用于现有 deepsea_vlm 项目，暂不修改。deep_sea_explorer 后续单独建立 Python 3.11 环境。

## 7. 暂未确认事项

- 服务器 SSH 地址和端口的正式记录。
- 文生图模型的准确名称、版本和许可证。
- GTE 模型的准确名称、版本和许可证。
- MiniLM 模型的准确名称、版本和许可证。
- 模型服务最大并发数。
- 最大视频大小和最长视频时长。
- 最终模型服务端口。
- 正式服务账号和目录规划。
