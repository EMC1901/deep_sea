# 关键画面检测重构与 LoRA 模型接入开发参考

> 更新时间：2026-08-10
> 用途：记录本阶段从“定时描述视频”迁移到“关键画面事件驱动”以及接入微调 Qwen3-VL LoRA 的实现、部署和验收依据。后续开发应以本文件、`关键画面检测方案.md` 和实际代码共同为准。

## 1. 本阶段结果

- 监测主路径改为接收独立 JPEG 帧，不为帧监测组装短 MP4，也不以中文描述的文本向量相似度去重。
- 通过“已知目标检测/跟踪”和“场景变化检测”生成候选事件；只有视觉文本模型确认后才保存正式截图并发布讲解。
- 每个会话限制为一个在途模型任务和一个可替换的等待候选，避免模型推理期间无限积压。
- 服务器当前使用基础模型 `Qwen3-VL-2B-Instruct` 加载 `qwen3_vl_2b_lora_second_round_10000_v1/best_adapter`。
- 使用真实 720p Y4M 素材完成端到端验收；服务器定向回归测试为 `37 passed`。

## 2. 实际处理路径

```text
浏览器 getUserMedia
  -> Canvas 定时编码最新 JPEG
  -> POST /videoanalyze（X-Session-ID）
  -> VideoIngestionService 验证并转交单帧
  -> MonitoringService
       -> YOLO（可选）检测已知对象
       -> 轻量 IoU 轨迹状态维持
       -> OpenCV 场景变化检测
       -> 生成并持久化候选 JPEG
  -> 每会话异步候选队列
  -> Qwen3-VL：参考 JPEG + 候选 JPEG + 元数据
  -> 校验/归一化结构化结果
  -> 保存正式截图和 SQLite 事件记录
  -> GET /memos 返回中文画面讲解和样本
```

前端为每个页面生成独立会话 ID；监测开始后按前端运行配置采样、上传最新 JPEG，并轮询同一会话的 `/memos`。调用帧监测时不要附带问答 `prompt`，否则会落入保留的旧视频兼容分支。

## 3. 关键画面检测实现

### 3.1 候选事件条件

`MonitoringService.process_frame()` 对每帧执行以下伪代码：

```text
保存并验证 JPEG
tracks = detector.detect(frame) 后更新轨迹
new_tracks = 连续出现达到确认阈值且尚未确认过的轨迹
metrics = 当前帧与本会话场景参考帧比较

if 第一次收到帧：保存为场景参考帧，不触发事件
if new_tracks 非空：加入 yolo 触发信号
if metrics.changed：加入 scene 触发信号
if 没有触发信号：返回 monitoring
否则：立即保存候选 JPEG，提交异步模型队列，返回 candidate_pending
```

候选事件包含会话、当前/参考图路径、YOLO 变化、场景指标、触发来源和视觉签名。触发来源可以是 `yolo`、`scene` 或两者组合。

### 3.2 已知目标检测与跟踪

- `YoloObjectDetector` 仅在配置了可用 `YOLO_MODEL_PATH` 时加载 Ultralytics YOLO；未配置或模型失败时退化为 `NullObjectDetector`，不会阻断场景变化检测。
- `YOLO_CONFIDENCE` 默认 `0.35`。
- `ByteTrackState` 以同类目标的 IoU 进行匹配，默认 IoU 阈值 `0.3`、连续确认帧数 `3`；同一目标小幅移动保持同一 `track_id`，不会重复产生新轨迹事件。
- **重要限制**：当前 `ByteTrackState` 是具有 ByteTrack 生命周期语义的轻量 IoU 跟踪器，并非第三方原版 ByteTrack 实现。若后续需要遮挡恢复、低置信度关联或严格的 ByteTrack 指标，应替换为真实 ByteTrack 并补充深海数据集验证。

### 3.3 场景变化检测与冗余帧过滤

`SceneChangeDetector` 在默认 `160x90` 低分辨率上处理，优先使用 OpenCV CUDA；若 OpenCV 未提供所需 CUDA 算子则稳定回退 CPU。它先补偿全局运动，再计算多种互补指标：

1. 相位相关补偿平移；
2. ECC 仿射配准补偿小角度旋转、缩放和剪切；
3. 裁掉配准边缘后，计算感知哈希、HSV、边缘、4x4 网格差异、SSIM；
4. 以 Farneback 光流的残余幅度和一致性压制纯整体运动；
5. 对加权变化分数连续确认。默认 `SCENE_CHANGE_THRESHOLD=0.22`、`SCENE_CONFIRM_FRAMES=3`。

因此，镜头平移/轻微旋转、抖动和亮度小波动会优先被配准和连续确认过滤；底质、微地形、未知要素等即使没有 YOLO 框，仍可通过 `scene` 触发候选。场景检测只判断“视觉状态可能改变”，不替代模型的调查价值判断。

## 4. 异步队列、接受规则与持久化

- `PerSessionEventQueue` 使用线程池；同一会话最多一个正在评估的候选和一个等待候选。模型任务在途时，新候选覆盖等待槽中的不同视觉签名候选。
- 模型返回后，只有同时满足 `survey_value=true`、事件类型为 `new_element` 或 `major_scene_change`、且存在新要素或场景改变，并且不等于已接受事件签名，才接受事件。
- 接受后保存：
  - 正式截图：`data/captures/<session_id>/...jpg`
  - 候选截图：`data/candidates/<session_id>/...jpg`
  - 事件库：`data/events.sqlite3`
- SQLite 事件记录包含事件/会话时间、事件与触发类型、要素类别和名称、中文描述、置信度、截图路径、YOLO 轨迹 ID、视觉指纹。
- 接受的结果发布为 memo；其中生物与环境要素会分别成为样本并更新会话统计。

## 5. 微调 Qwen3-VL 接入

### 5.1 当前模型和检查点

服务器模型目录为 `/sevenH/deepsea_vlm`。已发现的 LoRA 检查点包括：

```text
qwen3_vl_2b_lora_first_round_v1
qwen3_vl_2b_lora_second_round_10000_v1      <- 当前选择
qwen3_vl_2b_lora_second_round_smoke32_v1
qwen3_vl_2b_lora_second_round_smoke32_v2
qwen3_vl_2b_lora_third_round_80000_v1
qwen3_vl_2b_lora_third_round_smoke32_v1
```

当前运行组合：

```text
QWEN_MODEL_PATH=/sevenH/deepsea_vlm/models/Qwen3-VL-2B-Instruct
QWEN_ADAPTER_PATH=/sevenH/deepsea_vlm/outputs/checkpoints/qwen3_vl_2b_lora_second_round_10000_v1/best_adapter
```

必须保留“基础模型路径 + 适配器路径”的两段配置；适配器目录不是可独立加载的完整模型。`QwenAdapter` 先离线加载基础 Qwen，再通过 PEFT 的 `PeftModel.from_pretrained()` 载入 LoRA，最后将合并后的推理对象移至 CUDA。服务器项目 `.venv` 已安装 `peft 0.20.0`。

### 5.2 事件模型输入输出契约

模型收到两张 JPEG（已确认场景参考图、当前候选图）和 JSON 元数据（YOLO 变化、场景指标、触发类型），输出以下逻辑字段：

```json
{
  "survey_value": true,
  "event_type": "new_element",
  "scene_changed": true,
  "new_elements": [{"category": "organism", "name": "名称", "is_new": true}],
  "description": "简体中文当前场景生态描述。",
  "confidence": 0.91
}
```

允许的类别是 `organism`、`seabed_substrate`、`micro_topography`、`other`；事件类型是 `new_element`、`major_scene_change`、`none`。解析器会拒绝无 JSON、无描述、无效置信度、无效类型，以及“调查价值为真但没有可验证变化”的结果。

第二轮 LoRA 已观察到两项与通用契约的差异，代码已做受限兼容：

- 漏掉冗余字段 `scene_changed` 时，以 `survey_value` 作为默认值；
- 同时给出 `survey_value=true`、有效 `new_elements` 却错误填写 `event_type=none` 时，**仅**归一化为 `new_element`。

不要扩大这两条兼容逻辑，也不要因为模型输出了 `survey_value=true` 就跳过结构化证据校验；更换 LoRA 时必须重新采样其实际输出并新增解析器测试。

## 6. 代码定位

| 责任 | 主要文件 |
| --- | --- |
| 前端摄像头采样、JPEG 上传、memo 轮询 | `frontend/index.html` |
| 帧接口和单帧入口 | `src/deep_sea_explorer/api/routes/video.py` |
| 事件编排、接受和 memo 发布 | `src/deep_sea_explorer/services/monitoring.py` |
| YOLO、轨迹、场景比较、候选签名 | `src/deep_sea_explorer/services/key_frame_detection.py` |
| 每会话单在途/单等待队列 | `src/deep_sea_explorer/services/candidate_queue.py` |
| 候选/正式截图及 SQLite | `src/deep_sea_explorer/infrastructure/storage/event_store.py` |
| 本地 Qwen + LoRA 加载、提示词和 JSON 解析 | `src/deep_sea_explorer/infrastructure/models/local/adapters.py` |
| 环境变量与本地容器组装 | `src/deep_sea_explorer/config.py`、`src/deep_sea_explorer/container.py` |

## 7. 部署与人工验收

### 7.1 运行边界

- 服务器项目：`/projects/deep-sea-explorer-codex/app`
- 仅使用 `ssh deepsea-codex`，仅以普通开发用户执行项目命令。
- API、前端和语音服务仅绑定回环地址，通过 SSH 隧道供本机访问。
- 不操作 Apache、Nginx、systemd、CUDA 驱动、系统 FFmpeg，也不读取或写入旧项目 `/projects/deep-sea-realtime`。
- API 重启前先用 `ps` 确认项目 Gunicorn 主进程；只终止该进程。以 `setsid -f` 启动，避免 SSH 会话退出时终止服务。
- 启动保持 `MODEL_BACKEND=local`、`MODEL_SERVICE_ENABLED=false`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`IMAGE_GENERATION_ENABLED=false`，并设置上述基础模型和适配器路径。

### 7.2 自动化测试

在服务器项目目录执行：

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/unit/test_local_model_runtime.py \
  tests/unit/test_monitoring_service.py \
  tests/unit/test_key_frame_detection.py \
  tests/contract/test_main_api_contract.py \
  tests/unit/test_settings_and_services.py
```

本阶段最近一次结果为 `37 passed`。覆盖内容包括：轨迹 ID 稳定性、候选替换队列、平移/旋转补偿、局部变化检测、事件样本发布、LoRA 路径传递及 LoRA JSON 兼容。

### 7.3 真实视频验收

使用 `runtime/actual-deep-sea-camera-full-720p.y4m`。端到端接口测试的最小模式为：同一参考画面上传 3 次，再上传变化画面 3 次；预期依次出现连续确认和 `candidate_pending`，随后 `/memos` 出现正式中文讲解与截图。

启动 Chrome 虚拟摄像头进行人工测试：

```powershell
.\scripts\start-video-camera-test.ps1 `
  -VideoPath .\runtime\actual-deep-sea-camera-full-720p.y4m `
  -Url http://127.0.0.1:19100/
```

脚本会创建独立 Chrome profile 并自动授予模拟摄像头权限；在页面点击“开始监测”。服务器 `/health` 应返回 `status=ok` 且模型状态为 `loaded/ready`。

本阶段实测：参考帧连续上传不产生候选；变化帧第 3 次确认后进入 `candidate_pending`；LoRA 成功给出“鱼在深海沉积物附近的大型脊椎动物骨架上方游动”的中文讲解并保存正式截图。

## 8. 日志、调参与维护注意事项

- API 访问日志记录请求方法、路径、状态、来源；候选评估失败记录会话、异常类型和简要原因；不要长期记录原始模型输出、图片、Token 或 `.env` 内容。
- 场景指标会作为 `/videoanalyze` 响应的一部分提供，包括 `confirmed_frames`、SSIM、相位/仿射补偿、光流、各距离和 CPU/CUDA 后端，可用于调节阈值。
- 调整阈值必须先用平移、旋转、光照波动、局部新要素和明显换景样本回归；不能仅根据单段视频降低阈值。
- 当前代码仍保留 `MonitoringService.process_session()` 的旧 MP4 + memo embedding 兼容路径，用于带 `prompt` 的旧功能。关键画面监测必须走无 `prompt` 的单 JPEG 路径；若后续彻底移除旧路径，应单独评估问答和历史功能影响。
- 服务器与本地工作区可能含未提交的服务器专属改动。同步、提交或重启前均先检查 `git status`，不得使用 `git reset --hard`、`git clean` 或覆盖式复制。
