# DeepSea VLM 项目详细目录树

## 1. 项目整体目录关系

项目采用“代码与大型资源分离”的结构：

```text
/projects/deepsea_vlm
│
├── 代码、脚本、文档、测试、Conda 环境
│
├── data    ───────────────→ /sevenH/deepsea_vlm/data
├── models  ───────────────→ /sevenH/deepsea_vlm/models
└── outputs ───────────────→ /sevenH/deepsea_vlm/outputs
```

其中：

- `/projects/deepsea_vlm`：项目代码仓库；

- `/sevenH/deepsea_vlm`：模型、数据、缓存和输出；

- `data`、`models`、`outputs` 是软链接，不是重复存储。

---

# 2. 代码仓库目录树

```text
/projects/deepsea_vlm
├── .conda/
├── .git/
├── .gitignore
├── README.md
├── configs/
├── data -> /sevenH/deepsea_vlm/data
├── docs/
├── models -> /sevenH/deepsea_vlm/models
├── outputs -> /sevenH/deepsea_vlm/outputs
├── scripts/
├── src/
└── tests/
```

---

## 2.1 `.conda`

项目独立 Conda 环境：

```text
/projects/deepsea_vlm/.conda
```

占用约：

```text
5.6GB
```

当前已知结构：

```text
.conda/
├── bin/
├── compiler_compat/
├── conda-meta/
├── etc/
│   └── conda/
│       └── activate.d/
│           └── deepsea_env.sh
├── include/
├── lib/
│   └── python3.10/
│       └── site-packages/
├── man/
├── share/
├── ssl/
└── x86_64-conda-linux-gnu/
```

激活脚本内容：

```bash
export PIP_CACHE_DIR=/sevenH/deepsea_vlm/cache/pip
export TORCH_HOME=/sevenH/deepsea_vlm/cache/torch
export HF_HOME=/sevenH/deepsea_vlm/cache/huggingface
export PYTHONNOUSERSITE=1
```

部分已安装包位于：

```text
.conda/lib/python3.10/site-packages/
├── fathomnet/
├── fathomnet-1.10.0.dist-info/
├── qwen_vl_utils/
├── torch/
├── torchvision/
├── transformers/
├── accelerate/
├── safetensors/
├── PIL/
└── coco_lib/
    └── imagecaptioning.py
```

可执行文件中包括：

```text
.conda/bin/
├── python
├── pip
└── fathomnet-generate
```

该环境当前主要版本：

```text
Python         3.10.0
PyTorch        2.6.0+cu124
torchvision    0.21.0+cu124
Transformers   4.57.1
Accelerate     1.12.0
Safetensors    0.7.0
Pillow         12.0.0
```

---

## 2.2 `.git`

Git 仓库内部目录：

```text
/projects/deepsea_vlm/.git
```

未详细展开其内部对象文件。

当前仓库信息：

```text
分支：main
远程：git@github.com:EMC1901/deepsea-vlm.git
```

最近提交：

```text
d02a170  feat: complete base model environment validation
f8196c8  chore: ignore project conda environment
e0dc6f2  chore: initialize deepsea vlm project
```

---

## 2.3 `.gitignore`

文件：

```text
/projects/deepsea_vlm/.gitignore
```

内容：

```text
# Python
__pycache__/
*.py[cod]
*.so
.pytest_cache/
.mypy_cache/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# System files
.DS_Store
Thumbs.db

# Secrets
.env
*.key
secrets.json

# Large project resources
models
data
outputs
cache

# Logs and temporary files
*.log
tmp/
temp/

# Project Conda environment
.conda/
```

---

## 2.4 `README.md`

文件：

```text
/projects/deepsea_vlm/README.md
```

当前内容概述：

```text
项目名称：DeepSea VLM

项目定位：
基于 Qwen3-VL-2B-Instruct 的深海图像与视频离线批量分析技术验证项目。

第一阶段目标：
- 深海生物大类识别
- 生境与地貌识别
- 低质量画面判断
- 异常与未知目标筛选
- 目标框定位
- 图片与视频批量处理
- JSON 和 CSV 结构化输出
```

当前 README 仍写着“阶段 1：开发环境与基座模型准备”，已经落后于实际进度。

---

## 2.5 `configs`

当前目录：

```text
/projects/deepsea_vlm/configs/
```

当前为空：

```text
configs/
```

尚未建立：

```text
model.yaml
data.yaml
train_lora.yaml
inference.yaml
evaluation.yaml
```

目前模型路径、数据路径和输出路径主要硬编码在脚本中。

---

## 2.6 `docs`

目录：

```text
/projects/deepsea_vlm/docs
```

文件树：

```text
docs/
├── core_versions.txt
├── environment-conda-explicit.txt
├── environment-pip-freeze.txt
└── 阶段1_环境与基座模型验证记录.md
```

### `core_versions.txt`

```text
docs/core_versions.txt
```

保存项目核心软件版本，文件约 152B。

### `environment-conda-explicit.txt`

```text
docs/environment-conda-explicit.txt
```

保存 Conda 环境显式依赖列表，约 2.5KB。

### `environment-pip-freeze.txt`

```text
docs/environment-pip-freeze.txt
```

保存 pip 包版本快照，约 1.3KB。

### `阶段1_环境与基座模型验证记录.md`

```text
docs/阶段1_环境与基座模型验证记录.md
```

记录内容包括：

```text
- Ubuntu 20.04.6 LTS
- RTX 3090
- CUDA Toolkit 12.4
- Python 3.10.0
- PyTorch 2.6.0+cu124
- Transformers 4.57.1
- Qwen3-VL-2B-Instruct
- 模型参数量约 2.13B
- BF16 加载
- GPU 加载显存约 3.96GiB
- 图片推理成功
- 推理峰值显存约 4.07GiB
```

---

## 2.7 `scripts`

目录：

```text
/projects/deepsea_vlm/scripts
```

当前结构：

```text
scripts/
├── 01_export_fathomnet_concepts.py
├── 02_export_fathomnet_concept_counts.py
├── 03_classify_fathomnet_top_concepts.py
├── 04_summarize_fathomnet_taxa.py
├── 05_build_fathomnet_candidate_pool.py
├── 06_download_fathomnet_concept_samples.py
├── 07_build_caecosagitta_eval_dataset.py
├── 08_run_caecosagitta_baseline.py
├── 09_score_caecosagitta_baseline.py
├── 11_validate_benthicnet_images.py
├── run_image_smoke_test.py
└── __pycache__/
```

没有编号为 `10` 的脚本。

### `01_export_fathomnet_concepts.py`

作用：

```text
查询 FathomNet 全部概念名称
→ 保存为 CSV
```

输出：

```text
/sevenH/deepsea_vlm/data/raw/fathomnet/
└── fathomnet_concepts_raw.csv
```

### `02_export_fathomnet_concept_counts.py`

作用：

```text
查询每个 FathomNet 概念的目标框数量
→ 按数量从高到低排序
→ 保存为 CSV
```

输出：

```text
/sevenH/deepsea_vlm/data/raw/fathomnet/
└── fathomnet_concept_counts.csv
```

### `03_classify_fathomnet_top_concepts.py`

作用：

```text
读取高频 FathomNet 概念
→ 调用 WoRMS
→ 判断是否为可信动物分类
→ 输出分类筛选结果
```

输入：

```text
fathomnet_concept_counts.csv
```

输出：

```text
/sevenH/deepsea_vlm/data/interim/fathomnet/
└── fathomnet_top300_taxonomy_screen.csv
```

### `04_summarize_fathomnet_taxa.py`

作用：

```text
统计可信动物概念
→ 按分类层级汇总
→ 按动物门类汇总
```

输入：

```text
fathomnet_top300_taxonomy_screen.csv
```

主要用于终端统计输出。

### `05_build_fathomnet_candidate_pool.py`

作用：

```text
按动物门类配额
→ 按分类层级和目标框数量选择
→ 构建平衡候选池
```

输出：

```text
/sevenH/deepsea_vlm/data/interim/fathomnet/
└── fathomnet_balanced_candidate_pool.csv
```

### `06_download_fathomnet_concept_samples.py`

目标概念：

```text
Caecosagitta macrocephala
```

作用：

```text
查询 FathomNet
→ 选择 12 个样本
→ 下载完整图片
→ 裁剪目标区域
→ 生成 manifest
→ 生成 contact sheet
```

输出目录：

```text
/sevenH/deepsea_vlm/data/interim/fathomnet/
└── concept_samples/
    └── caecosagitta_macrocephala/
        ├── full/
        ├── crops/
        ├── manifest.csv
        └── contact_sheet.jpg
```

### `07_build_caecosagitta_eval_dataset.py`

作用：

```text
读取下载样本
→ 查询 WoRMS 分类链
→ 建立 12 条 JSONL 评测数据
```

输出：

```text
/sevenH/deepsea_vlm/data/processed/fathomnet/benchmarks/
└── deepsea_id_caecosagitta_12.jsonl
```

### `08_run_caecosagitta_baseline.py`

作用：

```text
加载 Qwen3-VL-2B-Instruct
→ 读取 12 条评测数据
→ 进行图片分类推理
→ 提取 JSON
→ 保存预测结果
```

输出：

```text
/sevenH/deepsea_vlm/outputs/baselines/
└── qwen3_vl_caecosagitta_12_predictions.jsonl
```

### `09_score_caecosagitta_baseline.py`

作用：

```text
读取模型预测
→ 按 kingdom 到 species 分层评分
→ 保存逐条评分
→ 生成汇总
```

输出：

```text
/sevenH/deepsea_vlm/outputs/baselines/
├── qwen3_vl_caecosagitta_12_scored.jsonl
└── qwen3_vl_caecosagitta_12_summary.json
```

### `11_validate_benthicnet_images.py`

作用：

```text
扫描全部 BenthicNet JPG
→ 使用 Pillow 验证图片
→ 统计格式、模式、尺寸
→ 记录异常项
```

输入：

```text
/sevenH/deepsea_vlm/data/raw/benthicnet/
└── extracted/compiled_labelled_512pix/
```

输出：

```text
/sevenH/deepsea_vlm/data/interim/benthicnet/image_validation/
├── invalid_images.csv
└── validation_summary.txt
```

### `run_image_smoke_test.py`

作用：

```text
加载本地 Qwen3-VL
→ 读取 smoke_test.png
→ 描述文字、图形、颜色和位置
→ 保存 JSON 结果
```

输入：

```text
/projects/deepsea_vlm/tests/assets/smoke_test.png
```

输出：

```text
/sevenH/deepsea_vlm/outputs/baselines/
└── qwen3_vl_image_smoke_test.json
```

### `scripts/__pycache__`

语法检查后生成的 Python 字节码缓存：

```text
scripts/__pycache__/
├── 01_export_fathomnet_concepts.cpython-310.pyc
├── 02_export_fathomnet_concept_counts.cpython-310.pyc
├── 03_classify_fathomnet_top_concepts.cpython-310.pyc
├── 04_summarize_fathomnet_taxa.cpython-310.pyc
├── 05_build_fathomnet_candidate_pool.cpython-310.pyc
├── 06_download_fathomnet_concept_samples.cpython-310.pyc
├── 07_build_caecosagitta_eval_dataset.cpython-310.pyc
├── 08_run_caecosagitta_baseline.cpython-310.pyc
├── 09_score_caecosagitta_baseline.cpython-310.pyc
├── 11_validate_benthicnet_images.cpython-310.pyc
└── run_image_smoke_test.cpython-310.pyc
```

这些文件已由 `.gitignore` 忽略。

---

## 2.8 `src`

当前目录：

```text
/projects/deepsea_vlm/src/
```

当前为空：

```text
src/
```

尚未建立正式模块，例如：

```text
src/
├── data/
├── models/
├── training/
├── inference/
├── evaluation/
└── utils/
```

---

## 2.9 `tests`

目录结构：

```text
tests/
└── assets/
    └── smoke_test.png
```

当前只有一张基础模型测试图：

```text
/projects/deepsea_vlm/tests/assets/smoke_test.png
```

尚没有：

```text
test_*.py
pytest 配置
数据格式测试
训练样本测试
模型输出测试
```

---

# 3. 大型资源根目录

```text
/sevenH/deepsea_vlm
├── cache/
├── data/
├── models/
└── outputs/
```

当前总体占用约：

```text
59GB
```

其中：

```text
cache      约 1.3GB
data       约 54GB
models     约 4.0GB
outputs    约 428KB
```

---

# 4. 缓存目录树

```text
/sevenH/deepsea_vlm/cache
├── conda/
├── huggingface/
├── pip/
└── torch/
```

容量大致为：

```text
cache/
├── conda/         约 595MB
├── huggingface/   当前很小
├── pip/           约 736MB
└── torch/         当前很小
```

说明当前约 1.3GB 缓存主要来自：

```text
Conda 包缓存
pip 下载缓存
```

模型已经直接保存在 `models`，因此 Hugging Face 缓存当前不大。

---

# 5. 模型目录树

```text
/sevenH/deepsea_vlm/models
└── Qwen3-VL-2B-Instruct/
```

详细文件：

```text
Qwen3-VL-2B-Instruct/
├── .gitattributes
├── README.md
├── chat_template.json
├── config.json
├── configuration.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── preprocessor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── video_preprocessor_config.json
└── vocab.json
```

目录约：

```text
4.0GB
```

当前没有：

```text
adapters/
checkpoints/
merged_models/
其他模型/
```

---

# 6. 数据目录总树

```text
/sevenH/deepsea_vlm/data
├── interim/
├── processed/
├── raw/
└── splits/
```

展开后：

```text
data/
├── raw/
│   ├── benthicnet/
│   ├── customer_annotations/
│   └── fathomnet/
│
├── interim/
│   ├── benthicnet/
│   └── fathomnet/
│
├── processed/
│   └── fathomnet/
│
└── splits/
```

---

# 7. BenthicNet 原始数据目录

```text
/sevenH/deepsea_vlm/data/raw/benthicnet
├── documentation/
├── extracted/
├── full_labelled_512px.tar
├── full_labelled_512px.tar.sha256
├── full_labelled_512px_download.log
└── full_labelled_512px_filelist.txt
```

---

## 7.1 `documentation`

```text
documentation/
└── README.txt
```

README 内容包括：

```text
- BenthicNet 数据集说明
- 作者和联系方式
- 数据采集时间：1965—2022
- 全球地理范围
- 许可证说明
- 官方论文和 DOI
- 官方目录结构
- CSV 文件说明
- 预训练模型说明
- 数据处理方法
```

当前服务器没有完整下载官方 README 中提到的：

```text
all_licenses_ref.csv
licenses/
finalized_csvs.zip
```

---

## 7.2 原始压缩包

```text
full_labelled_512px.tar
```

约：

```text
27GB
```

对应校验文件：

```text
full_labelled_512px.tar.sha256
```

内容记录 SHA256：

```text
8a86b8bab263f481a37fa09d8324acfac1b59f88b28bab02717fd73df53fffcb
```

校验结果：

```text
full_labelled_512px.tar: OK
```

下载日志：

```text
full_labelled_512px_download.log
```

记录：

```text
2026-07-23：首次下载中断
2026-07-24：继续下载并完成
```

文件清单：

```text
full_labelled_512px_filelist.txt
```

约：

```text
21MB
246,834 行
```

其中既包含图片条目，也包含目录条目。

---

## 7.3 解压图片目录

```text
extracted/
└── compiled_labelled_512pix/
```

详细结构形式：

```text
compiled_labelled_512pix/
├── 20191020_Broughton_Island/
├── 20191024_Seal_Rocks/
├── 20200616_Seal_Rocks/
├── 20200812_Seal_Rocks/
├── Bastos/
├── Batemans201011/
├── Batemans201211/
├── Batemans201411/
├── Bay_of_Fundy_2019/
├── Bedford_2017/
├── CatlinSeaview_ATL_ABW/
├── CatlinSeaview_ATL_AIA/
├── CatlinSeaview_ATL_BES/
├── CatlinSeaview_ATL_BHS/
├── CatlinSeaview_ATL_BLZ/
├── CatlinSeaview_ATL_BMU/
├── CatlinSeaview_ATL_CUW/
├── CatlinSeaview_ATL_GLP/
├── CatlinSeaview_ATL_MAF/
├── CatlinSeaview_ATL_MEX/
├── CatlinSeaview_ATL_SXM/
├── CatlinSeaview_ATL_TCA/
├── CatlinSeaview_ATL_VCT/
├── CatlinSeaview_IND_CHA/
├── CatlinSeaview_IND_MDV/
├── CatlinSeaview_PAC_AUS/
├── CatlinSeaview_PAC_IDN/
├── CatlinSeaview_PAC_PHL/
├── CatlinSeaview_PAC_SLB/
├── CatlinSeaview_PAC_TLS/
├── CatlinSeaview_PAC_TWN/
├── CatlinSeaview_PAC_USA/
├── Chesterfield/
├── Dellwood_H1682/
├── Dellwood_H1683/
├── Dellwood_South_H1690/
├── DFO_Eelgrass/
├── EAC_2021/
├── EMR202001/
├── Explorer_H1691/
├── fk180731/
├── FK200308/
├── Frobisher/
├── Georges_Bank_2000/
├── Georges_Bank_2002/
├── German_Bank_2003/
├── German_Bank_2006/
├── German_Bank_2010/
├── Hakai_ROV_2019/
├── Hakai_Video_2020/
├── Hawaii_Archipelago_2019/
├── Hawaii_CRAMP_2015/
├── Hogkins_H1685/
├── Julia_2020/
├── Kona_2004/
├── Mariana_2017/
├── NOAA_HabCam_2015/
├── nrcan-2000042/
├── nrcan-2000047/
├── nrcan-2001ROPOS/
├── nrcan-2002021/
├── nrcan-2002026/
├── nrcan-2002066/
├── nrcan-2003009/
├── nrcan-2003015/
├── nrcan-2003029/
├── nrcan-2003054/
├── nrcan-2003068/
├── nrcan-2004010/
├── nrcan-2004014/
├── nrcan-2004018/
├── nrcan-2004024/
├── nrcan-2004037/
├── nrcan-2005011PGC/
├── nrcan-2005023/
├── nrcan-2005030/
├── nrcan-2005033B/
├── nrcan-2006002PGC/
├── nrcan-2006039/
├── nrcan-2006040/
├── nrcan-2006054/
├── nrcan-2007016/
├── nrcan-2007048/
├── nrcan-2008015/
├── nrcan-2008027/
├── nrcan-2008052/
├── nrcan-2008303/
├── nrcan-2009044/
├── nrcan-2010020/
├── nrcan-2010023/
├── nrcan-2010034/
├── nrcan-2011002PGC/
├── nrcan-2013002PGC/
├── nrcan-2015002PGC/
├── nrcan-2015004PGC/
├── nrcan-69016/
├── nrcan-73003PHASE2/
├── nrcan-73006/
├── nrcan-75009PHASE1/
├── nrcan-76016/
├── nrcan-77011/
├── nrcan-78012/
├── nrcan-78QUEST/
├── nrcan-83019/
├── nrcan-85005/
├── nrcan-87014/
├── nrcan-90035/
├── nrcan-97060/
├── pangaea-839225/
├── pangaea-841459/
├── pangaea-846142/
├── pangaea-846143/
├── pangaea-846144/
├── pangaea-846146/
├── pangaea-846147/
├── pangaea-846185/
├── pangaea-846186/
├── pangaea-846264/
├── pangaea-846266/
├── pangaea-867188/
├── pangaea-878000/
├── pangaea-878001/
├── pangaea-878003/
├── pangaea-878004/
├── pangaea-878006/
├── pangaea-878007/
├── pangaea-878008/
├── pangaea-878009/
├── pangaea-878010/
├── pangaea-878011/
├── pangaea-878012/
├── pangaea-878013/
├── pangaea-878014/
├── pangaea-878015/
├── pangaea-878016/
├── pangaea-878017/
├── pangaea-878018/
├── pangaea-878019/
├── pangaea-892599/
├── pangaea-892600/
├── pangaea-892604/
├── pangaea-892607/
├── pangaea-892608/
├── pangaea-892615/
├── pangaea-892619/
├── pangaea-892623/
├── pangaea-892625/
├── pangaea-892627/
├── pangaea-894800/
├── pangaea-894801/
├── pangaea-895121/
├── pangaea-895124/
├── pangaea-895147/
├── pangaea-895154/
├── pangaea-895157/
├── pangaea-895160/
├── pangaea-897047/
├── pangaea-897590/
├── pangaea-899670/
├── pangaea-900446/
├── pangaea-904715/
├── pangaea-907013/
├── pangaea-912471/
├── PS201012/
├── PS201211/
├── PS201502/
├── Qikiqtarjuaq/
├── RLS_.../
├── Sabrina_2017/
├── Samoa_2017/
├── SEQueensland201010/
├── Sgaan_H1684/
├── Sgaan_H1686/
├── Shreya_2020/
├── SolitaryIs201208/
├── ssk16-01/
├── ssk17-01/
├── ssk18-01/
├── St_Anns_Bank/
├── Sydney201211/
├── Sydney201303/
├── Tasmania200810/
├── Tasmania200903/
├── Tasmania200906/
├── Tasmania201006/
├── Tasmania201106/
├── Tasmania201205/
├── Tasmania201306/
├── Tasmania201406/
├── Tasmania201502/
├── Tasmania201610/
├── Tasmania201707/
├── Tasmania201808/
├── Tasmania202001/
├── Tasmania202208/
├── TasVic201602SS/
├── WA201104/
├── WA201204/
├── WA201304/
├── Wager/
└── Wake_2017/
```

其中 `RLS_...` 下还包含大量地区和年份目录，例如：

```text
RLS_Adelaide_2015/
RLS_Adelaide_2016/
RLS_Ambon_2015/
RLS_Batemans Marine Park_2008/
RLS_Batemans Marine Park_2009/
RLS_Cape Howe_2011/
RLS_Carrie Bow, Belize_2015/
RLS_Cook Islands_2012/
RLS_Great Barrier Reef_2019/
RLS_Ningaloo Marine Park_2019/
RLS_Port Phillip Bay_2020/
RLS_Port Stephens_2021/
RLS_Sydney_2020/
RLS_Tenerife_2013/
RLS_Tonga_2012/
RLS_Vancouver_2010/
RLS_Wilsons Promontory_2020/
```

每个来源目录内部通常还会继续包含：

```text
来源目录/
└── 航次、站点或图像序列目录/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── image_003.jpg
    └── ...
```

例如：

```text
Wake_2017/
└── WAK-438/
    ├── WAK-438_2017_A_01.jpg
    ├── WAK-438_2017_A_02.jpg
    ├── WAK-438_2017_A_03.jpg
    └── ...
```

当前实际普通 JPG 文件：

```text
240,526 张
```

全部可正常识别，没有真实损坏图片。

---

# 8. 客户标注目录树

```text
/sevenH/deepsea_vlm/data/raw/customer_annotations
└── captioning_labelled/
```

详细文件：

```text
captioning_labelled/
├── captioning_0_9999.json
├── captioning_10000_19999.json
├── captioning_20000_33999.json
├── captioning_34000_39999.json
├── captioning_40000_79999.json
├── captioning_80000_99999.json
├── captioning_100000_119999.json
├── captioning_120000_139999.json
├── captioning_140000_159999.json
├── captioning_160000_199999.json
└── image_elements.json
```

文件大小：

```text
captioning_0_9999.json             约 4.5MB
captioning_10000_19999.json        约 5.1MB
captioning_20000_33999.json        约 7.5MB
captioning_34000_39999.json        约 3.4MB
captioning_40000_79999.json        约 19MB
captioning_80000_99999.json        约 12MB
captioning_100000_119999.json      约 8.5MB
captioning_120000_139999.json      约 8.7MB
captioning_140000_159999.json      约 9.9MB
captioning_160000_199999.json      约 14MB
image_elements.json                约 45MB
```

目录总计约：

```text
135MB
```

这 11 个文件目前仍属于原始数据，尚未生成：

```text
清洗结果
图片匹配索引
训练数据
验证集
测试集
```

---

# 9. FathomNet 原始数据目录

```text
/sevenH/deepsea_vlm/data/raw/fathomnet
├── fathomnet_concept_counts.csv
└── fathomnet_concepts_raw.csv
```

### `fathomnet_concepts_raw.csv`

保存 FathomNet 概念名称列表。

### `fathomnet_concept_counts.csv`

保存：

```text
concept
bounding_box_count
```

即每个概念对应的目标框数量。

---

# 10. 中间数据目录

```text
/sevenH/deepsea_vlm/data/interim
├── benthicnet/
└── fathomnet/
```

---

## 10.1 BenthicNet 中间结果

```text
interim/benthicnet/
└── image_validation/
    ├── invalid_images.csv
    └── validation_summary.txt
```

### `validation_summary.txt`

内容概述：

```text
image_root:
  /sevenH/deepsea_vlm/data/raw/benthicnet/
  extracted/compiled_labelled_512pix

原始扫描项：240527
真实有效 JPG：240526
误判目录：1
检查耗时：431.53 秒
图片格式：JPEG
颜色模式：RGB
```

### `invalid_images.csv`

记录的异常项不是损坏图片，而是：

```text
一个名称以 .jpg 结尾的目录
```

异常路径：

```text
fk180731/
└── 20180804_093404_20180804_143258_20180805_123456_20180809_083837_ae2000f_sx3/
    └── sxA180804_image0000948_FC.png.png .jpg/
```

---

## 10.2 FathomNet 中间结果

```text
interim/fathomnet/
├── fathomnet_balanced_candidate_pool.csv
├── fathomnet_top300_taxonomy_screen.csv
└── concept_samples/
    └── caecosagitta_macrocephala/
```

详细展开：

```text
concept_samples/
└── caecosagitta_macrocephala/
    ├── concept_review.json
    ├── contact_sheet.jpg
    ├── manifest.csv
    ├── crops/
    │   ├── sample_001.jpg
    │   ├── sample_002.jpg
    │   ├── sample_003.jpg
    │   ├── sample_004.jpg
    │   ├── sample_005.jpg
    │   ├── sample_006.jpg
    │   ├── sample_007.jpg
    │   ├── sample_008.jpg
    │   ├── sample_009.jpg
    │   ├── sample_010.jpg
    │   ├── sample_011.jpg
    │   └── sample_012.jpg
    └── full/
        ├── sample_001.jpg
        ├── sample_002.jpg
        ├── sample_003.jpg
        ├── sample_004.jpg
        ├── sample_005.jpg
        ├── sample_006.jpg
        ├── sample_007.jpg
        ├── sample_008.jpg
        ├── sample_009.jpg
        ├── sample_010.jpg
        ├── sample_011.jpg
        └── sample_012.jpg
```

### `concept_review.json`

保存对 `Caecosagitta macrocephala` 的人工评审信息。

### `contact_sheet.jpg`

将 12 张裁剪样本拼成一张接触表，便于人工检查。

### `manifest.csv`

记录样本来源、图片路径、目标框等信息。

### `crops`

保存目标框周围裁剪图。

### `full`

保存对应完整原图。

---

# 11. 处理后数据目录

```text
/sevenH/deepsea_vlm/data/processed
└── fathomnet/
    └── benchmarks/
        └── deepsea_id_caecosagitta_12.jsonl
```

当前只有 FathomNet 评测数据。

尚不存在：

```text
processed/customer_annotations/
processed/benthicnet/
processed/training/
processed/evaluation/
```

---

# 12. 数据集划分目录

```text
/sevenH/deepsea_vlm/data/splits/
```

当前为空：

```text
splits/
```

尚未生成：

```text
train.jsonl
validation.jsonl
test.jsonl
golden_test.jsonl
```

---

# 13. 输出目录树

```text
/sevenH/deepsea_vlm/outputs
├── baselines/
├── checkpoints/
├── logs/
└── reports/
```

---

## 13.1 `baselines`

```text
outputs/baselines/
├── qwen3_vl_image_smoke_test.json
├── qwen3_vl_caecosagitta_12_predictions.jsonl
├── qwen3_vl_caecosagitta_12_scored.jsonl
└── qwen3_vl_caecosagitta_12_summary.json
```

### `qwen3_vl_image_smoke_test.json`

基础图片描述测试结果。

### `qwen3_vl_caecosagitta_12_predictions.jsonl`

12 张 Caecosagitta 图片的模型预测。

### `qwen3_vl_caecosagitta_12_scored.jsonl`

逐条评分结果。

### `qwen3_vl_caecosagitta_12_summary.json`

整体汇总指标。

---

## 13.2 `checkpoints`

```text
outputs/checkpoints/
```

当前为空。

尚未产生：

```text
LoRA Adapter
QLoRA Adapter
Trainer 状态
优化器状态
训练检查点
```

---

## 13.3 `logs`

```text
outputs/logs/
├── check_model_load.log
├── download_qwen3_vl_2b.log
├── image_smoke_test.log
└── install_pytorch.log
```

### `install_pytorch.log`

记录 PyTorch 安装过程。

### `download_qwen3_vl_2b.log`

记录 Qwen3-VL-2B-Instruct 下载过程，约 348KB。

### `check_model_load.log`

记录模型加载验证。

### `image_smoke_test.log`

记录图片推理测试。

---

## 13.4 `reports`

```text
outputs/reports/
└── environment_before_qwen3vl.txt
```

记录安装 Qwen3-VL 前的环境状态。

当前尚未生成：

```text
客户数据分析报告
正式基线报告
训练报告
微调前后对比报告
最终验收报告
```

---

# 14. 当前目录树简化总览

```text
/projects/deepsea_vlm
├── .conda/                         # 项目 Python 环境，约 5.6GB
├── .git/                           # Git 仓库
├── .gitignore
├── README.md
├── configs/                        # 当前为空
├── data -> /sevenH/deepsea_vlm/data
├── docs/
│   ├── core_versions.txt
│   ├── environment-conda-explicit.txt
│   ├── environment-pip-freeze.txt
│   └── 阶段1_环境与基座模型验证记录.md
├── models -> /sevenH/deepsea_vlm/models
├── outputs -> /sevenH/deepsea_vlm/outputs
├── scripts/
│   ├── 01_export_fathomnet_concepts.py
│   ├── 02_export_fathomnet_concept_counts.py
│   ├── 03_classify_fathomnet_top_concepts.py
│   ├── 04_summarize_fathomnet_taxa.py
│   ├── 05_build_fathomnet_candidate_pool.py
│   ├── 06_download_fathomnet_concept_samples.py
│   ├── 07_build_caecosagitta_eval_dataset.py
│   ├── 08_run_caecosagitta_baseline.py
│   ├── 09_score_caecosagitta_baseline.py
│   ├── 11_validate_benthicnet_images.py
│   └── run_image_smoke_test.py
├── src/                            # 当前为空
└── tests/
    └── assets/
        └── smoke_test.png


/sevenH/deepsea_vlm
├── cache/
│   ├── conda/
│   ├── huggingface/
│   ├── pip/
│   └── torch/
│
├── models/
│   └── Qwen3-VL-2B-Instruct/
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       ├── chat_template.json
│       └── 其他模型配置文件
│
├── data/
│   ├── raw/
│   │   ├── benthicnet/
│   │   │   ├── full_labelled_512px.tar
│   │   │   ├── full_labelled_512px.tar.sha256
│   │   │   ├── full_labelled_512px_filelist.txt
│   │   │   ├── full_labelled_512px_download.log
│   │   │   ├── documentation/
│   │   │   │   └── README.txt
│   │   │   └── extracted/
│   │   │       └── compiled_labelled_512pix/
│   │   │           └── 240,526 张 JPG
│   │   │
│   │   ├── customer_annotations/
│   │   │   └── captioning_labelled/
│   │   │       ├── 10 个 captioning 分片
│   │   │       └── image_elements.json
│   │   │
│   │   └── fathomnet/
│   │       ├── fathomnet_concepts_raw.csv
│   │       └── fathomnet_concept_counts.csv
│   │
│   ├── interim/
│   │   ├── benthicnet/
│   │   │   └── image_validation/
│   │   │       ├── invalid_images.csv
│   │   │       └── validation_summary.txt
│   │   │
│   │   └── fathomnet/
│   │       ├── fathomnet_top300_taxonomy_screen.csv
│   │       ├── fathomnet_balanced_candidate_pool.csv
│   │       └── concept_samples/
│   │           └── caecosagitta_macrocephala/
│   │               ├── full/
│   │               ├── crops/
│   │               ├── manifest.csv
│   │               ├── contact_sheet.jpg
│   │               └── concept_review.json
│   │
│   ├── processed/
│   │   └── fathomnet/
│   │       └── benchmarks/
│   │           └── deepsea_id_caecosagitta_12.jsonl
│   │
│   └── splits/                     # 当前为空
│
└── outputs/
    ├── baselines/
    │   ├── qwen3_vl_image_smoke_test.json
    │   ├── qwen3_vl_caecosagitta_12_predictions.jsonl
    │   ├── qwen3_vl_caecosagitta_12_scored.jsonl
    │   └── qwen3_vl_caecosagitta_12_summary.json
    ├── checkpoints/                # 当前为空
    ├── logs/
    │   ├── install_pytorch.log
    │   ├── download_qwen3_vl_2b.log
    │   ├── check_model_load.log
    │   └── image_smoke_test.log
    └── reports/
        └── environment_before_qwen3vl.txt
```

---

# 15. 当前仍缺少的目录和文件

按照后续工作计划，项目中还没有建立：

```text
configs/
├── paths.yaml
├── data_analysis.yaml
├── baseline.yaml
├── train_lora.yaml
└── evaluation.yaml

src/
├── data/
│   ├── annotation_loader.py
│   ├── image_index.py
│   ├── annotation_matcher.py
│   └── dataset_splitter.py
├── models/
│   └── qwen3vl_loader.py
├── training/
│   └── train_lora.py
├── inference/
│   └── run_baseline.py
├── evaluation/
│   ├── label_metrics.py
│   └── caption_metrics.py
└── utils/
    └── paths.py

tests/
├── test_annotation_format.py
├── test_image_matching.py
├── test_label_hierarchy.py
└── test_output_schema.py

/sevenH/deepsea_vlm/data/interim/customer_annotations/
├── annotation_summary.json
├── duplicate_records.csv
├── unmatched_annotations.csv
└── image_index.csv

/sevenH/deepsea_vlm/data/processed/customer_annotations/
├── cleaned_annotations.jsonl
├── training_samples.jsonl
└── label_vocabulary.json

/sevenH/deepsea_vlm/data/splits/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── golden_test.jsonl
```

这些是后续规划目录，目前尚未实际创建。
