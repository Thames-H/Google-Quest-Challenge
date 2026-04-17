# Google QUEST Challenge

基于 `DeBERTa-v3` 双分支分层聚合模型的 Google QUEST Q&A Labeling 训练与推理工程。

本仓库面向 Kaggle `Google QUEST Q&A Labeling` 任务，目标是对每条问答对的 30 个主观标签进行联合预测，并导出符合提交格式的 `submission.csv`。

## 1. 项目概览

当前实现采用以下主线方案：

- `GroupKFold`：避免同一问题文本在训练和验证中同时出现。
- `DeBERTa-v3-base` 双分支：
  - 问题分支输入：`question_title + question_body`
  - 回答分支输入：`question_title + answer`
- `chunked hierarchical pooling`：将长文本切块后分别编码，再在块级别进行聚合。
- 元特征融合：类别、站点、域名、长度、URL 数量、代码块比例、词汇重叠等。
- 混合损失：`SmoothL1/BCE + margin ranking loss`
- 后处理：`rank-based distribution matching`

代码已经包含：

- 训练入口：`train.py`
- 推理入口：`predict.py`
- 可复用 checkpoint 的训练流程
- smoke 配置和正式 Linux 配置
- pytest 回归测试

## 2. 目录结构

```text
google-quest-challenge/
├─ configs/                    # YAML 配置
├─ docs/                       # 补充文档
├─ quest/                      # 数据、模型、loss、pipeline 核心实现
├─ tests/                      # 回归测试
├─ train.py                    # 训练入口
├─ predict.py                  # 推理入口
└─ requirements.txt            # 依赖
```

推荐将代码和数据分开放置：

```text
/workspace/Google-Quest-Challenge
/data/google-quest-challenge
```

或 Windows 本地对应：

```text
F:\面壁challenge\google-quest-challenge
F:\面壁challenge\google-quest-data\google-quest-challenge
```

## 3. 数据格式

数据目录中应至少包含：

```text
train.csv
test.csv
sample_submission.csv
```

代码默认假设标签列来自 `sample_submission.csv` 中除 `qa_id` 外的所有列。

## 4. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

核心依赖包括：

- `torch`
- `transformers`
- `tokenizers`
- `sentencepiece`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `tqdm`
- `pyyaml`

## 5. 模型说明

模型位于 `quest/model.py`，核心是双分支编码器：

- `question_encoder`：编码 `title + body`
- `answer_encoder`：编码 `title + answer`

每个分支都先对切块后的文本编码，再通过块级 attention pooling 得到文档表示；随后拼接元特征表示，并分别走：

- `question_*` 标签头
- `answer_*` 标签头
- 共享标签头

该设计比把所有文本简单拼成一个序列更贴近 QUEST 任务的结构。

## 6. 配置文件

常用配置：

- `configs/deberta_dual_hierarchical_smoke.yaml`
- `configs/deberta_dual_hierarchical_linux.yaml`

正式 Linux 配置默认包含：

- `folds: 5`
- `seeds: [42, 2024]`
- `epochs: 3`
- `batch_size: 2`
- `grad_accum_steps: 4`
- `fp16: true`
- `gradient_checkpointing: true`

如果只想快速跑通，可以先减少：

- `folds`
- `seeds`
- `epochs`

## 7. 训练

### 7.1 Smoke Run

```bash
python train.py \
  --config configs/deberta_dual_hierarchical_smoke.yaml \
  --data-dir /data/google-quest-challenge \
  --debug
```

### 7.2 正式训练

```bash
python train.py \
  --config configs/deberta_dual_hierarchical_linux.yaml \
  --data-dir /data/google-quest-challenge
```

### 7.3 使用本地模型目录

如果不希望运行时在线下载 Hugging Face 模型，可以先下载到本地，再通过 `--model-dir` 指定：

```bash
python train.py \
  --config configs/deberta_dual_hierarchical_linux.yaml \
  --data-dir /data/google-quest-challenge \
  --model-dir /path/to/local/deberta-v3-base
```

模型来源解析优先级为：

1. `--model-dir`
2. `QUEST_MODEL_DIR`
3. `config.model_dir`
4. `config.backbone`

## 8. 推理与提交文件

训练完成后，需要单独运行推理脚本导出 `submission.csv`：

```bash
python predict.py \
  --config configs/deberta_dual_hierarchical_linux.yaml \
  --data-dir /data/google-quest-challenge \
  --checkpoint-dir artifacts_deberta/checkpoints \
  --output submission.csv
```

注意：

- `train.py` 不会直接输出 `submission.csv`
- `predict.py` 会读取 checkpoint 目录中的所有 `model_seed*_fold*.pt` 做平均预测

## 9. 训练产物

默认产物目录示例：

```text
artifacts_deberta/
├─ checkpoints/
│  └─ model_seed{seed}_fold{fold}.pt
├─ oof/
│  └─ oof_seed{seed}.csv
└─ metrics/
   └─ cv_summary.json
```

关键行为：

- 每个 `seed + fold` 会单独保存 checkpoint
- 已完成的 fold 再次运行 `train.py` 时会优先复用，不会重复训练
- 当前支持的是“按 fold 复用”，不是“按 step 断点续训”

## 10. 常见问题

### Q1. 为什么看不到 GPU 在跑？

训练流程会先创建 dataset 和 tokenizer，再构建模型并迁移到 GPU。如果卡在 tokenizer、模型下载或数据准备阶段，`nvidia-smi` / `nvtop` 里可能暂时看不到训练进程。

### Q2. 为什么会报 tokenizer / sentencepiece 相关错误？

`deberta-v3` 依赖 `sentencepiece`。如果 fast tokenizer 加载失败，代码会自动回退到显式的 `DebertaV2Tokenizer`，但环境里仍然需要正确安装 `sentencepiece`。

### Q3. 为什么会报混合精度错误？

当前代码已经显式将 backbone 以 `float32` 加载，避免环境默认把参数加载成 `float16` 后与 `GradScaler` 冲突。如果服务器上仍然出现类似问题，请确认同步的是最新代码。

### Q4. 训练是不是完全不使用 test 数据？

模型训练与验证不会直接使用 `test.csv` 样本；但当前元特征词表的构建会联合 `train + test` 的无标签字段（如 `category/host/domain`），数值统计量仍只使用训练集。

### Q5. 显存太高怎么办？

代码内置了简单的 OOM fallback，会逐步：

- 增大 `grad_accum_steps`
- 减少 `answer_max_chunks`
- 减少 `question_max_chunks`

## 11. 测试

运行回归测试：

```bash
pytest tests -q
```

当前仓库测试覆盖了：

- 配置解析
- 数据集与 GroupKFold
- tokenizer fallback
- 模型前向
- 训练/推理 pipeline
- checkpoint 复用
- 分布匹配

## 12. 当前实现边界

当前版本重点是“完整可跑的训练/推理主线”，尚未包含：

- continued pretraining
- 伪标签 teacher-student 流程
- 多 backbone 集成
- DDP / 多卡同步训练
- step 级断点续训

