# 语音匿名化 Pipeline 运行指南

## 概述

本项目使用 **解耦 + 风格聚类** 方案构建 Target Pool，然后进行语音匿名化合成。

## 环境配置

```bash
# 激活 conda 环境
conda activate anon_test

# 或使用完整路径
/root/miniconda3/envs/anon_test/bin/python
```

## 数据集说明

当前支持两个特征空间：

| 数据集 | 特征路径 | Utterances | Speakers | Frames |
|--------|----------|------------|----------|--------|
| LibriSpeech | `cache/features/wavlm/` | 500 | 5 | 332,773 |
| IEMOCAP | `cache/features/iemocap/` | 10,039 | 10 | 2,233,391 |

---

## Pipeline 步骤

### Step 1: 特征提取 (已完成)

如需重新提取特征：

```bash
# LibriSpeech 特征提取
python pipelines/offline/feature_extraction.py

# IEMOCAP 特征提取
python scripts/extract_iemocap_features.py
```

---

### Step 2: 风格聚类模型训练

训练 Pattern 模型进行风格聚类：

```bash
# v6 版本 (Sinkhorn-Knopp 均匀分配)
python tests/test_v32_style_clustering_v6.py

# v5.2 版本 (Prototype Pattern)
python tests/test_v32_style_clustering_v5.py
```

**配置修改** (在脚本 main 函数中):
- `max_utts`: 训练使用的 utterance 数量
- `dataset`: 选择 `'iemocap'` 或 `'wavlm'`
- `total_epochs`: 训练轮数

**输出**:
- 模型权重: `outputs/v6_tests/samm_v6_model.pt`
- 训练曲线: `outputs/v6_tests/training_history_v5.2.png`
- 聚类可视化: `outputs/v6_tests/samm_clusters_v5.2.png`

---

### Step 3: 构建 Target Pool

使用训练好的 Pattern 模型构建 Target Pool：

```bash
# 使用 v6 模型
python scripts/build_pool_v6.py

# 使用 v5.2 模型
python scripts/build_pool_v52.py
```

**配置修改** (在脚本 main 函数中):
```python
config = {
    'model_path': 'outputs/v6_tests/samm_v6_model.pt',  # 模型路径
    'cache_dir': 'cache/features/iemocap',              # 特征目录
    'output_dir': 'data/samm_anon/target_pool_v6',      # 输出目录
}
```

**输出**:
- `features.npy`: 特征向量
- `patterns.npy`: Pattern 分配
- `genders.npy`: 性别标签
- `faiss.index`: FAISS 检索索引
- `metadata.json`: 元数据

---

### Step 4: 语音合成测试

使用 Target Pool 进行语音匿名化合成：

```bash
# 使用 v6 Pool
python scripts/synthesize_v6.py --audio <输入音频路径> --output <输出路径>

# 使用 v5.2 Pool
python scripts/synthesize_v52.py --audio <输入音频路径> --output <输出路径>
```

**示例**:
```bash
python scripts/synthesize_v52.py \
    --audio outputs/20260121_163701/anonymized_audio/original.wav \
    --output outputs/v52_synth/test_anon.wav
```

---

## 快速开始 (完整流程)

```bash
cd /root/autodl-tmp/anon_test

# 1. 训练风格聚类模型 (使用 IEMOCAP)
python tests/test_v32_style_clustering_v6.py

# 2. 构建 Target Pool
python scripts/build_pool_v6.py

# 3. 语音合成测试
python scripts/synthesize_v6.py --audio <your_audio.wav> --output output.wav
```

---

## 文件结构

```
anon_test/
├── cache/features/
│   ├── wavlm/          # LibriSpeech 特征
│   └── iemocap/        # IEMOCAP 特征
├── outputs/
│   ├── v6_tests/       # v6 模型输出
│   └── v52_tests/      # v5.2 模型输出
├── data/samm_anon/
│   ├── target_pool_v6/ # v6 Target Pool
│   └── target_pool_v52/# v5.2 Target Pool
├── scripts/
│   ├── extract_iemocap_features.py  # IEMOCAP 特征提取
│   ├── build_pool_v6.py             # v6 Pool 构建
│   ├── build_pool_v52.py            # v5.2 Pool 构建
│   ├── synthesize_v6.py             # v6 语音合成
│   └── synthesize_v52.py            # v5.2 语音合成
└── tests/
    ├── test_v32_style_clustering_v6.py  # v6 训练脚本
    └── test_v32_style_clustering_v5.py  # v5.2 训练脚本
```

---

## 常见问题

### Q: 如何切换数据集？

修改训练脚本中的 `load_utterance_data` 调用：
```python
# 使用 IEMOCAP
data = load_utterance_data(cache_dir, max_utts=2000, dataset='iemocap')

# 使用 LibriSpeech
data = load_utterance_data(cache_dir, max_utts=500, dataset='wavlm')
```

### Q: 如何修改 Pool 构建的特征来源？

修改 `build_pool_v6.py` 中的 `cache_dir`：
```python
config = {
    'cache_dir': 'cache/features/iemocap',  # 或 'cache/features/wavlm'
}
```

### Q: 模型训练效果不好怎么办？

1. 增加训练数据量 (`max_utts`)
2. 调整训练轮数 (`total_epochs`)
3. 检查 Pattern 使用分布是否均匀
