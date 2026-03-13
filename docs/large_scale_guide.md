# 大规模数据集训练指南

## 概述

从小规模数据集（43个说话人，train-clean-100）切换到大规模数据集（train-clean-360）。

## 目录结构

```
checkpoints/
├── large/          # 大规模数据集权重
│   ├── speaker_embeddings_pca.npy
│   ├── eta_projection.pt
│   ├── style_extractor.pkl
│   └── ...
└── WavLM-Large.pt  # 共享的预训练模型

cache/
└── large/          # 大规模数据集缓存
    └── features/
        └── wavlm/
            └── train_clean_360/
                ├── features.h5
                └── metadata.json
```

## 使用方法

### 方式 1: 一键运行（推荐）

```bash
cd /root/autodl-tmp/anon_test
./run_large_scale.sh
```

### 方式 2: 分步运行

```bash
# Step 1: WavLM 特征提取
python scripts/run_offline.py --config configs/large_scale.yaml --step 1

# Step 2: 提取说话人嵌入
python scripts/step1_extract_speaker_embeddings.py --config configs/large_scale.yaml

# Step 3: 计算 Eta 投影
python scripts/step2_compute_eta_projection.py --config configs/large_scale.yaml

# Step 4: 构建风格提取器
python scripts/step4_build_style_extractor.py --config configs/large_scale.yaml

# Step 5: 预计算话语风格
python scripts/step5_precompute_utterance_styles.py --config configs/large_scale.yaml

# Step 6: 构建 phone clusters
python scripts/step5_build_phone_clusters.py --config configs/large_scale.yaml
```

## 配置说明

### 数据集选择

在 `configs/large_scale.yaml` 中修改：

```yaml
offline:
  train_split: "train-clean-360"  # 或 "train-other-500"
```

- `train-clean-100`: ~28k 话语, 251 说话人（小规模）
- `train-clean-360`: ~104k 话语, 921 说话人（推荐）
- `train-other-500`: ~148k 话语, 1166 说话人（最大）

### 输出路径

所有大规模数据集的输出都在 `large/` 子目录下，不会覆盖小规模数据集的权重。

## 后续步骤

训练完成后，使用大规模数据集进行去匿名：

```bash
# 使用大规模数据集的权重进行推理
python scripts/step6_style_guided_retrieval.py \
  --config configs/large_scale.yaml \
  --input <input_audio> \
  --output <output_audio>
```

## 注意事项

1. **内存需求**: train-clean-360 约需 60-80GB 内存
2. **存储空间**: 特征文件约 20-30GB
3. **训练时间**: 完整流程约 2-4 小时（单 GPU）
