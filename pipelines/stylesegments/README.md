# Style Segments Pipeline

统一的特征提取和风格分割管线，整合了原 `offline/feature_extraction.py` 和 `scripts/step1-6.py`。

## 目录结构

```
pipelines/stylesegments/
├── __init__.py              # 模块导出
├── feature_extraction.py    # WavLM多层特征提取
├── speaker_embeddings.py    # 说话人嵌入+PCA
├── eta_projection.py        # Eta-WavLM投影计算
├── phone_precompute.py      # 音素预计算
├── style_extractor.py       # 段级风格提取器
├── phone_clusters.py        # Phone-level聚类
├── runner.py                # 统一运行脚本
└── README.md                # 本文档
```

## 主要改进

1. **多层提取支持**: 同时提取 WavLM 第6层和第24层特征
2. **模块化设计**: 每个步骤独立模块，便于维护
3. **统一管线**: 一个命令运行所有步骤
4. **优化存储**: 每层特征独立存储在 `layer_6/` 和 `layer_24/` 子目录

## 使用方法

### 运行完整管线

```bash
cd /root/autodl-tmp/anon_test
python pipelines/stylesegments/runner.py --config configs/base.yaml
```

### 运行特定步骤

```bash
# 只运行步骤1和2
python pipelines/stylesegments/runner.py --steps 1,2

# 使用采样（10%数据）
python pipelines/stylesegments/runner.py --sample-ratio 0.1
```

## 步骤说明

1. **Step 1**: WavLM特征提取（多层）
   - 输出: `cache/features/wavlm/{split}/layer_{6,24}/features.h5`

2. **Step 2**: 说话人嵌入提取
   - 输出: `checkpoints/speaker_embeddings_pca.npy`

3. **Step 3**: 音素预计算
   - 输出: `checkpoints/layer_{6,24}/phones.npy`

4. **Step 4**: Eta投影计算
   - 输出: `checkpoints/layer_{6,24}/eta_projection.pt`

5. **Step 5**: 风格提取器
   - 输出: `checkpoints/layer_{6,24}/style_extractor.pkl`

6. **Step 6**: Phone聚类
   - 输出: `checkpoints/layer_{6,24}/phone_clusters.pkl`

## 配置

在 `configs/base.yaml` 中设置：

```yaml
ssl:
  layers: [6, 24]  # 提取的层级

offline:
  train_split: 'train-clean-100'
  batch_size: 8
  feature_extraction:
    sample_ratio: 1.0
    save_interval: 1000
```
