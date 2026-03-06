# 语音匿名化系统架构文档（Segment 方案）

## 概述

基于 Eta-WavLM 和段级风格提取的语音匿名化系统，通过风格引导检索实现内容保留和说话人匿名化。

---

## 数据集划分

### 训练集（Step 1-5）
- **数据源**: LibriSpeech train-clean-360
- **路径**: `cache/features/wavlm/train_clean_360/`
- **用途**:
  - 训练 speaker embedding PCA
  - 训练 Eta 投影矩阵
  - 训练 phone predictor
  - 训练 segment style extractor
  - 预计算话语级风格向量（作为检索目标池）

### 测试集（Step 6）
- **推荐数据源**: LibriSpeech test-clean
- **路径**: `../datasets/LibriSpeech/test-clean/`
- **用途**: 待匿名化的源音频

---

## 系统流程

### Step 1: Speaker Embedding PCA
**脚本**: `scripts/step1_build_speaker_pca.py`

**输入**:
- `cache/features/ecapa/train_clean_360/embeddings.npy` (256维 ECAPA-TDNN embeddings)

**输出**:
- `checkpoints/speaker_embeddings_pca.npy` (降维后的 speaker embeddings)
- `checkpoints/speaker_pca_model.pkl` (PCA 模型)
- `checkpoints/utt_indices.npy` (话语索引映射)

**功能**: 将 256 维 speaker embedding 降维到 P 维（默认 64），用于后续 Eta 投影

---

### Step 2: Eta Projection
**脚本**: `scripts/step2_train_eta_projection.py`

**输入**:
- `cache/features/wavlm/train_clean_360/features.h5` (WavLM 特征)
- `checkpoints/speaker_embeddings_pca.npy` (Step 1)

**输出**:
- `checkpoints/eta_projection.pt` (包含 A*, b*)

**功能**: 训练线性投影 A*, b*，使得 η = s - (A*d + b*) 去除说话人信息

**核心公式**:
```
η = s - (A*d + b*)
```
其中:
- s: WavLM 特征 (1024维)
- d: speaker embedding PCA (P维)
- η: 去说话人后的内容+风格表示

---

### Step 3: Phone Predictor
**脚本**: `scripts/step3_train_phone_decoder.py`

**输入**:
- `cache/features/wavlm/train_clean_360/features.h5`
- 对应的 phone 标注（从 forced alignment）

**输出**:
- `checkpoints/phone_decoder.pt` (phone predictor 模型)

**功能**: 训练 phone 分类器，用于 phone-constrained kNN 检索

---

### Step 4: Segment Style Extractor
**脚本**: `scripts/step4_build_style_extractor.py`

**输入**:
- `cache/features/wavlm/train_clean_360/features.h5`
- `checkpoints/eta_projection.pt` (Step 2)
- `checkpoints/speaker_embeddings_pca.npy` (Step 1)
- `checkpoints/phone_decoder.pt` (Step 3)

**输出**:
- `checkpoints/style_extractor.pkl` (SegmentStyleExtractor)
- `outputs/style_analysis/segment_style_clustering.png` (验证图)

**功能**:
- 按 phone 边界分段
- 提取段级风格 embedding: [mean(η), std(η), delta_mean(η)]
- PCA 降维到 64 维
- 验证聚类结构

**关键指标**:
- Silhouette score: 聚类质量
- ARI_speaker ≈ 0: 风格与说话人身份解耦
- ARI_phone: 风格与音素的相关性

---

### Step 5: Precompute Utterance Styles
**脚本**: `scripts/step5_precompute_utterance_styles.py`

**输入**:
- `cache/features/wavlm/train_clean_360/features.h5`
- `cache/features/wavlm/train_clean_360/metadata.json`
- `checkpoints/eta_projection.pt`
- `checkpoints/speaker_embeddings_pca.npy`
- `checkpoints/style_extractor.pkl`
- `checkpoints/phone_decoder.pt`

**输出**:
- `checkpoints/utterance_styles.npz`:
  - `styles`: (N_utt, 64) 每个话语的风格向量
  - `speaker_ids`: (N_utt,) 说话人 ID
  - `genders`: (N_utt,) 性别
  - `utt_boundaries`: (N_utt, 2) features.h5 中的起止帧

**功能**: 为训练集所有话语预计算风格向量，作为 Step 6 的检索目标池

---

### Step 6: Style-Guided Retrieval
**脚本**: `scripts/step6_style_guided_retrieval.py`

**输入**:
- 源音频（test-clean）
- `checkpoints/eta_projection.pt`
- `checkpoints/style_extractor.pkl`
- `checkpoints/utterance_styles.npz` (Step 5)
- `cache/features/wavlm/train_clean_360/features.h5` (目标池)

**输出**:
- 匿名化的 WavLM 特征 h_anon
- 风格相似度指标

**功能**:
1. 提取源音频的 WavLM 特征 s_src
2. 提取源说话人 embedding d_src
3. 计算 η_src = s_src - (A*d_src + b*)
4. 提取源话语风格 z_src
5. 从 utterance_styles.npz 中检索:
   - 不同说话人
   - 同性别
   - 风格最相似的目标话语
6. 在目标话语内做 phone-constrained kNN:
   - 对每个 phone，只在目标话语的相同 phone 帧中检索
   - 使用 cosine 相似度
7. 输出匿名化特征 h_anon

**三种模式**:
- **System A (baseline)**: 随机选择目标话语（无风格引导）
- **System B (proposed)**: 风格引导的目标话语选择
- **System C (transfer)**: 指定风格模板

---

## 目录结构

```
anon_test/
├── cache/
│   └── features/
│       ├── ecapa/train_clean_360/
│       │   └── embeddings.npy
│       └── wavlm/train_clean_360/
│           ├── features.h5
│           └── metadata.json
├── checkpoints/
│   ├── speaker_embeddings_pca.npy
│   ├── speaker_pca_model.pkl
│   ├── utt_indices.npy
│   ├── eta_projection.pt
│   ├── phone_decoder.pt
│   ├── style_extractor.pkl
│   └── utterance_styles.npz
├── outputs/
│   └── style_analysis/
│       └── segment_style_clustering.png
├── scripts/
│   ├── step1_build_speaker_pca.py
│   ├── step2_train_eta_projection.py
│   ├── step3_train_phone_decoder.py
│   ├── step4_build_style_extractor.py
│   ├── step5_precompute_utterance_styles.py
│   └── step6_style_guided_retrieval.py
└── docs/
    └── architecture_segment.md
```

---

## 运行流程

### 训练阶段（Step 1-5）

```bash
# Step 1: Speaker PCA
python scripts/step1_build_speaker_pca.py

# Step 2: Eta Projection
python scripts/step2_train_eta_projection.py

# Step 3: Phone Predictor
python scripts/step3_train_phone_decoder.py

# Step 4: Style Extractor
python scripts/step4_build_style_extractor.py

# Step 5: Precompute Styles
python scripts/step5_precompute_utterance_styles.py
```

### 测试阶段（Step 6）

```bash
# 匿名化单个音频
python scripts/step6_style_guided_retrieval.py \
    --system B \
    --audio ../datasets/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac \
    --output outputs/anon/1089-134686-0000_anon.npy

# 批量匿名化
python scripts/step6_style_guided_retrieval.py \
    --system B \
    --test-dir ../datasets/LibriSpeech/test-clean/ \
    --output-dir outputs/anon/
```

---

## 关键参数

### Step 1
- `--pca-dim`: Speaker embedding PCA 维度（默认 64）

### Step 2
- `--max-frames`: 训练用的最大帧数（默认无限制）
- `--learning-rate`: 学习率（默认 0.001）

### Step 4
- `--pca-dim`: Style embedding 维度（默认 64）
- `--min-segment-frames`: 最小段长度（默认 3 帧）

### Step 6
- `--system`: A (baseline) / B (proposed) / C (transfer)
- `--top-k`: kNN 的 k 值（默认 1）
- `--style-weight`: 风格相似度权重（默认 1.0）

---

## 评估指标

### 匿名化质量
- **EER (Equal Error Rate)**: 说话人验证错误率，越高越好
- **Linkability**: 源-目标链接性，越低越好

### 内容保留
- **WER (Word Error Rate)**: ASR 词错误率，越低越好
- **Phone Error Rate**: 音素错误率

### 风格保留
- **Style Similarity**: 源-目标风格余弦相似度，越高越好
- **Prosody Correlation**: 韵律特征相关性

---

## 下一步：测试准备

### 1. 准备测试数据
```bash
# 确认测试集路径
ls ../datasets/LibriSpeech/test-clean/
```

### 2. 提取测试集特征
需要为 test-clean 提取 WavLM 特征和 ECAPA embeddings

### 3. 运行匿名化
使用 Step 6 对测试集进行匿名化

### 4. 评估
- ASR 评估（WER）
- Speaker verification 评估（EER）
- 风格保留评估

---

## 注意事项

1. **数据集分离**: 训练集和测试集必须完全分离，避免数据泄漏
2. **性别匹配**: Step 6 默认只检索同性别的目标话语
3. **Phone 约束**: kNN 检索限制在相同 phone 内，保证内容对齐
4. **风格解耦**: Step 4 验证 ARI_speaker ≈ 0，确保风格不泄漏身份信息

---

## 参考

- Eta-WavLM: 基于线性投影的说话人信息去除
- Segment Style: 基于 phone 边界的段级风格提取
- Phone-constrained kNN: 保证内容对齐的检索策略
