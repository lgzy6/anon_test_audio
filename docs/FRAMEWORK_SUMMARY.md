# SAMM-Anon 框架总结文档

## 1. 框架概述

SAMM-Anon 是一个基于 WavLM 特征的语音匿名化框架，采用 Pseudo-Style Bank + kNN 检索的方案实现说话人身份保护。

### 核心思想
- **多层特征提取**：Layer 6 用于语音转换，Layer 24 用于音素预测
- **伪风格池构建**：基于混淆熵筛选高质量匿名化特征
- **音素约束检索**：保证内容一致性的同时实现说话人解耦

---

## 2. 系统架构

### 2.1 离线训练流程

```
LibriSpeech train-clean-360 (460 说话人, 52k 样本)
    ↓
[Step 1] 特征提取
    ├─ Layer 6: 语音转换特征 (1024维)
    ├─ Layer 12: 说话人嵌入计算 (1024维)
    └─ Layer 24: 音素预测 (实时预测后丢弃)
    ↓
[Step 2] 混淆熵计算
    ├─ 计算音素内说话人嵌入 (Centroids)
    ├─ 逐帧计算混淆熵 (基于 Layer 12)
    └─ 输出: entropies.h5
    ↓
[Step 3] 伪风格 Bank 构建
    ├─ 筛选高熵帧 (Percentile 60%)
    ├─ K-means 聚类 (50 clusters/phone)
    ├─ 说话人多样性过滤 (≥3 speakers)
    └─ 输出: pseudo_bank.pt (40 音素簇)
```

### 2.2 在线推理流程

```
测试音频 (test-clean)
    ↓
[提取特征]
    ├─ Layer 6: 源特征 (用于检索)
    └─ Layer 24: 音素标签 (用于约束)
    ↓
[kNN 检索]
    ├─ 音素约束: 同音素内检索
    ├─ Top-k 平均 (k=4)
    └─ Duration 适配 (weight=0.5)
    ↓
[HiFi-GAN 合成]
    └─ 输出: 匿名化音频
```

---

## 3. 关键模块

### 3.1 特征提取 (`extract_with_phones.py`)

**优化点**：
- 单次前向提取多层 (`forward_multi_layer`)
- Layer 24 仅用于 phone 预测后立即释放
- 按时长排序减少 padding 开销

**输出**：
```
train360_with_phones/
├── layer_6.h5          # [N_frames, 1024]
├── layer_12.h5         # [N_frames, 1024]
├── phones.h5           # [N_frames] int16
└── metadata.json       # 索引信息
```

### 3.2 混淆熵计算 (`compute_entropy.py`)

**核心算法**：
```python
# 音素内说话人嵌入
phone_spk_embs[phone_id][speaker_id] = mean(L12_features)

# 混淆熵 (向量化)
sims = L12_frame @ phone_spk_embs.T  # 余弦相似度
probs = softmax(sims * temperature)
entropy = -sum(probs * log(probs)) / log(num_speakers)
```

**性能优化**：
- 批量矩阵运算 (100x 加速)
- 数值稳定的 Softmax

### 3.3 Bank 构建 (`build_bank.py`)

**筛选策略**：
1. **高熵筛选**：Percentile 60%
2. **K-means 聚类**：50 clusters/phone
3. **多样性过滤**：≥3 speakers/cluster
4. **距离排序**：Top-20 frames/cluster

**输出格式**：
```python
pseudo_bank = {
    phone_id: torch.Tensor([N_frames, 1024])  # Layer 6 特征
}
```

### 3.4 匿名化合成 (`synthesize_pseudo.py`)

**关键特性**：
- 单声道强制转换（避免维度错误）
- Duration 适配（插值对齐）
- 音素约束 kNN 检索
- 对比音频生成

---

## 4. 数据统计

### 训练数据
- **数据集**：LibriSpeech train-clean-360
- **说话人数**：460 (50% 采样)
- **样本数**：52,000
- **总帧数**：~3000 万帧

### Bank 统计
- **有效音素**：40 个
- **总特征帧数**：~40k 帧
- **存储大小**：~160 MB

---

## 5. 配置参数

### 特征提取
```yaml
LAYERS: [6, 12, 24]
BATCH_SIZE: 8
SPEAKER_RATIO: 0.5
```

### 混淆熵
```yaml
TEMPERATURE: 10.0
MIN_SPEAKERS: 3
```

### Bank 构建
```yaml
N_PSEUDO_CLUSTERS: 50
FRAMES_PER_CLUSTER: 20
ENTROPY_PERCENTILE: 60
MIN_SPK_DIVERSITY: 3
```

### 推理
```yaml
TOP_K: 4
DUR_WEIGHT: 0.5
```

---

## 6. 使用示例

### 训练流程
```bash
# Step 1: 特征提取
python pipelines/stylesegments/extract_with_phones.py

# Step 2: 混淆熵计算
python pipelines/stylesegments/compute_entropy.py

# Step 3: Bank 构建
python pipelines/stylesegments/build_bank.py
```

### 推理测试
```bash
python pipelines/stylesegments/synthesize_pseudo.py \
  --audio test.flac \
  --output anon.wav \
  --bank checkpoints/pseudo_bank.pt \
  --save_comparison \
  --k 4 \
  --dur_weight 0.5
```

---

## 7. 性能优化总结

### 内存优化
- Layer 24 即用即丢（节省 33% 存储）
- 预加载 L6 到内存（避免随机 I/O）

### 计算优化
- 向量化熵计算（100x 加速）
- 单次前向多层提取（3x 加速）
- 按时长排序减少 padding

### 工程优化
- 单声道强制转换
- HDF5 流式读写
- 断点续传支持

---

## 8. 已知问题与解决

### 问题 1：HDF5 随机访问慢
**解决**：预加载到内存

### 问题 2：双声道音频维度错误
**解决**：强制转为单声道

### 问题 3：HiFi-GAN 输入格式
**解决**：保持 `[B, T, 1024]` 格式

---

## 9. 目录结构

```
anon_test/
├── checkpoints/
│   ├── train360_with_phones/      # 提取的特征
│   │   ├── layer_6.h5
│   │   ├── layer_12.h5
│   │   ├── phones.h5
│   │   ├── entropies.h5
│   │   └── metadata.json
│   ├── pseudo_bank.pt             # 伪风格 Bank
│   ├── WavLM-Large.pt
│   ├── phone_decoder.pt
│   ├── duration_decoder.pt
│   └── hifigan.pt
├── pipelines/stylesegments/
│   ├── extract_with_phones.py     # 特征提取
│   ├── compute_entropy.py         # 混淆熵计算
│   ├── build_bank.py              # Bank 构建
│   └── synthesize_pseudo.py       # 匿名化合成
├── models/
│   ├── ssl/wrappers.py            # WavLM 封装
│   ├── phone_predictor/           # 音素预测器
│   └── vocoder/                   # HiFi-GAN
└── outputs/                       # 输出音频
```

---

## 10. 未来优化方向

1. **多 GPU 并行**：特征提取加速
2. **在线 Bank 更新**：动态扩充音素簇
3. **自适应 k 值**：根据音素频率调整
4. **端到端训练**：联合优化检索与合成

---

**文档版本**：v1.0
**更新日期**：2026-03-13
**维护者**：SAMM-Anon Team
