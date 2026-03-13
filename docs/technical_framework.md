# 语音匿名化系统技术框架文档

## 1. 核心技术思想

本系统通过**解耦说话人身份与语义内容**，实现基于风格引导的 phone-level 特征替换匿名化。

**关键创新点：**
- Eta-WavLM 线性投影解耦说话人信息
- 段级风格表示捕获韵律特征
- Phone-constrained kNN 保持语义一致性
- 聚类加速的大规模检索

---

## 2. 系统架构

### 2.1 离线预处理阶段

```
LibriSpeech 训练集
    ↓
[Step 1] 说话人嵌入提取 (ECAPA-TDNN + PCA)
    ↓
[Step 2] Eta-WavLM 投影学习 (s = A^T·d + b + η)
    ↓
[Step 3.5] 音素预测 (Phone Predictor)
    ↓
[Step 4] 段级风格提取器构建 (PCA 64维)
    ↓
[Step 5a] 目标池风格预计算
    ↓
[Step 5b] Phone-level 聚类 (K-Means k=10)
    ↓
目标池准备完成
```

### 2.2 在线匿名化阶段

```
源音频输入
    ↓
特征提取 (WavLM + ECAPA + Phone)
    ↓
风格向量计算 (段级统计量)
    ↓
目标话语检索 (余弦相似度)
    ↓
Phone-level kNN 匹配 (Top-k 平均)
    ↓
音频合成 (HiFiGAN)
    ↓
匿名音频输出
```

---

## 3. 技术模块详解

### 3.1 Eta-WavLM 解耦模块

**数学模型：**
```
s = A^T · d + b + η
```

- `s ∈ R^(T×1024)`: WavLM 特征（混合表示）
- `d ∈ R^128`: 说话人嵌入（PCA 降维）
- `A ∈ R^(128×1024)`: 投影矩阵
- `b ∈ R^1024`: 偏置向量
- `η ∈ R^(T×1024)`: 解耦后的内容特征

**求解方法：**
正规方程求解最小二乘问题
```
(D̃^T D̃) Ã = D̃^T S
其中 D̃ = [d_pca, 1]
```

**实现文件：** `scripts/step2_compute_eta_projection.py`

---

### 3.2 段级风格提取器

**核心思想：**
风格是慢变量，帧级特征无法体现韵律差异。

**方法：**
1. 按 phone 边界分段
2. 每段提取统计量：
   ```
   [mean(η), std(η), delta_mean(η)]
   ```
3. PCA 降维到 64 维

**输出：**
- 话语级风格向量 `v ∈ R^64`
- 用于目标话语检索

**实现文件：** `scripts/step4_build_style_extractor.py`

---

### 3.3 Phone-level 聚类加速

**问题：**
对 100k+ 话语做帧级 kNN 检索计算量巨大。

**解决方案：**
为每个目标话语的每个音素预聚类：
```
phone_clusters[utt_id][phone_id] = K-Means 簇中心 (k=10)
```

**检索流程：**
```python
for phone_id in source_phones:
    # 只在对应音素的簇中心中检索
    clusters = phone_clusters[target_utt][phone_id]
    topk_indices = cosine_similarity(source_frame, clusters).topk(k=4)
    anon_frame = clusters[topk_indices].mean()
```

**实现文件：** `scripts/step5_build_phone_clusters.py`

---

### 3.4 风格引导检索

**目标：**
选择"说话方式相似"的目标话语。

**方法：**
```python
# 1. 提取源音频风格
source_style = style_extractor.extract(source_eta, source_phones)

# 2. 余弦相似度检索
similarities = cosine_similarity(source_style, pool_styles)

# 3. 性别过滤（可选）
valid_idx = [i for i, g in enumerate(genders) if g == source_gender]

# 4. 选择最相似话语
target_idx = argmax(similarities[valid_idx])
```

**实现文件：** `scripts/step6_style_guided_retrieval.py`

---

## 4. 端到端匿名化流程

**脚本：** `scripts/test_e2e_anonymization_v2.py`

### 4.1 输入参数
```bash
python scripts/test_e2e_anonymization_v2.py \
  --audio <源音频路径> \
  --output <输出路径> \
  --gender {f|m} \
  --k 4 \
  --device cuda
```

### 4.2 执行步骤

**Step 1: 加载模型**
- WavLM-Large (layer 15)
- ECAPA-TDNN
- Speaker PCA
- Eta 投影矩阵
- Phone Predictor
- Style Extractor
- HiFiGAN 声码器

**Step 2: 加载目标池**
- 话语风格向量 (104,014 × 64)
- Phone 聚类数据
- 元数据（性别、说话人 ID）

**Step 3: 提取源特征**
```python
wavlm_feats = wavlm(waveform)           # (T, 1024)
spk_emb = ecapa(waveform)               # (192,) → PCA → (128,)
eta = wavlm_feats - (A^T·d + b)         # (T, 1024)
phones = phone_predictor(wavlm_feats)   # (T,)
style = style_extractor(eta, phones)    # (64,)
```

**Step 4: 选择目标话语**
```python
target_idx = argmax(cosine_sim(style, pool_styles[gender_filtered]))
```

**Step 5: Phone-level kNN 匹配**
```python
for phone_id in unique(source_phones):
    src_mask = (source_phones == phone_id)
    tgt_clusters = phone_clusters[target_idx][phone_id]

    # Top-k 平均
    topk = cosine_similarity(src_feats[src_mask], tgt_clusters).topk(k=4)
    h_anon[src_mask] = tgt_clusters[topk].mean(dim=1)
```

**Step 6: 音频合成**
```python
waveform = vocoder(h_anon)
torchaudio.save(output_path, waveform, 16000)
```

---

## 5. 评估方法

### 5.1 语义保留评估

**工具：** `tools/testing/evaluate_semantic_preservation.py`

**方法：**
1. Whisper ASR 转录原始和匿名音频
2. 计算 WER (Word Error Rate)
3. 计算 Jaccard 相似度

**命令：**
```bash
python tools/testing/evaluate_semantic_preservation.py \
  --original <原始音频> \
  --anonymized <匿名音频> \
  --output <评估结果目录>
```

**评级标准：**
- WER < 10%: 🟢 优秀 (Excellent)
- WER < 30%: 🟡 良好 (Good)
- WER < 50%: 🟠 一般 (Fair)
- WER ≥ 50%: 🔴 较差 (Poor)

### 5.2 隐私保护评估

**指标：**
- EER (Equal Error Rate): 说话人验证错误率
- Linkability: 匿名前后话语关联度

**工具：** 需要额外的 ASV 系统

---

## 6. 当前性能表现

### 6.1 测试配置
- 源音频: LibriSpeech test-clean/61/70968/61-70968-0000.flac
- 目标池: train-clean-360 (104,014 话语)
- 参数: k=4, gender=m

### 6.2 评估结果

**原始文本：**
> "He began a confused complaint against the wizard who had vanished behind the curtain on the left."

**匿名文本：**
> "You be shotguned when you get through to the roofless ar .. You are going to get me for the sticker then."

**指标：**
- WER: 117.65% 🔴
- Jaccard 相似度: 3.23%
- 时长保持: 4.91s → 4.90s ✓

**评级：** 🔴 较差 (Poor) - 严重语义丢失

---

## 7. 问题分析与改进方向

### 7.1 当前问题

**语义崩溃原因：**
1. **Top-k 平均导致特征模糊**
   - k=4 时多个簇中心平均后失去判别性
   - 建议测试 k=1 或 k=2

2. **Phone 聚类质量不足**
   - K-Means k=10 可能无法捕获音素内部变化
   - 簇中心不能代表该音素的语义

3. **风格相似 ≠ 内容匹配**
   - 风格向量只捕获韵律，不保证内容相关性
   - 需要引入内容约束

### 7.2 改进建议

**短期优化：**
- 测试 k=1（最近邻，无平均）
- 增加聚类数 k=20 或 k=50
- 添加内容相似度约束

**中期改进：**
- 使用 FAISS 加速检索
- 引入 duration predictor 调整时长
- 多目标话语融合

**长期方向：**
- 端到端神经网络替代 kNN
- 对抗训练增强隐私保护
- 多语言支持

---

## 8. 文件结构

```
anon_test/
├── scripts/
│   ├── step1_extract_speaker_embeddings.py
│   ├── step2_compute_eta_projection.py
│   ├── step3_5_precompute_phones.py
│   ├── step4_build_style_extractor.py
│   ├── step5_precompute_utterance_styles.py
│   ├── step5_build_phone_clusters.py
│   ├── step6_style_guided_retrieval.py
│   └── test_e2e_anonymization_v2.py
├── tools/
│   └── testing/
│       └── evaluate_semantic_preservation.py
├── checkpoints/
│   ├── WavLM-Large.pt
│   ├── speaker_pca_model.pkl
│   ├── eta_projection.pt
│   ├── phone_decoder.pt
│   ├── style_extractor.pkl
│   ├── utterance_styles.npz
│   ├── phone_clusters.pkl
│   └── hifigan.pt
└── cache/
    └── features/
        └── wavlm/
            └── train_clean_360/
                ├── features.h5
                └── metadata.json
```

---

## 9. 系统参数配置

### 9.1 模型参数

**WavLM 配置：**
- 模型: WavLM-Large
- 提取层: Layer 15
- 特征维度: 1024
- 采样率: 16000 Hz
- Hop size: 320 (20ms)

**说话人编码器：**
- 模型: ECAPA-TDNN (speechbrain)
- 原始维度: 192
- PCA 降维: 128

**Eta-WavLM 投影：**
- 投影类型: 线性投影
- 投影矩阵: A ∈ R^(128×1024)
- 正则化系数: 1.0e-6

**Phone Predictor：**
- 音素类别数: 41
- 隐藏层维度: 256
- 网络层数: 2

**段级风格提取器：**
- 分段方式: phone 边界
- 最小段长: 3 帧
- 原始特征维度: 4096 (mean + std + delta_mean + delta_std)
- PCA 降维: 64
- 统计特征:
  - mean: 均值 (1024 维)
  - std: 标准差 (1024 维)
  - delta_abs_mean: 一阶差分绝对值均值 (1024 维)
  - delta_std: 一阶差分标准差 (1024 维)

**Phone-level 聚类：**
- 聚类算法: MiniBatch K-Means
- 每音素簇数: k=10
- 距离度量: 欧氏距离

**声码器：**
- 模型: HiFiGAN
- 输入: WavLM 特征 (1024 维)
- 输出: 16kHz 波形

### 9.2 数据集配置

**训练集（目标池）：**
- 数据集: LibriSpeech train-clean-360
- 话语数: 104,014
- 说话人数: 921
- 总时长: ~360 小时

**测试集：**
- 数据集: LibriSpeech test-clean
- 话语数: 2,620
- 说话人数: 40

### 9.3 检索参数

**风格引导检索：**
- 相似度度量: 余弦相似度
- 性别约束: 启用
- 说话人约束: 启用（避免选择同一说话人）

**Phone-level kNN：**
- Top-k: 4（当前配置）
- 聚合方式: 算术平均
- 归一化: L2 归一化

### 9.4 计算资源

**硬件配置：**
- GPU: CUDA 设备
- 批处理大小: 8（特征提取）
- 工作线程: 4
- 内存固定: 启用

**离线预处理时间（train-clean-360）：**
- Step 1 (说话人嵌入): ~2 小时
- Step 2 (Eta 投影): ~30 分钟
- Step 3.5 (音素预测): ~3 小时
- Step 4 (风格提取器): ~1 小时
- Step 5a (风格预计算): ~2 小时
- Step 5b (Phone 聚类): ~4 小时
- 总计: ~12-15 小时

**在线匿名化时间（单条话语）：**
- 特征提取: ~0.5 秒
- 风格检索: ~0.1 秒
- kNN 匹配: ~2-5 秒
- 音频合成: ~0.3 秒
- 总计: ~3-6 秒

### 9.5 存储需求

**模型文件：**
- WavLM-Large.pt: ~1.2 GB
- ECAPA 模型: ~50 MB
- HiFiGAN: ~150 MB
- 其他模型: ~100 MB
- 小计: ~1.5 GB

**目标池数据（train-clean-360）：**
- features.h5: ~40 GB
- utterance_styles.npz: ~26 MB
- phone_clusters.pkl: ~15 GB
- metadata.json: ~20 MB
- 小计: ~55 GB

**总存储需求: ~60 GB**

---

## 10. 依赖环境

**核心依赖：**
- Python 3.10
- PyTorch 2.0+
- torchaudio
- speechbrain
- openai-whisper
- scikit-learn
- h5py

**预训练模型：**
- WavLM-Large
- ECAPA-TDNN (speechbrain)
- HiFiGAN vocoder

---

## 10. 参考文献

1. Eta-WavLM: Speaker disentanglement via linear projection
2. VoicePrivacy Challenge 2024
3. spkanon: Phone-level clustering for anonymization
4. HiFiGAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

---

**文档版本：** v1.0
**更新日期：** 2026-03-08
**作者：** anon_test 项目组
