# 大规模风格池构建方案

## 1. 方案概述

### 1.1 目标
- 从 5,000 话语扩展到 100,000+ 话语
- 构建分层风格池，提升匹配精度
- 优化检索速度和内存占用

### 1.2 预期收益
- **匹配精度提升**：更多样化的风格选择
- **音质改善**：减少 kNN fallback，更精准的帧匹配
- **可控性增强**：风格簇可解释，支持风格迁移

---

## 2. 数据规模估算

### 2.1 LibriSpeech train-clean-360

| 项目 | 数量 |
|------|------|
| 话语数 | ~104,000 |
| 说话人数 | 921 |
| 总时长 | ~360 小时 |
| 平均话语长度 | ~12 秒 |

### 2.2 存储需求

| 数据类型 | 单条大小 | 总大小 |
|---------|---------|--------|
| WavLM 特征 (FP32) | ~500KB | ~50GB |
| WavLM 特征 (FP16) | ~250KB | ~25GB |
| Speaker Embedding | ~5KB | ~500MB |
| 风格向量 (64维) | ~256B | ~25MB |
| Phone 序列 | ~50KB | ~5GB |

**推荐配置：**
- 磁盘空间：100GB+
- 内存：32GB+
- GPU：16GB+ VRAM

---

## 3. 实施步骤

### Step 0: 准备全量数据

```bash
# 下载 LibriSpeech train-clean-360 (如果还没有)
cd /root/autodl-tmp/datasets/LibriSpeech
# 确认数据完整性
ls train-clean-360/ | wc -l  # 应该有 921 个说话人目录
```

### Step 1: 全量特征提取（分批处理）

**修改配置：**
```yaml
# configs/base.yaml
offline:
  train_split: "train-clean-360"
  feature_extraction:
    max_utterances: null  # 处理全部
    batch_size: 8
    save_interval: 1000   # 每1000条保存一次
    use_fp16: true        # 使用 FP16 节省空间
```

**运行：**
```bash
# 提取 WavLM 特征
python scripts/run_offline.py --step 1 --config configs/base.yaml

# 预计时间：~10-15 小时（取决于 GPU）
```

### Step 2: Speaker Embedding + Eta 投影

```bash
# Step 1: Speaker PCA (全量)
python scripts/step1_extract_speaker_embeddings.py

# Step 2: Eta 投影
python scripts/step2_compute_eta_projection.py
```

### Step 3: Phone Predictor（使用预训练）

```bash
# 已有 checkpoints/phone_decoder.pt，跳过
```

### Step 4: Style Extractor（全量训练）

```bash
# 使用全量数据训练
python scripts/step4_build_style_extractor.py
```

### Step 5: 全量风格预计算 + 聚类

**新增功能：分层风格池**

```bash
# 5.1 预计算所有话语的风格向量
python scripts/step5_precompute_utterance_styles.py

# 5.2 风格聚类（新增）
python scripts/step5_cluster_styles.py \
    --n-clusters 256 \
    --output checkpoints/style_clusters.pkl
```

**输出：**
- `checkpoints/utterance_styles.npz` (100k, 64)
- `checkpoints/style_clusters.pkl`:
  - `cluster_centers`: (256, 64)
  - `cluster_labels`: (100k,)
  - `cluster_indices`: dict[cluster_id → utterance_ids]

### Step 6: 两阶段检索

**修改检索逻辑：**
```python
# 粗检索：找到最近的 M 个簇
source_style → top_M_clusters (M=8)

# 细检索：在这 M 个簇内做 kNN
for cluster_id in top_M_clusters:
    candidate_utts = cluster_indices[cluster_id]
    best_utt = knn_search(source_style, candidate_utts)
```

---

## 4. 分层风格池架构

### 4.1 数据结构

```
checkpoints/
├── utterance_styles.npz          # 全量风格向量
│   ├── styles: (100k, 64)
│   ├── speaker_ids: (100k,)
│   ├── genders: (100k,)
│   └── boundaries: (100k, 2)
│
├── style_clusters.pkl             # 风格聚类
│   ├── cluster_centers: (256, 64)
│   ├── cluster_labels: (100k,)
│   ├── cluster_indices: dict
│   └── cluster_stats: dict
│
└── cache/features/wavlm/train_clean_360/
    └── features.h5                # 原始 WavLM 特征
```

### 4.2 检索流程

```
源音频
  ↓
提取风格 z_src (64维)
  ↓
粗检索：z_src → top_M 簇中心 (M=8)
  ↓
细检索：在 M 个簇内找最相似话语
  ↓
kNN 匿名化
  ↓
HiFi-GAN 合成
```

---

## 5. 性能优化

### 5.1 内存优化

**问题：** 100k 话语的 WavLM 特征无法全部加载到内存

**解决方案：**
1. 使用 HDF5 按需加载
2. 只在内存中保存风格向量（25MB）
3. kNN 时才加载对应话语的 WavLM 特征

### 5.2 检索加速

**粗检索：** O(256) - 快速
**细检索：** O(100k/256) ≈ O(400) - 可接受

**对比：**
- 原方案：O(100k) - 太慢
- 新方案：O(256 + 400) - 快 100 倍

### 5.3 分布式处理（可选）

如果单机太慢，可以分布式处理：

```bash
# 节点1：处理 speaker 0-300
python scripts/step5_precompute_utterance_styles.py \
    --speaker-range 0 300

# 节点2：处理 speaker 301-600
python scripts/step5_precompute_utterance_styles.py \
    --speaker-range 301 600

# 节点3：处理 speaker 601-921
python scripts/step5_precompute_utterance_styles.py \
    --speaker-range 601 921

# 合并结果
python scripts/merge_style_results.py
```

---

## 6. 验证与评估

### 6.1 风格聚类质量

```bash
# 可视化风格簇
python scripts/visualize_style_clusters.py \
    --output outputs/style_clusters.png

# 评估指标：
# - Silhouette score
# - 簇内方差
# - 簇间距离
```

### 6.2 检索质量

```bash
# 测试检索精度
python scripts/evaluate_retrieval.py \
    --test-set test-clean \
    --metrics style_similarity,speaker_distance

# 对比：
# - 全量检索 vs 分层检索
# - 风格相似度
# - 检索速度
```

### 6.3 匿名化质量

```bash
# 端到端测试
python scripts/test_e2e_anonymization.py \
    --audio test.flac \
    --output anon.wav \
    --use-hierarchical-pool

# 评估：
# - WER (内容保留)
# - EER (说话人匿名化)
# - MOS (音质)
```

---

## 7. 预期时间表

| 步骤 | 预计时间 | 备注 |
|------|---------|------|
| Step 1: 特征提取 | 10-15 小时 | GPU 依赖 |
| Step 2: Eta 投影 | 2-3 小时 | |
| Step 4: Style Extractor | 1-2 小时 | |
| Step 5: 风格预计算 | 3-5 小时 | |
| Step 5: 风格聚类 | 10-30 分钟 | |
| **总计** | **~20 小时** | 可过夜运行 |

---

## 8. 风险与应对

### 8.1 内存不足

**风险：** 处理 100k 话语时 OOM

**应对：**
- 使用 FP16
- 分批处理
- 增加 swap 空间

### 8.2 磁盘空间不足

**风险：** 特征文件过大

**应对：**
- 使用压缩（gzip）
- 定期清理中间文件
- 使用外部存储

### 8.3 聚类质量差

**风险：** 256 个簇不够或太多

**应对：**
- 尝试不同的 K 值（128, 256, 512）
- 使用层次聚类
- 评估 Silhouette score

---

## 9. 下一步行动

### 立即执行：

1. **检查磁盘空间**
   ```bash
   df -h /root/autodl-tmp
   ```

2. **修改配置文件**
   ```bash
   # 编辑 configs/base.yaml
   # 设置 max_utterances: null
   ```

3. **启动特征提取**
   ```bash
   nohup python scripts/run_offline.py --step 1 > logs/step1.log 2>&1 &
   ```

### 后续步骤：

4. 创建 `step5_cluster_styles.py`
5. 修改 `test_e2e_anonymization.py` 支持分层检索
6. 运行完整评估

---

## 10. 参考资料

- LibriSpeech 数据集：http://www.openslr.org/12/
- kNN-VC 论文：https://arxiv.org/abs/2305.18975
- FAISS 加速检索：https://github.com/facebookresearch/faiss
