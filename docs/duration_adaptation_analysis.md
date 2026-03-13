# Duration 适配分析与实现方案

## 问题诊断

### 当前框架的缺陷

**test_e2e_anonymization_v2.py (第183-200行):**
```python
# 问题：直接帧对帧匹配，忽略时长信息
for phone_id in unique_phones:
    src_mask = (source_phones == phone_id)
    tgt_clusters = phone_clusters[phone_id]
    # 直接对每帧独立进行 kNN
    h_anon[src_mask] = tgt_clusters[topk_indices].mean(dim=1)
```

**导致的问题：**
1. 源音频和目标音频的相同音素可能有不同时长
2. 直接替换导致韵律失配和语义丢失
3. WER 117.65% 表明严重的语义崩溃

---

## 参考实现分析

### spkanon converter.py 的关键机制

**核心流程 (第237-287行):**

```python
# 1. 获取连续音素段及其实际时长
unique, counts = torch.unique_consecutive(phones, return_counts=True)

# 2. 预测合理时长
pred_durations = duration_predictor(phones)

# 3. 插值实际时长和预测时长
interpolated_durations = dur_w * pred_durations + (1 - dur_w) * phone_durations

# 4. 通过线性插值调整帧数
feat_indices = torch.linspace(start, end, n_frames)
adjusted_feats = src_feats[feat_indices]

# 5. 在调整后的特征上进行 kNN 匹配
```

**关键参数：**
- `dur_w`: duration 预测权重 (0.0-1.0)
  - 0.0: 完全保持原始时长
  - 1.0: 完全使用预测时长
  - 0.5: 折中方案（推荐）

---

## 实现方案

### 新增文件

**scripts/test_e2e_with_duration.py**

### 核心修改

**1. 加载 duration predictor:**
```python
duration_predictor = PhonePredictor.load(
    str(ckpt_dir / 'duration_decoder.pt'),
    device=device
)
```

**2. 音素分段:**
```python
unique_phones, phone_durations = [], []
for phone in source_phones:
    # 统计连续相同音素的时长
```

**3. 时长插值:**
```python
pred_durations = duration_predictor(phones_tensor)
interpolated_durations = dur_w * pred_durations + (1 - dur_w) * actual_durations
```

**4. 特征调整:**
```python
indices = torch.linspace(start, end, new_duration)
adjusted_feats = src_feats[indices]
```

**5. kNN 匹配:**
```python
# 在调整后的特征上进行匹配
h_anon = knn_match(adjusted_feats, target_clusters)
```

---

## 使用方法

### 基本用法

```bash
python scripts/test_e2e_with_duration.py \
  --audio <源音频> \
  --output <输出路径> \
  --gender {f|m} \
  --k 4 \
  --dur_weight 0.5
```

### 参数说明

- `--dur_weight`: Duration 预测权重
  - `0.0`: 保持原始时长（无适配）
  - `0.5`: 折中方案（推荐）
  - `1.0`: 完全使用预测时长

### 对比测试

```bash
# 原版（无 duration 适配）
python scripts/test_e2e_anonymization_v2.py \
  --audio test.flac --output out_v2.wav --gender m --k 4

# 新版（带 duration 适配）
python scripts/test_e2e_with_duration.py \
  --audio test.flac --output out_dur.wav --gender m --k 4 --dur_weight 0.5
```

---

## 预期改进

### 理论优势

1. **韵律保持**: 通过时长预测保持自然韵律
2. **语义对齐**: 调整后的特征与目标更匹配
3. **减少失真**: 避免直接帧替换导致的不连续

### 需要验证的指标

- WER (Word Error Rate)
- 音频质量主观评分
- 时长保持率

---

## 技术细节

### Duration Predictor 架构

- 输入: Phone 序列 (batch_size, seq_len)
- 输出: 每个 phone 的预测帧数 (batch_size, seq_len)
- 权重: checkpoints/duration_decoder.pt

### 线性插值原理

```python
# 原始: 10 帧 -> 目标: 15 帧
indices = [0.0, 0.7, 1.4, 2.1, ..., 9.0]  # 15 个点
adjusted = src_feats[indices.long()]       # 取整索引
```

---

## 下一步优化

1. **测试不同 dur_weight 值** (0.0, 0.3, 0.5, 0.7, 1.0)
2. **评估语义保留效果** (WER 对比)
3. **调整 k 值** (1, 2, 4) 与 duration 的组合效果
4. **考虑音素级别的自适应权重**
