# Online匿名推理启动指南

## 📋 前置条件检查

### ✅ 已完成的Offline训练模型

根据你的项目状态，以下模型已经训练完成：

```bash
# 检查模型文件
ls -lh data/samm_anon/checkpoints/
```

**必需的模型文件：**
- ✅ `speaker_subspace.pt` (263KB) - Eta-WavLM说话人子空间
- ✅ `codebook.pt` (2.1MB) - SAMM码本
- ✅ `pattern_matrix.pt` (1.1MB) - SAMM模式矩阵
- ✅ `target_pool/` - kNN检索池（包含FAISS索引）

**预训练模型：**
- ✅ `checkpoints/WavLM-Large.pt` (1.2GB)
- ✅ `checkpoints/phone_decoder.pt` (3.9MB)
- ✅ `checkpoints/duration_decoder.pt` (1.6MB)

**⚠️ 缺失的模型：**
- ❌ Vocoder模型 (用于最终音频合成)

---

## 🚀 快速开始

### 方法1：使用现有的SpeechAnonymizer类（推荐）

项目已经实现了完整的online推理流程：`pipelines/online/anonymizer.py`

#### 步骤1：更新配置文件路径

编辑 `configs/online_inference_config.yaml`，确保路径指向正确的模型文件：

```yaml
paths:
  wavlm: "checkpoints/WavLM-Large.pt"
  subspace: "data/samm_anon/checkpoints/speaker_subspace.pt"  # 更新
  codebook: "data/samm_anon/checkpoints/codebook.pt"          # 更新
  pattern_matrix: "data/samm_anon/checkpoints/pattern_matrix.pt"  # 更新
  phone_predictor: "checkpoints/phone_decoder.pt"             # 更新
  duration_predictor: "checkpoints/duration_decoder.pt"       # 更新
  target_pool: "data/samm_anon/checkpoints/target_pool/"      # 更新
  vocoder: "checkpoints/hifigan_knnvc.pt"  # 可选，暂时不可用
```

#### 步骤2：运行匿名推理（不使用vocoder）

```bash
# 测试匿名推理（输出特征，不合成音频）
python pipelines/online/anonymizer.py \
  --input test.wav \
  --output output_anon_features.pt \
  --config configs/online_inference_config.yaml \
  --gender M
```

**注意：** 由于vocoder模型缺失，当前只能输出匿名化特征，无法直接生成音频。

---

## 📝 分步测试方案

### 测试1：验证前两个阶段（SSL + Eta-WavLM）

使用现有的测试脚本：

```bash
python test_online_stage1_2.py
```

**预期输出：**
```
SSL features: torch.Size([1, T, 1024])
Cleaned features: torch.Size([1, T, 1024])
Relative change: X.XX%
```

### 测试2：完整流程测试（不含vocoder）

创建一个新的测试脚本来验证完整流程。

---

## 🔧 解决Vocoder缺失问题

### 选项1：使用kNN-VC的HiFi-GAN（推荐）

如果你有kNN-VC项目的vocoder模型：

```bash
# 从kNN-VC项目复制vocoder
cp /path/to/knn-vc/checkpoints/hifigan.pt checkpoints/hifigan_knnvc.pt
```

### 选项2：使用通用Vocoder

可以使用其他预训练的vocoder，如：
- BigVGAN
- HiFi-GAN (通用版本)
- WaveGlow

### 选项3：暂时跳过vocoder

如果只需要验证匿名化效果，可以：
1. 输出匿名化特征
2. 使用外部vocoder进行合成
3. 或者直接评估特征空间的匿名化效果

---

## 📊 验证匿名化效果

### 方法1：特征空间验证

```python
import torch
import numpy as np

# 加载原始特征和匿名化特征
h_original = np.load("outputs/stage1_ssl_features.npy")
h_anon = np.load("outputs/stage4_anon_features.npy")

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(
    h_original.reshape(-1, 1024),
    h_anon.reshape(-1, 1024)
).diagonal().mean()

print(f"平均余弦相似度: {similarity:.4f}")
# 期望值：0.3-0.6（太高=匿名化不足，太低=内容丢失）
```

### 方法2：说话人验证

如果有说话人识别模型，可以验证：
- 原始音频 vs 匿名化音频的说话人相似度（应该很低）
- 匿名化音频的内容保留度（ASR准确率）

---

## 🎯 完整的Online推理流程

### 流程图

```
输入音频 (16kHz)
    ↓
[Stage 1] WavLM特征提取 → h [T, 1024]
    ↓
[Stage 2] Eta-WavLM投影 → h_clean [T, 1024]
    ↓
[Stage 3.1] SAMM符号分配 → z [T]
    ↓
[Stage 3.1'] Phone预测 → phones [T]
    ↓
[Stage 3.2] 音素时长提取 → durations
    ↓
[Stage 3.3] 时长匿名化 → anon_durations
    ↓
[Stage 3.4] 符号掩码 → z_masked
    ↓
[Stage 3.5] 模式正则化 → z_smooth
    ↓
[Stage 4.1] 约束kNN检索 → h_anon [T, 1024]
    ↓
[Stage 4.2] 时长调整 → h_anon_adjusted [T', 1024]
    ↓
[Stage 5] Vocoder合成 → 匿名化音频 (可选)
```

### 各阶段的作用

1. **SSL特征提取**: 提取语音的高层表示
2. **Eta-WavLM**: 去除说话人信息，保留内容
3. **SAMM符号化**: 将特征量化为离散符号
4. **Phone预测**: 提取音素信息用于约束检索
5. **时长匿名化**: 修改音素时长以增强隐私
6. **符号掩码**: 对符号序列进行掩码处理
7. **模式正则化**: 平滑符号序列
8. **约束kNN检索**: 从目标池中检索匹配的特征
9. **时长调整**: 根据匿名化时长调整特征序列
10. **Vocoder合成**: 将特征转换为音频波形

---

## 💡 常见问题

### Q1: 为什么需要vocoder？

A: Vocoder将匿名化的特征（[T, 1024]）转换为可播放的音频波形。没有vocoder时，只能输出特征文件。

### Q2: 可以使用其他vocoder吗？

A: 可以，但需要确保：
- Vocoder训练时使用的特征类型与WavLM Layer 15特征兼容
- 或者需要添加特征转换层

### Q3: 如何评估匿名化效果？

A: 主要评估指标：
- **隐私保护**: EER (Equal Error Rate) - 说话人验证错误率
- **内容保留**: WER (Word Error Rate) - ASR识别准确率
- **音质**: MOS (Mean Opinion Score) - 主观评分

### Q4: 推理速度慢怎么办？

A: 优化建议：
- 使用GPU加速
- 批量处理多个音频
- 减少kNN的k值
- 使用更小的target pool

---

## 📚 下一步

1. **测试完整流程**: 运行测试脚本验证各阶段
2. **准备vocoder**: 获取或训练vocoder模型
3. **批量处理**: 处理测试集进行评估
4. **性能优化**: 根据需求优化推理速度

---

## 🔗 相关文档

- `ONLINE_INFERENCE_ANALYSIS.md` - Online推理实现分析
- `CALL_FLOW_GUIDE.md` - 调用流程指南
- `POOL_BUILDING_ANALYSIS.md` - Target Pool构建分析
