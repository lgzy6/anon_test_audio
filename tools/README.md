# Tools 工具集

本目录包含 SAMM-Anon 项目的各类辅助工具。

## 目录结构

```
tools/
├── diagnostic/          # 诊断工具
├── testing/             # 测试工具
├── comparison/          # 对比分析工具
└── utils/              # 其他实用工具
```

---

## 1. diagnostic/ - 诊断工具

### diagnose_system.py
系统诊断工具，检查配置、模型、Target Pool 状态。

**用法:**
```bash
python tools/diagnostic/diagnose_system.py --config configs/base.yaml
```

**功能:**
- 检查配置文件正确性
- 验证 Target Pool 元数据
- 检查 Phone Clusters 状态
- 诊断 kNN 检索器行为

---

### inspect_pools.py
Target Pool 版本信息查看工具。

**用法:**
```bash
python tools/diagnostic/inspect_pools.py
```

**功能:**
- 列出所有 Target Pool 版本
- 显示每个版本的元数据、特征数、大小
- 检查 Phone Clusters 和索引文件

---

### check_setup.py
环境依赖检查工具。

**用法:**
```bash
python tools/diagnostic/check_setup.py
```

**功能:**
- 验证 Python 依赖
- 检查 CUDA/GPU 可用性
- 验证预训练模型路径

---

## 2. testing/ - 测试工具

### test_random_sample.py
使用随机 LibriSpeech 音频测试匿名化。

**用法:**
```bash
python tools/testing/test_random_sample.py --config configs/test.yaml
```

**功能:**
- 从 LibriSpeech test-clean 随机选择音频
- 执行匿名化
- 生成匿名音频

---

### test_random_speaker.py
测试随机说话人匿名化。

**用法:**
```bash
python tools/testing/test_random_speaker.py --config configs/test.yaml --num-samples 5
```

---

### test_online_inference.py
在线推理流程测试。

**用法:**
```bash
python tools/testing/test_online_inference.py --audio <path> --config configs/base.yaml
```

---

### evaluate_semantic_preservation.py
语义保留评估工具 (使用 Whisper ASR 计算 WER)。

**用法:**
```bash
python tools/testing/evaluate_semantic_preservation.py \
  --original <original.wav> \
  --anonymized <anonymized.wav> \
  --output <output_dir>
```

**功能:**
- 使用 Whisper 转录原始和匿名音频
- 计算 WER (Word Error Rate)
- 计算 Jaccard 相似度
- 生成详细报告

---

## 3. comparison/ - 对比分析工具

### ablation_test.py
逐层消融测试，诊断语义丢失的具体阶段。

**用法:**
```bash
python tools/comparison/ablation_test.py \
  --config configs/test.yaml \
  --audio <test_audio.wav> \
  --output outputs/ablation
```

**功能:**
- Stage 1: WavLM 特征提取
- Stage 2: Eta-WavLM 投影
- Stage 4: kNN 检索
- Stage 5: Duration 调整
- 每个阶段生成音频 + 计算 WER

---

### compare_pool_versions.py
对比不同 Target Pool 版本的隐私与效用。

**用法:**
```bash
python tools/comparison/compare_pool_versions.py \
  --config configs/test.yaml \
  --output outputs/pool_comparison
```

**功能:**
- 自动发现所有 Target Pool 版本
- 对每个版本执行匿名化测试
- 计算 WER 和语义保留指标
- 生成对比报告

---

### compare_configs.py
对比不同配置文件的参数差异。

**用法:**
```bash
python tools/comparison/compare_configs.py configs/base.yaml configs/production.yaml
```

---

## 4. utils/ - 实用工具

### run_online_inference.py
在线推理执行脚本。

**用法:**
```bash
python tools/utils/run_online_inference.py \
  --input <audio.wav> \
  --output <output.wav> \
  --config configs/base.yaml
```

---

### check_duration_ckpt.py
检查 Duration Predictor 模型。

**用法:**
```bash
python tools/utils/check_duration_ckpt.py --ckpt checkpoints/duration_decoder.pt
```

---

## 常用命令速查

```bash
# 1. 快速诊断系统
python tools/diagnostic/diagnose_system.py --config configs/base.yaml

# 2. 查看所有 Target Pool 版本
python tools/diagnostic/inspect_pools.py

# 3. 测试随机音频
python tools/testing/test_random_sample.py --config configs/test.yaml

# 4. 评估语义保留
python tools/testing/evaluate_semantic_preservation.py \
  --original outputs/test/original.wav \
  --anonymized outputs/test/anonymized.wav \
  --output outputs/evaluation

# 5. 消融测试
python tools/comparison/ablation_test.py \
  --config configs/test.yaml \
  --output outputs/ablation

# 6. Target Pool 对比
python tools/comparison/compare_pool_versions.py \
  --config configs/test.yaml \
  --output outputs/pool_comparison
```

---

## 注意事项

1. 所有工具都需要在项目根目录运行
2. 大多数工具需要指定配置文件 `--config`
3. 评估工具需要 Whisper 模型 (自动下载)
4. 对比工具会生成大量输出，建议使用专用输出目录
