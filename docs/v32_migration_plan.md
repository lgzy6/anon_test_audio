# DS-SAMM-Anon v3.2 迁移方案

> **创建时间**: 2026-01-27
> **状态**: 待实施

---

## 一、架构对比

### 1.1 核心差异

| 维度 | v3.0 (当前) | v3.2 (目标) |
|------|-------------|-------------|
| **解耦目标** | 去除说话人 | 去除内容 |
| **投影基底** | 说话人中心 U_s | 音素中心 U_c |
| **SAMM 输入** | H_clean (含内容) | H_style (纯风格) |
| **检索 Query** | 被 Masking 污染 | 原始 H (不变) |
| **目标池大小** | 213GB | ~100MB |

### 1.2 数据流对比

**v3.0 流程 (有问题)**:
```
H → Eta投影 → H_clean → SAMM Masking → 检索 → Vocoder
                              ↓
                        Query 被污染 → WER 爆炸
```

**v3.2 流程 (双流分离)**:
```
        ┌→ Phone Predictor → Phone IDs (语义锚点，不变)
        │
H_wavlm ┤
        │
        └→ 正交投影 → H_style → SAMM → Pattern ID (风格导航)
                                           ↓
                              选择 Target Pool → 1-NN 检索 → Vocoder
```

---

## 二、模块修改清单

### 2.1 需要新增的模块

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `models/disentangle/content_projector.py` | 内容子空间投影 | P0 |
| `models/samm/pattern_classifier.py` | Pattern 分类器 | P1 |
| `pipelines/offline/pool_building_v32.py` | 量化池构建 | P0 |
| `models/knn_vc/retriever_v32.py` | 中心检索器 | P0 |
| `models/knn_vc/duration_reset.py` | 时长重置 | P1 |

### 2.2 需要修改的模块

| 文件 | 修改内容 |
|------|----------|
| `pipelines/online/anonymizer.py` | 重构为双流架构 |
| `pipelines/offline/runner.py` | 添加 v3.2 步骤 |
| `configs/base.yaml` | 添加 v3.2 配置项 |

### 2.3 可完全复用的模块

- `models/ssl/` - WavLM 特征提取
- `models/phone_predictor/` - 音素预测
- `models/vocoder/` - HiFi-GAN
- `models/samm/codebook.py` - VQ 量化逻辑

---

## 三、实施路线

### Phase 1: 验证解耦可行性 (1-2天)

**目标**: 证明"去除内容"的投影是有效的

**测试脚本**: `tests/test_v32_disentanglement.py`

**成功标准**:
- H_style 上的音素分类准确率 < H_original 的 50%

### Phase 2: 构建量化池 (2-3天)

**目标**: 实现 1000x 压缩的目标池

**关键参数**:
- n_patterns: 8-16
- centers_per_phone: 64
- 预计存储: ~100MB

### Phase 3: 端到端集成 (3-5天)

**目标**: 完整流程跑通

**评估指标**:
- WER < 15% (语义保留)
- EER > 15% (隐私保护)
