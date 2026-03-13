# Style Segments Pipeline - 迁移总结

## 完成的工作

### 1. 创建新的管线目录
```
pipelines/stylesegments/
├── __init__.py              # 模块导出
├── feature_extraction.py    # WavLM多层特征提取 (优化)
├── speaker_embeddings.py    # 说话人嵌入+PCA
├── eta_projection.py        # Eta-WavLM投影
├── phone_precompute.py      # 音素预计算
├── style_extractor.py       # 段级风格提取器
├── phone_clusters.py        # Phone聚类
├── runner.py                # 统一运行脚本
├── verify.py                # 验证脚本
└── README.md                # 使用文档
```

### 2. 核心优化

**多层提取支持**
- 同时提取 WavLM Layer 6 和 Layer 24
- 每层独立存储: `cache/features/wavlm/{split}/layer_{6,24}/`
- 每层独立处理: `checkpoints/layer_{6,24}/`

**代码精简**
- 总代码量: ~1020行 (原来分散在多个文件)
- 移除冗余代码，保留核心功能
- 统一接口，便于维护

### 3. 配置更新

`configs/base.yaml` 新增:
```yaml
ssl:
  layers: [6, 24]  # 多层提取配置
```

### 4. 使用方式

**完整管线**
```bash
cd /root/autodl-tmp/anon_test
python pipelines/stylesegments/runner.py --config configs/base.yaml
```

**指定步骤**
```bash
# 只运行特征提取和说话人嵌入
python pipelines/stylesegments/runner.py --steps 1,2

# 使用10%数据快速测试
python pipelines/stylesegments/runner.py --sample-ratio 0.1
```

### 5. 输出结构

```
cache/features/wavlm/{split}/
├── layer_6/
│   └── features.h5          # Layer 6 特征
├── layer_24/
│   └── features.h5          # Layer 24 特征
└── metadata.json            # 共享元数据

checkpoints/
├── speaker_embeddings_pca.npy
├── speaker_ids.npy
├── utt_indices.npy
├── layer_6/
│   ├── phones.npy
│   ├── eta_projection.pt
│   ├── style_extractor.pkl
│   └── phone_clusters.pkl
└── layer_24/
    ├── phones.npy
    ├── eta_projection.pt
    ├── style_extractor.pkl
    └── phone_clusters.pkl
```

## 与原实现的对应关系

| 原文件 | 新模块 |
|--------|--------|
| `offline/feature_extraction.py` | `stylesegments/feature_extraction.py` |
| `scripts/step1_extract_speaker_embeddings.py` | `stylesegments/speaker_embeddings.py` |
| `scripts/step2_compute_eta_projection.py` | `stylesegments/eta_projection.py` |
| `scripts/step3_5_precompute_phones.py` | `stylesegments/phone_precompute.py` |
| `scripts/step4_build_style_extractor.py` | `stylesegments/style_extractor.py` |
| `scripts/step5_build_phone_clusters.py` | `stylesegments/phone_clusters.py` |
| 无 | `stylesegments/runner.py` (新增统一入口) |

## 下一步

原有的 `offline/` 和 `scripts/step*.py` 文件可以保留作为参考，或在确认新管线工作正常后删除。
