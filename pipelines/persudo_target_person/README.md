# Pseudo Target Person 匿名化流程

基于伪风格 Bank 的语音匿名化系统

## 使用方法

### 1. 构建 Pseudo Bank（如果还没有）

```bash
python pipelines/persudo_target_person/build_bank.py
```

### 2. 批量匿名化

```bash
python pipelines/persudo_target_person/run_anonymization.py \
  --input /path/to/test_set \
  --output outputs/persudo_target \
  --bank checkpoints/pseudo_bank.pt \
  --k 4 \
  --dur_weight 0.5
```

输出结构与输入相同，例如：
```
outputs/persudo_target/
├── speaker1/
│   ├── audio1.wav
│   └── audio2.wav
├── speaker2/
│   └── audio3.wav
├── batch_results.json
└── eval_mapping.json
```

### 3. 评估

```bash
python pipelines/persudo_target_person/evaluate_asv.py \
  --mapping outputs/persudo_target/eval_mapping.json \
  --output outputs/persudo_target/evaluation_results.json
```

## 参数说明

- `--k`: kNN 的 k 值（默认 4）
- `--dur_weight`: 时长预测权重 0-1（默认 0.5）
- `--output`: 输出基础目录（默认 outputs/persudo_target）

