#!/bin/bash
# 批量构建男女独立的 Bank

DATA_DIR="/root/autodl-tmp/anon_test/checkpoints/trainother500_with_phones"
CONDA_ENV="anon_test"

echo "=========================================="
echo "批量构建性别独立 Bank"
echo "=========================================="

# 激活环境
conda activate $CONDA_ENV

# 1. 计算男性混合熵
echo "[1/4] 计算男性混合熵..."
python compute_entropy.py \
    --data-dir "$DATA_DIR" \
    --gender m \
    --temperature 10.0 \
    --min-speakers 3

# 2. 计算女性混合熵
echo "[2/4] 计算女性混合熵..."
python compute_entropy.py \
    --data-dir "$DATA_DIR" \
    --gender f \
    --temperature 10.0 \
    --min-speakers 3

# 3. 构建男性 Bank
echo "[3/4] 构建男性 Bank..."
python build_bank.py \
    --data-dir "$DATA_DIR" \
    --gender m \
    --clusters 50 \
    --frames-per-cluster 20 \
    --entropy-percentile 75

# 4. 构建女性 Bank
echo "[4/4] 构建女性 Bank..."
python build_bank.py \
    --data-dir "$DATA_DIR" \
    --gender f \
    --clusters 50 \
    --frames-per-cluster 20 \
    --entropy-percentile 75

echo "=========================================="
echo "完成！生成文件："
echo "  - entropies.gender-m.h5"
echo "  - entropies.gender-f.h5"
echo "  - pseudo_bank.gender-m.pt"
echo "  - pseudo_bank.gender-f.pt"
echo "=========================================="
