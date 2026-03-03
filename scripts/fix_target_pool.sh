#!/bin/bash
# 修复 Target Pool 的完整流程

set -e

echo "=========================================="
echo "修复 Target Pool"
echo "=========================================="

cd /root/autodl-tmp/anon_test

# Step 1: 生成 Phone 预测
echo ""
echo "[Step 1] 生成 Phone 预测..."
/root/miniconda3/envs/anon_test/bin/python scripts/generate_phone_predictions.py

# Step 2: 重建 Target Pool
echo ""
echo "[Step 2] 重建 Target Pool..."
/root/miniconda3/envs/anon_test/bin/python scripts/rebuild_target_pool.py

# Step 3: 更新符号链接
echo ""
echo "[Step 3] 更新 Pool 路径..."
POOL_DIR="data/samm_anon/checkpoints/target_pool"
FIXED_DIR="data/samm_anon/checkpoints/target_pool_fixed"

if [ -d "$FIXED_DIR" ]; then
    # 备份旧的
    if [ -d "$POOL_DIR" ] && [ ! -L "$POOL_DIR" ]; then
        mv "$POOL_DIR" "${POOL_DIR}_broken"
        echo "  旧 Pool 已备份到 ${POOL_DIR}_broken"
    fi

    # 创建符号链接
    ln -sfn "$(basename $FIXED_DIR)" "$POOL_DIR"
    echo "  ✓ Pool 路径已更新"
fi

echo ""
echo "=========================================="
echo "修复完成!"
echo "=========================================="
