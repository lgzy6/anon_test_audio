#!/bin/bash
set -e

MODE="${1:-same}"  # 默认 same，可选: same, cross, mix, all

DATA_DIR="/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk"
TEST_AUDIO="/root/autodl-tmp/datasets/LibriSpeech/test-clean/61/70970/61-70970-0000.flac"
OUTPUT_DIR="/root/autodl-tmp/anon_test/outputs/global_phonetic_test"

# ========== 统一参数 (V1/V2 共用) ==========
DUR_WEIGHT=0.3
TEMPERATURE=0.1
LAMBDA_ID=0.5
TOP_N=100
TOP_K=8
SRC_GENDER=m

# ========== V2 模块参数 ==========
POOL_RATIO=0.6       # P0: bank 子池采样比例
QUANT_K=8             # P1: 查询量化聚类数
NOISE_SCALE=0.05      # P2: 帧级噪声标准差

COMMON_ARGS="--audio $TEST_AUDIO --src-gender $SRC_GENDER --mode $MODE \
    --dur-weight $DUR_WEIGHT --temperature $TEMPERATURE --lambda-id $LAMBDA_ID \
    --top-n $TOP_N --top-k $TOP_K --output-dir $OUTPUT_DIR"

echo "============================================"
echo "  Global Phonetic Pipeline  (mode=${MODE})"
echo "  统一参数: dur=${DUR_WEIGHT} T=${TEMPERATURE} λ=${LAMBDA_ID}"
echo "============================================"

# ========== Step 0~3: 数据准备 (同之前) ==========

# Step 0: 提取特征
if [ ! -f "${DATA_DIR}/metadata.json" ]; then
    echo "[Step 0] 提取特征..."
    python extract_with_phones_v2.py
else
    echo "[Step 0] 特征已存在，跳过"
fi

# Step 1: 说话人质心
if [ ! -f "${DATA_DIR}/speaker_centroids.npz" ]; then
    echo "[Step 1] 计算说话人质心..."
    python compute_speaker_centroids.py --data-dir $DATA_DIR
else
    echo "[Step 1] 质心已存在，跳过"
fi

# Step 2 & 3: 按模式按需构建
build_if_needed() {
    local gender=$1
    local mask_file bank_file

    if [ "$gender" = "none" ]; then
        mask_file="${DATA_DIR}/identity_mask.h5"
        bank_file="${DATA_DIR}/pseudo_bank_v2.pt"
    else
        mask_file="${DATA_DIR}/identity_mask.gender-${gender}.h5"
        bank_file="${DATA_DIR}/pseudo_bank_v2.gender-${gender}.pt"
    fi

    if [ ! -f "$bank_file" ]; then
        if [ ! -f "$mask_file" ]; then
            echo "  Identity 过滤 (${gender})..."
            python compute_identity_filter.py --data-dir $DATA_DIR --tau 0.80 --gender $gender
        else
            echo "  Identity 掩码已存在 (${gender})，跳过"
        fi
        echo "  构建 Bank (${gender})..."
        python build_bank.py --data-dir $DATA_DIR --l24-clusters 8 --gender $gender
    else
        echo "  Bank 已存在 (${gender})，跳过"
    fi
}

echo "[Step 2&3] 按需构建..."
if [ "$MODE" = "mix" ]; then
    build_if_needed none
elif [ "$MODE" = "same" ] || [ "$MODE" = "cross" ]; then
    build_if_needed m
    build_if_needed f
elif [ "$MODE" = "all" ]; then
    build_if_needed m
    build_if_needed f
    build_if_needed none
fi

# ========== Step 4: 消融实验 ==========
echo ""
echo "============================================"
echo "  Step 4: 消融实验 (V0 → V1 → V2)"
echo "============================================"

# --- 4.0 V0 纯 kNN 基线 ---
echo ""
echo ">>> [1/9] V0 纯 kNN 基线 (无L12打分, 无duration匿名)"
python synthesize_pseudo.py $COMMON_ARGS --version v0

# --- 4.1 V1 联合打分 ---
echo ""
echo ">>> [2/9] V1 联合打分 (+L12打分, +duration匿名)"
python synthesize_pseudo.py $COMMON_ARGS --version v1

# --- 4.2 V2 base (全关 = V1, 一致性校验) ---
echo ""
echo ">>> [3/9] V2 base (全关, 应与V1一致)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p0 --disable-p1 --disable-p2

# --- 4.3~4.5 单模块测试 (定位自然度影响) ---
echo ""
echo ">>> [4/9] V2 + P0 only (随机子池)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p1 --disable-p2 \
    --pool-sample-ratio $POOL_RATIO

echo ""
echo ">>> [5/9] V2 + P1 only (查询量化)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p0 --disable-p2 \
    --query-quant-k $QUANT_K

echo ""
echo ">>> [6/9] V2 + P2 only (帧级噪声)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p0 --disable-p1 \
    --noise-scale $NOISE_SCALE

# --- 4.6~4.8 两两组合 ---
echo ""
echo ">>> [7/9] V2 + P0+P1 (子池+量化)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p2 \
    --pool-sample-ratio $POOL_RATIO --query-quant-k $QUANT_K

echo ""
echo ">>> [8/9] V2 + P0+P2 (子池+噪声)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --disable-p1 \
    --pool-sample-ratio $POOL_RATIO --noise-scale $NOISE_SCALE

echo ""
echo ">>> [9/9] V2 + P0+P1+P2 (全开)"
python synthesize_pseudo.py $COMMON_ARGS --version v2 \
    --pool-sample-ratio $POOL_RATIO --query-quant-k $QUANT_K --noise-scale $NOISE_SCALE

# ========== 汇总 ==========
echo ""
echo "============================================"
echo "  消融实验完成"
echo "============================================"
echo "  输出目录: $OUTPUT_DIR"
echo ""
echo "  生成的音频文件:"
ls -1 ${OUTPUT_DIR}/anon_*.wav 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done
echo ""
echo "  对比建议:"
echo "    1. V0 vs V1 → 确认 L12 打分 + duration 匿名的质量影响"
echo "    2. V1 vs V2_base → 确认一致性"
echo "    3. 逐个听 P0/P1/P2 → 定位自然度损失来源"
echo "    4. 选自然度OK的组合 → 送VPC评估隐私效果"
echo "============================================"
