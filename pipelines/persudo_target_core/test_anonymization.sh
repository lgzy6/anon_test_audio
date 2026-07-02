#!/bin/bash
# 测试同性别和跨性别匿名化

source /root/miniconda3/etc/profile.d/conda.sh
conda activate anon_test

INPUT_AUDIO="/root/autodl-tmp/datasets/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
OUTPUT_DIR="/root/autodl-tmp/anon_test/test_outputs"
BANK_DIR="/root/autodl-tmp/anon_test/checkpoints/trainother500_with_phones"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "测试音频匿名化"
echo "输入: $INPUT_AUDIO (男性说话人1089)"
echo "=========================================="

# 1. 同性别匿名化 (男->男)
echo "[1/2] 同性别匿名化 (男->男)..."
python synthesize_pseudo.py \
    --audio "$INPUT_AUDIO" \
    --output "$OUTPUT_DIR/anon_same_gender.wav" \
    --bank "$BANK_DIR/pseudo_bank.gender-m.pt" \
    --k 4 \
    --dur_weight 0.3

# 2. 跨性别匿名化 (男->女)
echo "[2/2] 跨性别匿名化 (男->女)..."
python synthesize_pseudo.py \
    --audio "$INPUT_AUDIO" \
    --output "$OUTPUT_DIR/anon_cross_gender.wav" \
    --bank "$BANK_DIR/pseudo_bank.gender-f.pt" \
    --target-gender f \
    --k 4 \
    --dur_weight 0.3

# 3. 复制原始音频用于对比
echo "复制原始音频..."
cp "$INPUT_AUDIO" "$OUTPUT_DIR/original.flac"

echo "=========================================="
echo "完成！输出文件："
echo "  - original.flac (原始音频)"
echo "  - anon_same_gender.wav (同性别)"
echo "  - anon_cross_gender.wav (跨性别)"
echo "位置: $OUTPUT_DIR"
echo "=========================================="
