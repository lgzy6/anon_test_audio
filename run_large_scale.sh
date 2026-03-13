#!/bin/bash
# =============================================================================
# 大规模数据集完整流程脚本
# =============================================================================
# 用途: 从小规模(43说话人)切换到大规模数据集
# 流程: run_offline step1 → segment step1-5 → 去匿名生成
# =============================================================================

set -e  # 遇到错误立即退出

CONFIG="configs/large_scale.yaml"
BASE_DIR="/root/autodl-tmp/anon_test"

echo "=========================================="
echo "大规模数据集训练流程"
echo "配置文件: $CONFIG"
echo "=========================================="

# Step 0: 创建必要的目录
echo ""
echo "[Step 0] 创建输出目录..."
mkdir -p cache/large/features/wavlm
mkdir -p checkpoints/large
mkdir -p outputs/large
mkdir -p logs/large

# Step 1: run_offline 特征提取
echo ""
echo "[run_offline Step 1] WavLM 特征提取..."
python scripts/run_offline.py --config $CONFIG --step 1

# Step 2: segment step1 - 提取说话人嵌入
echo ""
echo "[Segment Step 1] 提取说话人嵌入..."
python scripts/step1_extract_speaker_embeddings.py --config $CONFIG

# Step 3: segment step2 - 计算 Eta 投影
echo ""
echo "[Segment Step 2] 计算 Eta 投影..."
python scripts/step2_compute_eta_projection.py --config $CONFIG

# Step 4: segment step4 - 构建风格提取器
echo ""
echo "[Segment Step 4] 构建风格提取器..."
python scripts/step4_build_style_extractor.py --config $CONFIG

# Step 5: segment step5 - 预计算话语风格
echo ""
echo "[Segment Step 5] 预计算话语风格..."
python scripts/step5_precompute_utterance_styles.py --config $CONFIG

# Step 6: segment step5 - 构建 phone clusters
echo ""
echo "[Segment Step 5b] 构建 phone clusters..."
python scripts/step5_build_phone_clusters.py --config $CONFIG

echo ""
echo "=========================================="
echo "✓ 大规模数据集训练完成！"
echo "权重保存在: checkpoints/large/"
echo "=========================================="
