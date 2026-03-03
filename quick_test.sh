#!/bin/bash
# 快速测试脚本 - 随机选择说话人并进行匿名化测试

echo "========================================"
echo "随机说话人匿名化快速测试"
echo "========================================"
echo ""

# 默认参数
MAX_FILES=3
SEED=42
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./quick_test.sh [--max-files N] [--seed N] [--device cuda|cpu]"
            exit 1
            ;;
    esac
done

echo "配置:"
echo "  - 最大文件数: $MAX_FILES"
echo "  - 随机种子: $SEED"
echo "  - 设备: $DEVICE"
echo ""

# 运行测试
python test_random_speaker.py \
    --max-files $MAX_FILES \
    --seed $SEED \
    --device $DEVICE \
    --target-gender same

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
