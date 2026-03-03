#!/bin/bash
# 一键测试并评估语义保留

echo "======================================================================"
echo "anon_test v2.1 - 语义保留完整测试"
echo "======================================================================"

cd /root/autodl-tmp/anon_test

# Step 1: 运行匿名化测试
echo -e "\n[Step 1/3] 运行匿名化测试..."
python test_random_sample.py

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: 匿名化测试失败"
    exit 1
fi

# Step 2: 安装 Whisper (如果未安装)
echo -e "\n[Step 2/3] 检查 Whisper 依赖..."
python -c "import whisper" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 openai-whisper..."
    pip install -q openai-whisper
fi

# Step 3: 运行语义评估
echo -e "\n[Step 3/3] 评估语义保留..."
python evaluate_semantic_preservation.py

echo -e "\n======================================================================"
echo "测试完成!"
echo "======================================================================"
