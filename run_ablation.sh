#!/bin/bash
# 快速运行消融测试

echo "======================================================================"
echo "逐层消融测试 - 诊断语义丢失阶段"
echo "======================================================================"

cd /root/autodl-tmp/anon_test

# 使用 test_no_eta.yaml 配置（禁用 Eta-WavLM 以匹配 Target Pool）
python ablation_test.py \
  --config configs/test_no_eta.yaml \
  --output outputs/ablation_$(date +%Y%m%d_%H%M%S)

echo ""
echo "======================================================================"
echo "测试完成！查看输出目录中的:"
echo "  - stage*_audio.wav: 每个阶段的合成音频"
echo "  - ablation_report.json: 详细报告"
echo "======================================================================"
