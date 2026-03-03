#!/bin/bash
# 快速测试脚本 - 使用禁用 Eta-WavLM 的配置

echo "======================================================================"
echo "快速验证测试 - 禁用 Eta-WavLM"
echo "======================================================================"

cd /root/autodl-tmp/anon_test

# 使用 test_no_eta.yaml 配置
CONFIG="configs/test_no_eta.yaml"
OUTPUT="outputs/quick_verify_$(date +%Y%m%d_%H%M%S)"

echo -e "\n使用配置: $CONFIG"
echo "输出目录: $OUTPUT"

# 运行匿名化
echo -e "\n[1/2] 运行匿名化..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig
import random

# 加载配置
config = AnonymizerConfig.from_yaml('$CONFIG')
print(f'配置: use_eta_wavlm={config.use_eta_wavlm}, use_top1={config.use_top1}, use_cosine={config.use_cosine}')

# 随机选择音频
audio_files = list(Path('/root/autodl-tmp/datasets/LibriSpeech/test-clean').glob('**/*.flac'))
audio_path = str(random.choice(audio_files))
print(f'测试音频: {audio_path}')

# 运行匿名化
anonymizer = SpeechAnonymizer(config)
result = anonymizer.anonymize_file(
    input_path=audio_path,
    output_path='$OUTPUT/anonymized.wav',
    source_gender='M',
    save_original=True
)
print('✓ 完成')
"

if [ $? -ne 0 ]; then
    echo "✗ 匿名化失败"
    exit 1
fi

# 运行评估
echo -e "\n[2/2] 运行语义评估..."
python evaluate_semantic_preservation.py \
  --original "$OUTPUT/original.wav" \
  --anonymized "$OUTPUT/anonymized.wav" \
  --output "$OUTPUT/evaluation"

echo -e "\n======================================================================"
echo "测试完成!"
echo "======================================================================"
echo "结果保存在: $OUTPUT"
