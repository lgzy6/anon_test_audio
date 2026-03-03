#!/usr/bin/env python3
"""
配置对比测试脚本 - 测试不同配置的语义保留效果
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 测试配置
CONFIGS = {
    "default": {
        "file": "configs/default.yaml",
        "desc": "默认配置 (Eta-WavLM enabled, Layer 15)",
        "expected": "可能失败 (特征空间不匹配)"
    },
    "no_eta": {
        "file": "configs/test_no_eta.yaml",
        "desc": "禁用 Eta-WavLM (匹配 Target Pool)",
        "expected": "应该改善"
    },
    "minimal": {
        "file": "configs/test_minimal.yaml",
        "desc": "最小化配置 (无掩码 + 禁用 Eta)",
        "expected": "最佳语义保留"
    },
    "layer6": {
        "file": "configs/test_layer6.yaml",
        "desc": "Layer 6 + 禁用 Eta",
        "expected": "平衡方案"
    }
}


def run_test(config_name, config_info, audio_path=None):
    """运行单个配置的测试"""
    print("\n" + "="*70)
    print(f"测试配置: {config_name}")
    print("="*70)
    print(f"描述: {config_info['desc']}")
    print(f"预期: {config_info['expected']}")
    print(f"配置文件: {config_info['file']}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"outputs/compare_{config_name}_{timestamp}"

    # 修改 test_random_sample.py 使其支持配置文件参数
    # 或直接使用 Python API
    print(f"\n运行匿名化...")

    # 使用 Python API
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig
import random

# 加载配置
config = AnonymizerConfig.from_yaml('{config_info['file']}')

# 随机选择音频
audio_files = list(Path('/root/autodl-tmp/datasets/LibriSpeech/test-clean').glob('**/*.flac'))
audio_path = str(random.choice(audio_files))

print(f'测试音频: {{audio_path}}')
print(f'配置: use_eta_wavlm={{config.use_eta_wavlm}}, use_top1={{config.use_top1}}')

# 运行匿名化
anonymizer = SpeechAnonymizer(config)
result = anonymizer.anonymize_file(
    input_path=audio_path,
    output_path='{output_dir}/anonymized.wav',
    source_gender='M',
    save_original=True
)
print('✓ 匿名化完成')
"""

    try:
        subprocess.run(['python', '-c', code], check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ 匿名化失败: {e}")
        return None

    # 运行评估
    print(f"\n运行语义评估...")
    try:
        result = subprocess.run(
            ['python', 'evaluate_semantic_preservation.py',
             '--original', f'{output_dir}/original.wav',
             '--anonymized', f'{output_dir}/anonymized.wav',
             '--output', f'{output_dir}/evaluation'],
            capture_output=True,
            text=True,
            check=True
        )

        # 提取 WER
        for line in result.stdout.split('\n'):
            if 'WER' in line and '%' in line:
                print(f"\n{'='*70}")
                print(f"结果: {config_name}")
                print(f"{'='*70}")
                print(line)

    except subprocess.CalledProcessError as e:
        print(f"✗ 评估失败: {e}")
        return None

    return output_dir


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="配置对比测试")
    parser.add_argument("--configs", "-c", nargs='+',
                       choices=list(CONFIGS.keys()) + ['all'],
                       default=['all'],
                       help="要测试的配置")
    parser.add_argument("--audio", "-a", help="指定测试音频")

    args = parser.parse_args()

    # 确定要测试的配置
    if 'all' in args.configs:
        configs_to_test = CONFIGS.keys()
    else:
        configs_to_test = args.configs

    print("="*70)
    print("配置对比测试")
    print("="*70)
    print(f"\n将测试以下配置:")
    for name in configs_to_test:
        print(f"  - {name}: {CONFIGS[name]['desc']}")

    results = {}

    # 运行测试
    for config_name in configs_to_test:
        config_info = CONFIGS[config_name]
        output_dir = run_test(config_name, config_info, args.audio)
        if output_dir:
            results[config_name] = output_dir

    # 汇总结果
    print("\n" + "="*70)
    print("测试完成 - 结果汇总")
    print("="*70)

    for config_name, output_dir in results.items():
        eval_file = Path(output_dir) / "evaluation" / "semantic_evaluation.json"
        if eval_file.exists():
            import json
            with open(eval_file) as f:
                data = json.load(f)
            wer = data['metrics']['wer']
            similarity = data['metrics']['jaccard_similarity']

            print(f"\n{config_name}:")
            print(f"  WER: {wer:.2%}")
            print(f"  相似度: {similarity:.2%}")
            print(f"  输出: {output_dir}")


if __name__ == "__main__":
    main()
