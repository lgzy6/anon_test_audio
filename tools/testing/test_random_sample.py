#!/usr/bin/env python3
"""
从 LibriSpeech test-clean 随机选择音频并运行匿名化测试
"""

import random
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def get_random_audio(dataset_dir="/root/autodl-tmp/datasets/LibriSpeech/test-clean"):
    """从 LibriSpeech test-clean 随机选择一个音频文件"""
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_dir}")
        return None

    # 找到所有 .flac 文件
    audio_files = list(dataset_path.glob("**/*.flac"))

    if not audio_files:
        print(f"错误: 未找到音频文件在 {dataset_dir}")
        return None

    # 随机选择
    selected = random.choice(audio_files)

    print(f"\n随机选择的音频文件:")
    print(f"  路径: {selected}")
    print(f"  说话人ID: {selected.parent.parent.name}")
    print(f"  章节ID: {selected.parent.name}")
    print(f"  文件名: {selected.name}")

    return selected


def run_anonymization_test(audio_path, output_dir=None):
    """运行匿名化测试"""
    import torch
    from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"outputs/test_v2.1_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("匿名化测试 (v2.1)")
    print("="*70)
    print(f"输入音频: {audio_path}")
    print(f"输出目录: {output_dir}")

    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载配置
    print("\n[1/3] 加载匿名化器...")
    config = AnonymizerConfig.from_yaml('configs/default.yaml')
    config.device = device

    print(f"  配置:")
    print(f"    - use_eta_wavlm: {config.use_eta_wavlm}")
    print(f"    - use_top1: {config.use_top1}")
    print(f"    - use_cosine: {config.use_cosine}")
    print(f"    - ssl_layer: {config.ssl_layer}")

    anonymizer = SpeechAnonymizer(config)

    # 执行匿名化
    print("\n[2/3] 执行匿名化...")
    result = anonymizer.anonymize_file(
        input_path=str(audio_path),
        output_path=str(output_path / "anonymized.wav"),
        source_gender='M',  # 可以根据 speaker ID 推断
        save_original=True
    )

    # 输出结果
    print("\n[3/3] 测试完成!")
    print(f"\n生成的文件:")
    for f in sorted(output_path.glob("*.wav")):
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")

    print(f"\n所有结果保存在: {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="随机选择 LibriSpeech 音频并测试匿名化")
    parser.add_argument("--dataset", "-d",
                       default="/root/autodl-tmp/datasets/LibriSpeech/test-clean",
                       help="LibriSpeech test 数据集路径")
    parser.add_argument("--output", "-o", default=None,
                       help="输出目录 (默认: outputs/test_v2.1_TIMESTAMP)")
    parser.add_argument("--audio", "-a", default=None,
                       help="指定音频文件路径 (不随机选择)")
    parser.add_argument("--num-samples", "-n", type=int, default=1,
                       help="测试的样本数量")

    args = parser.parse_args()

    # 如果指定了音频文件
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"错误: 音频文件不存在: {args.audio}")
            sys.exit(1)
        run_anonymization_test(audio_path, args.output)

    # 随机选择
    else:
        for i in range(args.num_samples):
            print(f"\n{'='*70}")
            print(f"样本 {i+1}/{args.num_samples}")
            print('='*70)

            audio_path = get_random_audio(args.dataset)
            if audio_path is None:
                sys.exit(1)

            # 为每个样本创建独立的输出目录
            if args.num_samples > 1:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = f"outputs/test_v2.1_{timestamp}_sample{i+1}"
            else:
                output_dir = args.output

            run_anonymization_test(audio_path, output_dir)

            print("\n" + "="*70 + "\n")
