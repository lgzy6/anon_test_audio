#!/usr/bin/env python3
"""
随机说话人匿名化测试脚本
从 LibriSpeech test-clean 随机选择一个说话人进行匿名化测试
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import sys
import random
from datetime import datetime
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig


def get_random_speaker(librispeech_path: str, seed: int = None):
    """
    从 LibriSpeech test-clean 随机选择一个说话人

    Args:
        librispeech_path: LibriSpeech test-clean 路径
        seed: 随机种子

    Returns:
        speaker_id: 说话人ID
        audio_files: 该说话人的所有音频文件列表
    """
    librispeech_path = Path(librispeech_path)

    if not librispeech_path.exists():
        raise FileNotFoundError(f"LibriSpeech路径不存在: {librispeech_path}")

    # 获取所有说话人目录
    speaker_dirs = [d for d in librispeech_path.iterdir() if d.is_dir()]

    if not speaker_dirs:
        raise ValueError(f"未找到说话人目录: {librispeech_path}")

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 随机选择一个说话人
    speaker_dir = random.choice(speaker_dirs)
    speaker_id = speaker_dir.name

    # 获取该说话人的所有音频文件
    audio_files = []
    for chapter_dir in speaker_dir.iterdir():
        if chapter_dir.is_dir():
            audio_files.extend(list(chapter_dir.glob("*.flac")))

    return speaker_id, audio_files


def create_output_directory(base_dir: str = "outputs"):
    """
    创建以时间戳命名的输出目录

    Args:
        base_dir: 基础输出目录

    Returns:
        output_dir: 创建的输出目录路径
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def anonymize_speaker_files(
    audio_files: list,
    anonymizer: SpeechAnonymizer,
    output_dir: Path,
    source_gender: str = None,
    max_files: int = None,
):
    """
    对说话人的音频文件进行匿名化处理

    Args:
        audio_files: 音频文件列表
        anonymizer: 匿名化器实例
        output_dir: 输出目录
        source_gender: 源说话人性别 ('M' 或 'F')
        max_files: 最大处理文件数（None表示处理所有）

    Returns:
        results: 处理结果字典
    """
    # 创建子目录
    anon_audio_dir = output_dir / "anonymized_audio"
    anon_audio_dir.mkdir(exist_ok=True)

    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    # 限制处理文件数
    if max_files is not None:
        audio_files = audio_files[:max_files]

    results = {
        "total_files": len(audio_files),
        "processed_files": 0,
        "failed_files": 0,
        "file_results": []
    }

    print(f"\n{'='*70}")
    print(f"开始处理 {len(audio_files)} 个音频文件")
    print(f"{'='*70}\n")

    for idx, audio_file in enumerate(audio_files, 1):
        try:
            print(f"[{idx}/{len(audio_files)}] 处理: {audio_file.name}")

            # 生成输出文件名
            output_filename = audio_file.stem + "_anon.wav"
            output_path = anon_audio_dir / output_filename

            # 执行匿名化
            result = anonymizer.anonymize_file(
                input_path=str(audio_file),
                output_path=str(output_path),
                source_gender=source_gender,
            )

            # 保存特征（可选）
            if 'features' in result:
                feature_path = features_dir / f"{audio_file.stem}_features.npy"
                np.save(feature_path, result['features'].numpy())

            results["processed_files"] += 1
            results["file_results"].append({
                "input": str(audio_file),
                "output": str(output_path),
                "status": "success"
            })

            print(f"  ✓ 完成\n")

        except Exception as e:
            import traceback
            print(f"  ✗ 失败: {e}")
            traceback.print_exc()
            print()
            results["failed_files"] += 1
            results["file_results"].append({
                "input": str(audio_file),
                "status": "failed",
                "error": str(e)
            })

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从 LibriSpeech test-clean 随机选择说话人进行匿名化测试"
    )
    parser.add_argument(
        "--librispeech-path",
        "-l",
        default="/root/autodl-tmp/datasets/LibriSpeech/test-clean",
        help="LibriSpeech test-clean 路径"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/online_inference_config.yaml",
        help="匿名化配置文件路径"
    )
    parser.add_argument(
        "--output-base",
        "-o",
        default="outputs",
        help="输出基础目录"
    )
    parser.add_argument(
        "--gender",
        "-g",
        choices=['M', 'F'],
        default=None,
        help="源说话人性别"
    )
    parser.add_argument(
        "--target-gender",
        "-t",
        choices=['same', 'cross', 'M', 'F'],
        default='same',
        help="目标性别选择策略"
    )
    parser.add_argument(
        "--max-files",
        "-m",
        type=int,
        default=None,
        help="最大处理文件数（默认处理所有）"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="设备 (cuda/cpu)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("随机说话人匿名化测试")
    print("="*70)

    # Step 1: 随机选择说话人
    print("\n[Step 1] 随机选择说话人...")
    try:
        speaker_id, audio_files = get_random_speaker(
            args.librispeech_path,
            seed=args.seed
        )
        print(f"  ✓ 选中说话人: {speaker_id}")
        print(f"  ✓ 音频文件数: {len(audio_files)}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return

    # Step 2: 创建输出目录
    print("\n[Step 2] 创建输出目录...")
    output_dir = create_output_directory(args.output_base)
    print(f"  ✓ 输出目录: {output_dir}")

    # 保存元信息
    metadata = {
        "speaker_id": speaker_id,
        "total_files": len(audio_files),
        "source_gender": args.gender,
        "target_gender": args.target_gender,
        "max_files": args.max_files,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "librispeech_path": args.librispeech_path,
        "config": args.config,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ 元信息已保存")

    # Step 3: 加载匿名化器
    print("\n[Step 3] 加载匿名化器...")
    try:
        # 检查配置文件是否存在
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"  ⚠ 配置文件不存在: {config_path}")
            print(f"  使用默认配置...")
            config = AnonymizerConfig(device=args.device)
            config.target_gender = args.target_gender
        else:
            config = AnonymizerConfig.from_yaml(str(config_path))
            config.device = args.device
            config.target_gender = args.target_gender

        anonymizer = SpeechAnonymizer(config)
        print(f"  ✓ 匿名化器加载完成")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: 执行匿名化
    print("\n[Step 4] 执行匿名化...")
    results = anonymize_speaker_files(
        audio_files=audio_files,
        anonymizer=anonymizer,
        output_dir=output_dir,
        source_gender=args.gender,
        max_files=args.max_files,
    )

    # Step 5: 保存结果摘要
    print("\n[Step 5] 保存结果...")
    results_summary = {
        **metadata,
        **results
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  ✓ 结果已保存")

    # 打印总结
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"\n说话人ID: {speaker_id}")
    print(f"总文件数: {results['total_files']}")
    print(f"成功处理: {results['processed_files']}")
    print(f"失败文件: {results['failed_files']}")
    print(f"\n输出目录: {output_dir}")
    print(f"  - anonymized_audio/  # 匿名化音频")
    print(f"  - features/          # 匿名化特征")
    print(f"  - metadata.json      # 元信息")
    print(f"  - results.json       # 处理结果")


if __name__ == "__main__":
    main()
