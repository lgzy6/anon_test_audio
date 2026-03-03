#!/usr/bin/env python3
"""
检查匿名化系统设置
验证所有必需的模型文件和数据路径
"""

from pathlib import Path
import sys

def check_file(path: str, description: str) -> bool:
    """检查文件是否存在"""
    p = Path(path)
    exists = p.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if not exists:
        print(f"      路径: {path}")
    return exists

def check_directory(path: str, description: str) -> bool:
    """检查目录是否存在"""
    p = Path(path)
    exists = p.exists() and p.is_dir()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if not exists:
        print(f"      路径: {path}")
    return exists

def main():
    print("\n" + "="*70)
    print("匿名化系统设置检查")
    print("="*70)

    all_ok = True

    # 检查 LibriSpeech 数据集
    print("\n[1] LibriSpeech 数据集")
    all_ok &= check_directory(
        "/root/autodl-tmp/datasets/LibriSpeech/test-clean",
        "LibriSpeech test-clean"
    )

    # 检查模型文件
    print("\n[2] 模型文件")
    all_ok &= check_file(
        "checkpoints/WavLM-Large.pt",
        "WavLM-Large"
    )
    all_ok &= check_file(
        "data/samm_anon/checkpoints/speaker_subspace.pt",
        "Speaker Subspace (Eta-WavLM)"
    )
    all_ok &= check_file(
        "data/samm_anon/checkpoints/codebook.pt",
        "SAMM Codebook"
    )
    all_ok &= check_file(
        "data/samm_anon/checkpoints/pattern_matrix.pt",
        "Pattern Matrix"
    )
    all_ok &= check_file(
        "checkpoints/phone_decoder.pt",
        "Phone Predictor"
    )
    all_ok &= check_file(
        "checkpoints/duration_decoder.pt",
        "Duration Predictor"
    )

    # 检查 Target Pool
    print("\n[3] Target Pool")
    all_ok &= check_directory(
        "data/samm_anon/checkpoints/target_pool",
        "Target Pool 目录"
    )

    # 检查配置文件
    print("\n[4] 配置文件")
    config_exists = check_file(
        "configs/online_inference_config.yaml",
        "Online Inference Config"
    )
    if not config_exists:
        print("      提示: 配置文件不存在时会使用默认配置")

    # 检查 Vocoder (可选)
    print("\n[5] Vocoder (可选)")
    vocoder_exists = check_file(
        "checkpoints/hifigan.pt",
        "HiFi-GAN Vocoder"
    )
    if not vocoder_exists:
        print("      提示: 没有 vocoder 时只会输出特征文件，不会生成音频")

    # 总结
    print("\n" + "="*70)
    if all_ok:
        print("✓ 所有必需组件已就绪！")
        print("\n可以运行测试:")
        print("  python test_random_speaker.py --max-files 3")
        print("  或")
        print("  ./quick_test.sh")
    else:
        print("✗ 缺少必需组件，请检查上述标记为 ✗ 的项目")
        sys.exit(1)
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
