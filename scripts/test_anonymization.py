#!/usr/bin/env python3
"""
匿名化测试脚本 - 测试匿名效果并验证语义保留
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_anonymization(audio_path: str, output_dir: str = None):
    """运行匿名化测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if output_dir is None:
        output_dir = f"outputs/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("匿名化测试")
    print("=" * 60)
    print(f"输入音频: {audio_path}")
    print(f"输出目录: {output_dir}")

    # 1. 加载匿名化器
    print("\n[1/4] 加载匿名化器...")
    from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig

    config = AnonymizerConfig(device=device)
    anonymizer = SpeechAnonymizer(config)

    # 2. 执行匿名化
    print("\n[2/4] 执行匿名化...")
    result = anonymizer.anonymize_file(
        input_path=audio_path,
        output_path=str(output_path / "anonymized.wav"),
        source_gender='M',
        save_original=True
    )

    # 3. 语义诊断
    print("\n[3/4] 语义诊断...")
    diagnose_semantics(audio_path, result, output_path, device)

    # 4. 输出结果
    print("\n[4/4] 测试完成!")
    print(f"  原始音频: {output_path / 'original.wav'}")
    print(f"  匿名音频: {output_path / 'anonymized.wav'}")

    return result


def diagnose_semantics(audio_path, result, output_path, device):
    """语义诊断"""
    from models.ssl.wrappers import WavLMSSLExtractor

    # 加载原始音频特征
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(0).to(device)

    ssl = WavLMSSLExtractor(
        ckpt_path='checkpoints/WavLM-Large.pt',
        layer=15, device=device
    )

    with torch.no_grad():
        h_orig = ssl(waveform.unsqueeze(0)).squeeze(0)

    # 获取匿名化特征
    h_anon = result['features'].to(device)

    # 诊断 1: 帧连续性
    print("\n  --- 帧连续性诊断 ---")
    analyze_continuity(h_orig, "原始特征")
    analyze_continuity(h_anon, "匿名特征")

    # 诊断 2: 特征分布
    print("\n  --- 特征分布诊断 ---")
    analyze_distribution(h_orig, h_anon)

    # 保存诊断结果
    save_diagnosis(h_orig, h_anon, output_path)


def analyze_continuity(features, name):
    """分析帧连续性"""
    if features.dim() == 3:
        features = features.squeeze(0)

    features_norm = torch.nn.functional.normalize(features, dim=-1)
    cos_sim = (features_norm[:-1] * features_norm[1:]).sum(dim=-1)

    mean_sim = cos_sim.mean().item()
    std_sim = cos_sim.std().item()
    low_ratio = (cos_sim < 0.5).float().mean().item()

    print(f"  {name}:")
    print(f"    相邻帧cosine均值: {mean_sim:.4f}")
    print(f"    标准差: {std_sim:.4f}")
    print(f"    低相似度比例(<0.5): {low_ratio*100:.1f}%")

    if mean_sim < 0.7:
        print(f"    ⚠️ 警告: 帧连续性差")
    else:
        print(f"    ✓ 帧连续性正常")


def analyze_distribution(h_orig, h_anon):
    """分析特征分布"""
    if h_orig.dim() == 3:
        h_orig = h_orig.squeeze(0)
    if h_anon.dim() == 3:
        h_anon = h_anon.squeeze(0)

    # 截取相同长度
    min_len = min(len(h_orig), len(h_anon))
    h_orig = h_orig[:min_len]
    h_anon = h_anon[:min_len]

    # 计算统计量
    orig_mean = h_orig.mean().item()
    orig_std = h_orig.std().item()
    anon_mean = h_anon.mean().item()
    anon_std = h_anon.std().item()

    # 整体 cosine
    cos_sim = torch.nn.functional.cosine_similarity(
        h_orig.mean(0).unsqueeze(0),
        h_anon.mean(0).unsqueeze(0)
    ).item()

    print(f"  原始: mean={orig_mean:.3f}, std={orig_std:.3f}")
    print(f"  匿名: mean={anon_mean:.3f}, std={anon_std:.3f}")
    print(f"  整体cosine相似度: {cos_sim:.4f}")

    if cos_sim < 0.5:
        print(f"  ⚠️ 警告: 特征分布严重偏移")
    else:
        print(f"  ✓ 特征分布基本一致")


def save_diagnosis(h_orig, h_anon, output_path):
    """保存诊断结果"""
    np.save(output_path / "h_orig.npy", h_orig.cpu().numpy())
    np.save(output_path / "h_anon.npy", h_anon.cpu().numpy())
    print(f"\n  诊断数据已保存到 {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str,
                       default="outputs/20260121_163701/anonymized_audio/original.wav")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    test_anonymization(args.audio, args.output)
