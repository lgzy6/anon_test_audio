#!/usr/bin/env python3
"""
Online匿名推理测试脚本
完整测试SAMM + Eta-WavLM + kNN-VC流程
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.ssl.extractor import WavLMSSLExtractor
from models.eta_wavlm.projector import EtaWavLMProjector
from models.samm.codebook import SAMMCodebook
from models.samm.pattern_matrix import PatternMatrix
from models.samm.masking import ProsodyAwareMasking
from models.phone_predictor.predictor import PhonePredictor
from models.knn_vc.retriever import TargetPool, ConstrainedKNNRetriever
from models.knn_vc.duration import DurationAnonymizer, DurationAdjuster


def load_models(device="cuda"):
    """加载所有必需的模型"""
    print("\n" + "="*70)
    print("加载模型")
    print("="*70)

    models = {}

    # Stage 1: SSL特征提取
    print("\n[1/7] 加载WavLM...")
    models['ssl_extractor'] = WavLMSSLExtractor(
        ckpt_path="checkpoints/WavLM-Large.pt",
        layer=15,
        device=device
    )
    print("  ✓ WavLM加载完成")

    # Stage 2: Eta-WavLM投影
    print("\n[2/7] 加载Eta-WavLM投影器...")
    models['projector'] = EtaWavLMProjector(
        checkpoint_path="data/samm_anon/checkpoints/speaker_subspace.pt",
        device=device
    )
    print("  ✓ Eta-WavLM加载完成")

    # Stage 3: SAMM组件
    print("\n[3/7] 加载SAMM码本...")
    models['codebook'] = SAMMCodebook.load(
        "data/samm_anon/checkpoints/codebook.pt",
        device=device
    )
    print(f"  ✓ 码本加载完成 (size={models['codebook'].codebook_size})")

    print("\n[4/7] 加载Pattern Matrix...")
    models['pattern_matrix'] = PatternMatrix.load(
        "data/samm_anon/checkpoints/pattern_matrix.pt",
        device=device
    )
    print("  ✓ Pattern Matrix加载完成")

    # 掩码模块
    models['masking'] = ProsodyAwareMasking(
        token_mask_ratio=0.10,
        span_mask_ratio=0.15
    )

    # Stage 3': Phone Predictor
    print("\n[5/7] 加载Phone Predictor...")
    models['phone_predictor'] = PhonePredictor(
        checkpoint_path="checkpoints/phone_decoder.pt",
        device=device
    )
    print("  ✓ Phone Predictor加载完成")

    # Duration Anonymizer
    print("\n[6/7] 加载Duration Anonymizer...")
    models['duration_anonymizer'] = DurationAnonymizer(
        predictor_path="checkpoints/duration_decoder.pt",
        predictor_weight=0.7,
        noise_std=0.1,
        device=device
    )
    print("  ✓ Duration Anonymizer加载完成")

    # Stage 4: Target Pool和kNN检索
    print("\n[7/7] 加载Target Pool...")
    models['target_pool'] = TargetPool.load(
        "data/samm_anon/checkpoints/target_pool"
    )
    print(f"  ✓ Target Pool加载完成")
    print(f"    - 特征数量: {len(models['target_pool'].features):,}")

    models['retriever'] = ConstrainedKNNRetriever(
        target_pool=models['target_pool'],
        k=4,
        num_clusters=8,
        device=device
    )
    print("  ✓ kNN检索器初始化完成")

    return models


def anonymize_audio(audio_path, models, target_gender=0, output_dir="outputs/online_test", device="cuda"):
    """
    执行完整的匿名化流程

    Args:
        audio_path: 输入音频路径
        models: 加载的模型字典
        target_gender: 目标性别 (0=M, 1=F)
        output_dir: 输出目录
        device: 设备

    Returns:
        匿名化特征和中间结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("处理音频")
    print("="*70)

    # 加载音频
    print(f"\n加载音频: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        print(f"  重采样: {sr}Hz -> 16000Hz")
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"  音频长度: {waveform.shape[1]/sr:.2f}秒")

    # 保存原始音频到输出目录（用于对比）
    original_audio_path = output_dir / "original.wav"
    torchaudio.save(str(original_audio_path), waveform.cpu(), 16000)
    print(f"  ✓ 原始音频已保存: {original_audio_path}")

    waveform = waveform.to(device)

    results = {}

    # ========== Stage 1: SSL特征提取 ==========
    print("\n[Stage 1] SSL特征提取...")
    with torch.no_grad():
        h = models['ssl_extractor'](waveform)  # [1, T, 1024]
    print(f"  ✓ 特征形状: {h.shape}")
    results['h_ssl'] = h.cpu().numpy()
    np.save(output_dir / "stage1_ssl_features.npy", results['h_ssl'])

    # ========== Stage 2: Eta-WavLM说话人去除 ==========
    print("\n[Stage 2] Eta-WavLM说话人去除...")
    with torch.no_grad():
        h_clean = models['projector'](h)  # [1, T, 1024]

    diff = (h - h_clean).norm() / h.norm()
    print(f"  ✓ 去说话人完成")
    print(f"    相对变化: {diff:.2%}")
    results['h_clean'] = h_clean.cpu().numpy()
    np.save(output_dir / "stage2_cleaned_features.npy", results['h_clean'])

    # 去掉batch维度
    h = h.squeeze(0)  # [T, 1024]
    h_clean = h_clean.squeeze(0)  # [T, 1024]

    # ========== Stage 3: SAMM处理 ==========
    print("\n[Stage 3] SAMM符号化和掩码...")

    # Stage 3.1: 符号分配
    with torch.no_grad():
        z = models['codebook'].encode(h_clean)  # [T]
    print(f"  ✓ 符号序列长度: {len(z)}")
    print(f"    唯一符号数: {len(torch.unique(z))}")
    results['symbols'] = z.cpu().numpy()
    np.save(output_dir / "stage3_1_symbols.npy", results['symbols'])

    # Stage 3.1': Phone预测
    print("\n[Stage 3'] Phone预测...")
    with torch.no_grad():
        phones = models['phone_predictor'](h)  # 使用原始特征
    print(f"  ✓ Phone序列长度: {len(phones)}")
    results['phones'] = phones.cpu().numpy()

    # Stage 3.2: 音素时长提取
    phone_ids, true_durations = models['phone_predictor'].get_phone_durations(phones)
    phone_segments = models['phone_predictor'].get_phone_segments(phones)
    print(f"  ✓ 音素数量: {len(phone_ids)}")
    print(f"    平均时长: {true_durations.float().mean():.2f} 帧")

    # Stage 3.3: 时长匿名化
    print("\n[Stage 3.3] 时长匿名化...")
    with torch.no_grad():
        anon_durations = models['duration_anonymizer'].anonymize(
            phone_ids.to(device),
            true_durations.to(device)
        )
    print(f"  ✓ 匿名化时长完成")
    print(f"    原始总时长: {true_durations.sum().item():.0f} 帧")
    print(f"    匿名化总时长: {anon_durations.sum().item():.0f} 帧")
    results['anon_durations'] = anon_durations.cpu().numpy()

    # Stage 3.4: 符号掩码
    print("\n[Stage 3.4] 符号掩码...")
    z_masked, _, mask_indicator = models['masking'](
        z,
        torch.ones_like(z, dtype=torch.float)
    )
    print(f"  ✓ 掩码比例: {mask_indicator.float().mean():.2%}")

    # Stage 3.5: 模式正则化
    print("\n[Stage 3.5] 模式正则化...")
    z_smooth = models['pattern_matrix'].smooth_sequence(z_masked, mask_indicator)
    print(f"  ✓ 平滑完成")
    results['z_smooth'] = z_smooth.cpu().numpy()

    return h, h_clean, z_smooth, phones, phone_ids, anon_durations, phone_segments, results, output_dir


def run_stage4_retrieval(h_clean, z_smooth, phones, phone_segments, anon_durations,
                         models, target_gender, output_dir, device="cuda"):
    """执行Stage 4: kNN检索和时长调整"""

    # ========== Stage 4.1: 约束kNN检索 ==========
    print("\n[Stage 4.1] 约束kNN检索...")
    print(f"  目标性别: {'Male' if target_gender == 0 else 'Female'}")

    with torch.no_grad():
        h_anon = models['retriever'].retrieve_batch(
            h_clean,
            phones,
            z_smooth,
            target_gender
        )

    print(f"  ✓ 检索完成")
    print(f"    检索特征形状: {h_anon.shape}")

    # ========== Stage 4.2: 时长调整 ==========
    print("\n[Stage 4.2] 时长调整...")
    h_anon_adjusted = DurationAdjuster.adjust_features(
        h_anon,
        phone_segments,
        anon_durations
    )

    print(f"  ✓ 时长调整完成")
    print(f"    调整前: {h_anon.shape}")
    print(f"    调整后: {h_anon_adjusted.shape}")

    # 保存最终特征
    np.save(output_dir / "stage4_anon_features.npy", h_anon_adjusted.cpu().numpy())

    return h_anon_adjusted


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Online匿名推理测试")
    parser.add_argument("--input", "-i", required=True, help="输入音频路径")
    parser.add_argument("--output-dir", "-o", default="outputs/online_test", help="输出目录")
    parser.add_argument("--gender", "-g", choices=['M', 'F'], default='M', help="目标性别")
    parser.add_argument("--device", "-d", default="cuda", help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: CUDA不可用，使用CPU（速度会很慢）")

    target_gender = 0 if args.gender == 'M' else 1

    print("\n" + "="*70)
    print("Online匿名推理测试")
    print("="*70)
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标性别: {args.gender}")
    print(f"设备: {device}")

    # 加载模型
    models = load_models(device=device)

    # 执行匿名化 (Stage 1-3)
    h, h_clean, z_smooth, phones, phone_ids, anon_durations, phone_segments, results, output_dir = \
        anonymize_audio(args.input, models, target_gender, args.output_dir, device)

    # 执行Stage 4
    h_anon_adjusted = run_stage4_retrieval(
        h_clean, z_smooth, phones, phone_segments, anon_durations,
        models, target_gender, output_dir, device
    )

    # 总结
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print(f"\n所有中间结果已保存到: {output_dir}")
    print("\n生成的文件:")
    for f in sorted(output_dir.glob("*.npy")):
        size = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size:.2f} MB)")

    print("\n注意: 由于缺少vocoder模型，当前只输出匿名化特征。")
    print("如需生成音频，请准备vocoder模型并使用pipelines/online/anonymizer.py")


if __name__ == "__main__":
    main()
