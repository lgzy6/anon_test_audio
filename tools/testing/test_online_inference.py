#!/usr/bin/env python3
"""
Online匿名推理测试脚本
测试完整的匿名化流程（不包含vocoder）
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
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
from models.knn_vc.duration import DurationPredictor, DurationAnonymizer, DurationAdjuster


def test_online_inference(audio_path: str, output_dir: str = "outputs/online_test"):
    """
    测试完整的online匿名推理流程

    Args:
        audio_path: 输入音频文件路径
        output_dir: 输出目录
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Online匿名推理测试")
    print("="*70)

    # ========== 加载模型 ==========
    print("\n[1/8] 加载WavLM模型...")
    ssl_extractor = WavLMSSLExtractor(
        ckpt_path="checkpoints/WavLM-Large.pt",
        layer=15,
        device=device
    )
    print("  ✓ WavLM加载完成")

    print("\n[2/8] 加载Eta-WavLM投影器...")
    projector = EtaWavLMProjector(
        checkpoint_path="data/samm_anon/checkpoints/speaker_subspace.pt",
        device=device
    )
    print("  ✓ Eta-WavLM加载完成")

    print("\n[3/8] 加载SAMM码本...")
    codebook = SAMMCodebook.load(
        "data/samm_anon/checkpoints/codebook.pt",
        device=device
    )
    print(f"  ✓ 码本加载完成 (size={codebook.codebook_size})")

    print("\n[4/8] 加载Pattern Matrix...")
    pattern_matrix = PatternMatrix.load(
        "data/samm_anon/checkpoints/pattern_matrix.pt",
        device=device
    )
    print("  ✓ Pattern Matrix加载完成")

    print("\n[5/8] 加载Phone Predictor...")
    phone_predictor = PhonePredictor(
        checkpoint_path="checkpoints/phone_decoder.pt",
        device=device
    )
    print("  ✓ Phone Predictor加载完成")

    print("\n[6/8] 加载Duration Anonymizer...")
    duration_anonymizer = DurationAnonymizer(
        predictor_path="checkpoints/duration_decoder.pt",
        predictor_weight=0.7,
        noise_std=0.1,
        device=device
    )
    print("  ✓ Duration Anonymizer加载完成")

    print("\n[7/8] 加载Target Pool...")
    target_pool = TargetPool.load("data/samm_anon/checkpoints/target_pool")
    print(f"  ✓ Target Pool加载完成")
    print(f"    - 特征数量: {len(target_pool.features):,}")
    print(f"    - FAISS索引: {'已加载' if target_pool.faiss_index else '未加载'}")

    print("\n[8/8] 初始化kNN检索器...")
    retriever = ConstrainedKNNRetriever(
        target_pool=target_pool,
        k=4,
        num_clusters=8,
        device=device
    )
    print("  ✓ kNN检索器初始化完成")

    # ========== 加载音频 ==========
    print("\n" + "="*70)
    print("处理音频")
    print("="*70)

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

    # ========== Stage 1: SSL特征提取 ==========
    print("\n[Stage 1] SSL特征提取...")
    with torch.no_grad():
        h = ssl_extractor(waveform.to(device))  # [1, T, 1024]
    print(f"  ✓ 特征形状: {h.shape}")

    # 保存中间结果
    np.save(output_dir / "stage1_ssl_features.npy", h.cpu().numpy())

    # ========== Stage 2: Eta-WavLM说话人去除 ==========
    print("\n[Stage 2] Eta-WavLM说话人去除...")
    with torch.no_grad():
        h_clean = projector(h)  # [1, T, 1024]

    diff = (h - h_clean).norm() / h.norm()
    print(f"  ✓ 去说话人完成")
    print(f"    相对变化: {diff:.2%}")

    np.save(output_dir / "stage2_cleaned_features.npy", h_clean.cpu().numpy())

    # ========== Stage 3.1: SAMM符号分配 ==========
    print("\n[Stage 3.1] SAMM符号分配...")
    with torch.no_grad():
        z = codebook.encode(h_clean.squeeze(0))  # [T]
    print(f"  ✓ 符号序列长度: {len(z)}")
    print(f"    唯一符号数: {len(torch.unique(z))}")

    np.save(output_dir / "stage3_1_symbols.npy", z.cpu().numpy())
