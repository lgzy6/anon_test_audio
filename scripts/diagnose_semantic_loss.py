#!/usr/bin/env python3
"""
语义损失诊断脚本
验证匿名化流程中语义信息在哪个阶段被破坏
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SemanticDiagnostics:
    """语义诊断工具"""

    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}

    def diagnose_frame_continuity(self, features: torch.Tensor, name: str):
        """
        诊断 1: 相邻帧 cosine 连续性
        如果相邻帧 cosine 相似度剧烈波动，vocoder 必然失败
        """
        if features.dim() == 3:
            features = features.squeeze(0)

        # 计算相邻帧 cosine 相似度
        features_norm = torch.nn.functional.normalize(features, dim=-1)
        cos_sim = (features_norm[:-1] * features_norm[1:]).sum(dim=-1)

        result = {
            'mean': cos_sim.mean().item(),
            'std': cos_sim.std().item(),
            'min': cos_sim.min().item(),
            'max': cos_sim.max().item(),
            'low_sim_ratio': (cos_sim < 0.5).float().mean().item(),
        }

        self.results[f'{name}_continuity'] = result

        print(f"\n=== {name} 帧连续性诊断 ===")
        print(f"  相邻帧 cosine 均值: {result['mean']:.4f}")
        print(f"  相邻帧 cosine 标准差: {result['std']:.4f}")
        print(f"  最小值: {result['min']:.4f}, 最大值: {result['max']:.4f}")
        print(f"  低相似度帧比例 (<0.5): {result['low_sim_ratio']*100:.1f}%")

        # 判断
        if result['mean'] < 0.7:
            print(f"  ⚠️ 警告: 帧连续性差，vocoder 可能失败")
        elif result['std'] > 0.2:
            print(f"  ⚠️ 警告: 帧相似度波动大，可能产生抖动")
        else:
            print(f"  ✓ 帧连续性正常")

        return cos_sim

    def diagnose_phone_alignment(
        self,
        source_phones: torch.Tensor,
        retrieved_phones: torch.Tensor,
        name: str = "kNN"
    ):
        """
        诊断 2: Phone 对齐率
        retrieved_phone == source_phone 的比例
        """
        if source_phones.shape != retrieved_phones.shape:
            min_len = min(len(source_phones), len(retrieved_phones))
            source_phones = source_phones[:min_len]
            retrieved_phones = retrieved_phones[:min_len]

        match_ratio = (source_phones == retrieved_phones).float().mean().item()

        result = {
            'match_ratio': match_ratio,
            'total_frames': len(source_phones),
        }

        self.results[f'{name}_phone_alignment'] = result

        print(f"\n=== {name} Phone 对齐诊断 ===")
        print(f"  Phone 匹配率: {match_ratio*100:.1f}%")
        print(f"  总帧数: {len(source_phones)}")

        if match_ratio < 0.6:
            print(f"  ❌ 严重: Phone 匹配率过低，语义必然丢失")
        elif match_ratio < 0.8:
            print(f"  ⚠️ 警告: Phone 匹配率偏低")
        else:
            print(f"  ✓ Phone 对齐正常")

        return match_ratio

    def diagnose_feature_distribution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_name: str,
        target_name: str
    ):
        """
        诊断 3: 特征分布偏移
        检查处理前后特征分布是否一致
        """
        if source_features.dim() == 3:
            source_features = source_features.squeeze(0)
        if target_features.dim() == 3:
            target_features = target_features.squeeze(0)

        src_mean = source_features.mean(dim=0)
        src_std = source_features.std(dim=0)
        tgt_mean = target_features.mean(dim=0)
        tgt_std = target_features.std(dim=0)

        mean_diff = (src_mean - tgt_mean).abs().mean().item()
        std_diff = (src_std - tgt_std).abs().mean().item()

        # 计算整体 cosine 相似度
        src_flat = source_features.mean(dim=0)
        tgt_flat = target_features.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            src_flat.unsqueeze(0), tgt_flat.unsqueeze(0)
        ).item()

        result = {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'cosine_sim': cos_sim,
        }

        self.results[f'{source_name}_vs_{target_name}'] = result

        print(f"\n=== {source_name} vs {target_name} 分布诊断 ===")
        print(f"  均值差异: {mean_diff:.4f}")
        print(f"  标准差差异: {std_diff:.4f}")
        print(f"  整体 cosine 相似度: {cos_sim:.4f}")

        if cos_sim < 0.5:
            print(f"  ❌ 严重: 特征分布严重偏移")
        elif mean_diff > 1.0:
            print(f"  ⚠️ 警告: 均值偏移较大")
        else:
            print(f"  ✓ 分布基本一致")

        return result

    def summary(self):
        """输出诊断总结"""
        print("\n" + "=" * 60)
        print("诊断总结")
        print("=" * 60)

        issues = []

        for key, val in self.results.items():
            if 'continuity' in key:
                if val['mean'] < 0.7:
                    issues.append(f"帧连续性差 ({key})")
            elif 'phone_alignment' in key:
                if val['match_ratio'] < 0.6:
                    issues.append(f"Phone 对齐率过低 ({key})")
            elif 'vs' in key:
                if val['cosine_sim'] < 0.5:
                    issues.append(f"特征分布偏移 ({key})")

        if issues:
            print("\n发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n未发现严重问题")

        return issues
