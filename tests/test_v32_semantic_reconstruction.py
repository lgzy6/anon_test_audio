#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 Test C: 语义重建测试 (Sanity Check)

验证目标:
- Target Pattern = Source Pattern (不做匿名，只做重建)
- 测试 WER 是否可接受

成功标准:
- WER < 10% (接近 WavLM 原始重构水平)
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SemanticReconstructionTest:
    """语义重建测试器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.results = {}

    def test_reconstruction_quality(
        self,
        features: np.ndarray,
        phones: np.ndarray,
        phone_clusters: dict,
    ) -> dict:
        """
        测试重建质量

        流程:
        1. 对每帧，在同音素的聚类中心中找最近邻
        2. 用最近邻替换原特征
        3. 计算重建误差
        """
        n_frames = len(features)
        reconstructed = np.zeros_like(features)
        distances = []

        for t in range(n_frames):
            phone_id = phones[t]
            h = features[t]

            if phone_id in phone_clusters:
                centers = phone_clusters[phone_id]
                # 找最近的中心
                dist = np.linalg.norm(centers - h, axis=1)
                nearest_idx = dist.argmin()
                reconstructed[t] = centers[nearest_idx]
                distances.append(dist[nearest_idx])
            else:
                reconstructed[t] = h
                distances.append(0)

        # 计算指标
        mse = np.mean((features - reconstructed) ** 2)
        mae = np.mean(np.abs(features - reconstructed))
        avg_dist = np.mean(distances)

        results = {
            'mse': float(mse),
            'mae': float(mae),
            'avg_nearest_dist': float(avg_dist),
            'n_frames': n_frames,
        }

        print(f"Reconstruction Quality:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Avg Nearest Distance: {avg_dist:.4f}")

        return results
