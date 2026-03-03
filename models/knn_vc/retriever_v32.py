#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 核心模块: 量化检索器

功能:
- 从量化池中检索最近的聚类中心
- 强制 Phone ID 约束
- 支持 Pattern 分区检索
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class QuantizedRetriever(nn.Module):
    """
    v3.2 量化检索器

    检索策略:
    1. 根据 Target Pattern 选择对应的子池
    2. 在子池中按 Phone ID 约束检索
    3. 返回最近的聚类中心 (1-NN)
    """

    def __init__(
        self,
        pool_path: str,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device_str = device
        self.centers = {}  # {(pattern, phone, gender): Tensor}

        self._load_pool(pool_path)

    def _load_pool(self, path: str):
        """加载量化池"""
        pool_path = Path(path)

        # 加载中心点
        data = np.load(pool_path / 'quantized_centers.npz')

        for str_key in data.files:
            parts = str_key.split('_')
            key = (int(parts[0]), int(parts[1]), int(parts[2]))
            self.centers[key] = torch.from_numpy(data[str_key]).float()

        print(f"[QuantizedRetriever] Loaded {len(self.centers)} center groups")

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.centers = {k: v.to(device) for k, v in self.centers.items()}
        return self

    @torch.inference_mode()
    def retrieve(
        self,
        h: torch.Tensor,
        phones: torch.Tensor,
        target_pattern: int,
        target_gender: int = 0,
    ) -> torch.Tensor:
        """
        检索匿名化特征

        Args:
            h: [T, D] 查询特征
            phones: [T] 音素标签
            target_pattern: 目标 Pattern ID
            target_gender: 目标性别 (0=M, 1=F)

        Returns:
            h_anon: [T, D] 匿名化特征
        """
        T, D = h.shape
        h_anon = torch.zeros_like(h)

        for t in range(T):
            phone_id = phones[t].item()
            key = (target_pattern, phone_id, target_gender)

            if key in self.centers:
                centers = self.centers[key]
                # 1-NN: 找最近的中心
                dist = torch.cdist(h[t:t+1], centers)
                nearest_idx = dist.argmin()
                h_anon[t] = centers[nearest_idx]
            else:
                # Fallback: 尝试不区分性别
                fallback_key = (target_pattern, phone_id, 0)
                if fallback_key in self.centers:
                    centers = self.centers[fallback_key]
                    dist = torch.cdist(h[t:t+1], centers)
                    nearest_idx = dist.argmin()
                    h_anon[t] = centers[nearest_idx]
                else:
                    h_anon[t] = h[t]  # 最终 fallback

        return h_anon

    @torch.inference_mode()
    def retrieve_batch(
        self,
        h: torch.Tensor,
        phones: torch.Tensor,
        target_pattern: int,
        target_gender: int = 0,
    ) -> torch.Tensor:
        """批量检索 (按音素分组优化)"""
        T, D = h.shape
        h_anon = torch.zeros_like(h)

        unique_phones = phones.unique()

        for phone_id in unique_phones:
            mask = (phones == phone_id)
            h_subset = h[mask]

            key = (target_pattern, phone_id.item(), target_gender)

            if key in self.centers:
                centers = self.centers[key]
                dist = torch.cdist(h_subset, centers)
                nearest_idx = dist.argmin(dim=-1)
                h_anon[mask] = centers[nearest_idx]
            else:
                h_anon[mask] = h_subset

        return h_anon
