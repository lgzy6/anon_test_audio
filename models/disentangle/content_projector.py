#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 核心模块: 内容子空间投影器

功能:
- 基于音素中心构建内容子空间
- 投影去除内容 → 得到纯风格特征 H_style
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class ContentSubspaceProjector(nn.Module):
    """
    v3.2 内容子空间投影器

    与 EtaWavLMProjector 的区别:
    - Eta: 基于说话人中心，投影去除说话人
    - 本模块: 基于音素中心，投影去除内容
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        n_phones: int = 41,
        variance_threshold: float = 0.95,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_phones = n_phones
        self.variance_threshold = variance_threshold

        # 将在 fit() 中初始化
        self.register_buffer('U_c', None)
        self.register_buffer('P_orth', None)
        self.register_buffer('phone_centroids', None)

        self.subspace_dim = 0
        self.explained_variance = 0.0

    def fit(self, features: torch.Tensor, phones: torch.Tensor):
        """
        学习内容子空间

        Args:
            features: [N, D] 特征
            phones: [N] 音素标签
        """
        features_np = features.cpu().numpy()
        phones_np = phones.cpu().numpy()

        # Step 1: 计算音素中心
        centroids = np.zeros((self.n_phones, self.feature_dim))
        for p in range(self.n_phones):
            mask = (phones_np == p)
            if mask.sum() > 0:
                centroids[p] = features_np[mask].mean(axis=0)

        # Step 2: SVD 分解
        centroids_centered = centroids - centroids.mean(axis=0)
        U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)

        # Step 3: 选择主成分
        total_var = (S ** 2).sum()
        cumsum = np.cumsum(S ** 2) / total_var
        n_comp = np.searchsorted(cumsum, self.variance_threshold) + 1
        n_comp = min(n_comp, self.n_phones - 1)

        self.subspace_dim = n_comp
        self.explained_variance = cumsum[n_comp - 1]

        # Step 4: 构建投影矩阵
        U_c = Vt[:n_comp].T
        P_orth = np.eye(self.feature_dim) - U_c @ U_c.T

        self.U_c = torch.from_numpy(U_c).float()
        self.P_orth = torch.from_numpy(P_orth).float()
        self.phone_centroids = torch.from_numpy(centroids).float()

        return self

    @torch.inference_mode()
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """投影到风格子空间 (去除内容)"""
        if self.P_orth is None:
            raise RuntimeError("Must call fit() first")
        return h @ self.P_orth.to(h.device)

    def get_style(self, h: torch.Tensor) -> torch.Tensor:
        """forward 的别名"""
        return self.forward(h)

    def get_content(self, h: torch.Tensor) -> torch.Tensor:
        """提取内容成分"""
        return h - self.forward(h)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'U_c': self.U_c,
            'P_orth': self.P_orth,
            'phone_centroids': self.phone_centroids,
            'subspace_dim': self.subspace_dim,
            'explained_variance': self.explained_variance,
            'feature_dim': self.feature_dim,
            'n_phones': self.n_phones,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """加载模型"""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            feature_dim=ckpt['feature_dim'],
            n_phones=ckpt['n_phones'],
        )
        model.U_c = ckpt['U_c']
        model.P_orth = ckpt['P_orth']
        model.phone_centroids = ckpt['phone_centroids']
        model.subspace_dim = ckpt['subspace_dim']
        model.explained_variance = ckpt['explained_variance']
        return model.to(device)
