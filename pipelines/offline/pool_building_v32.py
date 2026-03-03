#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 核心模块: 量化池构建器

功能:
- 按 Pattern 分区构建目标池
- 每个 (Pattern, Phone) 组合只保存 K 个聚类中心
- 存储从 213GB 压缩到 ~100MB
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from tqdm import tqdm
import json
import logging

logger = logging.getLogger(__name__)


class QuantizedPoolBuilder:
    """
    v3.2 量化目标池构建器

    核心思想:
    1. 按 Pattern ID 物理分区
    2. 每个分区内按 Phone ID 分组
    3. 每组只保存 K 个 KMeans 中心
    """

    def __init__(
        self,
        n_patterns: int = 8,
        n_phones: int = 41,
        centers_per_phone: int = 64,
        use_gender_split: bool = True,
        min_samples: int = 100,
        random_state: int = 42,
    ):
        self.n_patterns = n_patterns
        self.n_phones = n_phones
        self.centers_per_phone = centers_per_phone
        self.use_gender_split = use_gender_split
        self.min_samples = min_samples
        self.random_state = random_state

        # 构建结果
        self.centers = {}  # {(pattern, phone, gender): [K, D]}
        self.duration_stats = {}  # {(pattern, phone): (mean, std)}
        self.pattern_counts = defaultdict(int)

    def build(
        self,
        features: np.ndarray,
        phones: np.ndarray,
        patterns: np.ndarray,
        genders: Optional[np.ndarray] = None,
        durations: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        构建量化目标池

        Args:
            features: [N, D] 特征
            phones: [N] 音素标签
            patterns: [N] Pattern 标签
            genders: [N] 性别标签 (0=M, 1=F)
            durations: [N] 时长信息 (可选)

        Returns:
            构建统计信息
        """
        N, D = features.shape
        logger.info(f"Building quantized pool: {N} frames, {D} dims")

        if genders is None:
            genders = np.zeros(N, dtype=np.int32)

        # 统计各 Pattern 的帧数
        for p in range(self.n_patterns):
            self.pattern_counts[p] = (patterns == p).sum()
        logger.info(f"Pattern distribution: {dict(self.pattern_counts)}")

        # 按 (Pattern, Phone, Gender) 分组构建
        total_centers = 0
        skipped_groups = 0

        for pattern_id in tqdm(range(self.n_patterns), desc="Patterns"):
            pattern_mask = (patterns == pattern_id)

            for phone_id in range(self.n_phones):
                phone_mask = (phones == phone_id)

                gender_list = [0, 1] if self.use_gender_split else [0]

                for gender in gender_list:
                    if self.use_gender_split:
                        gender_mask = (genders == gender)
                        mask = pattern_mask & phone_mask & gender_mask
                    else:
                        mask = pattern_mask & phone_mask

                    n_samples = mask.sum()

                    if n_samples < self.min_samples:
                        skipped_groups += 1
                        continue

                    # KMeans 聚类
                    subset = features[mask]
                    n_clusters = min(self.centers_per_phone, n_samples // 2)

                    kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        batch_size=min(1024, n_samples),
                    )
                    kmeans.fit(subset)

                    key = (pattern_id, phone_id, gender)
                    self.centers[key] = kmeans.cluster_centers_.astype(np.float32)
                    total_centers += n_clusters

        logger.info(f"Built {total_centers} centers, skipped {skipped_groups} groups")

        return {
            'total_centers': total_centers,
            'skipped_groups': skipped_groups,
            'n_patterns': self.n_patterns,
        }

    def save(self, output_dir: str):
        """保存量化池"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存中心点
        centers_dict = {}
        for key, centers in self.centers.items():
            str_key = f"{key[0]}_{key[1]}_{key[2]}"
            centers_dict[str_key] = centers

        np.savez_compressed(
            output_path / 'quantized_centers.npz',
            **centers_dict
        )

        # 保存元数据
        meta = {
            'n_patterns': self.n_patterns,
            'n_phones': self.n_phones,
            'centers_per_phone': self.centers_per_phone,
            'pattern_counts': dict(self.pattern_counts),
        }
        with open(output_path / 'pool_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved quantized pool to {output_path}")

    @classmethod
    def load(cls, path: str) -> 'QuantizedPoolBuilder':
        """加载量化池"""
        pool_path = Path(path)

        with open(pool_path / 'pool_meta.json', 'r') as f:
            meta = json.load(f)

        builder = cls(
            n_patterns=meta['n_patterns'],
            n_phones=meta['n_phones'],
            centers_per_phone=meta['centers_per_phone'],
        )

        data = np.load(pool_path / 'quantized_centers.npz')
        for str_key in data.files:
            parts = str_key.split('_')
            key = (int(parts[0]), int(parts[1]), int(parts[2]))
            builder.centers[key] = data[str_key]

        return builder
