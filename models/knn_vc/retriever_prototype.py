# models/knn_vc/retriever_prototype.py

"""
Prototype Pool 专用检索器

适配 Speaker Prototype Pool 的检索策略:
- 输入: Speaker Prototype (~100K) 而非 Frame-level (~55M)
- 优势: 检索速度大幅提升，内存占用极低
- 核心: 利用 prototype 的说话人元数据进行约束检索
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


class PrototypeKNNRetriever(nn.Module):
    """
    Prototype Pool 专用检索器

    与 ConstrainedKNNRetriever 的区别:
    - 检索对象是 speaker prototype 而非原始帧
    - 利用 speaker_id 进行说话人级别的约束
    - 支持随机/固定目标说话人选择
    """

    def __init__(
        self,
        target_pool_path: str,
        k: int = 4,
        num_clusters: int = 8,
        use_phone_constraint: bool = True,
        use_gender_constraint: bool = True,
        target_speaker_mode: str = 'random',  # 'random', 'fixed', 'pool'
        fixed_target_speaker: Optional[str] = None,
        temperature: float = 1.0,
        device: str = 'cuda',
    ):
        """
        Args:
            target_pool_path: Prototype Pool 路径
            k: kNN 的 k 值
            num_clusters: 聚类数 (用于 phone clusters)
            use_phone_constraint: 是否使用音素约束
            use_gender_constraint: 是否使用性别约束
            target_speaker_mode: 目标说话人选择模式
                - 'random': 每次随机选择一个目标说话人
                - 'fixed': 使用固定的目标说话人
                - 'pool': 从整个 pool 中检索 (不限制说话人)
            fixed_target_speaker: 固定目标说话人 ID
            temperature: softmax 温度
            device: 设备
        """
        super().__init__()

        self.k = k
        self.num_clusters = num_clusters
        self.use_phone = use_phone_constraint
        self.use_gender = use_gender_constraint
        self.target_speaker_mode = target_speaker_mode
        self.fixed_target_speaker = fixed_target_speaker
        self.temperature = temperature
        self.device_str = device

        # 加载 Prototype Pool
        self._load_prototype_pool(target_pool_path)

    def _load_prototype_pool(self, path: str):
        """加载 Prototype Pool"""
        pool_path = Path(path)

        if not pool_path.is_dir():
            raise ValueError(f"Prototype pool must be a directory: {path}")

        # 检测 pool 类型
        metadata_path = pool_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.pool_type = metadata.get('pool_type', 'frame_level')
        else:
            self.pool_type = 'frame_level'

        print(f"[Prototype Retriever] Pool type: {self.pool_type}")

        # 加载 prototypes
        if (pool_path / 'prototypes.npy').exists():
            prototypes = np.load(pool_path / 'prototypes.npy')
        elif (pool_path / 'features.npy').exists():
            prototypes = np.load(pool_path / 'features.npy')
        else:
            raise FileNotFoundError(f"No prototypes/features file in {pool_path}")

        self.register_buffer('prototypes', torch.from_numpy(prototypes).float())

        # 加载性别
        genders = np.load(pool_path / 'genders.npy')
        self.register_buffer('genders', torch.from_numpy(genders).long())

        # 加载音素
        phones_path = pool_path / 'phones.npy'
        if phones_path.exists():
            phones = np.load(phones_path)
            self.register_buffer('phones', torch.from_numpy(phones).long())
        else:
            self.phones = None

        # 加载符号
        symbols_path = pool_path / 'symbols.npy'
        if symbols_path.exists():
            symbols = np.load(symbols_path)
            self.register_buffer('symbols', torch.from_numpy(symbols).long())
        else:
            self.symbols = None

        # 加载说话人 ID (Prototype Pool 特有)
        speaker_ids_path = pool_path / 'prototype_speaker_ids.npy'
        if speaker_ids_path.exists():
            speaker_ids = np.load(speaker_ids_path, allow_pickle=True)
            self.speaker_ids = speaker_ids
            self.unique_speakers = list(set(speaker_ids))
            print(f"  - Speakers: {len(self.unique_speakers)}")
        else:
            self.speaker_ids = None
            self.unique_speakers = None

        # 加载 phone clusters
        clusters_path = pool_path / 'phone_clusters.pt'
        if clusters_path.exists():
            self.phone_clusters = torch.load(clusters_path, map_location='cpu')
            # 确保是 tensor
            for k, v in self.phone_clusters.items():
                if isinstance(v, np.ndarray):
                    self.phone_clusters[k] = torch.from_numpy(v).float()
        else:
            self.phone_clusters = None

        # 构建说话人到 prototype 索引的映射
        if self.speaker_ids is not None:
            self.speaker_to_indices = {}
            for idx, spk in enumerate(self.speaker_ids):
                if spk not in self.speaker_to_indices:
                    self.speaker_to_indices[spk] = []
                self.speaker_to_indices[spk].append(idx)

            # 转换为 tensor
            for spk in self.speaker_to_indices:
                self.speaker_to_indices[spk] = torch.tensor(
                    self.speaker_to_indices[spk], dtype=torch.long
                )

        print(f"[Prototype Retriever] Loaded pool:")
        print(f"  - Prototypes: {self.prototypes.shape}")
        print(f"  - Phones: {'Yes' if self.phones is not None else 'No'}")
        print(f"  - Phone Clusters: {'Yes' if self.phone_clusters else 'No'}")

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        if self.phone_clusters is not None:
            self.phone_clusters = {
                k: v.to(device) for k, v in self.phone_clusters.items()
            }
        if hasattr(self, 'speaker_to_indices') and self.speaker_to_indices:
            self.speaker_to_indices = {
                k: v.to(device) for k, v in self.speaker_to_indices.items()
            }
        return self

    def select_target_speaker(self, target_gender: int = 0) -> Optional[str]:
        """选择目标说话人"""
        if self.target_speaker_mode == 'fixed':
            return self.fixed_target_speaker

        elif self.target_speaker_mode == 'random':
            if self.unique_speakers is None:
                return None

            # 按性别过滤说话人
            valid_speakers = []
            for spk in self.unique_speakers:
                indices = self.speaker_to_indices[spk]
                if len(indices) > 0:
                    spk_gender = self.genders[indices[0]].item()
                    if spk_gender == target_gender:
                        valid_speakers.append(spk)

            if len(valid_speakers) == 0:
                valid_speakers = self.unique_speakers

            # 随机选择
            idx = torch.randint(len(valid_speakers), (1,)).item()
            return valid_speakers[idx]

        else:  # 'pool' mode
            return None

    @torch.inference_mode()
    def retrieve(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        target_gender: int = 0,
        target_speaker: Optional[str] = None,
    ) -> torch.Tensor:
        """
        检索匿名化特征

        Args:
            h_clean: [T, D] 去说话人特征
            phones: [T] 音素标签
            symbols: [T] SAMM 符号 (可选)
            target_gender: 0=M, 1=F
            target_speaker: 目标说话人 ID (可选)
        Returns:
            h_anon: [T, D] 匿名化特征
        """
        # 选择目标说话人
        if target_speaker is None:
            target_speaker = self.select_target_speaker(target_gender)

        # 使用 phone clusters (如果可用)
        if self.phone_clusters is not None:
            return self._retrieve_from_clusters(
                h_clean, phones, target_gender, target_speaker
            )

        # 否则使用 kNN
        return self._retrieve_knn(
            h_clean, phones, symbols, target_gender, target_speaker
        )

    def _retrieve_from_clusters(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        target_gender: int,
        target_speaker: Optional[str],
    ) -> torch.Tensor:
        """从 phone clusters 检索"""
        T, D = h_clean.shape
        h_anon = torch.zeros_like(h_clean)

        # 按音素分组处理
        unique_phones = phones.unique()

        for phone_id in unique_phones:
            phone_mask = (phones == phone_id)
            h_subset = h_clean[phone_mask]

            # 获取聚类中心
            key = f"{phone_id.item()}_{target_gender}"
            if key not in self.phone_clusters:
                key = str(phone_id.item())

            if key in self.phone_clusters:
                centers = self.phone_clusters[key]
                # 批量最近邻
                dist = torch.cdist(h_subset, centers)  # [N, K]
                nearest_idx = dist.argmin(dim=-1)
                h_anon[phone_mask] = centers[nearest_idx]
            else:
                # Fallback: 使用 kNN
                h_anon[phone_mask] = self._retrieve_knn_subset(
                    h_subset, phone_id.item(), target_gender, target_speaker
                )

        return h_anon

    def _retrieve_knn(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        symbols: Optional[torch.Tensor],
        target_gender: int,
        target_speaker: Optional[str],
    ) -> torch.Tensor:
        """kNN 检索"""
        T, D = h_clean.shape
        h_anon = torch.zeros_like(h_clean)

        for t in range(T):
            phone_id = phones[t].item()
            h_anon[t] = self._retrieve_knn_single(
                h_clean[t], phone_id, target_gender, target_speaker
            )

        return h_anon

    def _retrieve_knn_single(
        self,
        h: torch.Tensor,
        phone_id: int,
        target_gender: int,
        target_speaker: Optional[str],
    ) -> torch.Tensor:
        """单帧 kNN 检索"""
        # 构建约束掩码
        device = self.prototypes.device
        mask = torch.ones(len(self.prototypes), dtype=torch.bool, device=device)

        # 说话人约束 (如果指定)
        if target_speaker is not None and target_speaker in self.speaker_to_indices:
            speaker_indices = self.speaker_to_indices[target_speaker].to(device)
            speaker_mask = torch.zeros_like(mask)
            speaker_mask[speaker_indices] = True
            mask &= speaker_mask

        # 性别约束
        if self.use_gender:
            mask &= (self.genders == target_gender)

        # 音素约束
        if self.use_phone and self.phones is not None:
            phone_mask = (self.phones == phone_id)
            if (mask & phone_mask).sum() >= self.k:
                mask &= phone_mask

        # 获取候选
        candidates = self.prototypes[mask]

        if len(candidates) == 0:
            # Fallback: 仅性别约束
            mask = (self.genders == target_gender)
            candidates = self.prototypes[mask]

        if len(candidates) == 0:
            return h  # 最终 fallback

        # kNN
        dist = ((h.unsqueeze(0) - candidates) ** 2).sum(dim=-1)
        k_actual = min(self.k, len(candidates))
        _, topk_idx = dist.topk(k_actual, largest=False)

        # 加权平均
        topk_dist = dist[topk_idx]
        weights = torch.softmax(-topk_dist / self.temperature, dim=0)
        h_anon = (candidates[topk_idx] * weights.unsqueeze(-1)).sum(dim=0)

        return h_anon

    def _retrieve_knn_subset(
        self,
        h_subset: torch.Tensor,
        phone_id: int,
        target_gender: int,
        target_speaker: Optional[str],
    ) -> torch.Tensor:
        """批量 kNN 检索"""
        N, D = h_subset.shape
        h_anon = torch.zeros_like(h_subset)

        for i in range(N):
            h_anon[i] = self._retrieve_knn_single(
                h_subset[i], phone_id, target_gender, target_speaker
            )

        return h_anon

    def retrieve_batch(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        target_gender: int = 0,
    ) -> torch.Tensor:
        """批量检索 (兼容接口)"""
        return self.retrieve(h_clean, phones, symbols, target_gender)


def create_retriever(
    target_pool_path: str,
    use_prototype: bool = True,
    **kwargs
) -> nn.Module:
    """
    工厂函数：根据 pool 类型创建合适的检索器

    Args:
        target_pool_path: Pool 路径
        use_prototype: 是否强制使用 Prototype 检索器
        **kwargs: 传递给检索器的参数

    Returns:
        检索器实例
    """
    pool_path = Path(target_pool_path)

    # 检测 pool 类型
    metadata_path = pool_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        pool_type = metadata.get('pool_type', 'frame_level')
    else:
        pool_type = 'frame_level'

    if pool_type == 'speaker_prototype' or use_prototype:
        print(f"[Factory] Creating PrototypeKNNRetriever for {pool_type} pool")
        return PrototypeKNNRetriever(target_pool_path, **kwargs)
    else:
        print(f"[Factory] Creating ConstrainedKNNRetriever for {pool_type} pool")
        from .retriever import ConstrainedKNNRetriever
        return ConstrainedKNNRetriever(target_pool_path, **kwargs)
