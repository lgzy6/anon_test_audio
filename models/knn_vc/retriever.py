# models/knn_vc/retriever.py

"""
Constrained kNN-VC Retriever
约束近邻检索器 - 融合 SAMM 符号约束 + Private kNN-VC 音素聚类
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


class ConstrainedKNNRetriever(nn.Module):
    """
    约束 kNN 检索器 [Online Stage 4]

    两级检索策略:
    - Level 1: Phone Cluster (粗粒度，高召回)
    - Level 2: Constrained kNN (细粒度，可选)

    优化项 (v2.1):
    - use_top1: 使用 Top-1 而非加权平均，保留语义离散性
    - use_cosine: 使用余弦相似度而非 L2 距离
    """

    def __init__(
        self,
        target_pool_path: str,
        k: int = 4,
        num_clusters: int = 8,
        use_phone_clusters: bool = True,
        use_knn_refinement: bool = False,
        use_symbol_constraint: bool = True,
        use_gender_constraint: bool = True,
        temperature: float = 1.0,
        use_top1: bool = True,  # 新增: Top-1 策略开关
        use_cosine: bool = True,  # 新增: 余弦相似度开关
        device: str = 'cuda',
    ):
        super().__init__()

        self.k = k
        self.num_clusters = num_clusters
        self.use_phone_clusters = use_phone_clusters
        self.use_knn_refinement = use_knn_refinement
        self.use_symbol = use_symbol_constraint
        self.use_gender = use_gender_constraint
        self.temperature = temperature
        self.use_top1 = use_top1  # 新增
        self.use_cosine = use_cosine  # 新增
        self.device_str = device

        # 加载 Target Pool
        self._load_target_pool(target_pool_path)

        # 移动到指定设备
        self.to(device)
    
    def _load_target_pool(self, path: str):
        """加载目标池"""
        pool_path = Path(path)
        
        if pool_path.is_dir():
            # 目录形式 (兼容新的 pool_building 输出)
            
            # 加载特征 (支持 .npy 或 .h5)
            if (pool_path / 'features.npy').exists():
                features = np.load(pool_path / 'features.npy')
            elif (pool_path / 'features.h5').exists():
                import h5py
                with h5py.File(pool_path / 'features.h5', 'r') as f:
                    features = f['features'][:]
            else:
                raise FileNotFoundError(f"No features file in {pool_path}")
            
            pool = {
                'features': torch.from_numpy(features).float(),
                'symbols': torch.from_numpy(
                    np.load(pool_path / 'symbols.npy')
                ).long(),
                'genders': torch.from_numpy(
                    np.load(pool_path / 'genders.npy')
                ).long(),
            }
            
            # 可选：音素标签
            phones_path = pool_path / 'phones.npy'
            if phones_path.exists():
                pool['phones'] = torch.from_numpy(
                    np.load(phones_path)
                ).long()
            
            # 可选：音素聚类
            clusters_path = pool_path / 'phone_clusters.pt'
            if clusters_path.exists():
                pool['phone_clusters'] = torch.load(
                    clusters_path, map_location='cpu'
                )
            
            # 可选：符号索引 (加速检索)
            symbol_index_path = pool_path / 'symbol_index.pkl'
            if symbol_index_path.exists():
                import pickle
                with open(symbol_index_path, 'rb') as f:
                    self.symbol_index = pickle.load(f)
        else:
            # 单文件形式
            pool = torch.load(path, map_location='cpu')
        
        # 注册为 buffer
        self.register_buffer('features', pool['features'])
        self.register_buffer('symbols', pool['symbols'])
        self.register_buffer('genders', pool['genders'])
        
        if 'phones' in pool:
            self.register_buffer('phones', pool['phones'])
        else:
            self.phones = None
        
        # 音素聚类中心
        self.phone_clusters = pool.get('phone_clusters', None)
        if self.phone_clusters is not None:
            # 转换为 tensor dict
            for k, v in self.phone_clusters.items():
                if isinstance(v, np.ndarray):
                    self.phone_clusters[k] = torch.from_numpy(v).float()
        
        print(f"[KNN Retriever] Loaded target pool:")
        print(f"  - Features: {self.features.shape}")
        print(f"  - Phones: {'Yes' if self.phones is not None else 'No'}")
        print(f"  - Phone Clusters: {'Yes' if self.phone_clusters else 'No'}")
    
    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        if self.phone_clusters is not None:
            self.phone_clusters = {
                k: v.to(device) for k, v in self.phone_clusters.items()
            }
        return self
    
    @torch.inference_mode()
    def retrieve(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        target_gender: int = 0,
    ) -> torch.Tensor:
        """
        检索匿名化特征
        
        Args:
            h_clean: [T, D] 去说话人特征
            phones: [T] 音素标签
            symbols: [T] SAMM 符号 (可选)
            target_gender: 0=M, 1=F
        Returns:
            h_anon: [T, D] 匿名化特征
        """
        T, D = h_clean.shape
        h_anon = torch.zeros_like(h_clean)
        
        for t in range(T):
            phone_id = phones[t].item()
            
            # Level 1: Phone Cluster 检索
            if self.use_phone_clusters and self.phone_clusters is not None:
                h_anon[t] = self._retrieve_from_cluster(
                    h_clean[t], phone_id, target_gender
                )
            
            # Level 2: kNN 精调 (可选)
            elif self.use_knn_refinement:
                symbol = symbols[t].item() if symbols is not None else None
                h_anon[t] = self._retrieve_knn(
                    h_clean[t], phone_id, symbol, target_gender
                )
            
            # Fallback: 直接 kNN
            else:
                h_anon[t] = self._retrieve_knn(
                    h_clean[t], phone_id, None, target_gender
                )
        
        return h_anon
    
    def _retrieve_from_cluster(
        self,
        h: torch.Tensor,
        phone_id: int,
        target_gender: int,
    ) -> torch.Tensor:
        """从音素聚类中心检索"""
        # 获取该音素的聚类中心
        key = f"{phone_id}_{target_gender}"
        if key not in self.phone_clusters:
            key = str(phone_id)  # fallback: 不区分性别
        
        if key in self.phone_clusters:
            centers = self.phone_clusters[key]  # [K, D]
            # 找最近的聚类中心
            dist = ((h.unsqueeze(0) - centers) ** 2).sum(dim=-1)
            nearest_idx = dist.argmin()
            return centers[nearest_idx]
        
        # 最终 fallback
        return self._retrieve_knn(h, phone_id, None, target_gender)
    
    def _retrieve_knn(
        self,
        h: torch.Tensor,
        phone_id: int,
        symbol: Optional[int],
        target_gender: int,
    ) -> torch.Tensor:
        """约束 kNN 检索"""
        # 构建约束掩码
        mask = torch.ones(len(self.features), dtype=torch.bool, 
                         device=self.features.device)
        
        # 性别约束 (硬约束)
        if self.use_gender:
            mask &= (self.genders == target_gender)
        
        # 音素约束 (如果有)
        if self.phones is not None:
            phone_mask = (self.phones == phone_id)
            if (mask & phone_mask).sum() >= self.k:
                mask &= phone_mask
        
        # 符号约束 (软约束) - 修复: 跳过掩码符号 -1
        if self.use_symbol and symbol is not None and symbol >= 0:
            symbol_mask = (self.symbols == symbol)
            if (mask & symbol_mask).sum() >= self.k:
                mask &= symbol_mask
        
        # 获取候选
        candidates = self.features[mask]
        
        if len(candidates) == 0:
            # Fallback: 仅性别约束
            mask = (self.genders == target_gender)
            candidates = self.features[mask]
        
        if len(candidates) == 0:
            return h  # 最终 fallback
        
        # kNN 检索 - v2.1 优化
        if self.use_cosine:
            # 余弦相似度 (privateknnvc 方式)
            h_norm = h / (h.norm() + 1e-8)
            candidates_norm = candidates / (candidates.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (h_norm.unsqueeze(0) * candidates_norm).sum(dim=-1)

            if self.use_top1:
                # Top-1: 取最相似的单个目标 (推荐)
                nearest_idx = sim.argmax()
                h_anon = candidates[nearest_idx]
            else:
                # 加权平均 (保留作为 fallback)
                k_actual = min(self.k, len(candidates))
                _, topk_idx = sim.topk(k_actual, largest=True)
                topk_sim = sim[topk_idx]
                weights = torch.softmax(topk_sim / self.temperature, dim=0)
                h_anon = (candidates[topk_idx] * weights.unsqueeze(-1)).sum(dim=0)
        else:
            # L2 距离 (原始方式)
            dist = ((h.unsqueeze(0) - candidates) ** 2).sum(dim=-1)

            if self.use_top1:
                # Top-1: 取最近的单个目标
                nearest_idx = dist.argmin()
                h_anon = candidates[nearest_idx]
            else:
                # 加权平均
                k_actual = min(self.k, len(candidates))
                _, topk_idx = dist.topk(k_actual, largest=False)
                topk_dist = dist[topk_idx]
                weights = torch.softmax(-topk_dist / self.temperature, dim=0)
                h_anon = (candidates[topk_idx] * weights.unsqueeze(-1)).sum(dim=0)

        return h_anon
    
    def retrieve_batch(
        self,
        h_clean: torch.Tensor,
        phones: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        target_gender: int = 0,
    ) -> torch.Tensor:
        """批量检索 (向量化，更快)"""
        if not self.use_phone_clusters or self.phone_clusters is None:
            # 无聚类时退化为逐帧
            return self.retrieve(h_clean, phones, symbols, target_gender)
        
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
                h_anon[phone_mask] = h_subset
        
        return h_anon