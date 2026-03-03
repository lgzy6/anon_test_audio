# models/samm/target_selector.py

"""
TargetSelector - Pattern-based Target Pool Selection
基于 Pattern 的目标池选择器

核心思想:
- SAMM 的 Pattern Matrix 用于建模说话人的语调/韵律模式
- 不再用于 Mask Query，而是用于选择"不同风格"的 Target Pool
- 通过选择与源说话人不同 Pattern 的目标池，实现更好的匿名化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PatternClassifier(nn.Module):
    """
    Pattern 分类器

    基于帧级特征预测说话人属于哪个 Pattern 聚类
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        n_patterns: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_patterns = n_patterns

        # 简单的 MLP 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_patterns),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        预测 Pattern

        Args:
            h: [T, D] 或 [B, T, D] 特征序列
        Returns:
            pattern_id: 预测的 Pattern ID
        """
        # 聚合为 utterance 级别表示
        if h.dim() == 2:
            h_agg = h.mean(dim=0)  # [D]
        else:
            h_agg = h.mean(dim=1)  # [B, D]

        logits = self.classifier(h_agg)
        return logits.argmax(dim=-1)


class TargetSelector:
    """
    目标池选择器 [Online Stage 3 新角色]

    利用 SAMM 的 Pattern 知识选择匿名化目标池:
    1. 预测源说话人的 Pattern
    2. 选择一个不同的 Pattern 作为目标
    3. 返回对应 Pattern 的特征池

    这样可以确保匿名化后的语音具有不同的说话风格，
    而不仅仅是不同的音色。
    """

    def __init__(
        self,
        pool_dir: str,
        n_patterns: int = 8,
        selection_strategy: str = 'farthest',  # 'farthest', 'random', 'fixed'
        fixed_target_pattern: Optional[int] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            pool_dir: 目标池目录 (应包含 pool_pattern_0/, pool_pattern_1/, ...)
            n_patterns: Pattern 数量
            selection_strategy: 目标选择策略
                - 'farthest': 选择与源 Pattern 距离最远的
                - 'random': 随机选择一个不同的 Pattern
                - 'fixed': 使用固定的目标 Pattern
            fixed_target_pattern: 当 strategy='fixed' 时使用
            device: 计算设备
        """
        self.pool_dir = Path(pool_dir)
        self.n_patterns = n_patterns
        self.strategy = selection_strategy
        self.fixed_target = fixed_target_pattern
        self.device = device

        # 加载 Pattern 中心 (用于预测源 Pattern)
        self.pattern_centroids = self._load_pattern_centroids()

        # 预加载所有 Pattern 池的路径
        self.pattern_pools = self._discover_pattern_pools()

        # 当前加载的池
        self._current_pool = None
        self._current_pattern_id = None

        logger.info(f"[TargetSelector] Initialized with {len(self.pattern_pools)} pattern pools")
        logger.info(f"  Strategy: {selection_strategy}")

    def _load_pattern_centroids(self) -> Optional[torch.Tensor]:
        """加载 Pattern 聚类中心"""
        centroid_path = self.pool_dir / 'pattern_centroids.pt'

        if centroid_path.exists():
            data = torch.load(centroid_path, map_location='cpu')
            if isinstance(data, dict):
                centroids = data.get('centroids', data.get('pattern_centroids'))
            else:
                centroids = data
            logger.info(f"  Loaded pattern centroids: {centroids.shape}")
            return centroids.to(self.device)

        logger.warning(f"  Pattern centroids not found at {centroid_path}")
        return None

    def _discover_pattern_pools(self) -> Dict[int, Path]:
        """发现所有 Pattern 池目录"""
        pools = {}

        # 查找 pool_pattern_X/ 目录
        for p in self.pool_dir.iterdir():
            if p.is_dir() and p.name.startswith('pool_pattern_'):
                try:
                    pattern_id = int(p.name.split('_')[-1])
                    pools[pattern_id] = p
                except ValueError:
                    continue

        # 如果没有分区池，使用整个 pool_dir 作为唯一池
        if not pools:
            logger.warning("  No partitioned pools found, using single pool mode")
            pools[0] = self.pool_dir

        return pools

    @torch.inference_mode()
    def predict_pattern(self, h: torch.Tensor) -> int:
        """
        预测输入特征属于哪个 Pattern

        Args:
            h: [T, D] 特征序列
        Returns:
            pattern_id: 预测的 Pattern ID
        """
        if self.pattern_centroids is None:
            # 没有 centroids，返回随机 pattern
            return np.random.randint(0, self.n_patterns)

        # 计算 utterance 级别表示
        h_agg = h.mean(dim=0).to(self.device)  # [D]

        # 找最近的 Pattern 中心
        dist = torch.cdist(
            h_agg.unsqueeze(0),
            self.pattern_centroids
        )  # [1, n_patterns]

        pattern_id = dist.argmin().item()
        return pattern_id

    def select_target_pattern(self, source_pattern: int) -> int:
        """
        选择目标 Pattern

        Args:
            source_pattern: 源说话人的 Pattern ID
        Returns:
            target_pattern: 目标 Pattern ID (与源不同)
        """
        available = [p for p in self.pattern_pools.keys() if p != source_pattern]

        if not available:
            # 只有一个池，无法选择不同的
            return source_pattern

        if self.strategy == 'fixed' and self.fixed_target is not None:
            if self.fixed_target in available:
                return self.fixed_target
            return available[0]

        elif self.strategy == 'random':
            return np.random.choice(available)

        elif self.strategy == 'farthest':
            if self.pattern_centroids is None:
                return np.random.choice(available)

            # 选择与源 Pattern 距离最远的
            source_centroid = self.pattern_centroids[source_pattern]
            max_dist = -1
            farthest = available[0]

            for p in available:
                if p < len(self.pattern_centroids):
                    dist = torch.norm(
                        self.pattern_centroids[p] - source_centroid
                    ).item()
                    if dist > max_dist:
                        max_dist = dist
                        farthest = p

            return farthest

        else:
            return np.random.choice(available)

    def get_pool_path(self, pattern_id: int) -> Path:
        """获取指定 Pattern 的池路径"""
        if pattern_id in self.pattern_pools:
            return self.pattern_pools[pattern_id]

        # Fallback: 使用第一个可用的池
        return list(self.pattern_pools.values())[0]

    def select_pool(
        self,
        h: torch.Tensor,
        source_gender: Optional[str] = None,
    ) -> Tuple[Path, int, int]:
        """
        选择目标池 (完整流程)

        Args:
            h: [T, D] 源特征
            source_gender: 源性别
        Returns:
            pool_path: 目标池路径
            source_pattern: 源 Pattern ID
            target_pattern: 目标 Pattern ID
        """
        # 1. 预测源 Pattern
        source_pattern = self.predict_pattern(h)

        # 2. 选择目标 Pattern
        target_pattern = self.select_target_pattern(source_pattern)

        # 3. 获取目标池路径
        pool_path = self.get_pool_path(target_pattern)

        logger.debug(f"  Pattern: {source_pattern} -> {target_pattern}")

        return pool_path, source_pattern, target_pattern


class AdaptiveTargetSelector(TargetSelector):
    """
    自适应目标选择器

    根据源说话人的特征动态调整目标选择策略:
    - 高能量说话人 -> 选择低能量目标
    - 快语速说话人 -> 选择慢语速目标
    """

    def __init__(
        self,
        pool_dir: str,
        n_patterns: int = 8,
        device: str = 'cuda',
    ):
        super().__init__(
            pool_dir=pool_dir,
            n_patterns=n_patterns,
            selection_strategy='farthest',
            device=device,
        )

        # 加载 Pattern 属性 (如果有)
        self.pattern_attributes = self._load_pattern_attributes()

    def _load_pattern_attributes(self) -> Optional[Dict]:
        """加载 Pattern 属性描述"""
        attr_path = self.pool_dir / 'pattern_attributes.json'

        if attr_path.exists():
            import json
            with open(attr_path, 'r') as f:
                return json.load(f)

        return None

    def select_target_pattern(self, source_pattern: int) -> int:
        """
        自适应选择目标 Pattern

        如果有属性信息，选择属性最不同的 Pattern
        否则退化为 'farthest' 策略
        """
        if self.pattern_attributes is None:
            return super().select_target_pattern(source_pattern)

        available = [p for p in self.pattern_pools.keys() if p != source_pattern]
        if not available:
            return source_pattern

        # 基于属性差异选择
        source_attrs = self.pattern_attributes.get(str(source_pattern), {})

        best_target = available[0]
        max_diff = -1

        for p in available:
            target_attrs = self.pattern_attributes.get(str(p), {})

            # 计算属性差异
            diff = 0
            for key in source_attrs:
                if key in target_attrs:
                    diff += abs(source_attrs[key] - target_attrs[key])

            if diff > max_diff:
                max_diff = diff
                best_target = p

        return best_target


def create_target_selector(
    pool_dir: str,
    config: Optional[Dict] = None,
    device: str = 'cuda',
) -> TargetSelector:
    """
    工厂函数: 创建 TargetSelector

    Args:
        pool_dir: 目标池目录
        config: 配置字典
        device: 计算设备
    Returns:
        TargetSelector 实例
    """
    if config is None:
        config = {}

    n_patterns = config.get('n_patterns', 8)
    strategy = config.get('selection_strategy', 'farthest')
    use_adaptive = config.get('use_adaptive', False)

    if use_adaptive:
        return AdaptiveTargetSelector(
            pool_dir=pool_dir,
            n_patterns=n_patterns,
            device=device,
        )

    return TargetSelector(
        pool_dir=pool_dir,
        n_patterns=n_patterns,
        selection_strategy=strategy,
        device=device,
    )
