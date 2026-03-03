# models/samm/masking.py
"""
SAMM (Self-supervised Anonymization with Masked Modeling) 掩码模块
用于增强语音匿名化的不可链接性
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaskConfig:
    """掩码配置"""
    token_mask_ratio: float = 0.10      # 随机 token 掩码比例
    span_mask_ratio: float = 0.15       # span 掩码比例
    min_span: int = 3                   # 最小 span 长度
    max_span: int = 10                  # 最大 span 长度
    duration_noise_std: float = 0.15    # 时长扰动标准差
    duration_quant_step: float = 0.02   # 时长量化步长
    rhythm_window: int = 5              # 节奏打乱窗口
    rhythm_shuffle_prob: float = 0.3    # 节奏打乱概率


class SAMMMasker:
    """
    SAMM 掩码器
    
    功能:
    1. Token-level 随机掩码
    2. Span-level 连续掩码
    3. 基于 Pattern Matrix 的掩码填充
    4. 时长扰动
    5. 节奏打乱
    """
    
    def __init__(
        self,
        codebook_path: str,
        pattern_path: str,
        config: Optional[MaskConfig] = None,
        device: str = 'cuda',
    ):
        self.config = config or MaskConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载 Codebook
        codebook_ckpt = torch.load(codebook_path, map_location='cpu')
        self.codebook = codebook_ckpt['codebook'].float().to(self.device)
        self.codebook_size = self.codebook.shape[0]
        
        # 加载 Pattern Matrix
        pattern_ckpt = torch.load(pattern_path, map_location='cpu')
        self.pattern_matrix = pattern_ckpt['M'].float().numpy()  # [K, K]
        self.marginal = pattern_ckpt['marginal'].float().numpy()  # [K]
        
        logger.info(f"SAMM Masker initialized: codebook={self.codebook_size}, device={self.device}")
    
    def apply(
        self,
        features: np.ndarray,
        return_details: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        应用 SAMM 掩码
        
        Args:
            features: [T, D] 输入特征
            return_details: 是否返回详细掩码信息
        
        Returns:
            masked_features: [T, D] 掩码后特征
            info: 掩码信息字典
        """
        T, D = features.shape
        
        # 1. 量化到 codebook
        symbols = self._quantize(features)  # [T]
        
        # 2. 生成掩码
        mask = self._generate_mask(T)  # [T] bool
        
        # 3. 使用 Pattern Matrix 填充掩码位置
        filled_symbols = self._fill_masked(symbols, mask)
        
        # 4. 从 codebook 取回特征
        masked_features = self._symbols_to_features(filled_symbols)
        
        # 5. 混合: 非掩码位置保持原特征
        masked_features[~mask] = features[~mask]
        
        info = {
            'mask': mask,
            'mask_ratio': mask.sum() / T,
            'original_symbols': symbols,
            'filled_symbols': filled_symbols,
        }
        
        return masked_features, info
    
    @torch.no_grad()
    def _quantize(self, features: np.ndarray) -> np.ndarray:
        """量化特征到 codebook 索引"""
        x = torch.from_numpy(features).float().to(self.device)  # [T, D]
        
        # 计算距离
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [T, 1]
        c_norm = (self.codebook ** 2).sum(dim=1).unsqueeze(0)  # [1, K]
        dist = x_norm + c_norm - 2.0 * (x @ self.codebook.T)  # [T, K]
        
        # 取最近邻
        symbols = torch.argmin(dist, dim=1).cpu().numpy()  # [T]
        
        return symbols
    
    def _generate_mask(self, T: int) -> np.ndarray:
        """生成组合掩码 (token + span)"""
        mask = np.zeros(T, dtype=bool)
        
        # Token-level 掩码
        n_token_mask = int(T * self.config.token_mask_ratio)
        if n_token_mask > 0:
            token_indices = np.random.choice(T, n_token_mask, replace=False)
            mask[token_indices] = True
        
        # Span-level 掩码
        n_span_mask = int(T * self.config.span_mask_ratio)
        masked_so_far = mask.sum()
        
        while masked_so_far < n_token_mask + n_span_mask:
            # 随机选择 span 起点
            start = np.random.randint(0, T)
            
            # 随机 span 长度
            span_len = np.random.randint(self.config.min_span, self.config.max_span + 1)
            end = min(start + span_len, T)
            
            mask[start:end] = True
            masked_so_far = mask.sum()
        
        return mask
    
    def _fill_masked(self, symbols: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """使用 Pattern Matrix 填充掩码位置"""
        filled = symbols.copy()
        masked_indices = np.where(mask)[0]
        
        for idx in masked_indices:
            # 获取上下文
            prev_sym = filled[idx - 1] if idx > 0 else None
            next_sym = symbols[idx + 1] if idx < len(symbols) - 1 else None
            
            # 基于 Pattern Matrix 采样
            if prev_sym is not None and next_sym is not None:
                # 双向约束: P(z | z_prev) * P(z_next | z)
                forward_prob = self.pattern_matrix[prev_sym]
                backward_prob = self.pattern_matrix[:, next_sym]
                joint_prob = forward_prob * backward_prob
                joint_prob = joint_prob / (joint_prob.sum() + 1e-10)
            elif prev_sym is not None:
                # 只有前向
                joint_prob = self.pattern_matrix[prev_sym]
            elif next_sym is not None:
                # 只有后向
                joint_prob = self.pattern_matrix[:, next_sym]
                joint_prob = joint_prob / (joint_prob.sum() + 1e-10)
            else:
                # 使用边缘分布
                joint_prob = self.marginal
            
            # 采样
            filled[idx] = np.random.choice(self.codebook_size, p=joint_prob)
        
        return filled
    
    def _symbols_to_features(self, symbols: np.ndarray) -> np.ndarray:
        """将 symbol 序列转换为特征"""
        return self.codebook[symbols].cpu().numpy()
    
    # =========================================================================
    # 高级掩码方法
    # =========================================================================
    
    def apply_duration_perturbation(
        self,
        features: np.ndarray,
        phone_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        时长扰动
        在 phone 边界处进行插值或删除
        """
        if phone_boundaries is None:
            # 简单做法: 均匀分段
            T = len(features)
            seg_len = 10
            phone_boundaries = [(i, min(i + seg_len, T)) for i in range(0, T, seg_len)]
        
        perturbed_segments = []
        
        for start, end in phone_boundaries:
            segment = features[start:end]
            orig_len = len(segment)
            
            # 随机扰动时长
            noise = np.random.normal(0, self.config.duration_noise_std)
            new_len = max(1, int(orig_len * (1 + noise)))
            
            # 量化
            new_len = int(np.round(new_len / self.config.duration_quant_step) * self.config.duration_quant_step * 50)
            new_len = max(1, new_len)
            
            # 插值
            if new_len != orig_len:
                indices = np.linspace(0, orig_len - 1, new_len)
                segment = np.array([
                    segment[int(i)] * (1 - (i % 1)) + segment[min(int(i) + 1, orig_len - 1)] * (i % 1)
                    for i in indices
                ])
            
            perturbed_segments.append(segment)
        
        return np.concatenate(perturbed_segments, axis=0)
    
    def apply_rhythm_shuffle(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        节奏打乱
        在小窗口内打乱帧顺序
        """
        T = len(features)
        result = features.copy()
        
        window = self.config.rhythm_window
        
        for i in range(0, T - window, window):
            if np.random.random() < self.config.rhythm_shuffle_prob:
                # 打乱窗口内的顺序
                perm = np.random.permutation(window)
                result[i:i+window] = result[i:i+window][perm]
        
        return result
    
    def full_pipeline(
        self,
        features: np.ndarray,
        enable_duration: bool = True,
        enable_rhythm: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        完整 SAMM 处理流程
        """
        info = {}
        
        # 1. Token/Span 掩码
        features, mask_info = self.apply(features)
        info['mask'] = mask_info
        
        # 2. 时长扰动
        if enable_duration:
            orig_len = len(features)
            features = self.apply_duration_perturbation(features)
            info['duration_change'] = len(features) / orig_len
        
        # 3. 节奏打乱
        if enable_rhythm:
            features = self.apply_rhythm_shuffle(features)
            info['rhythm_shuffled'] = True
        
        return features, info


# =============================================================================
# 便捷函数
# =============================================================================

def create_samm_masker(config: Dict) -> SAMMMasker:
    """从配置创建 SAMM Masker"""
    from pathlib import Path
    
    checkpoint_dir = Path(config['paths']['checkpoints_dir'])
    samm_cfg = config.get('samm', {}).get('masking', {})
    
    mask_config = MaskConfig(
        token_mask_ratio=samm_cfg.get('token_mask_ratio', 0.10),
        span_mask_ratio=samm_cfg.get('span_mask_ratio', 0.15),
        min_span=samm_cfg.get('min_span', 3),
        max_span=samm_cfg.get('max_span', 10),
        duration_noise_std=samm_cfg.get('duration_noise_std', 0.15),
        duration_quant_step=samm_cfg.get('duration_quant_step', 0.02),
        rhythm_window=samm_cfg.get('rhythm_window', 5),
        rhythm_shuffle_prob=samm_cfg.get('rhythm_shuffle_prob', 0.3),
    )
    
    return SAMMMasker(
        codebook_path=str(checkpoint_dir / 'codebook.pt'),
        pattern_path=str(checkpoint_dir / 'pattern_matrix.pt'),
        config=mask_config,
        device=config.get('device', 'cuda'),
    )


# =============================================================================
# 向后兼容包装类
# =============================================================================

class ProsodyAwareMasking:
    """
    简化的掩码接口 - 用于向后兼容
    只执行 token 和 span 掩码，不依赖 codebook 和 pattern matrix
    """

    def __init__(
        self,
        token_mask_ratio: float = 0.10,
        span_mask_ratio: float = 0.15,
        min_span: int = 3,
        max_span: int = 10,
        mask_token_id: int = -1,
    ):
        self.token_mask_ratio = token_mask_ratio
        self.span_mask_ratio = span_mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self.mask_token_id = mask_token_id

    def __call__(
        self,
        symbols: torch.Tensor,
        durations: torch.Tensor = None,
    ):
        """
        执行掩码操作

        Args:
            symbols: [T] 符号序列
            durations: [T] 时长序列（可选）

        Returns:
            masked_symbols: [T] 掩码后的符号
            masked_durations: [T] 掩码后的时长（如果提供）
            mask_indicator: [T] 掩码指示器 (1=被掩码, 0=未掩码)
        """
        T = len(symbols)
        device = symbols.device

        # 初始化掩码指示器
        mask_indicator = torch.zeros(T, dtype=torch.bool, device=device)

        # Token-level 随机掩码
        num_token_masks = int(T * self.token_mask_ratio)
        if num_token_masks > 0:
            token_indices = torch.randperm(T, device=device)[:num_token_masks]
            mask_indicator[token_indices] = True

        # Span-level 连续掩码
        num_span_masks = int(T * self.span_mask_ratio)
        masked_count = 0

        while masked_count < num_span_masks:
            span_len = torch.randint(self.min_span, self.max_span + 1, (1,)).item()
            start = torch.randint(0, max(1, T - span_len), (1,)).item()
            end = min(start + span_len, T)

            mask_indicator[start:end] = True
            masked_count += (end - start)

        # 应用掩码
        masked_symbols = symbols.clone()
        masked_symbols[mask_indicator] = self.mask_token_id

        masked_durations = durations
        if durations is not None:
            masked_durations = durations.clone()

        return masked_symbols, masked_durations, mask_indicator