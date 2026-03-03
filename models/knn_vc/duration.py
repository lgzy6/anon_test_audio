# models/knn_vc/duration.py

"""
Duration 模块 - 时长预测与匿名化
融合 Private kNN-VC 的预测器 + SAMM 的扰动方法
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class DurationPredictor(nn.Module):
    """
    时长预测器
    
    基于音素序列预测时长（说话人无关）
    训练在单说话人数据上，消除说话人时长特征
    """
    
    def __init__(
        self,
        num_phones: int = 41,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.phone_embedding = nn.Embedding(num_phones, embedding_dim)
        
        layers = []
        in_dim = embedding_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
    
    @torch.inference_mode()
    def forward(self, phone_ids: torch.Tensor) -> torch.Tensor:
        """
        预测时长
        
        Args:
            phone_ids: [num_phones] 音素ID序列
        Returns:
            durations: [num_phones] 预测的帧数
        """
        emb = self.phone_embedding(phone_ids)
        dur = self.predictor(emb).squeeze(-1)
        return torch.relu(dur)  # 时长必须非负
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DurationPredictor':
        ckpt = torch.load(path, map_location='cpu')
        if 'config' in ckpt:
            model = cls(**ckpt['config'])
            model.load_state_dict(ckpt['model'])
        else:
            model = cls()
            model.load_state_dict(ckpt)
        return model.to(device).eval()


class DurationAnonymizer:
    """
    时长匿名化器 [Online Stage 3.3 + 4.2]
    
    融合预测器时长和真实时长，添加扰动
    """
    
    def __init__(
        self,
        predictor: Optional[DurationPredictor] = None,
        predictor_weight: float = 0.7,
        noise_std: float = 0.1,
        quant_step: float = 1.0,  # 帧为单位
        min_duration: int = 1,
        max_duration: int = 50,
    ):
        self.predictor = predictor
        self.w = predictor_weight
        self.noise_std = noise_std
        self.quant_step = quant_step
        self.min_dur = min_duration
        self.max_dur = max_duration
    
    def anonymize(
        self,
        phone_ids: torch.Tensor,
        true_durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        时长匿名化

        d_anon = w * d_pred + (1-w) * d_true + noise

        Args:
            phone_ids: [num_phones] 音素ID
            true_durations: [num_phones] 真实帧数
        Returns:
            anon_durations: [num_phones] 匿名化帧数
        """
        if self.predictor is not None and self.w > 0:
            d_pred = self.predictor(phone_ids)
            # 确保 true_durations 与 d_pred 在同一设备上
            true_durations = true_durations.to(d_pred.device)
            d_mixed = self.w * d_pred + (1 - self.w) * true_durations.float()
        else:
            d_mixed = true_durations.float()
        
        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(d_mixed) * self.noise_std
            d_noisy = d_mixed * (1 + noise)
        else:
            d_noisy = d_mixed
        
        # 量化
        if self.quant_step > 0:
            d_quant = torch.round(d_noisy / self.quant_step) * self.quant_step
        else:
            d_quant = d_noisy
        
        # 限制范围
        d_anon = torch.clamp(d_quant, min=self.min_dur, max=self.max_dur)
        
        return d_anon.long()


class DurationAdjuster:
    """
    时长调整器
    
    根据匿名化时长重采样特征帧
    """
    
    @staticmethod
    def adjust_features(
        features: torch.Tensor,
        phone_segments: List[Tuple[int, int, int]],
        target_durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        按音素边界调整特征时长
        
        Args:
            features: [T, D] 特征序列
            phone_segments: List of (start, end, phone_id)
            target_durations: [num_phones] 目标帧数
        Returns:
            adjusted: [T', D] 调整后的特征
        """
        adjusted_segments = []
        
        for i, (start, end, phone_id) in enumerate(phone_segments):
            segment = features[start:end]  # [seg_len, D]
            true_len = end - start
            target_len = int(target_durations[i].item())
            
            if target_len <= 0:
                target_len = 1
            
            if target_len == true_len:
                adjusted_segments.append(segment)
            else:
                # v3.0: 使用线性插值实现更平滑的时长调整
                # segment: [true_len, D] -> [target_len, D]
                import torch.nn.functional as F

                # 转换为 [1, D, true_len] 用于 interpolate
                seg_transposed = segment.T.unsqueeze(0)  # [1, D, true_len]

                # 线性插值
                resized = F.interpolate(
                    seg_transposed.float(),
                    size=target_len,
                    mode='linear',
                    align_corners=True
                )  # [1, D, target_len]

                # 转换回 [target_len, D]
                adjusted_segments.append(resized.squeeze(0).T)
        
        if len(adjusted_segments) == 0:
            return features
        
        return torch.cat(adjusted_segments, dim=0)
    
    @staticmethod
    def adjust_sequence(
        sequence: torch.Tensor,
        phone_segments: List[Tuple[int, int, int]],
        target_durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        调整 1D 序列（如符号序列）
        
        Args:
            sequence: [T] 序列
            phone_segments: List of (start, end, phone_id)
            target_durations: [num_phones] 目标帧数
        Returns:
            adjusted: [T'] 调整后的序列
        """
        adjusted_segments = []
        
        for i, (start, end, phone_id) in enumerate(phone_segments):
            segment = sequence[start:end]
            true_len = end - start
            target_len = int(target_durations[i].item())
            
            if target_len <= 0:
                target_len = 1
            
            if target_len == true_len:
                adjusted_segments.append(segment)
            else:
                # v3.0: 使用线性插值 (1D 序列版本)
                # 对于 1D 序列，使用最近邻以保持离散性
                indices = torch.linspace(0, true_len - 1, target_len)
                indices = indices.round().long().clamp(0, true_len - 1)
                adjusted_segments.append(segment[indices])
        
        if len(adjusted_segments) == 0:
            return sequence
        
        return torch.cat(adjusted_segments, dim=0)