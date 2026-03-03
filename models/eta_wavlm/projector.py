# models/eta_wavlm/projector.py

import torch
import torch.nn as nn


class EtaWavLMProjector(nn.Module):
    """
    Eta-WavLM 说话人成分去除 [Online Stage 2]
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        U_s = ckpt['U_s']  # [D, D_s]
        
        self.register_buffer('U_s', U_s)
        
        # 预计算正交投影矩阵 P_orth = I - U_s @ U_s^T
        P_orth = torch.eye(U_s.shape[0]) - U_s @ U_s.T
        self.register_buffer('P_orth', P_orth)
        
        self.feature_dim = U_s.shape[0]
        self.subspace_dim = U_s.shape[1]
        
        self.to(device)
    
    @torch.inference_mode()
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        去除说话人成分
        
        Args: 
            h: [B, T, D] 或 [T, D] WavLM 特征
        
        Returns:
            h_clean: 同形状，去说话人特征
        """
        return h @ self.P_orth
    
    def get_speaker_component(self, h: torch.Tensor) -> torch.Tensor:
        """提取说话人成分 (调试用)"""
        return h - self.forward(h)
    
    @property
    def explained_variance_ratio(self) -> float:
        """返回子空间解释方差比例 (如果有)"""
        # 可从 checkpoint 加载时保存
        return getattr(self, '_evr', None)