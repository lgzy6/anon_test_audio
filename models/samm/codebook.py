# models/samm/codebook.py

"""SAMM Codebook - Online Inference Only"""

import torch
import torch.nn as nn


class SAMMCodebook(nn.Module):
    """
    SAMM Codebook [Online Stage 3.1]
    
    将连续特征量化为离散符号
    """
    
    def __init__(self, codebook_size: int = 512, feature_dim: int = 1024):
        super().__init__()
        self.K = codebook_size
        self.D = feature_dim
        self.register_buffer('codebook', torch.zeros(codebook_size, feature_dim))
    
    @torch.inference_mode()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """
        特征 -> 符号索引
        
        Args: 
            h: [B, T, D] 或 [T, D]
        Returns: 
            z: [B, T] 或 [T]
        """
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(0)
        
        B, T, D = h.shape
        h_flat = h.reshape(-1, D)
        
        # ||h - c||^2 = ||h||^2 + ||c||^2 - 2*h·c
        h_sq = (h_flat ** 2).sum(dim=-1, keepdim=True)
        c_sq = (self.codebook ** 2).sum(dim=-1, keepdim=True).T
        dist = h_sq + c_sq - 2 * (h_flat @ self.codebook.T)
        
        z = dist.argmin(dim=-1).reshape(B, T)
        
        return z.squeeze(0) if squeeze else z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """符号索引 -> 量化特征"""
        return self.codebook[z]
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'SAMMCodebook':
        ckpt = torch.load(path, map_location='cpu')
        model = cls(
            codebook_size=ckpt['codebook_size'],
            feature_dim=ckpt.get('feature_dim', ckpt['codebook'].shape[1]),
        )
        model.codebook = ckpt['codebook']
        return model.to(device)