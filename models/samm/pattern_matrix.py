# models/samm/pattern_matrix.py

"""Pattern Matrix - Online Inference Only"""

import torch
import torch.nn as nn


class PatternMatrix(nn.Module):
    """
    Pattern Matrix [Online Stage 3.4]
    
    基于转移概率填充被掩码的符号
    """
    
    def __init__(self, codebook_size: int = 512):
        super().__init__()
        self.K = codebook_size
        self.register_buffer('M', torch.zeros(codebook_size, codebook_size))
        self.register_buffer('marginal', torch.ones(codebook_size) / codebook_size)
    
    @torch.inference_mode()
    def smooth_sequence(
        self,
        z_masked: torch.Tensor,
        mask_indicator: torch.Tensor,
    ) -> torch.Tensor:
        """
        填充被掩码位置
        
        Args: 
            z_masked: [T] 掩码后的符号序列
            mask_indicator: [T] True 表示被掩码
        Returns: 
            z_smooth: [T] 填充后的序列
        """
        T = len(z_masked)
        z_smooth = z_masked.clone()
        
        # 前向填充
        for t in range(T):
            if mask_indicator[t]:
                if t > 0 and not mask_indicator[t - 1]:
                    prev_sym = z_smooth[t - 1].item()
                    prob = self.M[prev_sym]
                    z_smooth[t] = torch.multinomial(prob, 1).item()
                else:
                    z_smooth[t] = torch.multinomial(self.marginal, 1).item()
        
        # 后向修正（双向融合）
        for t in range(T - 2, -1, -1):
            if mask_indicator[t] and t < T - 1 and not mask_indicator[t + 1]:
                next_sym = z_smooth[t + 1].item()
                backward_prob = self.M[:, next_sym]
                backward_prob = backward_prob / (backward_prob.sum() + 1e-10)
                
                if t > 0 and not mask_indicator[t - 1]:
                    prev_sym = z_smooth[t - 1].item()
                    forward_prob = self.M[prev_sym]
                    combined = forward_prob * backward_prob
                    combined = combined / (combined.sum() + 1e-10)
                    z_smooth[t] = torch.multinomial(combined, 1).item()
        
        return z_smooth
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PatternMatrix':
        ckpt = torch.load(path, map_location='cpu')
        model = cls(codebook_size=ckpt['codebook_size'])
        model.M = ckpt['M']
        model.marginal = ckpt['marginal']
        return model.to(device)