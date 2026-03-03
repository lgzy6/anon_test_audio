# models/samm/prosody.py

"""韵律特征提取与匿名化"""

import torch
import numpy as np
from typing import Tuple, Optional


class ProsodyExtractor:
    """
    韵律特征提取器 [Online Stage 3.2]
    
    提取 F0 和时长信息
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,  # 20ms for 16kHz
        f0_min: float = 50.0,
        f0_max: float = 600.0,
    ):
        self.sr = sample_rate
        self.hop = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def extract_f0(self, waveform: np.ndarray) -> np.ndarray:
        """
        提取 F0 (使用简单的方法，可替换为 WORLD/DIO)
        
        Args:
            waveform: [L] 音频波形
        Returns:
            f0: [T] 基频序列
        """
        # 简化实现：可以用 pyworld 或 librosa
        # 这里先返回占位符
        try:
            import pyworld as pw
            _f0, t = pw.dio(
                waveform.astype(np.float64),
                self.sr,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=self.hop / self.sr * 1000,
            )
            f0 = pw.stonemask(waveform.astype(np.float64), _f0, t, self.sr)
            return f0.astype(np.float32)
        except ImportError:
            # Fallback: 返回全零（无 F0）
            num_frames = len(waveform) // self.hop
            return np.zeros(num_frames, dtype=np.float32)
    
    def estimate_duration(
        self,
        num_ssl_frames: int,
        hop_length: int = 320,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        估计每帧时长（简化版：均匀分配）
        
        实际应用中可以用 MFA 对齐获取真实时长
        """
        frame_duration = hop_length / sample_rate  # 秒
        return torch.full((num_ssl_frames,), frame_duration)


class ProsodyAnonymizer:
    """
    韵律匿名化器 [Online Stage 4.2]
    """
    
    def __init__(
        self,
        f0_shift_range: Tuple[float, float] = (-50, 50),
        duration_noise_std: float = 0.1,
        duration_quant_step: float = 0.02,
    ):
        self.f0_shift_range = f0_shift_range
        self.duration_noise_std = duration_noise_std
        self.duration_quant_step = duration_quant_step
    
    def anonymize_f0(self, f0: np.ndarray) -> np.ndarray:
        """F0 匿名化：标准化 + 随机偏移"""
        voiced = f0 > 0
        if voiced.sum() == 0:
            return f0
        
        # 标准化
        mean_f0 = f0[voiced].mean()
        std_f0 = f0[voiced].std() + 1e-6
        f0_norm = np.zeros_like(f0)
        f0_norm[voiced] = (f0[voiced] - mean_f0) / std_f0
        
        # 随机偏移
        shift = np.random.uniform(*self.f0_shift_range)
        target_mean = 150.0 + shift  # 中性目标
        target_std = 30.0
        
        f0_anon = np.zeros_like(f0)
        f0_anon[voiced] = f0_norm[voiced] * target_std + target_mean
        
        return f0_anon
    
    def anonymize_duration(self, duration: torch.Tensor) -> torch.Tensor:
        """时长匿名化：噪声 + 量化"""
        noise = torch.randn_like(duration) * self.duration_noise_std
        d_noisy = duration * (1 + noise)
        
        # 量化
        d_quant = torch.round(d_noisy / self.duration_quant_step) * self.duration_quant_step
        
        return torch.clamp(d_quant, min=0.02, max=0.5)