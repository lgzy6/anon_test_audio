# models/vocoder/hifigan.py

"""
HiFi-GAN Vocoder 包装类
用于将 WavLM 特征转换为波形
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .hifigan_model import Generator, AttrDict


class HiFiGAN(nn.Module):
    """
    HiFi-GAN Vocoder 包装类

    支持从 kNN-VC 预训练权重加载
    """

    def __init__(
        self,
        config: Union[str, dict, AttrDict],
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # 加载配置
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = json.load(f)
        if isinstance(config, dict):
            config = AttrDict(config)

        self.config = config
        self.hop_size = config.hop_size
        self.sample_rate = config.sampling_rate

        # 初始化生成器
        self.generator = Generator(config)
        self.generator.to(device)

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ) -> 'HiFiGAN':
        """
        从检查点加载模型

        Args:
            checkpoint_path: 模型权重路径
            config_path: 配置文件路径 (可选，默认与权重同目录)
            device: 设备
        """
        ckpt_path = Path(checkpoint_path)

        # 自动查找配置文件
        if config_path is None:
            config_path = ckpt_path.parent / "hifigan.json"

        # 创建实例
        vocoder = cls(str(config_path), device)

        # 加载权重
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        vocoder.generator.load_state_dict(state_dict)

        # 推理优化
        vocoder.generator.eval()
        vocoder.generator.remove_weight_norm()

        return vocoder

    @torch.inference_mode()
    def forward(self, features: Tensor) -> Tensor:
        """
        将 WavLM 特征转换为波形

        Args:
            features: [T, D] 或 [B, T, D] WavLM 特征

        Returns:
            [L] 或 [B, L] 波形
        """
        # 处理输入维度
        squeeze_output = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze_output = True

        features = features.to(self.device)

        # 生成波形
        waveform = self.generator(features)  # [B, 1, L]

        # 处理输出维度
        waveform = waveform.squeeze(1)  # [B, L]
        if squeeze_output:
            waveform = waveform.squeeze(0)  # [L]

        return waveform

    def synthesize(
        self,
        features: Tensor,
        n_frames: Optional[int] = None,
    ) -> Tensor:
        """
        合成波形并裁剪到正确长度

        Args:
            features: WavLM 特征
            n_frames: 特征帧数 (用于计算输出长度)

        Returns:
            波形
        """
        waveform = self.forward(features)

        if n_frames is not None:
            expected_len = n_frames * self.hop_size
            waveform = waveform[..., :expected_len]

        return waveform

    def to(self, device: str) -> 'HiFiGAN':
        """移动到指定设备"""
        self.device = device
        self.generator.to(device)
        return self
