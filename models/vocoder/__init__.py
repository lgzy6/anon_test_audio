# models/vocoder/__init__.py

"""
Vocoder 模块 - 用于将 WavLM 特征转换为波形
"""

from .hifigan import HiFiGAN

__all__ = ['HiFiGAN']
