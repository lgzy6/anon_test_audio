# models/vocoder/hifigan_model.py

"""
HiFi-GAN Generator 模型实现
基于 kNN-VC 的 HiFi-GAN 实现优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """计算保持序列长度的 padding"""
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    """初始化卷积层权重"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class AttrDict(dict):
    """支持属性访问的字典"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ResBlock1(nn.Module):
    """残差块类型1 - 使用3个不同膨胀率的卷积"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[i],
                padding=get_padding(kernel_size, dilation[i])
            )) for i in range(3)
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )) for _ in range(3)
        ])
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class ResBlock2(nn.Module):
    """残差块类型2 - 使用2个不同膨胀率的卷积"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[i],
                padding=get_padding(kernel_size, dilation[i])
            )) for i in range(2)
        ])
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)


class Generator(nn.Module):
    """HiFi-GAN Generator - 将 WavLM 特征转换为波形"""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # 输入投影: WavLM 维度 -> HiFi-GAN 维度
        self.lin_pre = nn.Linear(h.hubert_dim, h.hifi_dim)

        # 预卷积
        self.conv_pre = weight_norm(
            Conv1d(h.hifi_dim, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # 选择残差块类型
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        # 上采样层
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                h.upsample_initial_channel // (2 ** i),
                h.upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2
            )))

        # 残差块
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))

        # 后卷积
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # 初始化权重
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, T, D] WavLM 特征 (batch, seq_len, dim)

        Returns:
            [B, 1, T*hop_size] 波形
        """
        # 线性投影
        x = self.lin_pre(x)
        # 转换维度: [B, T, D] -> [B, D, T]
        x = x.permute(0, 2, 1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """移除权重归一化以加速推理"""
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
