"""
Feature Purifier: 基于梯度反转的说话人信息净化器

架构:
  L24 (1024-d) -> Encoder -> Z_clean (512-d)
                                |
                    +-----------+-----------+
                    |                       |
              Phone Classifier       Speaker Classifier
              (保留内容)               (经GRL对抗去除身份)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """梯度反转层：前向不变，反向乘以 -alpha"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class FeaturePurifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_phones: int = 72,
        num_speakers: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # A. 净化编码器: L24 -> Z_clean
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # B. 音素分支 (保留语言内容)
        self.phone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phones),
        )

        # C. 说话人对抗分支 (经 GRL 去除身份)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

    def forward(self, x, alpha=1.0):
        z_clean = self.encoder(x)
        phone_logits = self.phone_classifier(z_clean)
        z_reversed = grad_reverse(z_clean, alpha)
        speaker_logits = self.speaker_classifier(z_reversed)
        return z_clean, phone_logits, speaker_logits

    def encode(self, x):
        """推理时只需要 encoder 输出"""
        return self.encoder(x)