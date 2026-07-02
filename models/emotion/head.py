"""models/emotion/head.py

句级情感门控用的情感头。

接口铁律（与 pipeline 的 _sad_prob 对齐，禁止改动）：
  输入 : mean-pooled WavLM-Large 第 24 层特征 [B, 1024]
  输出 : logits [B, 4]，顺序固定为 (SAD, NEU, ANG, HAP)，SAD=0
  调用 : EmotionHead.load(path, device) → head(feat) 返回 [1,4]

设计要点：
  - 输入标准化（训练集 mean/std）通过 buffer 烤进 forward，
    这样推理侧 pipeline 直接喂「原始」均值池化特征即可，无需任何改动。
  - 训练数据来自 CREMA-D / ESD（外部数据），禁用 IEMOCAP（= SER 评估集）。
"""

import torch
import torch.nn as nn

# 标签顺序铁律：必须与 pipeline 的 SAD_IDX=0 一致
EMOTION_LABELS = ['sad', 'neu', 'ang', 'hap']
LABEL2IDX = {name: i for i, name in enumerate(EMOTION_LABELS)}


class EmotionHead(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, n_classes=4, dropout=0.3):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hidden=hidden,
                        n_classes=n_classes, dropout=dropout)

        # 输入标准化统计量（训练集），随 checkpoint 保存/恢复
        self.register_buffer('feat_mean', torch.zeros(in_dim))
        self.register_buffer('feat_std',  torch.ones(in_dim))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def set_norm_stats(self, mean, std):
        """训练结束后由训练脚本写入训练集统计量。"""
        with torch.no_grad():
            self.feat_mean.copy_(mean.flatten())
            self.feat_std.copy_(std.flatten().clamp_min(1e-6))

    def forward(self, x):
        # x: [B,1024] 原始均值池化特征 → 内部标准化 → MLP
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x)

    def save(self, path):
        torch.save({'state_dict': self.state_dict(), 'config': self.cfg}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        ckpt = torch.load(str(path), map_location='cpu')
        cfg = ckpt.get('config', {})
        model = cls(**cfg) if cfg else cls()
        model.load_state_dict(ckpt['state_dict'])
        model.to(device).eval()
        return model