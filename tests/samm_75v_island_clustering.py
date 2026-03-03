#!/usr/bin/env python3
"""
DS-SAMM v7.4: 增强离散岛状特征空间

核心改进 (相比v7.3):
1. Speaker Adversarial Loss: 显式去除说话人信息
2. 增强的Island Loss: 更大的margin和权重
3. Cross-Speaker Consistency Loss: 确保pattern真正说话人不变
4. Pattern Orthogonality Loss: 强制patterns在空间中均匀分布
5. 改进的温度调度: 更平缓的退火曲线
"""

import sys
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 日志配置
# ============================================================

def setup_logging(log_dir: Path, timestamp: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'training_v74_{timestamp}.log'
    logger = logging.getLogger('SAMM_v74')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ============================================================
# Dataset
# ============================================================

class SpeakerUtteranceDataset(Dataset):
    def __init__(self, data_list: list, max_len: int = 500):
        self.data = data_list
        self.max_len = max_len
        self.speaker_to_indices = defaultdict(list)
        for i, d in enumerate(data_list):
            self.speaker_to_indices[d['speaker']].append(i)
        self.speakers = list(self.speaker_to_indices.keys())
        self.speaker_to_id = {s: i for i, s in enumerate(self.speakers)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        feat = d['features']
        if feat.shape[0] > self.max_len:
            start = np.random.randint(0, feat.shape[0] - self.max_len)
            feat = feat[start:start + self.max_len]
        return {
            'features': torch.from_numpy(feat).float(),
            'speaker_id': self.speaker_to_id[d['speaker']],
            'length': feat.shape[0]
        }


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    padded = []
    for b in batch:
        feat = b['features']
        if feat.shape[0] < max_len:
            pad = torch.zeros(max_len - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, pad], dim=0)
        padded.append(feat)
    return {
        'features': torch.stack(padded),
        'speaker_ids': torch.tensor([b['speaker_id'] for b in batch]),
        'lengths': torch.tensor([b['length'] for b in batch])
    }


# ============================================================
# 【新增组件1】梯度反转层
# ============================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层：前向传播不变，反向传播取负
    用于对抗学习去除说话人信息
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


# ============================================================
# 【新增组件2】说话人对抗模块
# ============================================================

class SpeakerAdversarialModule(nn.Module):
    """
    说话人对抗学习模块
    通过梯度反转让embedding无法区分说话人
    """
    def __init__(self, d_model: int = 256, n_speakers_max: int = 50):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_speakers_max)
        )
        self.register_buffer('gradient_reverse_lambda', torch.tensor(1.0))

    def forward(self, embeddings, speaker_ids):
        reversed_emb = GradientReversalLayer.apply(embeddings, self.gradient_reverse_lambda)
        logits = self.discriminator(reversed_emb)
        
        # 动态调整维度
        max_speaker_in_batch = speaker_ids.max().item() + 1
        if max_speaker_in_batch > logits.shape[1]:
            # 说话人ID超出范围，返回零损失
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        logits = logits[:, :max_speaker_in_batch]
        
        # 添加数值稳定性
        loss = F.cross_entropy(logits, speaker_ids, reduction='mean')
        return loss.clamp(min=0, max=10)
    
    def set_lambda(self, lambda_val: float):
        """动态调整梯度反转强度"""
        self.gradient_reverse_lambda.fill_(lambda_val)


# ============================================================
# 【核心组件】Sinkhorn-Knopp归一化
# ============================================================

def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 3, epsilon: float = 0.1):
    """
    Sinkhorn-Knopp归一化: 强制行列和均衡 (数值稳定版本)
    """
    with torch.no_grad():
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits / epsilon)
        Q = Q + 1e-8
        N, K = Q.shape
        
        for _ in range(n_iters):
            col_sum = Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
            Q = Q / col_sum
            row_sum = Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Q = Q / row_sum
        
        return Q


# ============================================================
# 【核心组件】DEC-style Target Distribution
# ============================================================

def compute_target_distribution(q: torch.Tensor, power: float = 2.0):
    """
    DEC-style target distribution: 让高置信度的assignment更hard
    数值稳定版本
    """
    q = q.clamp(min=1e-6, max=1.0 - 1e-6)
    f = q.sum(dim=[0, 1]).clamp(min=1e-6)
    p = (q ** power) / f.unsqueeze(0).unsqueeze(0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p = p.clamp(min=1e-6, max=1.0 - 1e-6)
    return p


# ============================================================
# 【增强版】Island Loss
# ============================================================

class IslandLossV2(nn.Module):
    """
    增强版Island Loss
    
    改进:
    1. 更大的margin要求
    2. 可调节的compactness和separation权重
    3. 数值稳定性增强
    """
    def __init__(self, margin: float = 1.2, compact_weight: float = 2.0, sep_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.compact_weight = compact_weight
        self.sep_weight = sep_weight
    
    def forward(self, embeddings, assignments, centroids):
        B, D = embeddings.shape
        K = assignments.shape[1]
        
        # 确保归一化
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        centroids = F.normalize(centroids, dim=-1, p=2)
        
        # Compactness: 使用距离的平方，更稳定
        sim_to_centroids = torch.matmul(embeddings, centroids.T).clamp(-1, 1)
        dists_sq = (1 - sim_to_centroids).clamp(min=0)  # 确保非负
        assignments_safe = assignments.clamp(min=1e-8)
        compact_loss = (dists_sq * assignments_safe).sum() / (B + 1e-6)
        
        # Separation: 使用 ReLU 限制范围
        centroid_sim = torch.matmul(centroids, centroids.T).clamp(-1, 1)
        mask = 1 - torch.eye(K, device=centroids.device)
        sep_violations = F.relu(centroid_sim - (1 - self.margin)) * mask
        sep_loss = sep_violations.sum() / (mask.sum() + 1e-6)
        
        total = self.compact_weight * compact_loss + self.sep_weight * sep_loss
        return total.clamp(min=0, max=100)  # 限制上界


# ============================================================
# 【增强版】VQ Pattern Matrix
# ============================================================

class VQPatternMatrixV74(nn.Module):
    """
    v7.4 Pattern矩阵: 增强离散岛状结构
    
    新增:
    1. 改进的温度调度（更平缓）
    2. Pattern正交性约束
    """
    def __init__(self, d_model: int = 256, n_patterns: int = 32, bottleneck_dim: int = 64):
        super().__init__()
        self.n_patterns = n_patterns
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        self.to_bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 正交初始化patterns
        init = torch.randn(bottleneck_dim, n_patterns)
        U, _, _ = torch.svd(init)
        self.patterns = nn.Parameter(U[:, :n_patterns].T.contiguous())
        self.centroids = nn.Parameter(torch.randn(n_patterns, d_model) * 0.1)

        self.register_buffer('ema_cluster_size', torch.ones(n_patterns))
        self.register_buffer('ema_dw', self.patterns.data.clone())
        self.register_buffer('pattern_usage', torch.ones(n_patterns) / n_patterns)
        
        self.ema_decay = 0.99
        self.epsilon = 1e-5

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """改进的温度退火: 使用余弦退火，更平缓"""
        warmup = total_epochs * 0.15
        if epoch < warmup:
            return 1.0
        progress = (epoch - warmup) / (total_epochs - warmup)
        # 余弦退火: 1.0 -> 0.2
        return 0.2 + 0.8 * (1 + np.cos(np.pi * progress)) / 2

    def forward(self, x: torch.Tensor, epoch: int = 0, total_epochs: int = 100, use_sinkhorn: bool = True):
        B, T, D = x.shape
        Q = self.to_bottleneck(x)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(self.patterns, dim=-1)

        temperature = self.get_temperature(epoch, total_epochs)
        logits = torch.matmul(Q_norm, K_norm.T) / temperature

        if self.training:
            noise = torch.randn_like(logits) * 0.03
            logits = logits + noise

        soft_assignments = F.softmax(logits, dim=-1)
        
        # Sinkhorn-Knopp均衡化
        if use_sinkhorn and self.training and epoch > total_epochs * 0.2:
            logits_flat = logits.view(-1, self.n_patterns)
            sk_assignments = sinkhorn_knopp(logits_flat, n_iters=3, epsilon=0.1)
            sk_assignments = sk_assignments.view(B, T, -1)
            mix_ratio = min(0.5, (epoch - total_epochs * 0.2) / (total_epochs * 0.3))
            soft_assignments = (1 - mix_ratio) * soft_assignments + mix_ratio * sk_assignments

        hard_indices = logits.argmax(dim=-1)
        hard_assignments = F.one_hot(hard_indices, self.n_patterns).float()

        progress = epoch / total_epochs
        if self.training:
            if progress < 0.25:
                assignments = soft_assignments
            elif progress < 0.5:
                target = compute_target_distribution(soft_assignments.detach(), power=2.0)
                assignments = 0.5 * soft_assignments + 0.5 * target
            else:
                assignments = hard_assignments + soft_assignments - soft_assignments.detach()
        else:
            assignments = hard_assignments

        if self.training:
            self._ema_update(Q, hard_assignments)

        pattern_emb_low = torch.matmul(assignments, self.patterns)
        pattern_emb = self.from_bottleneck(pattern_emb_low)
        
        return pattern_emb, assignments, logits, hard_indices, Q, soft_assignments

    def _ema_update(self, Q: torch.Tensor, hard_assignments: torch.Tensor):
        with torch.no_grad():
            B, T, _ = Q.shape
            encodings = hard_assignments.view(-1, self.n_patterns)
            batch_cluster_size = encodings.sum(0)
            
            self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + \
                                    (1 - self.ema_decay) * batch_cluster_size
            
            Q_flat = Q.view(-1, self.bottleneck_dim)
            batch_dw = encodings.T @ Q_flat
            self.ema_dw = self.ema_decay * self.ema_dw + (1 - self.ema_decay) * batch_dw
            
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / \
                           (n + self.n_patterns * self.epsilon) * n
            cluster_size = cluster_size.clamp(min=1.0)
            self.patterns.data = F.normalize(self.ema_dw / cluster_size.unsqueeze(1), dim=-1)
            
            usage = batch_cluster_size / (batch_cluster_size.sum() + 1e-8)
            self.pattern_usage = 0.9 * self.pattern_usage + 0.1 * usage

    def kl_loss(self, soft_assignments: torch.Tensor):
        """DEC-style KL散度损失 - 数值稳定版"""
        # 使用更大的clamp值，AMP下更安全
        q = soft_assignments.clamp(min=1e-6, max=1.0 - 1e-6)
        target = compute_target_distribution(q.detach(), power=2.0)
        target = target.clamp(min=1e-6, max=1.0 - 1e-6)

        # 使用更稳定的KL计算方式
        kl = (target * (target.log() - q.log())).sum(dim=-1).mean()

        # 检查并处理异常值
        if torch.isnan(kl) or torch.isinf(kl):
            return torch.tensor(0.0, device=q.device, requires_grad=True)
        return kl.clamp(min=0.0, max=10.0)

    def commitment_loss(self, Q: torch.Tensor, assignments: torch.Tensor):
        pattern_emb_low = torch.matmul(assignments, self.patterns)
        return F.mse_loss(Q, pattern_emb_low.detach())

    def batch_diversity_loss(self, assignments: torch.Tensor):
        batch_usage = assignments.mean(dim=[0, 1]).clamp(min=1e-8, max=1.0)
        target = torch.ones_like(batch_usage) / self.n_patterns
        return F.mse_loss(batch_usage, target)

    def entropy_regularization(self, assignments: torch.Tensor):
        """熵正则化 - 数值稳定版"""
        p = assignments.mean(dim=1).clamp(min=1e-6, max=1.0 - 1e-6)
        entropy = -(p * p.log()).sum(dim=-1)

        # 检查异常值
        if torch.isnan(entropy).any() or torch.isinf(entropy).any():
            return torch.tensor(0.0, device=assignments.device, requires_grad=True)
        return entropy.mean().clamp(min=0.0, max=10.0)
    
    def pattern_orthogonality_loss(self):
        patterns_norm = F.normalize(self.patterns, dim=-1)
        gram = torch.matmul(patterns_norm, patterns_norm.T)
        eye = torch.eye(self.n_patterns, device=self.patterns.device)
        
        # 只惩罚非对角元素
        off_diagonal = (gram * (1 - eye)) ** 2
        return off_diagonal.mean().clamp(min=0, max=10)

    def reinit_dead_patterns_by_splitting(self, threshold: float = 0.01):
        usage = self.pattern_usage
        dead_mask = usage < threshold
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return 0
        
        if n_dead >= self.n_patterns:
            self.patterns.data = F.normalize(torch.randn_like(self.patterns), dim=-1)
            self.pattern_usage.fill_(1.0 / self.n_patterns)
            return int(n_dead)
        
        alive_indices = torch.where(~dead_mask)[0]
        dead_indices = torch.where(dead_mask)[0]
        alive_usage = usage[alive_indices]
        sorted_order = alive_usage.argsort(descending=True)
        sorted_alive = alive_indices[sorted_order]
        
        n_to_split = min(int(n_dead), len(sorted_alive))
        
        for i in range(n_to_split):
            dominant_idx = sorted_alive[i % len(sorted_alive)]
            dead_idx = dead_indices[i]
            
            dominant_pattern = self.patterns.data[dominant_idx].clone()
            noise = F.normalize(torch.randn_like(dominant_pattern), dim=-1)
            perturbation = noise * 0.15 * dominant_pattern.norm()
            
            new_pattern = F.normalize(dominant_pattern + perturbation, dim=-1)
            self.patterns.data[dead_idx] = new_pattern
            self.patterns.data[dominant_idx] = F.normalize(dominant_pattern - perturbation * 0.5, dim=-1)
            
            split_usage = usage[dominant_idx] / 2
            self.pattern_usage[dead_idx] = split_usage
            self.pattern_usage[dominant_idx] = split_usage
            
            self.ema_dw[dead_idx] = self.ema_dw[dominant_idx] / 2
            self.ema_dw[dominant_idx] = self.ema_dw[dominant_idx] / 2
            self.ema_cluster_size[dead_idx] = self.ema_cluster_size[dominant_idx] / 2
            self.ema_cluster_size[dominant_idx] = self.ema_cluster_size[dominant_idx] / 2
            
            self.centroids.data[dead_idx] = self.centroids.data[dominant_idx].clone()
        
        return n_to_split


# ============================================================
# 基础组件
# ============================================================

class SymbolizationLayer(nn.Module):
    def __init__(self, input_dim: int, n_symbols: int = 64, codebook_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, codebook_dim)
        self.codebook = nn.Parameter(torch.randn(n_symbols, codebook_dim) * 0.1)

    def forward(self, x: torch.Tensor, hard: bool = False):
        h = self.proj(x)
        h_norm = F.normalize(h, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        logits = torch.matmul(h_norm, cb_norm.T) / 0.5
        if hard:
            indices = logits.argmax(dim=-1)
            quantized = self.codebook[indices]
        else:
            soft = F.gumbel_softmax(logits, tau=0.5, hard=False, dim=-1)
            quantized = torch.matmul(soft, self.codebook)
            indices = logits.argmax(dim=-1)
        return quantized, indices


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = res + x
        return x + self.ffn(self.norm2(x))


# ============================================================
# 【v7.4编码器】增强说话人不变性
# ============================================================

class SAMMEncoderV74(nn.Module):
    """
    v7.4编码器：增强离散岛状结构 + 说话人不变性
    """
    def __init__(self, input_dim=1024, d_model=256, n_symbols=64,
                 n_heads=4, n_layers=2, n_patterns=32, bottleneck_dim=64,
                 n_speakers_max=50):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.bottleneck_dim = bottleneck_dim

        self.symbol = SymbolizationLayer(input_dim, n_symbols, d_model)
        self.layers = nn.ModuleList([
            MaskedSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pattern = VQPatternMatrixV74(d_model, n_patterns, bottleneck_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        # 句子级投影
        self.sent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 【新增】说话人对抗模块
        self.speaker_adv = SpeakerAdversarialModule(d_model, n_speakers_max)

    def _encode_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, _ = self.symbol(x, hard=False)
        h = h + self.pos_enc[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor, epoch: int = 0, total_epochs: int = 100):
        h = self._encode_to_hidden(x)
        emb, assign, logits, hard_idx, Q, soft_assign = self.pattern(h, epoch, total_epochs)
        sent_emb = self.sent_proj(h.mean(dim=1))
        return emb, assign, logits, h, hard_idx, Q, soft_assign, sent_emb

    def get_utterance_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_to_hidden(x)
            return self.sent_proj(h.mean(dim=1))

    def get_pattern_distribution(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_to_hidden(x)
            _, assign, _, _, _, _ = self.pattern(h, epoch=100, total_epochs=100)
            return assign.mean(dim=1)


# ============================================================
# 【v7.4训练器】增强损失函数
# ============================================================

class SAMMTrainerV74:
    def __init__(self, model: SAMMEncoderV74, device='cuda',
                 total_epochs=100, warmup_epochs=20, logger=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.logger = logger
        self.use_amp = use_amp and device == 'cuda'

        # 增强的Island Loss
        self.island_loss = IslandLossV2(margin=1.2, compact_weight=2.0, sep_weight=1.0)

        self.opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_epochs, eta_min=1e-5
        )
        self.scaler = GradScaler() if self.use_amp else None
        self.history = []

    def log(self, msg: str, level: str = 'info'):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    def _init_patterns_from_data(self, dataloader: DataLoader):
        self.log("Initializing patterns from data using KMeans...")
        self.model.eval()
        all_Q, all_sent = [], []
        max_samples = 5000

        with torch.no_grad():
            for batch in dataloader:
                x = batch['features'].to(self.device)
                h = self.model._encode_to_hidden(x)
                Q = self.model.pattern.to_bottleneck(h)
                sent = self.model.sent_proj(h.mean(dim=1))
                all_Q.append(Q.mean(dim=1).cpu().numpy())
                all_sent.append(sent.cpu().numpy())
                if sum(q.shape[0] for q in all_Q) >= max_samples:
                    break

        all_Q = np.concatenate(all_Q, axis=0)[:max_samples]
        all_sent = np.concatenate(all_sent, axis=0)[:max_samples]

        n_patterns = self.model.n_patterns
        
        kmeans_q = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        kmeans_q.fit(all_Q)
        centers_q = torch.from_numpy(kmeans_q.cluster_centers_).float().to(self.device)
        self.model.pattern.patterns.data = F.normalize(centers_q, dim=-1)
        self.model.pattern.ema_dw.data = self.model.pattern.patterns.data.clone()
        
        kmeans_s = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        kmeans_s.fit(all_sent)
        centers_s = torch.from_numpy(kmeans_s.cluster_centers_).float().to(self.device)
        self.model.pattern.centroids.data = F.normalize(centers_s, dim=-1)
        
        self.model.pattern.ema_cluster_size.fill_(1.0)
        self.model.pattern.pattern_usage.fill_(1.0 / n_patterns)

        self.log(f"Initialized {n_patterns} patterns and centroids from KMeans")
        self.model.train()

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        all_metrics = []

        for batch in dataloader:
            x = batch['features'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)
            metrics = self._train_step(x, speaker_ids, epoch)
            all_metrics.append(metrics)

        avg = {}
        for k in all_metrics[0]:
            values = [m[k] for m in all_metrics]
            if isinstance(values[0], (int, float)):
                avg[k] = np.mean(values)
            else:
                avg[k] = values[0]

        if epoch == self.warmup_epochs - 1:
            self._init_patterns_from_data(dataloader)

        if (epoch + 1) % 5 == 0 and epoch >= self.warmup_epochs:
            n_reinit = self.model.pattern.reinit_dead_patterns_by_splitting(0.02)
            avg['reinit'] = n_reinit
            if n_reinit > 0:
                self.log(f"Epoch {epoch+1}: Split {n_reinit} patterns")
        else:
            avg['reinit'] = 0

        self.scheduler.step()
        self.history.append(avg)
        return avg

    def _train_step(self, x: torch.Tensor, speaker_ids: torch.Tensor, epoch: int):
        if self.use_amp:
            with autocast():
                loss, metrics = self._compute_loss(x, speaker_ids, epoch)
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss, metrics = self._compute_loss(x, speaker_ids, epoch)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        return metrics

    def _speaker_confusion_loss(self, sent_emb, speaker_ids):
        """让不同说话人的embedding centroid尽量接近"""
        unique_speakers = torch.unique(speaker_ids)
        if len(unique_speakers) < 2:
            return torch.tensor(0.0, device=sent_emb.device, requires_grad=True)
        
        centroids = []
        for spk_id in unique_speakers:
            mask = speaker_ids == spk_id
            if mask.sum() > 0:
                centroids.append(sent_emb[mask].mean(dim=0, keepdim=True))
        
        centroids = torch.cat(centroids, dim=0)  # (N_spk, D)
        centroids = F.normalize(centroids, dim=-1)
        
        # 计算成对距离（应该最小化距离，不是最大化）
        centroid_dists = torch.pdist(centroids, p=2)
        
        # 返回正值（鼓励距离减小）
        return centroid_dists.mean().clamp(min=0, max=10)
    
    def _cross_speaker_consistency_loss(self, sent_emb, assign, speaker_ids):
        B, T, K = assign.shape
        utt_assign = assign.mean(dim=1)
        dominant_patterns = utt_assign.argmax(dim=-1)
        
        loss = torch.tensor(0.0, device=sent_emb.device, requires_grad=True)
        count = 0
        
        for pid in torch.unique(dominant_patterns):
            mask = dominant_patterns == pid
            if mask.sum() < 2:
                continue
            
            pattern_embs = sent_emb[mask]
            pattern_speakers = speaker_ids[mask]
            
            if len(torch.unique(pattern_speakers)) < 2:
                continue
            
            centroid = pattern_embs.mean(dim=0)
            dists = torch.norm(pattern_embs - centroid, dim=-1, p=2)
            loss = loss + dists.mean()  # 累加到已有的 tensor
            count += 1
        
        if count == 0:
            return loss
        return (loss / count).clamp(min=0, max=10)

    def _compute_loss(self, x: torch.Tensor, speaker_ids: torch.Tensor, epoch: int):
        emb, assign, logits, hidden, hard_idx, Q, soft_assign, sent_emb = self.model(
            x, epoch=epoch, total_epochs=self.total_epochs
        )
        B, T, K = assign.shape

        pred_next = self.model.predictor(hidden[:, :-1, :])
        recon_loss = F.mse_loss(pred_next, hidden[:, 1:, :].detach())

        is_warmup = epoch < self.warmup_epochs
        
        if is_warmup:
            loss = recon_loss
            with torch.no_grad():
                unique = len(torch.unique(hard_idx))
                active = (self.model.pattern.pattern_usage > 0.01).sum().item()
            return loss, {
                'loss': loss.item(), 'recon': recon_loss.item(),
                'commitment': 0.0, 'batch_div': 0.0, 'island': 0.0,
                'kl': 0.0, 'entropy_reg': 0.0, 'speaker_adv': 0.0,
                'speaker_conf': 0.0, 'cross_spk': 0.0, 'pattern_orth': 0.0,
                'unique': unique, 'active': active, 'phase': 'warmup'
            }

        # ========== v7.4 主训练阶段 ==========
        
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
    
        # 更保守的权重调度
        island_weight = min(0.5, progress * 1.0)  # 降低最大权重
        kl_weight = min(0.1, progress * 0.2)
        speaker_weight = min(0.3, progress * 0.6)  # 降低对抗学习权重

        adv_lambda = min(0.5, progress * 0.8)
        self.model.speaker_adv.set_lambda(adv_lambda)
        
        # 基础损失
        commitment = self.model.pattern.commitment_loss(Q, assign)
        batch_div = self.model.pattern.batch_diversity_loss(assign)
        kl_loss = self.model.pattern.kl_loss(soft_assign)
        entropy_reg = self.model.pattern.entropy_regularization(assign)
        
        # Island Loss (增强权重)
        utt_assign = assign.mean(dim=1)
        island = self.island_loss(sent_emb, utt_assign, self.model.pattern.centroids)
        
        # 【新增】说话人对抗损失
        speaker_adv = self.model.speaker_adv(sent_emb, speaker_ids)
        
        # 【新增】说话人混淆损失
        speaker_conf = self._speaker_confusion_loss(sent_emb, speaker_ids)
        
        # 【新增】跨说话人一致性
        cross_spk = self._cross_speaker_consistency_loss(sent_emb, assign, speaker_ids)
        
        # 【新增】Pattern正交性
        pattern_orth = self.model.pattern.pattern_orthogonality_loss()
        
        # 动态权重调度
        island_weight = min(1.5, progress * 2.5)  # 增强island权重
        kl_weight = min(0.15, progress * 0.3)
        speaker_weight = min(0.8, progress * 1.2)  # 说话人不变性权重
        
        # 梯度反转lambda调度
        adv_lambda = min(1.0, progress * 1.5)
        self.model.speaker_adv.set_lambda(adv_lambda)
        
        loss = (
            recon_loss +
            0.15 * commitment +
            0.2 * batch_div +
            island_weight * island +
            kl_weight * kl_loss +
            0.05 * entropy_reg +
            speaker_weight * speaker_adv +      # 对抗损失
            0.3 * speaker_conf +                 # 混淆损失
            0.2 * cross_spk +                    # 跨说话人一致性
            0.1 * pattern_orth                   # Pattern正交性
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.log(f"Warning: NaN/Inf loss detected, using recon_loss only", 'warning')
            loss = recon_loss

        with torch.no_grad():
            unique = len(torch.unique(hard_idx))
            active = (self.model.pattern.pattern_usage > 0.01).sum().item()

        return loss, {
            'loss': loss.item(), 'recon': recon_loss.item(),
            'commitment': commitment.item(), 'batch_div': batch_div.item(),
            'island': island.item(), 'kl': kl_loss.item(),
            'entropy_reg': entropy_reg.item(),
            'speaker_adv': speaker_adv.item(),
            'speaker_conf': speaker_conf.item(),
            'cross_spk': cross_spk.item(),
            'pattern_orth': pattern_orth.item(),
            'unique': unique, 'active': active, 'phase': 'main'
        }


# ============================================================
# 数据加载
# ============================================================

def load_utterance_data(cache_dir: Path, max_utts: int = 10000):
    h5_path = cache_dir / 'features' / 'iemocap' / 'features.h5'
    meta_path = cache_dir / 'features' / 'iemocap' / 'metadata.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    utts = meta['utterances'][:max_utts]
    data = []
    with h5py.File(h5_path, 'r') as f:
        for u in utts:
            feat = f['features'][u['h5_start_idx']:u['h5_end_idx']][:]
            data.append({
                'features': feat,
                'speaker': u['speaker_id'],
                'gender': u.get('gender', 'unknown')
            })
    return data


# ============================================================
# 可视化和评估
# ============================================================

def visualize_results(model, data, output_dir, timestamp, device='cuda', logger=None):
    model.eval()
    embeddings, speakers_list, pattern_dists = [], [], []
    n_patterns = model.n_patterns

    with torch.no_grad():
        for d in data:
            x = torch.from_numpy(d['features']).float().unsqueeze(0).to(device)
            emb = model.get_utterance_embedding(x)
            pdist = model.get_pattern_distribution(x)
            embeddings.append(emb.cpu().numpy()[0])
            speakers_list.append(d['speaker'])
            pattern_dists.append(pdist.cpu().numpy()[0])

    embeddings = np.array(embeddings)
    pattern_dists = np.array(pattern_dists)
    pattern_ids = pattern_dists.argmax(axis=1)

    unique_patterns = len(np.unique(pattern_ids))
    total_usage = pattern_dists.sum(axis=0)
    active_patterns = (total_usage > total_usage.max() * 0.01).sum()

    # 【新增】计算说话人不变性
    spk_map = {s: i for i, s in enumerate(set(speakers_list))}
    spk_ids = np.array([spk_map[s] for s in speakers_list])
    
    speaker_invariance = silhouette_score(embeddings, spk_ids) if len(set(spk_ids)) > 1 else 0
    pattern_quality = silhouette_score(embeddings, pattern_ids) if unique_patterns > 1 else 0
    
    # Pattern纯度
    purities = []
    for pid in np.unique(pattern_ids):
        mask = pattern_ids == pid
        spks = spk_ids[mask]
        if len(spks) > 0:
            purity = np.bincount(spks).max() / len(spks)
            purities.append(purity)
    avg_purity = np.mean(purities) if purities else 0

    log_msg = f"\n{'='*50}\n"
    log_msg += f"=== v7.4 Results (Enhanced Island + Speaker Invariance) ===\n"
    log_msg += f"{'='*50}\n"
    log_msg += f"Unique Dominant Patterns: {unique_patterns}/{n_patterns}\n"
    log_msg += f"Active Patterns (>1%): {active_patterns}/{n_patterns}\n"
    log_msg += f"Speaker Invariance: {speaker_invariance:.4f} (lower is better)\n"
    log_msg += f"Pattern Quality: {pattern_quality:.4f}\n"
    log_msg += f"Pattern Purity: {avg_purity:.4f} (closer to 1/{len(spk_map)} = {1/len(spk_map):.3f} is better)"

    if logger:
        logger.info(log_msg)
    print(log_msg)

    # t-SNE可视化
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    scatter1 = axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=spk_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 0].set_title(f'By Speaker ({len(spk_map)} speakers)\nInvariance: {speaker_invariance:.3f}')
    plt.colorbar(scatter1, ax=axes[0, 0])

    scatter2 = axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pattern_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 1].set_title(f'By Pattern ({unique_patterns} unique)\nQuality: {pattern_quality:.3f}')
    plt.colorbar(scatter2, ax=axes[0, 1])

    n_clusters = min(8, unique_patterns) if unique_patterns > 1 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pattern_dists)
    scatter3 = axes[1, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title(f'By Pattern Distribution Cluster\nPurity: {avg_purity:.3f}')
    plt.colorbar(scatter3, ax=axes[1, 0])

    sorted_indices = np.argsort(total_usage)[::-1]
    sorted_usage = total_usage[sorted_indices]
    threshold = total_usage.max() * 0.01
    colors = ['steelblue' if u > threshold else 'lightgray' for u in sorted_usage]
    axes[1, 1].bar(range(n_patterns), sorted_usage, color=colors)
    axes[1, 1].axhline(y=total_usage.mean(), color='red', linestyle='--', label=f'Mean: {total_usage.mean():.1f}')
    axes[1, 1].axhline(y=threshold, color='orange', linestyle=':', label='1% threshold')
    axes[1, 1].set_title(f'Pattern Usage (Active: {active_patterns}/{n_patterns})')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = output_dir / f'samm_clusters_v74_{timestamp}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    if logger:
        logger.info(f"Saved cluster plot to {save_path}")

    compactness = []
    for pid in np.unique(pattern_ids):
        mask = pattern_ids == pid
        if mask.sum() > 1:
            cluster_emb = embeddings[mask]
            centroid = cluster_emb.mean(axis=0)
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            compactness.append(dists.mean())
    avg_compactness = np.mean(compactness) if compactness else 0

    return {
        'unique': unique_patterns, 
        'active': int(active_patterns),
        'embeddings': embeddings, 
        'speaker_invariance': float(speaker_invariance),
        'pattern_quality': float(pattern_quality),
        'pattern_purity': float(avg_purity),
        'compactness': float(avg_compactness)
    }


def plot_training_history(history, output_dir, timestamp, warmup_epochs):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    epochs = range(1, len(history) + 1)

    for ax in axes.flat:
        ax.axvline(x=warmup_epochs, color='green', linestyle='--', alpha=0.5, label='Warmup End')

    axes[0, 0].plot(epochs, [h['loss'] for h in history], 'b-')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h['recon'] for h in history], 'g-')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, [h['island'] for h in history], 'purple', label='Island')
    axes[0, 2].plot(epochs, [h['kl'] for h in history], 'orange', label='KL')
    axes[0, 2].set_title('Island & KL Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h['speaker_adv'] for h in history], 'red', label='Adversarial')
    axes[1, 0].plot(epochs, [h['speaker_conf'] for h in history], 'blue', label='Confusion')
    axes[1, 0].set_title('Speaker Invariance Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [h['cross_spk'] for h in history], 'cyan')
    axes[1, 1].set_title('Cross-Speaker Consistency')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, [h['pattern_orth'] for h in history], 'magenta')
    axes[1, 2].set_title('Pattern Orthogonality')
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].plot(epochs, [h['unique'] for h in history], 'g-', label='Unique', lw=2)
    axes[2, 0].plot(epochs, [h['active'] for h in history], 'b--', label='Active', lw=2)
    axes[2, 0].axhline(y=32, color='gray', linestyle=':', alpha=0.5)
    axes[2, 0].set_title('Pattern Usage')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(epochs, [h['commitment'] for h in history], 'teal')
    axes[2, 1].set_title('Commitment Loss')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(epochs, [h.get('reinit', 0) for h in history], 'red', marker='o', markersize=3)
    axes[2, 2].set_title('Pattern Splits')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_v74_{timestamp}.png', dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache'
    log_dir = base_dir / 'logs'
    output_dir = base_dir / 'outputs' / f'v74_enhanced_island_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(log_dir, timestamp)

    logger.info("=" * 60)
    logger.info("DS-SAMM v7.4: Enhanced Island + Speaker Invariance")
    logger.info("=" * 60)
    logger.info(f"Output dir: {output_dir}")

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    n_patterns = 32
    bottleneck_dim = 64
    max_utts = 8000
    total_epochs = 250
    warmup_epochs = 20
    batch_size = 64
    num_workers = 4

    logger.info(f"Config: n_patterns={n_patterns}, bottleneck={bottleneck_dim}")
    logger.info(f"Config: warmup={warmup_epochs}, total={total_epochs}")

    logger.info("[1] Loading data...")
    data = load_utterance_data(cache_dir, max_utts=max_utts)
    n_speakers = len(set(d['speaker'] for d in data))
    logger.info(f"Loaded {len(data)} utterances, {n_speakers} speakers")

    dataset = SpeakerUtteranceDataset(data, max_len=400)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )

    logger.info("[2] Creating model...")
    model = SAMMEncoderV74(
        input_dim=1024, d_model=256,
        n_patterns=n_patterns, bottleneck_dim=bottleneck_dim,
        n_layers=2, n_symbols=64, n_speakers_max=50
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    logger.info("[3] Training...")
    trainer = SAMMTrainerV74(
        model, device=device,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        logger=logger, use_amp=True
    )

    for epoch in range(total_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)

        if (epoch + 1) % 10 == 0 or epoch == warmup_epochs - 1:
            phase = metrics.get('phase', 'main')
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} [{phase.upper()}] | "
                f"Loss: {metrics['loss']:.4f} | Recon: {metrics['recon']:.4f}"
            )
            if phase == 'main':
                logger.info(
                    f"  Island: {metrics['island']:.4f} | Speaker ADV: {metrics['speaker_adv']:.4f} | "
                    f"Unique: {metrics['unique']}/{n_patterns}"
                )

    logger.info("[4] Plotting...")
    plot_training_history(trainer.history, output_dir, timestamp, warmup_epochs)

    logger.info("[5] Visualizing...")
    results = visualize_results(
        model, data[:2000], output_dir, timestamp,
        device=device, logger=logger
    )

    logger.info("[6] Saving...")
    model_path = output_dir / f'samm_v74_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 1024, 'd_model': 256,
            'n_patterns': n_patterns, 'bottleneck_dim': bottleneck_dim,
            'warmup_epochs': warmup_epochs,
        },
        'results': results,
        'history': trainer.history
    }, model_path)

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Results: Unique={results['unique']}, Active={results['active']}")
    logger.info(f"Speaker Invariance: {results['speaker_invariance']:.4f}")
    logger.info(f"Pattern Quality: {results['pattern_quality']:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()