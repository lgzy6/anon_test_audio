#!/usr/bin/env python3
"""
DS-SAMM v7.3: 离散岛状特征空间优化版

核心改进:
1. DEC-style Target Hardening: 让soft assignment逐渐变hard
2. Sinkhorn-Knopp归一化: 强制pattern均衡使用
3. Island Loss: 直接在embedding空间约束岛状结构
4. Centroid Contrastive Loss: 拉近同类，推远不同类centroid
5. 更aggressive的温度退火
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
    log_file = log_dir / f'training_v73_{timestamp}.log'
    logger = logging.getLogger('SAMM_v73')
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
# 【核心组件1】Sinkhorn-Knopp归一化
# ============================================================

def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 3, epsilon: float = 0.1):
    """
    Sinkhorn-Knopp归一化: 强制行列和均衡 (数值稳定版本)
    logits: (B*T, K) 的logits (未归一化)
    返回: 归一化后的assignment
    """
    with torch.no_grad():
        # 数值稳定: 减去最大值
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits / epsilon)
        
        # 防止全零
        Q = Q + 1e-8
        
        N, K = Q.shape
        
        for _ in range(n_iters):
            # 列归一化 (每个cluster的总权重相等)
            col_sum = Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
            Q = Q / col_sum
            
            # 行归一化 (每个样本的总权重为1)
            row_sum = Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Q = Q / row_sum
        
        return Q


# ============================================================
# 【核心组件2】DEC-style Target Distribution
# ============================================================

def compute_target_distribution(q: torch.Tensor, power: float = 2.0):
    """
    DEC-style target distribution: 让高置信度的assignment更hard (数值稳定版)
    q: (B, T, K) soft assignment
    返回: target distribution p
    """
    # 添加小常数防止除零
    q = q.clamp(min=1e-8)
    
    # 计算辅助target分布
    f = q.sum(dim=[0, 1]).clamp(min=1e-8)  # (K,) 每个cluster的频率
    
    # q^2 / f
    p = (q ** power) / f.unsqueeze(0).unsqueeze(0)
    
    # 归一化
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # 再次clamp确保数值稳定
    p = p.clamp(min=1e-8, max=1.0)
    
    return p


# ============================================================
# 【核心组件3】Island Loss - 形成离散岛状结构
# ============================================================

class IslandLoss(nn.Module):
    """
    Island Loss: 直接在embedding空间约束岛状结构
    
    1. Compactness: 同cluster样本靠近centroid
    2. Separation: 不同cluster的centroid相互远离
    """
    def __init__(self, margin: float = 0.5, compact_weight: float = 1.0, sep_weight: float = 0.5):
        super().__init__()
        self.margin = margin
        self.compact_weight = compact_weight
        self.sep_weight = sep_weight
    
    def forward(self, embeddings: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor):
        """
        embeddings: (B, D) utterance embeddings
        assignments: (B, K) soft assignments
        centroids: (K, D) cluster centroids
        """
        B, D = embeddings.shape
        K = assignments.shape[1]
        
        # 数值稳定的归一化
        embeddings = F.normalize(embeddings + 1e-8, dim=-1)
        centroids = F.normalize(centroids + 1e-8, dim=-1)
        
        # 1. Compactness Loss: 样本到其assigned centroid的距离
        # 使用cosine距离: 1 - cosine_sim
        sim_to_centroids = torch.matmul(embeddings, centroids.T)  # (B, K)
        dists_to_centroids = 1 - sim_to_centroids  # (B, K)
        
        # 加权距离，根据assignment概率
        assignments_safe = assignments.clamp(min=1e-8)
        compact_loss = (dists_to_centroids * assignments_safe).sum() / (B + 1e-8)
        
        # 2. Separation Loss: centroid之间的距离应该大于margin
        centroid_sim = torch.matmul(centroids, centroids.T)  # (K, K)
        mask = 1 - torch.eye(K, device=centroids.device)
        
        # 希望相似度小于(1-margin)，即距离大于margin
        # 使用hinge loss
        sep_violations = F.relu(centroid_sim - (1 - self.margin)) * mask
        n_pairs = mask.sum().clamp(min=1.0)
        sep_loss = sep_violations.sum() / n_pairs
        
        total = self.compact_weight * compact_loss + self.sep_weight * sep_loss
        
        # 防止nan/inf
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return total


# ============================================================
# 【核心组件4】VQ Pattern Matrix with Island Structure
# ============================================================

class VQPatternMatrixV73(nn.Module):
    """
    v7.3 Pattern矩阵: 离散岛状结构
    
    新增:
    1. Sinkhorn-Knopp均衡化
    2. DEC-style target hardening
    3. 更aggressive的温度退火
    4. 可学习的cluster centroids
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

        # 【新增】Utterance-level centroids (用于Island Loss)
        self.centroids = nn.Parameter(torch.randn(n_patterns, d_model) * 0.1)

        self.register_buffer('ema_cluster_size', torch.ones(n_patterns))
        self.register_buffer('ema_dw', self.patterns.data.clone())
        self.register_buffer('pattern_usage', torch.ones(n_patterns) / n_patterns)
        
        self.ema_decay = 0.99
        self.epsilon = 1e-5

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """更aggressive的温度退火: 1.0 → 0.1"""
        warmup = total_epochs * 0.15
        if epoch < warmup:
            return 1.0
        progress = (epoch - warmup) / (total_epochs - warmup)
        # 使用指数退火，更快达到低温
        return 0.1 + 0.9 * np.exp(-3 * progress)

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

        # 基础soft assignment
        soft_assignments = F.softmax(logits, dim=-1)
        
        # 【新增】Sinkhorn-Knopp均衡化 (在main阶段)
        if use_sinkhorn and self.training and epoch > total_epochs * 0.2:
            logits_flat = logits.view(-1, self.n_patterns)
            sk_assignments = sinkhorn_knopp(logits_flat, n_iters=3, epsilon=0.1)
            sk_assignments = sk_assignments.view(B, T, -1)
            # 混合原始和SK assignment，逐渐增加SK权重
            mix_ratio = min(0.5, (epoch - total_epochs * 0.2) / (total_epochs * 0.3))
            soft_assignments = (1 - mix_ratio) * soft_assignments + mix_ratio * sk_assignments

        hard_indices = logits.argmax(dim=-1)
        hard_assignments = F.one_hot(hard_indices, self.n_patterns).float()

        # 训练策略
        progress = epoch / total_epochs
        if self.training:
            if progress < 0.25:
                assignments = soft_assignments
            elif progress < 0.5:
                # DEC-style: 使用target distribution
                target = compute_target_distribution(soft_assignments.detach(), power=2.0)
                assignments = 0.5 * soft_assignments + 0.5 * target
            else:
                # Straight-through
                assignments = hard_assignments + soft_assignments - soft_assignments.detach()
        else:
            assignments = hard_assignments

        # EMA更新
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
        """DEC-style KL散度损失: 让assignment逐渐变hard (数值稳定版)"""
        # Clamp inputs
        q = soft_assignments.clamp(min=1e-8, max=1.0)
        
        target = compute_target_distribution(q.detach(), power=2.0)
        
        # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        # 使用F.kl_div更稳定
        kl = F.kl_div(
            q.log(),  # log(Q)
            target,   # P
            reduction='batchmean',
            log_target=False
        )
        
        # 确保非负
        return kl.clamp(min=0.0)

    def commitment_loss(self, Q: torch.Tensor, assignments: torch.Tensor):
        pattern_emb_low = torch.matmul(assignments, self.patterns)
        return F.mse_loss(Q, pattern_emb_low.detach())

    def batch_diversity_loss(self, assignments: torch.Tensor):
        batch_usage = assignments.mean(dim=[0, 1]).clamp(min=1e-8, max=1.0)
        target = torch.ones_like(batch_usage) / self.n_patterns
        return F.mse_loss(batch_usage, target)

    def entropy_regularization(self, assignments: torch.Tensor):
        """熵正则化: 每个样本的assignment应该尽量确定(低熵)"""
        p = assignments.mean(dim=1).clamp(min=1e-8, max=1.0)  # (B, K)
        entropy = -(p * p.log()).sum(dim=-1)
        return entropy.mean().clamp(max=10.0)  # 防止过大

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
            
            # 同步更新centroids
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
# 完整编码器
# ============================================================

class SAMMEncoderV73(nn.Module):
    def __init__(self, input_dim=1024, d_model=256, n_symbols=64,
                 n_heads=4, n_layers=2, n_patterns=32, bottleneck_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.bottleneck_dim = bottleneck_dim

        self.symbol = SymbolizationLayer(input_dim, n_symbols, d_model)
        self.layers = nn.ModuleList([
            MaskedSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pattern = VQPatternMatrixV73(d_model, n_patterns, bottleneck_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        # 句子级投影 (用于Island Loss)
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
# 训练器
# ============================================================

class SAMMTrainerV73:
    def __init__(self, model: SAMMEncoderV73, device='cuda',
                 total_epochs=100, warmup_epochs=20, logger=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.logger = logger
        self.use_amp = use_amp and device == 'cuda'

        # 【新增】Island Loss
        self.island_loss = IslandLoss(margin=0.8, compact_weight=1.0, sep_weight=0.5)

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
        
        # Pattern空间KMeans
        kmeans_q = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        kmeans_q.fit(all_Q)
        centers_q = torch.from_numpy(kmeans_q.cluster_centers_).float().to(self.device)
        self.model.pattern.patterns.data = F.normalize(centers_q, dim=-1)
        self.model.pattern.ema_dw.data = self.model.pattern.patterns.data.clone()
        
        # Sentence空间KMeans (用于centroids)
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

    def _compute_loss(self, x: torch.Tensor, speaker_ids: torch.Tensor, epoch: int):
        emb, assign, logits, hidden, hard_idx, Q, soft_assign, sent_emb = self.model(
            x, epoch=epoch, total_epochs=self.total_epochs
        )
        B, T, K = assign.shape

        # 重构损失
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
                'kl': 0.0, 'entropy_reg': 0.0,
                'unique': unique, 'active': active, 'phase': 'warmup'
            }

        # ========== 主训练阶段 (v7.3) ==========
        
        # 1. Commitment loss
        commitment = self.model.pattern.commitment_loss(Q, assign)

        # 2. Batch diversity (降低权重)
        batch_div = self.model.pattern.batch_diversity_loss(assign)

        # 3. 【新增】DEC-style KL散度损失
        kl_loss = self.model.pattern.kl_loss(soft_assign)

        # 4. 【新增】Island Loss
        utt_assign = assign.mean(dim=1)  # (B, K)
        island = self.island_loss(sent_emb, utt_assign, self.model.pattern.centroids)

        # 5. 【新增】熵正则化 (让assignment更确定)
        entropy_reg = self.model.pattern.entropy_regularization(assign)

        # 损失权重调度 (更保守的权重)
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        
        # Island loss权重逐渐增加，但上限降低
        island_weight = min(0.5, progress * 1.0)
        # KL loss权重逐渐增加，上限降低
        kl_weight = min(0.1, progress * 0.2)

        loss = (
            recon_loss +
            0.2 * commitment +
            0.3 * batch_div +
            island_weight * island +
            kl_weight * kl_loss +
            0.05 * entropy_reg
        )
        
        # 数值稳定性检查
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
# 可视化
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

    log_msg = f"\n{'='*50}\n"
    log_msg += f"=== v7.3 Results (Island Clustering) ===\n"
    log_msg += f"{'='*50}\n"
    log_msg += f"Unique Dominant Patterns: {unique_patterns}/{n_patterns}\n"
    log_msg += f"Active Patterns (>1%): {active_patterns}/{n_patterns}\n"
    log_msg += f"Pattern Usage Std: {total_usage.std():.2f}"

    if logger:
        logger.info(log_msg)
    print(log_msg)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    spk_map = {s: i for i, s in enumerate(set(speakers_list))}
    spk_ids = np.array([spk_map[s] for s in speakers_list])
    scatter1 = axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=spk_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 0].set_title(f'By Speaker ({len(spk_map)} speakers)')
    plt.colorbar(scatter1, ax=axes[0, 0])

    scatter2 = axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pattern_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 1].set_title(f'By Dominant Pattern ({unique_patterns} unique)')
    plt.colorbar(scatter2, ax=axes[0, 1])

    n_clusters = min(8, unique_patterns) if unique_patterns > 1 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pattern_dists)
    scatter3 = axes[1, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title('By Pattern Distribution Cluster')
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
    save_path = output_dir / f'samm_clusters_v73_{timestamp}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    if logger:
        logger.info(f"Saved cluster plot to {save_path}")

    if len(set(speakers_list)) > 1 and unique_patterns > 1:
        ari = adjusted_rand_score(spk_ids, pattern_ids)
        nmi = normalized_mutual_info_score(spk_ids, pattern_ids)
        sil = silhouette_score(embeddings, pattern_ids)
        
        # 【新增】计算cluster compactness
        compactness = []
        for pid in np.unique(pattern_ids):
            mask = pattern_ids == pid
            if mask.sum() > 1:
                cluster_emb = embeddings[mask]
                centroid = cluster_emb.mean(axis=0)
                dists = np.linalg.norm(cluster_emb - centroid, axis=1)
                compactness.append(dists.mean())
        avg_compactness = np.mean(compactness) if compactness else 0

        metrics_msg = f"\n=== Clustering Metrics ===\n"
        metrics_msg += f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil:.4f}\n"
        metrics_msg += f"Avg Compactness: {avg_compactness:.4f}"

        if logger:
            logger.info(metrics_msg)
        print(metrics_msg)

        return {'unique': unique_patterns, 'active': int(active_patterns),
                'embeddings': embeddings, 'ari': ari, 'nmi': nmi, 'sil': sil,
                'compactness': avg_compactness}

    return {'unique': unique_patterns, 'active': int(active_patterns), 'embeddings': embeddings}


def plot_training_history(history, output_dir, timestamp, warmup_epochs):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    epochs = range(1, len(history) + 1)

    for ax in axes.flat:
        ax.axvline(x=warmup_epochs, color='green', linestyle='--', alpha=0.5)

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

    axes[0, 3].plot(epochs, [h['batch_div'] for h in history], 'r-')
    axes[0, 3].set_title('Batch Diversity')
    axes[0, 3].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h['unique'] for h in history], 'g-', label='Unique', lw=2)
    axes[1, 0].plot(epochs, [h['active'] for h in history], 'b--', label='Active', lw=2)
    axes[1, 0].axhline(y=32, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_title('Pattern Usage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 35)

    axes[1, 1].plot(epochs, [h['commitment'] for h in history], 'teal')
    axes[1, 1].set_title('Commitment Loss')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, [h['entropy_reg'] for h in history], 'brown')
    axes[1, 2].set_title('Entropy Regularization')
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(epochs, [h.get('reinit', 0) for h in history], 'red', marker='o', markersize=3)
    axes[1, 3].set_title('Pattern Splits')
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_v73_{timestamp}.png', dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache'
    log_dir = base_dir / 'logs'
    output_dir = base_dir / 'outputs' / f'v73_island_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(log_dir, timestamp)

    logger.info("=" * 60)
    logger.info("DS-SAMM v7.3: Island Clustering")
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
    model = SAMMEncoderV73(
        input_dim=1024, d_model=256,
        n_patterns=n_patterns, bottleneck_dim=bottleneck_dim,
        n_layers=2, n_symbols=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    logger.info("[3] Training...")
    trainer = SAMMTrainerV73(
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
                    f"  Island: {metrics['island']:.4f} | KL: {metrics['kl']:.4f} | "
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
    model_path = output_dir / f'samm_v73_{timestamp}.pt'
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
    if 'ari' in results:
        logger.info(f"Metrics: ARI={results['ari']:.4f}, Silhouette={results['sil']:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()