#!/usr/bin/env python3
"""
DS-SAMM v7.0: 彻底修复版 - 解决模式坍塌问题

核心修复（三层手术）:
1. 架构层 - Bottleneck: 256维→64维，让欧氏距离重新生效
2. 策略层 - Warm-up: 前20 epoch只做重构，让Encoder先学会"说话"
3. 机制层 - Splitting: 废弃随机重置，改用分裂机制打破霸主垄断

额外优化:
- 更高起始温度(2.0)，更慢下降
- Batch级别多样性约束
- EMA更新Codebook
- 重构损失保持表示质量
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
    """配置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'training_v7_{timestamp}.log'

    logger = logging.getLogger('SAMM_v7')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


# ============================================================
# GPU优化: Dataset类
# ============================================================

class SpeakerUtteranceDataset(Dataset):
    """支持speaker信息的Dataset，用于对比学习"""

    def __init__(self, data_list: list, max_len: int = 500):
        self.data = data_list
        self.max_len = max_len

        # 构建speaker到样本索引的映射
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

        # 截断或padding
        if feat.shape[0] > self.max_len:
            start = np.random.randint(0, feat.shape[0] - self.max_len)
            feat = feat[start:start + self.max_len]

        speaker_id = self.speaker_to_id[d['speaker']]
        return {
            'features': torch.from_numpy(feat).float(),
            'speaker_id': speaker_id,
            'length': feat.shape[0]
        }


def collate_fn(batch):
    """动态padding的collate函数"""
    max_len = max(b['length'] for b in batch)

    padded_feats = []
    for b in batch:
        feat = b['features']
        if feat.shape[0] < max_len:
            pad = torch.zeros(max_len - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, pad], dim=0)
        padded_feats.append(feat)

    return {
        'features': torch.stack(padded_feats),
        'speaker_ids': torch.tensor([b['speaker_id'] for b in batch]),
        'lengths': torch.tensor([b['length'] for b in batch])
    }


# ============================================================
# 基础组件
# ============================================================

class SymbolizationLayer(nn.Module):
    """符号化层"""
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
    """自注意力层"""
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
# 【核心修复1】Bottleneck + Splitting Pattern矩阵
# ============================================================

class VQPatternMatrixV7(nn.Module):
    """
    v7.0 彻底修复版Pattern矩阵
    
    修复1 - Bottleneck: 256维→64维，缓解Hubness Problem
    修复2 - Splitting: 分裂霸主pattern，而非随机重置
    修复3 - 更高温度调度: 2.0→0.7
    修复4 - Batch级别多样性约束
    修复5 - EMA更新Codebook
    """

    def __init__(self, d_model: int = 256, n_patterns: int = 32, bottleneck_dim: int = 64):
        super().__init__()
        self.n_patterns = n_patterns
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # 【修复1】Bottleneck投影层
        self.to_bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        
        # 从bottleneck恢复到d_model
        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Pattern在低维空间（bottleneck_dim）
        # 正交初始化
        patterns_init = torch.randn(bottleneck_dim, n_patterns)
        U, _, _ = torch.svd(patterns_init)
        self.patterns = nn.Parameter(U[:, :n_patterns].T.contiguous())  # (K, bottleneck_dim)

        # EMA相关buffer
        self.register_buffer('ema_cluster_size', torch.ones(n_patterns))
        self.register_buffer('ema_dw', self.patterns.data.clone())
        self.register_buffer('pattern_usage', torch.ones(n_patterns) / n_patterns)
        
        # 配置
        self.ema_decay = 0.99
        self.epsilon = 1e-5

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """【修复3】更高起始温度，更慢下降: 2.0 → 0.7"""
        warmup_epochs = total_epochs * 0.1
        if epoch < warmup_epochs:
            return 2.0  # Warmup阶段保持高温
        
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(0.7, 2.0 - 1.3 * progress)

    def forward(self, x: torch.Tensor, epoch: int = 0, total_epochs: int = 100):
        B, T, D = x.shape

        # 【修复1】投影到低维bottleneck空间
        Q = self.to_bottleneck(x)  # (B, T, bottleneck_dim)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(self.patterns, dim=-1)  # (K, bottleneck_dim)

        temperature = self.get_temperature(epoch, total_epochs)
        
        # 训练时添加噪声防止winner-take-all
        logits = torch.matmul(Q_norm, K_norm.T) / temperature  # (B, T, K)
        if self.training:
            noise = torch.randn_like(logits) * 0.05
            logits = logits + noise

        # Soft和Hard分配
        soft_assignments = F.softmax(logits, dim=-1)
        hard_indices = logits.argmax(dim=-1)  # (B, T)
        hard_assignments = F.one_hot(hard_indices, self.n_patterns).float()

        # 训练策略
        progress = epoch / total_epochs
        if self.training:
            if progress < 0.3:
                # 前30%: 纯软分配
                assignments = soft_assignments
            else:
                # 后70%: Straight-through estimator
                assignments = hard_assignments + soft_assignments - soft_assignments.detach()
        else:
            assignments = hard_assignments

        # 【修复5】EMA更新Codebook（只在训练时）
        if self.training:
            with torch.no_grad():
                # 统计使用情况
                encodings = hard_assignments.view(-1, self.n_patterns)  # (B*T, K)
                
                # 更新EMA cluster size
                batch_cluster_size = encodings.sum(0)
                self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + \
                                        (1 - self.ema_decay) * batch_cluster_size
                
                # 更新EMA dw
                Q_flat = Q.view(-1, self.bottleneck_dim)  # (B*T, bottleneck_dim)
                batch_dw = encodings.T @ Q_flat  # (K, bottleneck_dim)
                self.ema_dw = self.ema_decay * self.ema_dw + (1 - self.ema_decay) * batch_dw
                
                # Laplace smoothing并更新patterns
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / \
                               (n + self.n_patterns * self.epsilon) * n
                # 只更新被使用的patterns，避免除以接近0的值
                cluster_size = cluster_size.clamp(min=1.0)
                self.patterns.data = self.ema_dw / cluster_size.unsqueeze(1)
                # 对patterns进行归一化，防止数值爆炸
                self.patterns.data = F.normalize(self.patterns.data, dim=-1)
                
                # 更新使用统计
                usage = batch_cluster_size / (batch_cluster_size.sum() + 1e-8)
                self.pattern_usage = 0.9 * self.pattern_usage + 0.1 * usage

        # 在低维空间获取pattern embedding
        pattern_emb_low = torch.matmul(assignments, self.patterns)  # (B, T, bottleneck_dim)
        
        # 恢复到高维空间
        pattern_emb = self.from_bottleneck(pattern_emb_low)  # (B, T, d_model)
        
        return pattern_emb, assignments, logits, hard_indices, Q

    def commitment_loss(self, Q: torch.Tensor, assignments: torch.Tensor):
        """VQ-VAE commitment loss: 让encoder输出靠近选中的pattern（在低维空间）"""
        pattern_emb_low = torch.matmul(assignments, self.patterns)
        return F.mse_loss(Q, pattern_emb_low.detach())

    def codebook_loss(self, Q: torch.Tensor, assignments: torch.Tensor):
        """VQ-VAE codebook loss: 让pattern靠近encoder输出（在低维空间）"""
        pattern_emb_low = torch.matmul(assignments, self.patterns)
        return F.mse_loss(pattern_emb_low, Q.detach())

    def batch_diversity_loss(self, assignments: torch.Tensor):
        """【修复4】Batch级别多样性约束 - 直接作用在assignment上"""
        # assignments: (B, T, K)
        batch_usage = assignments.mean(dim=[0, 1])  # (K,)
        # 添加数值稳定性
        batch_usage = batch_usage.clamp(min=1e-8, max=1.0)
        target = torch.ones_like(batch_usage) / self.n_patterns
        return F.mse_loss(batch_usage, target)

    def orthogonality_loss(self):
        """Pattern正交性损失"""
        K = F.normalize(self.patterns, dim=-1)
        sim = torch.matmul(K, K.T)
        mask = 1.0 - torch.eye(self.n_patterns, device=K.device)
        return (sim.abs() * mask).mean()

    def entropy_loss(self):
        """使用熵最大化"""
        usage = self.pattern_usage.clamp(min=1e-8)
        entropy = -(usage * usage.log()).sum()
        max_entropy = np.log(self.n_patterns)
        return (max_entropy - entropy) / max_entropy

    def reinit_dead_patterns_by_splitting(self, threshold: float = 0.01):
        """
        【修复2】分裂机制: 将霸主pattern一分为二
        
        原理: 霸主pattern已经在数据流形上，分裂后的新pattern也在流形上，
              比随机初始化更容易存活
        """
        usage = self.pattern_usage
        dead_mask = usage < threshold
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return 0
        
        if n_dead >= self.n_patterns:
            # 全部死亡，使用随机初始化
            self.patterns.data = F.normalize(
                torch.randn_like(self.patterns), dim=-1
            )
            self.pattern_usage.fill_(1.0 / self.n_patterns)
            return int(n_dead)
        
        # 找到存活的和死亡的pattern索引
        alive_mask = ~dead_mask
        alive_indices = torch.where(alive_mask)[0]
        dead_indices = torch.where(dead_mask)[0]
        
        # 按使用率排序存活的patterns（霸主们）
        alive_usage = usage[alive_indices]
        sorted_order = alive_usage.argsort(descending=True)
        sorted_alive = alive_indices[sorted_order]
        
        n_to_split = min(int(n_dead), len(sorted_alive))
        
        for i in range(n_to_split):
            # 选择一个霸主（循环使用）
            dominant_idx = sorted_alive[i % len(sorted_alive)]
            dead_idx = dead_indices[i]
            
            # 获取霸主pattern
            dominant_pattern = self.patterns.data[dominant_idx].clone()
            
            # 生成扰动方向
            noise = torch.randn_like(dominant_pattern)
            noise = F.normalize(noise, dim=-1)
            
            # 扰动大小: 10%
            perturbation = noise * 0.1 * dominant_pattern.norm()
            
            # 分裂: 新pattern = 霸主 + 扰动
            new_pattern = F.normalize(dominant_pattern + perturbation, dim=-1)
            self.patterns.data[dead_idx] = new_pattern
            
            # 同时微调霸主，让两者分开
            adjusted_dominant = F.normalize(dominant_pattern - perturbation * 0.5, dim=-1)
            self.patterns.data[dominant_idx] = adjusted_dominant
            
            # 重新分配使用率
            split_usage = usage[dominant_idx] / 2
            self.pattern_usage[dead_idx] = split_usage
            self.pattern_usage[dominant_idx] = split_usage
            
            # 同步更新EMA
            self.ema_dw[dead_idx] = self.ema_dw[dominant_idx] / 2
            self.ema_dw[dominant_idx] = self.ema_dw[dominant_idx] / 2
            self.ema_cluster_size[dead_idx] = self.ema_cluster_size[dominant_idx] / 2
            self.ema_cluster_size[dominant_idx] = self.ema_cluster_size[dominant_idx] / 2
        
        return n_to_split


# ============================================================
# 对比学习损失
# ============================================================

class ContrastiveSpeakerLoss(nn.Module):
    """对比学习损失: 同speaker样本应有相似的pattern分布"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pattern_dists: torch.Tensor, speaker_ids: torch.Tensor):
        B = pattern_dists.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=pattern_dists.device)

        pattern_dists = F.normalize(pattern_dists, dim=-1)
        sim_matrix = torch.matmul(pattern_dists, pattern_dists.T) / self.temperature

        speaker_mask = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        diag_mask = 1.0 - torch.eye(B, device=speaker_mask.device)
        speaker_mask = speaker_mask * diag_mask

        exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
        exp_sim = exp_sim * diag_mask

        pos_sim = (exp_sim * speaker_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        valid = (speaker_mask.sum(dim=1) > 0) & (all_sim > 1e-8)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pattern_dists.device)

        loss = -torch.log(pos_sim[valid] / all_sim[valid] + 1e-8).mean()
        return loss


class ImprovedSeparationLoss(nn.Module):
    """分离损失: 不同speaker的embedding应有足够距离"""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor):
        B = embeddings.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=-1)
        dist_matrix = 1.0 - torch.matmul(embeddings, embeddings.T)
        diff_speaker = (speaker_ids.unsqueeze(0) != speaker_ids.unsqueeze(1)).float()

        separation = F.relu(self.margin - dist_matrix) * diff_speaker
        n_pairs = diff_speaker.sum().clamp(min=1)

        return separation.sum() / n_pairs


# ============================================================
# 完整编码器
# ============================================================

class SAMMEncoderV7(nn.Module):
    """v7.0 彻底修复版SAMM编码器"""

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
        
        # 【修复1】使用带Bottleneck的Pattern矩阵
        self.pattern = VQPatternMatrixV7(d_model, n_patterns, bottleneck_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        # 句子级投影
        self.sent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 【修复2-Warmup】重构头: 预测下一帧
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
        emb, assign, logits, hard_idx, Q = self.pattern(h, epoch, total_epochs)
        return emb, assign, logits, h, hard_idx, Q

    def get_utterance_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_to_hidden(x)
            return self.sent_proj(h.mean(dim=1))

    def get_pattern_distribution(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, assign, _, _, _, _ = self.forward(x)
            return assign.mean(dim=1)


# ============================================================
# 训练器 (完整修复版)
# ============================================================

class SAMMTrainerV7:
    """v7.0 训练器 - 包含Warm-up策略"""

    def __init__(self, model: SAMMEncoderV7, device='cuda',
                 total_epochs=100, warmup_epochs=20, logger=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs  # 【修复2】Warm-up阶段
        self.logger = logger
        self.use_amp = use_amp and device == 'cuda'

        # 损失函数
        self.contrastive_loss = ContrastiveSpeakerLoss(temperature=0.1)
        self.separation_loss = ImprovedSeparationLoss(margin=0.5)

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
        """【关键修复】Warmup结束后，用KMeans从数据初始化patterns"""
        self.log("Initializing patterns from data using KMeans...")
        self.model.eval()

        # 收集bottleneck空间的表示
        all_Q = []
        max_samples = 5000

        with torch.no_grad():
            for batch in dataloader:
                x = batch['features'].to(self.device)
                h = self.model._encode_to_hidden(x)
                Q = self.model.pattern.to_bottleneck(h)  # (B, T, bottleneck_dim)
                # 取每个utterance的平均表示
                Q_mean = Q.mean(dim=1)  # (B, bottleneck_dim)
                all_Q.append(Q_mean.cpu().numpy())

                if sum(q.shape[0] for q in all_Q) >= max_samples:
                    break

        all_Q = np.concatenate(all_Q, axis=0)[:max_samples]

        # 用KMeans聚类
        n_patterns = self.model.n_patterns
        kmeans = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        kmeans.fit(all_Q)

        # 用聚类中心初始化patterns
        centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        centers = F.normalize(centers, dim=-1)

        self.model.pattern.patterns.data = centers
        self.model.pattern.ema_dw.data = centers.clone()
        self.model.pattern.ema_cluster_size.fill_(1.0)
        self.model.pattern.pattern_usage.fill_(1.0 / n_patterns)

        self.log(f"Initialized {n_patterns} patterns from KMeans centers")
        self.model.train()

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        all_metrics = []

        for batch in dataloader:
            x = batch['features'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)

            metrics = self._train_step(x, speaker_ids, epoch)
            all_metrics.append(metrics)

        # 计算平均值，跳过非数值字段
        avg = {}
        for k in all_metrics[0]:
            values = [m[k] for m in all_metrics]
            if isinstance(values[0], (int, float)):
                avg[k] = np.mean(values)
            else:
                avg[k] = values[0]  # 非数值字段取第一个值

        # 【关键修复】Warmup结束时，用KMeans初始化patterns
        if epoch == self.warmup_epochs - 1:
            self._init_patterns_from_data(dataloader)

        # 使用Splitting重初始化死亡pattern
        if (epoch + 1) % 5 == 0 and epoch >= self.warmup_epochs:
            n_reinit = self.model.pattern.reinit_dead_patterns_by_splitting(0.02)
            avg['reinit'] = n_reinit
            if n_reinit > 0:
                self.log(f"Epoch {epoch+1}: Split {n_reinit} patterns from dominant ones")
        else:
            avg['reinit'] = 0

        self.scheduler.step()
        self.history.append(avg)
        return avg

    def _train_step(self, x: torch.Tensor, speaker_ids: torch.Tensor, epoch: int):
        """单步训练"""
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
        """计算所有损失 - 修复重构损失"""
        emb, assign, logits, hidden, hard_idx, Q = self.model(
            x, epoch=epoch, total_epochs=self.total_epochs
        )
        B, T, K = assign.shape

        # ========== 【修复】重构损失：基于hidden而非emb ==========
        # hidden分布稳定，不受pattern优化影响
        pred_next = self.model.predictor(hidden[:, :-1, :])
        recon_loss = F.mse_loss(pred_next, hidden[:, 1:, :].detach())

        # ========== Warm-up策略 ==========
        is_warmup = epoch < self.warmup_epochs
        
        if is_warmup:
            # Warm-up阶段: 只做重构，不聚类
            loss = recon_loss
            
            with torch.no_grad():
                unique = len(torch.unique(hard_idx))
                active = (self.model.pattern.pattern_usage > 0.01).sum().item()
            
            return loss, {
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'commitment': 0.0,
                'codebook': 0.0,
                'batch_div': 0.0,
                'ortho': 0.0,
                'entropy': 0.0,
                'contrastive': 0.0,
                'separation': 0.0,
                'unique': unique,
                'active': active,
                'phase': 'warmup'
            }

        # ========== 主训练阶段（简化版）==========
        # 只保留3个核心损失：重构 + VQ commitment + 多样性

        # 1. VQ-VAE commitment loss（让encoder输出靠近pattern）
        commitment = self.model.pattern.commitment_loss(Q, assign)

        # 2. 多样性损失（防止模式坍塌）
        batch_div = self.model.pattern.batch_diversity_loss(assign)

        # 简化的损失组合
        loss = recon_loss + 0.25 * commitment + 1.0 * batch_div

        # 记录其他指标但不参与优化
        with torch.no_grad():
            codebook = self.model.pattern.codebook_loss(Q, assign)
            ortho = self.model.pattern.orthogonality_loss()
            entropy = self.model.pattern.entropy_loss()
            contrastive = torch.tensor(0.0)
            separation = torch.tensor(0.0)

        with torch.no_grad():
            unique = len(torch.unique(hard_idx))
            active = (self.model.pattern.pattern_usage > 0.01).sum().item()

        return loss, {
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'commitment': commitment.item(),
            'codebook': codebook.item(),
            'batch_div': batch_div.item(),
            'ortho': ortho.item(),
            'entropy': entropy.item(),
            'contrastive': contrastive.item(),
            'separation': separation.item(),
            'unique': unique,
            'active': active,
            'phase': 'main'
        }

# ============================================================
# 数据加载
# ============================================================

def load_utterance_data(cache_dir: Path, max_utts: int = 10000):
    """加载数据"""
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
    """可视化结果"""
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

    # 统计
    unique_patterns = len(np.unique(pattern_ids))
    total_usage = pattern_dists.sum(axis=0)
    active_patterns = (total_usage > total_usage.max() * 0.01).sum()

    log_msg = f"\n{'='*50}\n"
    log_msg += f"=== v7.0 Results (Bottleneck+Warmup+Splitting) ===\n"
    log_msg += f"{'='*50}\n"
    log_msg += f"Unique Dominant Patterns: {unique_patterns}/{n_patterns}\n"
    log_msg += f"Active Patterns (>1%): {active_patterns}/{n_patterns}\n"
    log_msg += f"Pattern Usage Std: {total_usage.std():.2f}\n"
    log_msg += f"Pattern Usage Min/Max: {total_usage.min():.2f}/{total_usage.max():.2f}"

    if logger:
        logger.info(log_msg)
    print(log_msg)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. By Speaker
    spk_map = {s: i for i, s in enumerate(set(speakers_list))}
    spk_ids = np.array([spk_map[s] for s in speakers_list])
    scatter1 = axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=spk_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 0].set_title(f'By Speaker ({len(spk_map)} speakers)')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # 2. By Dominant Pattern
    scatter2 = axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pattern_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 1].set_title(f'By Dominant Pattern ({unique_patterns} unique)')
    plt.colorbar(scatter2, ax=axes[0, 1])

    # 3. KMeans on pattern distribution
    n_clusters = min(8, unique_patterns) if unique_patterns > 1 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pattern_dists)
    scatter3 = axes[1, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title('By Pattern Distribution Cluster')
    plt.colorbar(scatter3, ax=axes[1, 0])

    # 4. Pattern Usage
    sorted_indices = np.argsort(total_usage)[::-1]
    sorted_usage = total_usage[sorted_indices]
    threshold = total_usage.max() * 0.01
    colors = ['steelblue' if u > threshold else 'lightgray' for u in sorted_usage]
    bars = axes[1, 1].bar(range(n_patterns), sorted_usage, color=colors)
    axes[1, 1].axhline(y=total_usage.mean(), color='red', linestyle='--', label=f'Mean: {total_usage.mean():.1f}')
    axes[1, 1].axhline(y=threshold, color='orange', linestyle=':', label=f'1% threshold')
    axes[1, 1].set_title(f'Pattern Usage (Active: {active_patterns}/{n_patterns})')
    axes[1, 1].set_xlabel('Pattern (sorted by usage)')
    axes[1, 1].set_ylabel('Total Usage')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = output_dir / f'samm_clusters_v7_{timestamp}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    if logger:
        logger.info(f"Saved cluster plot to {save_path}")

    # 聚类指标
    if len(set(speakers_list)) > 1 and unique_patterns > 1:
        ari = adjusted_rand_score(spk_ids, pattern_ids)
        nmi = normalized_mutual_info_score(spk_ids, pattern_ids)
        sil = silhouette_score(embeddings, pattern_ids)

        metrics_msg = f"\n=== Clustering Metrics ===\n"
        metrics_msg += f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil:.4f}"

        if logger:
            logger.info(metrics_msg)
        print(metrics_msg)

        return {'unique': unique_patterns, 'active': int(active_patterns), 
                'embeddings': embeddings, 'ari': ari, 'nmi': nmi, 'sil': sil}

    return {'unique': unique_patterns, 'active': int(active_patterns), 'embeddings': embeddings}


def plot_training_history(history, output_dir, timestamp, warmup_epochs):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    epochs = range(1, len(history) + 1)

    # 标记warmup阶段
    for ax in axes.flat:
        ax.axvline(x=warmup_epochs, color='green', linestyle='--', alpha=0.5, label='Warmup End')

    axes[0, 0].plot(epochs, [h['loss'] for h in history], 'b-')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h['recon'] for h in history], 'g-')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, [h['commitment'] for h in history], 'g-', label='Commit')
    axes[0, 2].plot(epochs, [h['codebook'] for h in history], 'r--', label='Codebook')
    axes[0, 2].set_title('VQ Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].plot(epochs, [h['batch_div'] for h in history], 'purple')
    axes[0, 3].set_title('Batch Diversity Loss')
    axes[0, 3].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h['unique'] for h in history], 'g-', label='Unique', linewidth=2)
    axes[1, 0].plot(epochs, [h['active'] for h in history], 'b--', label='Active', linewidth=2)
    axes[1, 0].axhline(y=32, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_title('Pattern Usage (Target: 32)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 35)

    axes[1, 1].plot(epochs, [h['contrastive'] for h in history], 'purple', label='Contrast')
    axes[1, 1].plot(epochs, [h['separation'] for h in history], 'orange', label='Separate')
    axes[1, 1].set_title('Speaker Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, [h['ortho'] for h in history], 'brown', label='Ortho')
    axes[1, 2].plot(epochs, [h['entropy'] for h in history], 'teal', label='Entropy')
    axes[1, 2].set_title('Regularization Losses')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(epochs, [h.get('reinit', 0) for h in history], 'red', marker='o', markersize=3)
    axes[1, 3].set_title('Pattern Splits')
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_v7_{timestamp}.png', dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache'
    log_dir = base_dir / 'logs'
    output_dir = base_dir / 'outputs' / f'v7_tests_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(log_dir, timestamp)

    logger.info("=" * 60)
    logger.info("DS-SAMM v7.0: Ultimate Fix (Bottleneck + Warmup + Splitting)")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output dir: {output_dir}")

    # 随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ========== 配置参数 ==========
    n_patterns = 32
    bottleneck_dim = 64      # 【修复1】Bottleneck维度
    max_utts = 8000
    total_epochs = 150
    warmup_epochs = 20       # 【修复2】Warm-up epochs
    batch_size = 64
    num_workers = 4

    logger.info(f"Config: n_patterns={n_patterns}, bottleneck_dim={bottleneck_dim}")
    logger.info(f"Config: warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
    logger.info(f"Config: max_utts={max_utts}, batch_size={batch_size}")

    # 加载数据
    logger.info("[1] Loading data...")
    try:
        data = load_utterance_data(cache_dir, max_utts=max_utts)
        n_speakers = len(set(d['speaker'] for d in data))
        logger.info(f"Loaded {len(data)} utterances, {n_speakers} speakers")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # 创建DataLoader
    dataset = SpeakerUtteranceDataset(data, max_len=400)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )

    # 创建模型
    logger.info("[2] Creating model...")
    model = SAMMEncoderV7(
        input_dim=1024, 
        d_model=256,
        n_patterns=n_patterns, 
        bottleneck_dim=bottleneck_dim,  # 【修复1】
        n_layers=2, 
        n_symbols=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # 训练
    logger.info("[3] Training...")
    logger.info(f"    Phase 1 (Warmup): Epochs 1-{warmup_epochs} - Reconstruction only")
    logger.info(f"    Phase 2 (Main):   Epochs {warmup_epochs+1}-{total_epochs} - Full training")
    
    trainer = SAMMTrainerV7(
        model, device=device,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,  # 【修复2】
        logger=logger, 
        use_amp=True
    )

    for epoch in range(total_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)

        if (epoch + 1) % 10 == 0 or epoch == warmup_epochs - 1:
            phase = metrics.get('phase', 'main')
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} [{phase.upper()}] | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Recon: {metrics['recon']:.4f}"
            )
            if phase == 'main':
                logger.info(
                    f"  BatchDiv: {metrics['batch_div']:.4f} | "
                    f"Contrast: {metrics['contrastive']:.4f} | "
                    f"Sep: {metrics['separation']:.4f}"
                )
            logger.info(
                f"  Unique: {metrics['unique']}/{n_patterns} | "
                f"Active: {metrics['active']}/{n_patterns}"
            )

    # 绘制训练历史
    logger.info("[4] Plotting training history...")
    plot_training_history(trainer.history, output_dir, timestamp, warmup_epochs)

    # 可视化结果
    logger.info("[5] Visualizing results...")
    results = visualize_results(
        model, data[:2000], output_dir, timestamp,
        device=device, logger=logger
    )

    # 保存模型
    logger.info("[6] Saving model...")
    model_path = output_dir / f'samm_v7_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 1024,
            'd_model': 256,
            'n_patterns': n_patterns,
            'bottleneck_dim': bottleneck_dim,
            'warmup_epochs': warmup_epochs,
        },
        'results': results,
        'history': trainer.history
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Results: Unique={results['unique']}/{n_patterns}, Active={results['active']}/{n_patterns}")
    if 'ari' in results:
        logger.info(f"Metrics: ARI={results['ari']:.4f}, NMI={results['nmi']:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()