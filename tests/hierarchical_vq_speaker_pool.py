#!/usr/bin/env python3
"""
层级矢量量化说话人池构建 (Hierarchical VQ Speaker Pool)

架构设计:
=========
Level 1 (粗粒度): 基于能量和情绪的全局韵律聚类
    - 8-16个一级cluster
    - 特征: 能量均值/方差、语速、F0范围、情绪强度

Level 2 (细粒度): 每个一级cluster内的细微韵律差别
    - 每个L1内16-32个二级pattern
    - 特征: 语调轮廓、节奏模式、重音分布

总pattern数: L1_clusters × L2_patterns = 8×32 = 256 或 16×16 = 256
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    log_file = log_dir / f'training_hvq_{timestamp}.log'
    logger = logging.getLogger('HVQ_Pool')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                           datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ============================================================
# 韵律特征提取器
# ============================================================

class ProsodyFeatureExtractor(nn.Module):
    """
    从WavLM特征中提取韵律相关特征

    输出:
    - 全局韵律特征 (用于L1聚类): 能量、语速、F0范围
    - 局部韵律特征 (用于L2聚类): 语调轮廓、节奏模式
    """
    def __init__(self, input_dim: int = 1024, d_model: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 能量/强度估计器
        self.energy_estimator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # F0相关特征估计器 (从SSL特征中学习)
        self.f0_estimator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # F0相关的多维特征
        )

        # 全局韵律编码器 (用于L1)
        self.global_prosody_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )

        # 局部韵律编码器 (用于L2)
        self.local_prosody_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D) WavLM特征

        返回:
        - global_prosody: (B, D//2) 全局韵律特征
        - local_prosody: (B, T, D) 局部韵律特征
        - energy: (B, T, 1) 帧级能量
        - f0_features: (B, T, 32) F0相关特征
        """
        B, T, D = x.shape

        # 帧级特征
        energy = self.energy_estimator(x)  # (B, T, 1)
        f0_features = self.f0_estimator(x)  # (B, T, 32)

        # 全局韵律: 聚合整个utterance
        # 使用统计量: mean, std, max, min
        x_mean = x.mean(dim=1)  # (B, D)
        x_std = x.std(dim=1)    # (B, D)
        x_max = x.max(dim=1)[0] # (B, D)

        energy_stats = torch.cat([
            energy.mean(dim=1),
            energy.std(dim=1),
            energy.max(dim=1)[0]
        ], dim=-1)  # (B, 3)

        f0_stats = torch.cat([
            f0_features.mean(dim=1),
            f0_features.std(dim=1)
        ], dim=-1)  # (B, 64)

        # 全局韵律编码
        global_input = x_mean + x_std  # 简单融合
        global_prosody = self.global_prosody_encoder(global_input)  # (B, D//2)

        # 局部韵律编码
        local_prosody = self.local_prosody_encoder(x)  # (B, T, D)

        return {
            'global_prosody': global_prosody,
            'local_prosody': local_prosody,
            'energy': energy,
            'f0_features': f0_features,
            'energy_stats': energy_stats,
            'f0_stats': f0_stats
        }


# ============================================================
# Level 1: 粗粒度聚类 (能量/情绪)
# ============================================================

class Level1Quantizer(nn.Module):
    """
    一级量化器: 基于能量和情绪的粗粒度聚类

    设计思路:
    - 将utterance按全局韵律特征分成8-16个大类
    - 例如: 高能量激动、高能量平稳、低能量激动、低能量平稳等
    """
    def __init__(self, d_model: int = 128, n_clusters: int = 8):
        super().__init__()
        self.n_clusters = n_clusters
        self.d_model = d_model

        # 一级codebook - 使用正交初始化确保cluster分散
        init = torch.randn(d_model, n_clusters)
        U, _, _ = torch.svd(init)
        self.codebook = nn.Parameter(U[:, :n_clusters].T.contiguous())

        # 用于计算assignment的投影
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # EMA更新相关
        self.register_buffer('ema_cluster_size', torch.ones(n_clusters))
        self.register_buffer('ema_centroids', self.codebook.data.clone())
        self.register_buffer('cluster_usage', torch.ones(n_clusters) / n_clusters)

        self.ema_decay = 0.99

    def forward(self, global_prosody: torch.Tensor, temperature: float = 1.0):
        """
        global_prosody: (B, D) 全局韵律特征

        返回:
        - l1_indices: (B,) 一级cluster索引
        - l1_soft: (B, K1) soft assignment
        - l1_embedding: (B, D) 量化后的embedding
        """
        B, D = global_prosody.shape

        # 投影到codebook空间
        h = self.proj(global_prosody)
        h_norm = F.normalize(h, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)

        # 计算相似度
        logits = torch.matmul(h_norm, cb_norm.T) / temperature  # (B, K1)

        # 训练时添加Gumbel噪声防止模式坍塌
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise * 0.5

        # Soft assignment
        soft_assign = F.softmax(logits, dim=-1)

        # Hard assignment
        hard_indices = logits.argmax(dim=-1)  # (B,)

        # Straight-through estimator
        hard_assign = F.one_hot(hard_indices, self.n_clusters).float()
        assign = hard_assign + soft_assign - soft_assign.detach()

        # 量化embedding
        l1_embedding = torch.matmul(assign, self.codebook)

        # EMA更新
        if self.training:
            self._ema_update(h, hard_assign)

        return {
            'indices': hard_indices,
            'soft_assign': soft_assign,
            'embedding': l1_embedding,
            'logits': logits
        }

    def _ema_update(self, h: torch.Tensor, hard_assign: torch.Tensor):
        with torch.no_grad():
            batch_size = hard_assign.sum(0)  # (K1,)

            self.ema_cluster_size = (
                self.ema_decay * self.ema_cluster_size +
                (1 - self.ema_decay) * batch_size
            )

            batch_centroids = hard_assign.T @ h  # (K1, D)
            self.ema_centroids = (
                self.ema_decay * self.ema_centroids +
                (1 - self.ema_decay) * batch_centroids
            )

            # 更新codebook
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.n_clusters * 1e-5) * n
            self.codebook.data = self.ema_centroids / cluster_size.unsqueeze(1).clamp(min=1.0)

            # 更新使用统计
            usage = batch_size / (batch_size.sum() + 1e-8)
            self.cluster_usage = 0.9 * self.cluster_usage + 0.1 * usage


# ============================================================
# Level 2: 细粒度聚类 (韵律细节)
# ============================================================

class Level2Quantizer(nn.Module):
    """
    二级量化器: 每个一级cluster内的细粒度韵律聚类

    设计思路:
    - 每个L1 cluster有独立的L2 codebook
    - 捕捉细微的韵律差别: 语调轮廓、节奏模式、重音分布
    """
    def __init__(self, d_model: int = 256, n_l1_clusters: int = 8,
                 n_l2_patterns: int = 32, bottleneck_dim: int = 64):
        super().__init__()
        self.n_l1_clusters = n_l1_clusters
        self.n_l2_patterns = n_l2_patterns
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # 每个L1 cluster有独立的L2 codebook
        # 使用nn.ParameterList存储
        self.l2_codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(n_l2_patterns, bottleneck_dim) * 0.1)
            for _ in range(n_l1_clusters)
        ])

        # 投影到bottleneck空间
        self.to_bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 每个L1的使用统计
        for i in range(n_l1_clusters):
            self.register_buffer(
                f'l2_usage_{i}',
                torch.ones(n_l2_patterns) / n_l2_patterns
            )

    def forward(self, local_prosody: torch.Tensor, l1_indices: torch.Tensor,
                temperature: float = 1.0):
        """
        local_prosody: (B, T, D) 局部韵律特征
        l1_indices: (B,) 一级cluster索引

        返回:
        - l2_indices: (B, T) 二级pattern索引
        - l2_soft: (B, T, K2) soft assignment
        - l2_embedding: (B, T, D) 量化后的embedding
        """
        B, T, D = local_prosody.shape

        # 投影到bottleneck
        h = self.to_bottleneck(local_prosody)  # (B, T, bottleneck)
        h_norm = F.normalize(h, dim=-1)

        # 为每个样本选择对应的L2 codebook
        l2_indices_list = []
        l2_soft_list = []
        l2_emb_list = []

        for b in range(B):
            l1_idx = l1_indices[b].item()
            codebook = self.l2_codebooks[l1_idx]  # (K2, bottleneck)
            cb_norm = F.normalize(codebook, dim=-1)

            # 计算相似度
            logits = torch.matmul(h_norm[b], cb_norm.T) / temperature  # (T, K2)

            # Soft assignment
            soft = F.softmax(logits, dim=-1)

            # Hard assignment
            hard_idx = logits.argmax(dim=-1)  # (T,)
            hard = F.one_hot(hard_idx, self.n_l2_patterns).float()

            # Straight-through
            assign = hard + soft - soft.detach()

            # 量化embedding
            emb = torch.matmul(assign, codebook)  # (T, bottleneck)

            l2_indices_list.append(hard_idx)
            l2_soft_list.append(soft)
            l2_emb_list.append(emb)

            # 更新使用统计
            if self.training:
                usage_buffer = getattr(self, f'l2_usage_{l1_idx}')
                batch_usage = hard.sum(0) / (T + 1e-8)
                usage_buffer.data = 0.9 * usage_buffer + 0.1 * batch_usage

        l2_indices = torch.stack(l2_indices_list)  # (B, T)
        l2_soft = torch.stack(l2_soft_list)        # (B, T, K2)
        l2_emb_low = torch.stack(l2_emb_list)      # (B, T, bottleneck)

        # 投影回原始维度
        l2_embedding = self.from_bottleneck(l2_emb_low)  # (B, T, D)

        return {
            'indices': l2_indices,
            'soft_assign': l2_soft,
            'embedding': l2_embedding,
            'bottleneck': l2_emb_low
        }


# ============================================================
# 层级VQ编码器
# ============================================================

class HierarchicalVQEncoder(nn.Module):
    """
    层级矢量量化编码器

    架构:
    Input -> ProsodyExtractor -> L1 Quantizer -> L2 Quantizer -> Output

    L1: 粗粒度 (能量/情绪) - 8-16 clusters
    L2: 细粒度 (韵律细节) - 每个L1内32 patterns
    """
    def __init__(self, input_dim: int = 1024, d_model: int = 256,
                 n_l1_clusters: int = 8, n_l2_patterns: int = 32,
                 bottleneck_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_l1_clusters = n_l1_clusters
        self.n_l2_patterns = n_l2_patterns

        # 韵律特征提取
        self.prosody_extractor = ProsodyFeatureExtractor(input_dim, d_model)

        # L1量化器 (全局韵律)
        self.l1_quantizer = Level1Quantizer(
            d_model=d_model // 2,
            n_clusters=n_l1_clusters
        )

        # L2量化器 (局部韵律)
        self.l2_quantizer = Level2Quantizer(
            d_model=d_model,
            n_l1_clusters=n_l1_clusters,
            n_l2_patterns=n_l2_patterns,
            bottleneck_dim=bottleneck_dim
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        """
        x: (B, T, input_dim) WavLM特征

        返回完整的层级量化结果
        """
        B, T, _ = x.shape

        # 1. 提取韵律特征
        prosody = self.prosody_extractor(x)

        # 2. L1量化 (全局)
        l1_result = self.l1_quantizer(
            prosody['global_prosody'],
            temperature=temperature
        )

        # 3. L2量化 (局部，条件于L1)
        l2_result = self.l2_quantizer(
            prosody['local_prosody'],
            l1_result['indices'],
            temperature=temperature
        )

        # 4. 融合L1和L2
        l1_expanded = l1_result['embedding'].unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([l2_result['embedding'], l1_expanded], dim=-1)
        output = self.fusion(fused)

        return {
            'output': output,
            'l1': l1_result,
            'l2': l2_result,
            'prosody': prosody
        }


# ============================================================
# 层级VQ损失函数
# ============================================================

class HierarchicalVQLoss(nn.Module):
    """层级VQ的损失函数"""
    def __init__(self, n_l1_clusters: int = 8, n_l2_patterns: int = 32):
        super().__init__()
        self.n_l1 = n_l1_clusters
        self.n_l2 = n_l2_patterns

    def l1_diversity_loss(self, l1_soft: torch.Tensor):
        """L1 cluster均衡使用"""
        usage = l1_soft.mean(dim=0)
        target = torch.ones_like(usage) / self.n_l1
        return F.mse_loss(usage, target)

    def l2_diversity_loss(self, l2_soft: torch.Tensor, l1_indices: torch.Tensor):
        """每个L1内的L2 pattern均衡使用"""
        B, T, K2 = l2_soft.shape
        losses = []
        for l1_idx in range(self.n_l1):
            mask = l1_indices == l1_idx
            if mask.sum() == 0:
                continue
            l2_usage = l2_soft[mask].mean(dim=[0, 1])
            target = torch.ones_like(l2_usage) / self.n_l2
            losses.append(F.mse_loss(l2_usage, target))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def l1_separation_loss(self, l1_codebook: torch.Tensor, margin: float = 1.0):
        """L1 cluster之间的分离"""
        cb_norm = F.normalize(l1_codebook, dim=-1)
        sim = torch.matmul(cb_norm, cb_norm.T)
        mask = 1 - torch.eye(self.n_l1, device=l1_codebook.device)
        violations = F.relu(sim - (1 - margin)) * mask
        return violations.sum() / mask.sum().clamp(min=1.0)


# ============================================================
# Dataset
# ============================================================

class SpeakerDataset(Dataset):
    def __init__(self, data_list: list, max_len: int = 500):
        self.data = data_list
        self.max_len = max_len
        self.speaker_to_id = {
            s: i for i, s in enumerate(set(d['speaker'] for d in data_list))
        }

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
# 训练器
# ============================================================

class HierarchicalVQTrainer:
    """层级VQ训练器"""
    def __init__(self, model: HierarchicalVQEncoder, device='cuda',
                 total_epochs=150, warmup_epochs=15, logger=None):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.logger = logger

        self.loss_fn = HierarchicalVQLoss(
            model.n_l1_clusters, model.n_l2_patterns
        )

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_epochs, eta_min=1e-5
        )
        self.scaler = GradScaler()
        self.history = []

    def log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def get_temperature(self, epoch: int) -> float:
        """温度退火"""
        if epoch < self.warmup_epochs:
            return 1.0
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return 0.2 + 0.8 * (1 + np.cos(np.pi * progress)) / 2

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        all_metrics = []
        temp = self.get_temperature(epoch)

        for batch in dataloader:
            x = batch['features'].to(self.device)
            metrics = self._train_step(x, epoch, temp)
            all_metrics.append(metrics)

        avg = {}
        for k in all_metrics[0]:
            values = [m[k] for m in all_metrics]
            if isinstance(values[0], (int, float)):
                avg[k] = np.mean(values)
            else:
                avg[k] = values[0]  # 字符串类型直接取第一个
        self.scheduler.step()
        self.history.append(avg)
        return avg

    def _train_step(self, x: torch.Tensor, epoch: int, temp: float):
        with autocast():
            result = self.model(x, temperature=temp)
            loss, metrics = self._compute_loss(result, epoch)

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return metrics

    def _compute_loss(self, result: dict, epoch: int):
        """计算层级VQ损失"""
        l1 = result['l1']
        l2 = result['l2']
        prosody = result['prosody']

        is_warmup = epoch < self.warmup_epochs
        progress = max(0, (epoch - self.warmup_epochs) /
                      (self.total_epochs - self.warmup_epochs))

        # 重构损失: 局部韵律特征重构
        recon_loss = F.mse_loss(
            l2['embedding'],
            prosody['local_prosody'].detach()
        )

        # L1 diversity - 从一开始就需要，防止模式坍塌
        l1_div = self.loss_fn.l1_diversity_loss(l1['soft_assign'])

        # L1 separation - 从一开始就需要
        l1_sep = self.loss_fn.l1_separation_loss(
            self.model.l1_quantizer.codebook, margin=1.2
        )

        with torch.no_grad():
            l1_unique = len(torch.unique(l1['indices']))

        if is_warmup:
            # warmup阶段也要有L1多样性损失
            loss = recon_loss + 0.5 * l1_div + 0.3 * l1_sep
            return loss, {
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'l1_div': l1_div.item(),
                'l2_div': 0.0,
                'l1_sep': l1_sep.item(),
                'l1_unique': l1_unique,
                'phase': 'warmup'
            }

        # L2 diversity (条件于L1)
        l2_div = self.loss_fn.l2_diversity_loss(
            l2['soft_assign'], l1['indices']
        )

        # 权重调度 - 增大L1多样性权重
        div_weight = 0.5 + min(0.5, progress * 0.5)
        sep_weight = 0.3 + min(0.3, progress * 0.3)

        loss = (
            recon_loss +
            div_weight * l1_div +
            0.3 * l2_div +
            sep_weight * l1_sep
        )

        return loss, {
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'l1_div': l1_div.item(),
            'l2_div': l2_div.item(),
            'l1_sep': l1_sep.item(),
            'l1_unique': l1_unique,
            'phase': 'main'
        }


# ============================================================
# 数据加载
# ============================================================

def load_utterance_data(cache_dir: Path, max_utts: int = 10000):
    """加载IEMOCAP数据"""
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

def visualize_hierarchical_clusters(model, data, output_dir, timestamp,
                                    device='cuda', logger=None):
    """可视化层级聚类结果"""
    from sklearn.manifold import TSNE

    model.eval()
    l1_indices_list = []
    l2_indices_list = []
    global_prosody_list = []
    speakers_list = []

    with torch.no_grad():
        for d in data[:2000]:
            x = torch.from_numpy(d['features']).float().unsqueeze(0).to(device)
            result = model(x, temperature=0.5)

            l1_indices_list.append(result['l1']['indices'].item())
            l2_dom = result['l2']['indices'][0].mode().values.item()
            l2_indices_list.append(l2_dom)
            global_prosody_list.append(
                result['prosody']['global_prosody'].cpu().numpy()[0]
            )
            speakers_list.append(d['speaker'])

    l1_indices = np.array(l1_indices_list)
    l2_indices = np.array(l2_indices_list)
    global_prosody = np.array(global_prosody_list)

    # 计算统计
    l1_unique = len(np.unique(l1_indices))
    combined_ids = l1_indices * model.n_l2_patterns + l2_indices
    combined_unique = len(np.unique(combined_ids))

    log_msg = f"\n{'='*50}\n"
    log_msg += f"=== Hierarchical VQ Results ===\n"
    log_msg += f"{'='*50}\n"
    log_msg += f"L1 Unique Clusters: {l1_unique}/{model.n_l1_clusters}\n"
    log_msg += f"Combined Unique Patterns: {combined_unique}\n"

    if logger:
        logger.info(log_msg)
    print(log_msg)

    # t-SNE可视化
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(global_prosody)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 按说话人着色
    spk_map = {s: i for i, s in enumerate(set(speakers_list))}
    spk_ids = np.array([spk_map[s] for s in speakers_list])
    scatter1 = axes[0, 0].scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=spk_ids, cmap='tab20', alpha=0.7, s=30
    )
    axes[0, 0].set_title(f'By Speaker ({len(spk_map)} speakers)')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # 按L1 cluster着色
    scatter2 = axes[0, 1].scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=l1_indices, cmap='tab10', alpha=0.7, s=30
    )
    axes[0, 1].set_title(f'By L1 Cluster ({l1_unique} unique)')
    plt.colorbar(scatter2, ax=axes[0, 1])

    # 按组合ID着色
    scatter3 = axes[1, 0].scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=combined_ids, cmap='viridis', alpha=0.7, s=30
    )
    axes[1, 0].set_title(f'By Combined L1+L2 ({combined_unique} unique)')
    plt.colorbar(scatter3, ax=axes[1, 0])

    # L1使用统计
    l1_counts = np.bincount(l1_indices, minlength=model.n_l1_clusters)
    axes[1, 1].bar(range(model.n_l1_clusters), l1_counts, color='steelblue')
    axes[1, 1].set_title('L1 Cluster Usage')
    axes[1, 1].set_xlabel('L1 Cluster ID')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    save_path = output_dir / f'hvq_clusters_{timestamp}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    if logger:
        logger.info(f"Saved plot to {save_path}")

    return {
        'l1_unique': l1_unique,
        'combined_unique': combined_unique,
        'l1_indices': l1_indices,
        'l2_indices': l2_indices
    }


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache'
    log_dir = base_dir / 'logs'
    output_dir = base_dir / 'outputs' / f'hvq_pool_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(log_dir, timestamp)

    logger.info("=" * 60)
    logger.info("Hierarchical VQ Speaker Pool Builder")
    logger.info("=" * 60)

    # 配置
    n_l1_clusters = 8      # 一级聚类数 (能量/情绪)
    n_l2_patterns = 32     # 每个L1内的二级pattern数
    total_patterns = n_l1_clusters * n_l2_patterns  # 总共256个pattern

    logger.info(f"Config: L1={n_l1_clusters}, L2={n_l2_patterns}")
    logger.info(f"Total patterns: {total_patterns}")

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # 加载数据
    logger.info("[1] Loading data...")
    data = load_utterance_data(cache_dir, max_utts=8000)
    n_speakers = len(set(d['speaker'] for d in data))
    logger.info(f"Loaded {len(data)} utterances, {n_speakers} speakers")

    # 创建数据加载器
    dataset = SpeakerDataset(data, max_len=400)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )

    # 创建模型
    logger.info("[2] Creating model...")
    model = HierarchicalVQEncoder(
        input_dim=1024,
        d_model=256,
        n_l1_clusters=n_l1_clusters,
        n_l2_patterns=n_l2_patterns,
        bottleneck_dim=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # 训练
    logger.info("[3] Training...")
    trainer = HierarchicalVQTrainer(
        model, device=device,
        total_epochs=150,
        warmup_epochs=15,
        logger=logger
    )

    for epoch in range(150):
        metrics = trainer.train_epoch(dataloader, epoch)
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/150 | Loss: {metrics['loss']:.4f} | "
                f"L1 Unique: {metrics['l1_unique']}"
            )

    # 可视化
    logger.info("[4] Visualizing...")
    results = visualize_hierarchical_clusters(
        model, data, output_dir, timestamp,
        device=device, logger=logger
    )

    # 保存模型
    logger.info("[5] Saving model...")
    model_path = output_dir / f'hvq_model_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_l1_clusters': n_l1_clusters,
            'n_l2_patterns': n_l2_patterns,
        },
        'results': results,
        'history': trainer.history
    }, model_path)

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"L1 Unique: {results['l1_unique']}/{n_l1_clusters}")
    logger.info(f"Combined Unique: {results['combined_unique']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
