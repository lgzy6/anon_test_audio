#!/usr/bin/env python3
"""
DS-SAMM v8.0: 修复版聚类 - 解决v7模式坍塌问题

核心修复:
1. 对比学习损失: 同speaker样本应分配相似pattern
2. 修复分离损失: 基于embedding距离而非pattern分配
3. VQ-VAE风格commitment loss: 更稳定的量化
4. 温度调度优化: 保持最低温度0.5
5. 改进死亡pattern重初始化: 使用随机正交向量
6. GPU优化: 混合精度训练 + DataLoader多进程
7. 大规模测试: 支持全量10000+数据
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
    log_file = log_dir / f'training_v8_{timestamp}.log'

    logger = logging.getLogger('SAMM_v8')
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
            # 随机截取
            start = np.random.randint(0, feat.shape[0] - self.max_len)
            feat = feat[start:start + self.max_len]

        speaker_id = self.speaker_to_id[d['speaker']]
        return {
            'features': torch.from_numpy(feat).float(),
            'speaker_id': speaker_id,
            'length': feat.shape[0]
        }

    def get_same_speaker_sample(self, idx):
        """获取同一speaker的另一个样本索引"""
        speaker = self.data[idx]['speaker']
        candidates = [i for i in self.speaker_to_indices[speaker] if i != idx]
        if candidates:
            return np.random.choice(candidates)
        return idx


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
    """符号化层 - 保持不变"""
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
    """自注意力层 - 保持不变"""
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
# 核心修复: VQ-VAE风格Pattern矩阵
# ============================================================

class VQPatternMatrix(nn.Module):
    """
    v8.0 VQ-VAE风格Pattern矩阵

    修复点:
    1. 使用VQ-VAE的commitment loss替代Hard Gumbel
    2. 温度调度更平缓，最低0.5
    3. 改进死亡pattern重初始化
    """

    def __init__(self, d_model: int = 256, n_patterns: int = 32):
        super().__init__()
        self.n_patterns = n_patterns
        self.d_model = d_model

        # 正交初始化
        patterns_init = torch.randn(d_model, n_patterns)
        U, _, _ = torch.svd(patterns_init)
        self.patterns = nn.Parameter(U[:, :n_patterns].T.contiguous())

        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # EMA prototypes
        self.register_buffer('ema_patterns', self.patterns.data.clone())
        self.register_buffer('pattern_usage', torch.ones(n_patterns) / n_patterns)
        self.register_buffer('pattern_ema_count', torch.zeros(n_patterns))

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """修复: 更平缓的温度调度，最低0.5"""
        progress = epoch / total_epochs
        # 从1.0线性降到0.5
        return max(0.5, 1.0 - 0.5 * progress)

    def forward(self, x: torch.Tensor, epoch: int = 0, total_epochs: int = 100):
        B, T, D = x.shape

        Q = self.query_proj(x)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(self.patterns, dim=-1)

        temperature = self.get_temperature(epoch, total_epochs)
        logits = torch.matmul(Q, K.T) / temperature  # (B, T, K)

        # 修复: 前50%使用纯软分配，后50%逐渐过渡到hard
        progress = epoch / total_epochs
        soft_assignments = F.softmax(logits, dim=-1)
        hard_indices = logits.argmax(dim=-1)  # (B, T)
        hard_assignments = F.one_hot(hard_indices, self.n_patterns).float()

        if self.training:
            if progress < 0.5:
                # 前50%: 纯软分配，保持梯度流动
                assignments = soft_assignments
            else:
                # 后50%: 逐渐过渡到hard
                hard_ratio = (progress - 0.5) * 2  # 0->1
                assignments = (1 - hard_ratio) * soft_assignments + \
                              hard_ratio * (hard_assignments + soft_assignments - soft_assignments.detach())
        else:
            assignments = hard_assignments

        # 更新使用统计
        if self.training:
            with torch.no_grad():
                usage = hard_assignments.sum(dim=[0, 1])
                usage = usage / (usage.sum() + 1e-8)
                self.pattern_usage = 0.9 * self.pattern_usage + 0.1 * usage
                self.ema_patterns = 0.99 * self.ema_patterns + 0.01 * self.patterns.data

        pattern_emb = torch.matmul(assignments, self.patterns)
        return pattern_emb, assignments, logits, hard_indices

    def commitment_loss(self, hidden: torch.Tensor, pattern_emb: torch.Tensor):
        """VQ-VAE commitment loss: 让encoder输出靠近选中的pattern"""
        return F.mse_loss(hidden, pattern_emb.detach())

    def codebook_loss(self, hidden: torch.Tensor, pattern_emb: torch.Tensor):
        """VQ-VAE codebook loss: 让pattern靠近encoder输出"""
        return F.mse_loss(pattern_emb, hidden.detach())

    def diversity_loss(self):
        """多样性损失 - 增强版"""
        K = F.normalize(self.patterns, dim=-1)
        sim = torch.matmul(K, K.T)
        mask = 1.0 - torch.eye(self.n_patterns, device=K.device)
        ortho = (sim.abs() * mask).mean()

        # 增强: 更强的均匀分布约束
        target = torch.ones_like(self.pattern_usage) / self.n_patterns
        usage_loss = F.kl_div(
            (self.pattern_usage + 1e-8).log(), target, reduction='sum'
        )

        # 新增: 熵最大化损失
        entropy = -(self.pattern_usage * (self.pattern_usage + 1e-8).log()).sum()
        max_entropy = np.log(self.n_patterns)
        entropy_loss = max_entropy - entropy

        return ortho + 0.5 * usage_loss + 0.3 * entropy_loss

    def reinit_dead_patterns(self, threshold: float = 0.01):
        """修复: 使用随机正交向量重初始化"""
        dead = self.pattern_usage < threshold
        n_dead = dead.sum().item()

        if 0 < n_dead < self.n_patterns:
            # 生成随机正交向量
            random_vecs = torch.randn(int(n_dead), self.d_model, device=self.patterns.device)
            random_vecs = F.normalize(random_vecs, dim=-1)

            # 添加噪声确保多样性
            noise = torch.randn_like(random_vecs) * 0.1
            new_patterns = F.normalize(random_vecs + noise, dim=-1)

            self.patterns.data[dead] = new_patterns
            self.pattern_usage[dead] = 1.0 / self.n_patterns

        return int(n_dead)


# ============================================================
# 修复: 对比学习损失
# ============================================================

class ContrastiveSpeakerLoss(nn.Module):
    """
    对比学习损失: 同speaker样本应有相似的pattern分布

    修复v7问题: 添加speaker级别的监督信号
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pattern_dists: torch.Tensor, speaker_ids: torch.Tensor):
        """
        Args:
            pattern_dists: (B, K) 每个样本的pattern分布
            speaker_ids: (B,) speaker标签
        """
        B = pattern_dists.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=pattern_dists.device)

        # 归一化pattern分布
        pattern_dists = F.normalize(pattern_dists, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(pattern_dists, pattern_dists.T) / self.temperature

        # 构建正样本mask: 同speaker的样本对 (避免inplace操作)
        speaker_mask = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        diag_mask = 1.0 - torch.eye(B, device=speaker_mask.device)
        speaker_mask = speaker_mask * diag_mask  # 排除自己

        # InfoNCE loss (避免inplace操作)
        exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
        exp_sim = exp_sim * diag_mask  # 排除对角线

        # 正样本的相似度
        pos_sim = (exp_sim * speaker_mask).sum(dim=1)
        # 所有样本的相似度
        all_sim = exp_sim.sum(dim=1)

        # 避免除零
        valid = (speaker_mask.sum(dim=1) > 0) & (all_sim > 1e-8)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pattern_dists.device)

        loss = -torch.log(pos_sim[valid] / all_sim[valid] + 1e-8).mean()
        return loss


class ImprovedSeparationLoss(nn.Module):
    """
    修复版分离损失: 基于embedding距离而非pattern分配

    v7问题: 当模式坍塌时diff_pattern全为False，损失为0
    修复: 直接基于embedding距离计算，不依赖pattern分配
    """
    def __init__(self, margin: float = 0.5, n_negatives: int = 16):
        super().__init__()
        self.margin = margin
        self.n_negatives = n_negatives

    def forward(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor):
        """
        Args:
            embeddings: (B, D) 句子级embedding
            speaker_ids: (B,) speaker标签
        """
        B = embeddings.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=-1)

        # 计算距离矩阵
        dist_matrix = 1.0 - torch.matmul(embeddings, embeddings.T)

        # 不同speaker的样本应该有足够距离
        diff_speaker = (speaker_ids.unsqueeze(0) != speaker_ids.unsqueeze(1)).float()

        # Triplet-style loss: 不同speaker距离应 > margin
        separation = F.relu(self.margin - dist_matrix) * diff_speaker
        n_pairs = diff_speaker.sum().clamp(min=1)

        return separation.sum() / n_pairs


# ============================================================
# 完整编码器
# ============================================================

class SAMMEncoderV8(nn.Module):
    """v8.0 修复版SAMM编码器"""

    def __init__(self, input_dim=1024, d_model=256, n_symbols=64,
                 n_heads=4, n_layers=2, n_patterns=32):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns

        self.symbol = SymbolizationLayer(input_dim, n_symbols, d_model)
        self.layers = nn.ModuleList([
            MaskedSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pattern = VQPatternMatrix(d_model, n_patterns)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        self.sent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
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
        emb, assign, logits, hard_idx = self.pattern(h, epoch, total_epochs)
        return emb, assign, logits, h, hard_idx

    def get_utterance_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_to_hidden(x)
            return self.sent_proj(h.mean(dim=1))

    def get_pattern_distribution(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, assign, _, _, _ = self.forward(x)
            return assign.mean(dim=1)


# ============================================================
# 训练器 (GPU优化版)
# ============================================================

class SAMMTrainerV8:
    """v8.0 训练器 - 混合精度 + DataLoader"""

    def __init__(self, model: SAMMEncoderV8, device='cuda',
                 total_epochs=100, logger=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.logger = logger
        self.use_amp = use_amp and device == 'cuda'

        # 损失函数
        self.contrastive_loss = ContrastiveSpeakerLoss(temperature=0.1)
        self.separation_loss = ImprovedSeparationLoss(margin=0.5)

        self.opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_epochs, eta_min=1e-5
        )

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None
        self.history = []

    def log(self, msg: str, level: str = 'info'):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        all_metrics = []

        for batch in dataloader:
            x = batch['features'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)

            metrics = self._train_step(x, speaker_ids, epoch)
            all_metrics.append(metrics)

        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        # 定期重初始化死亡pattern
        if (epoch + 1) % 10 == 0:
            n_reinit = self.model.pattern.reinit_dead_patterns(0.01)
            avg['reinit'] = n_reinit
            if n_reinit > 0:
                self.log(f"Epoch {epoch+1}: Reinitialized {n_reinit} dead patterns")
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
        """计算所有损失"""
        emb, assign, logits, hidden, hard_idx = self.model(
            x, epoch=epoch, total_epochs=self.total_epochs
        )
        B, T, K = assign.shape

        # 1. VQ-VAE损失
        commitment = self.model.pattern.commitment_loss(hidden, emb)
        codebook = self.model.pattern.codebook_loss(hidden, emb)

        # 2. 多样性损失
        div_loss = self.model.pattern.diversity_loss()

        # 3. 对比学习损失 (同speaker相似)
        pattern_dist = assign.mean(dim=1)  # (B, K)
        contrastive = self.contrastive_loss(pattern_dist, speaker_ids)

        # 4. 分离损失 (不同speaker分开)
        sent_emb = self.model.sent_proj(hidden.mean(dim=1))
        separation = self.separation_loss(sent_emb, speaker_ids)

        # 修复: 增加多样性损失权重，前期更重要
        progress = epoch / self.total_epochs
        div_w = 1.0 if progress < 0.3 else 0.5
        contrast_w = 0.3 if progress < 0.3 else 0.5

        loss = (
            0.1 * commitment +
            0.1 * codebook +
            div_w * div_loss +
            contrast_w * contrastive +
            0.2 * separation
        )

        with torch.no_grad():
            unique = len(torch.unique(hard_idx))
            active = (self.model.pattern.pattern_usage > 0.01).sum().item()

        return loss, {
            'loss': loss.item(),
            'commitment': commitment.item(),
            'codebook': codebook.item(),
            'diversity': div_loss.item(),
            'contrastive': contrastive.item(),
            'separation': separation.item(),
            'unique': unique,
            'active': active,
        }


# ============================================================
# 数据加载
# ============================================================

def load_utterance_data(cache_dir: Path, max_utts: int = 10000):
    """加载数据 - 支持大规模"""
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

    log_msg = f"\n=== v8.0 Results ===\n"
    log_msg += f"Unique Dominant Patterns: {unique_patterns}/{n_patterns}\n"
    log_msg += f"Active Patterns (>1%): {active_patterns}/{n_patterns}\n"
    log_msg += f"Pattern Usage Std: {total_usage.std():.2f}"

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
    axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=spk_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 0].set_title(f'By Speaker ({len(spk_map)} speakers)')

    # 2. By Dominant Pattern
    axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pattern_ids, cmap='tab20', alpha=0.7, s=30)
    axes[0, 1].set_title(f'By Dominant Pattern ({unique_patterns} unique)')

    # 3. KMeans on pattern distribution
    kmeans = KMeans(n_clusters=min(8, n_patterns), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pattern_dists)
    axes[1, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title('By Pattern Distribution Cluster')

    # 4. Pattern Usage
    sorted_usage = np.sort(total_usage)[::-1]
    colors = ['steelblue' if u > total_usage.max() * 0.01 else 'lightgray' for u in sorted_usage]
    axes[1, 1].bar(range(n_patterns), sorted_usage, color=colors)
    axes[1, 1].axhline(y=total_usage.mean(), color='red', linestyle='--', label='Mean')
    axes[1, 1].set_title(f'Pattern Usage (Active: {active_patterns}/{n_patterns})')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = output_dir / f'samm_clusters_v8_{timestamp}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    if logger:
        logger.info(f"Saved cluster plot to {save_path}")

    # 聚类指标
    if len(set(speakers_list)) > 1:
        ari = adjusted_rand_score(spk_ids, pattern_ids)
        nmi = normalized_mutual_info_score(spk_ids, pattern_ids)
        sil = silhouette_score(embeddings, pattern_ids) if unique_patterns > 1 else 0

        metrics_msg = f"\n=== Clustering Metrics ===\n"
        metrics_msg += f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil:.4f}"

        if logger:
            logger.info(metrics_msg)
        print(metrics_msg)

    return {'unique': unique_patterns, 'active': active_patterns, 'embeddings': embeddings}


def plot_training_history(history, output_dir, timestamp):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = range(1, len(history) + 1)

    axes[0, 0].plot(epochs, [h['loss'] for h in history], 'b-')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h['commitment'] for h in history], 'g-', label='Commit')
    axes[0, 1].plot(epochs, [h['codebook'] for h in history], 'r--', label='Codebook')
    axes[0, 1].set_title('VQ Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, [h['contrastive'] for h in history], 'purple', label='Contrast')
    axes[0, 2].plot(epochs, [h['separation'] for h in history], 'orange', label='Separate')
    axes[0, 2].set_title('Speaker Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h['unique'] for h in history], 'g-', label='Unique')
    axes[1, 0].plot(epochs, [h['active'] for h in history], 'b--', label='Active')
    axes[1, 0].set_title('Pattern Usage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [h['diversity'] for h in history], 'brown')
    axes[1, 1].set_title('Diversity Loss')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, [h.get('reinit', 0) for h in history], 'red')
    axes[1, 2].set_title('Pattern Reinit')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_v8_{timestamp}.png', dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache'
    log_dir = base_dir / 'logs'
    output_dir = base_dir / 'outputs' / f'v8_tests_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(log_dir, timestamp)

    logger.info("=" * 60)
    logger.info("DS-SAMM v8.0: Fixed Clustering")
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

    # 配置参数 - 大规模测试
    n_patterns = 32
    max_utts = 8000  # 增加数据量
    total_epochs = 150
    batch_size = 64
    num_workers = 4

    logger.info(f"Config: n_patterns={n_patterns}, max_utts={max_utts}")
    logger.info(f"Config: epochs={total_epochs}, batch_size={batch_size}")

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
    model = SAMMEncoderV8(
        input_dim=1024, d_model=256,
        n_patterns=n_patterns, n_layers=2, n_symbols=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # 训练
    logger.info("[3] Training...")
    trainer = SAMMTrainerV8(
        model, device=device,
        total_epochs=total_epochs, logger=logger, use_amp=True
    )

    for epoch in range(total_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Contrast: {metrics['contrastive']:.4f} | "
                f"Sep: {metrics['separation']:.4f}"
            )
            logger.info(
                f"  Unique: {metrics['unique']}/{n_patterns} | "
                f"Active: {metrics['active']}/{n_patterns}"
            )

    # 绘制训练历史
    logger.info("[4] Plotting training history...")
    plot_training_history(trainer.history, output_dir, timestamp)

    # 可视化结果
    logger.info("[5] Visualizing results...")
    results = visualize_results(
        model, data[:2000], output_dir, timestamp,
        device=device, logger=logger
    )

    # 保存模型
    logger.info("[6] Saving model...")
    model_path = output_dir / f'samm_v8_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 1024,
            'd_model': 256,
            'n_patterns': n_patterns
        },
        'results': results,
        'history': trainer.history
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Results: Unique={results['unique']}, Active={results['active']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
