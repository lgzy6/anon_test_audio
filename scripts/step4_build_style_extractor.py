#!/usr/bin/env python3
"""
步骤 4：构建段级风格提取器

从帧级 η 出发，按 phone 边界分段，提取段级 style embedding。
验证段级 embedding 是否具备聚类结构。

输入：
  - cache/features/wavlm/features.h5
  - cache/features/wavlm/metadata.json
  - checkpoints/eta_projection.pt (Step 2)
  - checkpoints/speaker_embeddings_pca.npy (Step 1)

输出：
  - checkpoints/style_extractor.pkl (PCA 模型 + 配置)
  - outputs/style_analysis/segment_clustering.png

使用方法：
    python scripts/step4_build_style_extractor.py
"""

import sys
import json
import argparse
import numpy as np
import torch
import pickle
import h5py
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor

BASE_DIR = Path(__file__).parent.parent


class SegmentStyleExtractor:
    """
    段级风格 Embedding 提取器
    
    核心思想：
      - 风格是慢变量，帧级看不出来
      - 按 phone 边界分段，每段取统计量
      - [mean, std, delta_mean] 捕获 "怎么说" 而非 "说什么"
    """
    
    def __init__(self, pca_dim=64, min_segment_frames=3):
        self.pca_dim = pca_dim
        self.min_segment_frames = min_segment_frames
        self.pca = None
        self.fitted = False
    
    def _find_phone_segments(self, phones):
        """找到连续相同 phone 的段边界"""
        if len(phones) == 0:
            return []
        boundaries = np.where(np.diff(phones) != 0)[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(phones)]])
        
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            phone_id = phones[start]
            if phone_id != 0 and (end - start) >= self.min_segment_frames:
                segments.append((start, end, int(phone_id)))
        return segments
    
    def _segment_to_raw_embedding(self, frames):
        """单段 → 原始 style embedding (统计 pooling)"""
        mu = frames.mean(axis=0)                    # 平均发音方式
        sigma = frames.std(axis=0) + 1e-8           # 发音稳定性
        if len(frames) > 1:
            delta = np.diff(frames, axis=0)
            delta_mu = delta.mean(axis=0)           # 动态变化趋势
        else:
            delta_mu = np.zeros_like(mu)
        return np.concatenate([mu, sigma, delta_mu])  # 3072 维
    
    def fit(self, eta_frames, phones_all, max_segments=50000):
        """在训练集上拟合 PCA"""
        print("[Step 4] Fitting SegmentStyleExtractor...")
        
        segments = self._find_phone_segments(phones_all)
        print(f"  Total segments found: {len(segments)}")
        
        if len(segments) > max_segments:
            idx = np.random.choice(len(segments), max_segments, replace=False)
            segments = [segments[i] for i in sorted(idx)]
        
        raw_embs = []
        for start, end, phone_id in tqdm(segments, desc="Building segment embeddings"):
            emb = self._segment_to_raw_embedding(eta_frames[start:end])
            raw_embs.append(emb)
        
        raw_embs = np.stack(raw_embs)
        raw_embs = normalize(raw_embs)  # L2 归一化
        
        actual_dim = min(self.pca_dim, raw_embs.shape[1], len(raw_embs) - 1)
        self.pca = PCA(n_components=actual_dim, random_state=42)
        self.pca.fit(raw_embs)
        self.fitted = True
        
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"  Raw dim: {raw_embs.shape[1]} → PCA dim: {actual_dim}")
        print(f"  Explained variance: {explained:.2%}")
        
        return self
    
    def extract_segment_styles(self, eta_utt, phones_utt):
        """提取一个话语中每个段的风格向量"""
        segments = self._find_phone_segments(phones_utt)
        results = []
        
        for start, end, phone_id in segments:
            raw = self._segment_to_raw_embedding(eta_utt[start:end])
            raw_norm = normalize(raw.reshape(1, -1))
            emb_pca = self.pca.transform(raw_norm)[0]
            results.append({
                'start': start, 'end': end,
                'phone': phone_id,
                'style_emb': emb_pca,
                'n_frames': end - start
            })
        return results
    
    def extract_utterance_style(self, eta_utt, phones_utt):
        """提取话语级风格向量 = 所有段的加权均值"""
        seg_results = self.extract_segment_styles(eta_utt, phones_utt)
        if not seg_results:
            return np.zeros(self.pca.n_components_)
        
        embs = np.stack([s['style_emb'] for s in seg_results])
        weights = np.array([s['n_frames'] for s in seg_results], dtype=float)
        weights /= weights.sum()
        
        return (embs * weights[:, None]).sum(axis=0)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'pca_dim': self.pca_dim,
                'min_segment_frames': self.min_segment_frames,
                'fitted': self.fitted,
            }, f)
        print(f"  Saved: {path}")
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(pca_dim=data['pca_dim'], 
                  min_segment_frames=data['min_segment_frames'])
        obj.pca = data['pca']
        obj.fitted = data['fitted']
        return obj


def validate_segment_clustering(style_extractor, eta_all, phones_all, 
                                 speaker_ids_per_frame, output_dir):
    """验证段级 embedding 的聚类质量"""
    print("\n[Validation] Segment-level clustering...")
    
    segments = style_extractor._find_phone_segments(phones_all)
    
    seg_embs = []
    seg_spks = []
    seg_phones = []
    
    for start, end, phone_id in segments[:20000]:
        raw = style_extractor._segment_to_raw_embedding(eta_all[start:end])
        raw_norm = normalize(raw.reshape(1, -1))
        emb = style_extractor.pca.transform(raw_norm)[0]
        seg_embs.append(emb)
        seg_spks.append(speaker_ids_per_frame[start])
        seg_phones.append(phone_id)
    
    seg_embs = np.stack(seg_embs)
    seg_spks = np.array(seg_spks)
    seg_phones = np.array(seg_phones)
    
    print(f"  Segments: {len(seg_embs)}, Speakers: {len(np.unique(seg_spks))}")
    
    # KMeans 聚类测试
    print(f"\n{'K':>5} {'Silhouette':>12} {'ARI_spk':>10} {'ARI_phone':>11}")
    print("-" * 45)
    
    best_k, best_sil = 8, -1
    for K in [4, 8, 16, 32]:
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(seg_embs)
        sil = silhouette_score(seg_embs, labels)
        ari_spk = adjusted_rand_score(seg_spks, labels)
        ari_phone = adjusted_rand_score(seg_phones, labels)
        print(f"{K:5d} {sil:12.4f} {ari_spk:10.4f} {ari_phone:11.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, K
    
    # t-SNE 可视化
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_idx = np.random.choice(len(seg_embs), min(3000, len(seg_embs)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embs_2d = tsne.fit_transform(seg_embs[vis_idx])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(seg_embs[vis_idx])
    axes[0].scatter(embs_2d[:, 0], embs_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
    axes[0].set_title(f'Style Clusters (K={best_k})')
    
    spk_ids_num = seg_spks[vis_idx]
    unique_spks = np.unique(spk_ids_num)
    spk_map = {s: i for i, s in enumerate(unique_spks)}
    spk_colors = [spk_map[s] for s in spk_ids_num]
    axes[1].scatter(embs_2d[:, 0], embs_2d[:, 1], c=spk_colors, cmap='tab20', s=10, alpha=0.5)
    axes[1].set_title(f'By Speaker ({len(unique_spks)} speakers)')
    
    axes[2].scatter(embs_2d[:, 0], embs_2d[:, 1], c=seg_phones[vis_idx], cmap='tab20', s=10, alpha=0.5)
    axes[2].set_title('By Phone')
    
    plt.tight_layout()
    save_path = output_dir / 'segment_style_clustering.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Visualization saved: {save_path}")
    
    return {'best_k': best_k, 'best_silhouette': best_sil}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--pca-dim', type=int, default=64)
    parser.add_argument('--max-frames', type=int, default=200000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 读 config
    config_path = Path(args.config) if args.config else BASE_DIR / 'configs' / 'base.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_split = config['offline'].get('train_split', 'train-clean-100')
        split_name = train_split.replace('-', '_')
        cache_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
        ckpt_dir = Path(config['paths']['checkpoints_dir'])
    else:
        cache_dir = BASE_DIR / 'cache' / 'features' / 'wavlm'
        ckpt_dir = BASE_DIR / 'checkpoints'

    # 加载 Eta 投影
    eta_ckpt = torch.load(ckpt_dir / 'eta_projection.pt', map_location=device)
    A_star = eta_ckpt['A_star'].to(device)
    b_star = eta_ckpt['b_star'].to(device)
    
    embeddings_pca = np.load(ckpt_dir / 'speaker_embeddings_pca.npy')
    utt_indices = np.load(ckpt_dir / 'utt_indices.npy')
    utt_to_pca = {int(idx): i for i, idx in enumerate(utt_indices)}
    
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']
    
    # Phone predictor
    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

    # 计算 η 并收集 phone 和 speaker 标签
    print("\n[1/3] Computing η across utterances...")
    all_eta, all_phones, all_speakers = [], [], []
    total_frames = 0
    
    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features_ds = f['features']
        for utt_idx, utt in enumerate(tqdm(utterances, desc="Computing η")):
            if utt_idx not in utt_to_pca:
                continue
            pca_idx = utt_to_pca[utt_idx]
            s_np = features_ds[utt['h5_start_idx']:utt['h5_end_idx']]
            if len(s_np) == 0:
                continue
            
            s = torch.from_numpy(s_np).float().to(device)
            d = torch.from_numpy(embeddings_pca[pca_idx]).float().to(device)
            
            with torch.no_grad():
                eta = (s - (d @ A_star + b_star).unsqueeze(0)).cpu().numpy()
                phones = phone_predictor(s).cpu().numpy()
            
            all_eta.append(eta)
            all_phones.append(phones)
            all_speakers.append(np.full(len(eta), utt['speaker_id']))

            total_frames += len(eta)
    
    all_eta = np.concatenate(all_eta)
    all_phones = np.concatenate(all_phones)
    all_speakers = np.concatenate(all_speakers)
    print(f"  Total: {len(all_eta)} frames, {len(np.unique(all_speakers))} speakers")
    
    # 构建段级风格提取器
    print("\n[2/3] Fitting SegmentStyleExtractor...")
    extractor = SegmentStyleExtractor(pca_dim=args.pca_dim)
    extractor.fit(all_eta, all_phones)
    extractor.save(ckpt_dir / 'style_extractor.pkl')
    
    # 验证
    print("\n[3/3] Validating segment clustering...")
    output_dir = BASE_DIR / 'outputs' / 'style_analysis'
    results = validate_segment_clustering(
        extractor, all_eta, all_phones, all_speakers, output_dir
    )
    
    print(f"\n{'='*50}")
    print(f"RESULT: Best K={results['best_k']}, Silhouette={results['best_silhouette']:.4f}")
    if results['best_silhouette'] > 0.05:
        print("✓ Segment-level style has clustering structure")
    else:
        print("◐ Weak structure, but still usable for style-guided retrieval")


if __name__ == '__main__':
    main()