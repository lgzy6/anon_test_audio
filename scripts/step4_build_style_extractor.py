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
        """旧版 fit（已弃用）"""
        raise NotImplementedError("Use fit_incremental() for large datasets")

    def fit_incremental(self, eta_iterator, batch_size=1000, pca_dim=64):
        """使用 IncrementalPCA 增量拟合，100% 数据覆盖"""
        from sklearn.decomposition import IncrementalPCA

        print("[Step 4] Fitting with IncrementalPCA (100% coverage)...")

        self.pca = IncrementalPCA(n_components=pca_dim, batch_size=batch_size)
        batch = []
        total_segs = 0

        for eta_utt, phones_utt, _ in tqdm(eta_iterator, desc="Incremental PCA"):
            segments = self._find_phone_segments(phones_utt)

            for start, end, _ in segments:
                raw_emb = self._segment_to_raw_embedding(eta_utt[start:end])
                raw_emb_norm = normalize(raw_emb.reshape(1, -1))[0]
                batch.append(raw_emb_norm)
                total_segs += 1

                if len(batch) >= batch_size:
                    self.pca.partial_fit(np.array(batch))
                    batch = []

        if batch:
            self.pca.partial_fit(np.array(batch))

        self.fitted = True
        print(f"  Fitted on {total_segs} segments, PCA dim: {pca_dim}")
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


def validate_incremental(extractor, eta_iterator, output_dir, max_samples=10000):
    """增量验证：采样验证聚类质量"""
    print("\n[Validation] Sampling segments for validation...")

    seg_embs = []
    seg_spks = []

    for eta, phones, spk_id in tqdm(eta_iterator, desc="Sampling"):
        seg_results = extractor.extract_segment_styles(eta, phones)
        for seg in seg_results:
            seg_embs.append(seg['style_emb'])
            seg_spks.append(spk_id)
            if len(seg_embs) >= max_samples:
                break
        if len(seg_embs) >= max_samples:
            break

    seg_embs = np.array(seg_embs)
    seg_spks = np.array(seg_spks)
    print(f"  Sampled {len(seg_embs)} segments from {len(np.unique(seg_spks))} speakers")

    # 聚类评估
    best_k, best_sil = 0, -1
    for k in [50, 100, 200]:
        if k >= len(seg_embs):
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(seg_embs)
        sil = silhouette_score(seg_embs, labels, sample_size=min(5000, len(seg_embs)))
        print(f"  K={k}: Silhouette={sil:.4f}")
        if sil > best_sil:
            best_k, best_sil = k, sil

    # 可视化
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(seg_embs)//4))
    embs_2d = tsne.fit_transform(seg_embs[:5000])

    unique_spks = np.unique(seg_spks[:5000])
    spk_to_idx = {spk: i for i, spk in enumerate(unique_spks)}
    spk_colors = np.array([spk_to_idx[spk] for spk in seg_spks[:5000]])

    plt.figure(figsize=(10, 8))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=spk_colors, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(label='Speaker')
    plt.title(f'Segment Styles (K={best_k}, Sil={best_sil:.3f})')
    plt.savefig(output_dir / 'segment_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {'best_k': best_k, 'best_silhouette': best_sil}


def validate_segment_clustering_simple(extractor, raw_features, speaker_labels, phone_labels, output_dir):
    """旧版验证（已弃用）"""
    raise NotImplementedError()


def validate_segment_clustering(extractor, eta_all, phones_all,
                                 speaker_ids_per_frame, output_dir):
    """旧版验证（已弃用）"""
    raise NotImplementedError()
    
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
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='Sample ratio for utterances (0.0-1.0)')
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

    # 采样话语索引
    if args.sample_ratio < 1.0:
        import random
        random.seed(42)
        valid_utts = [i for i in range(len(utterances)) if i in utt_to_pca]
        sample_size = int(len(valid_utts) * args.sample_ratio)
        sampled_utts = set(random.sample(valid_utts, sample_size))
        print(f"Sampled {len(sampled_utts)}/{len(valid_utts)} utterances")
    else:
        sampled_utts = None

    # 生成器：逐话语计算 η
    def eta_generator():
        with h5py.File(cache_dir / 'features.h5', 'r') as f:
            features_ds = f['features']
            for utt_idx, utt in enumerate(utterances):
                if utt_idx not in utt_to_pca:
                    continue
                if sampled_utts is not None and utt_idx not in sampled_utts:
                    continue
                s_np = features_ds[utt['h5_start_idx']:utt['h5_end_idx']]
                if len(s_np) < 5:
                    continue

                s = torch.from_numpy(s_np).float().to(device)
                d = torch.from_numpy(embeddings_pca[utt_to_pca[utt_idx]]).float().to(device)

                with torch.no_grad():
                    eta = (s - (d @ A_star + b_star).unsqueeze(0)).cpu().numpy()
                    phones = phone_predictor(s).cpu().numpy()

                yield eta, phones, utt['speaker_id']

    # 使用 IncrementalPCA 拟合
    print("\n[1/2] Fitting SegmentStyleExtractor (IncrementalPCA)...")
    extractor = SegmentStyleExtractor(pca_dim=args.pca_dim)
    extractor.fit_incremental(eta_generator(), batch_size=1000, pca_dim=args.pca_dim)
    extractor.save(ckpt_dir / 'style_extractor.pkl')

    # 验证（采样）
    print("\n[2/2] Validation (sampled)...")
    output_dir = BASE_DIR / 'outputs' / 'style_analysis'
    results = validate_incremental(extractor, eta_generator(), output_dir, max_samples=10000)
    
    print(f"\n{'='*50}")
    print(f"RESULT: Best K={results['best_k']}, Silhouette={results['best_silhouette']:.4f}")
    if results['best_silhouette'] > 0.05:
        print("✓ Segment-level style has clustering structure")
    else:
        print("◐ Weak structure, but still usable for style-guided retrieval")


if __name__ == '__main__':
    main()