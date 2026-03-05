#!/usr/bin/env python3
"""
步骤 4：在 η_style 上测试 VQ 可行性

判定标准：
  Silhouette > 0.1 → 有聚类结构
  ARI_speaker < 0.1 → 与说话人解耦
  Active codes >= 75% of K → 无 codebook collapse

测试方式：KMeans 多 K 值扫描，无需训练神经网络

依赖步骤 3 的输出（P_orth_on_eta.npy），或自动重新计算

使用方法：
    cd /root/autodl-tmp/anon_test
    python tests/test_eta_vq_feasibility.py
    python tests/test_eta_vq_feasibility.py --max-frames 30000
"""

import sys
import json
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_v32_disentanglement import ContentSubspaceProjectorV2
from models.phone_predictor.predictor import PhonePredictor

BASE_DIR = Path(__file__).parent.parent

# 从 config 读取路径
config_path = BASE_DIR / 'configs' / 'base.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_split = config['offline'].get('train_split', 'train-clean-100')
    split_name = train_split.replace('-', '_')
    CACHE_DIR = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
else:
    CACHE_DIR = BASE_DIR / 'cache' / 'features' / 'wavlm'

CKPT_DIR = BASE_DIR / 'checkpoints'


def load_and_compute_eta_style(max_frames=50000):
    """
    加载 features.h5 并计算 η_style

    如果 P_orth_on_eta.npy 存在则直接使用，否则重新学习
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载 Eta-WavLM 投影
    eta_ckpt = torch.load(CKPT_DIR / 'eta_projection.pt', map_location=device)
    A_star = eta_ckpt['A_star'].to(device)
    b_star = eta_ckpt['b_star'].to(device)

    embeddings_pca = np.load(CKPT_DIR / 'speaker_embeddings_pca.npy')
    utt_indices = np.load(CKPT_DIR / 'utt_indices.npy')

    with open(CACHE_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']

    utt_to_pca = {int(idx): i for i, idx in enumerate(utt_indices)}

    # Phone predictor
    phone_predictor = None
    phone_ckpt_path = CKPT_DIR / 'phone_decoder.pt'
    if phone_ckpt_path.exists():
        phone_predictor = PhonePredictor.load(str(phone_ckpt_path), device=device)

    # 计算 η
    all_eta = []
    all_s = []
    all_phones = []
    all_speaker_ids = []
    total_frames = 0

    # 按说话人分组，轮询采样
    spk_to_utts = {}
    for utt_idx in utt_to_pca.keys():
        spk = utterances[utt_idx]['speaker_id']
        if spk not in spk_to_utts:
            spk_to_utts[spk] = []
        spk_to_utts[spk].append(utt_idx)

    for spk in spk_to_utts:
        np.random.shuffle(spk_to_utts[spk])

    speakers = list(spk_to_utts.keys())
    utt_order = []
    max_len = max(len(utts) for utts in spk_to_utts.values())
    for i in range(max_len):
        for spk in speakers:
            if i < len(spk_to_utts[spk]):
                utt_order.append(spk_to_utts[spk][i])

    with h5py.File(CACHE_DIR / 'features.h5', 'r') as f:
        features_ds = f['features']

        for utt_meta_idx in tqdm(utt_order, desc="Computing η"):
            pca_idx = utt_to_pca[utt_meta_idx]
            utt = utterances[utt_meta_idx]
            start, end = utt['h5_start_idx'], utt['h5_end_idx']

            if end <= start:
                continue

            s_np = features_ds[start:end]
            T = s_np.shape[0]
            if T == 0:
                continue

            s = torch.from_numpy(s_np).float().to(device)
            d = torch.from_numpy(embeddings_pca[pca_idx]).float().to(device)
            eta = s - (d @ A_star + b_star).unsqueeze(0)

            if phone_predictor is not None:
                with torch.inference_mode():
                    phones = phone_predictor(s).cpu().numpy()
            else:
                phones = np.random.randint(0, 41, size=T)

            # 过滤静音帧
            non_silence_mask = phones != 0
            if non_silence_mask.sum() == 0:
                continue

            eta = eta[non_silence_mask]
            s_np = s_np[non_silence_mask]
            phones = phones[non_silence_mask]
            T_valid = eta.shape[0]

            all_eta.append(eta.cpu().numpy())
            all_s.append(s_np)
            all_phones.append(phones)
            all_speaker_ids.extend([utt['speaker_id']] * T_valid)

            total_frames += T_valid
            if total_frames >= max_frames:
                break

    all_eta = np.concatenate(all_eta, axis=0)[:max_frames]
    all_s = np.concatenate(all_s, axis=0)[:max_frames]
    all_phones = np.concatenate(all_phones)[:max_frames]
    all_speaker_ids = np.array(all_speaker_ids[:max_frames])

    print(f"η computed: {all_eta.shape}, speakers: {len(np.unique(all_speaker_ids))}")

    # 尝试加载已有 P_orth，否则重新学习
    p_orth_path = BASE_DIR / 'outputs' / 'eta_validation' / 'P_orth_on_eta.npy'
    if p_orth_path.exists():
        print(f"Loading P_orth from {p_orth_path}")
        P_orth = np.load(p_orth_path)
        eta_style = (all_eta @ P_orth).astype(np.float32)
    else:
        print("P_orth not found, learning content subspace on η...")
        projector = ContentSubspaceProjectorV2(
            n_phones=41, feature_dim=1024, silence_id=0, use_lda=False
        )
        projector.fit(all_eta, all_phones, max_per_class=2000)
        eta_style = projector.project_to_style(all_eta)

    return all_s, all_eta, eta_style, all_phones, all_speaker_ids


def entropy(counts):
    """计算分布熵"""
    p = counts / counts.sum()
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def test_vq_feasibility(eta_style, speaker_ids, phones, output_dir):
    """
    在 η_style 上测试 VQ 聚类可行性
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("VQ Feasibility Test on η_style")
    print("=" * 60)

    # 剔除 silence
    mask = phones != 0
    feats = eta_style[mask]
    spk = speaker_ids[mask]

    # 编码 speaker_id 为整数
    unique_spk = np.unique(spk)
    spk_to_int = {s: i for i, s in enumerate(unique_spk)}
    spk_int = np.array([spk_to_int[s] for s in spk])

    print(f"  Frames (non-silence): {len(feats)}")
    print(f"  Speakers: {len(unique_spk)}")

    # === 多 K 值 KMeans ===
    k_values = [4, 8, 16, 32]
    results = {}

    print(f"\n{'K':>4} {'Silhouette':>12} {'ARI_spk':>10} "
          f"{'Active':>8} {'Entropy':>10} {'Verdict':>12}")
    print("-" * 65)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feats)

        sil = silhouette_score(feats, labels) if len(np.unique(labels)) > 1 else 0
        ari_spk = adjusted_rand_score(spk_int, labels)

        counts = np.bincount(labels, minlength=k)
        active = (counts > len(labels) * 0.01).sum()
        ent = entropy(counts)

        results[k] = {
            'silhouette': float(sil),
            'ari_speaker': float(ari_spk),
            'active_codes': int(active),
            'total_codes': k,
            'entropy': float(ent),
            'max_entropy': float(np.log2(k)),
        }

        # 判定
        if sil > 0.1 and ari_spk < 0.1 and active >= k * 0.75:
            verdict = "FEASIBLE"
        elif sil > 0.05 and ari_spk < 0.2:
            verdict = "PARTIAL"
        else:
            verdict = "NOT VIABLE"

        results[k]['verdict'] = verdict
        print(f"{k:>4} {sil:>12.4f} {ari_spk:>10.4f} "
              f"{active:>4}/{k:<3} {ent:>10.3f} {verdict:>12}")

    # === 可视化 (K=8) ===
    best_k = 8
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(feats)

    n_sample = min(2000, len(feats))
    idx = np.random.choice(len(feats), n_sample, replace=False)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(feats[idx])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('η_style VQ Feasibility (K=8)', fontsize=14)

    axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=labels[idx], cmap='tab10', alpha=0.5, s=15)
    axes[0].set_title(f'KMeans Clusters (K={best_k})')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=spk_int[idx], cmap='tab20', alpha=0.5, s=15)
    axes[1].set_title(f'By Speaker ({len(unique_spk)} speakers)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # 码字使用直方图
    counts = np.bincount(labels, minlength=best_k)
    bars = axes[2].bar(range(best_k), counts, color='steelblue')
    axes[2].axhline(y=len(labels) / best_k, color='r', linestyle='--',
                    label='Uniform', alpha=0.7)
    axes[2].set_title(f'Code Usage (Active: {(counts > len(labels)*0.01).sum()}/{best_k})')
    axes[2].set_xlabel('Code Index')
    axes[2].set_ylabel('Frame Count')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'vq_feasibility.png', dpi=150)
    plt.close()

    # === 最终判定 ===
    r8 = results[8]
    print(f"\n{'=' * 60}")
    print(f"FINAL VERDICT (K=8):")
    print(f"  Silhouette:  {r8['silhouette']:.4f}  (> 0.1 = good structure)")
    print(f"  ARI_speaker: {r8['ari_speaker']:.4f}  (< 0.1 = speaker-free)")
    print(f"  Active:      {r8['active_codes']}/{r8['total_codes']}  (>= 75% = no collapse)")
    print(f"  Entropy:     {r8['entropy']:.3f} / {r8['max_entropy']:.3f}")

    if r8['silhouette'] > 0.1 and r8['ari_speaker'] < 0.1:
        recommendation = 'VQ'
        print("\n  -> VQ route FEASIBLE")
        print("  -> Recommend: K=8, bottleneck=256, commitment+diversity loss")
    elif r8['silhouette'] > 0.05:
        recommendation = 'BOTH'
        print("\n  -> Grey zone: try both VQ and continuous")
        print("  -> VQ: K=8, simplified loss")
        print("  -> Continuous: α-weighted cosine similarity")
    else:
        recommendation = 'CONTINUOUS'
        print("\n  -> VQ route NOT viable, use continuous α-weighted scheme")
        print("  -> Add style_sim channel in knn_retrieve_from_pool()")

    # 保存结果
    output_results = {
        'recommendation': recommendation,
        'k_results': {str(k): v for k, v in results.items()},
    }
    with open(output_dir / 'vq_feasibility_results.json', 'w') as f:
        json.dump(output_results, f, indent=2)

    print(f"\nResults saved: {output_dir}")
    return results, recommendation


def main():
    parser = argparse.ArgumentParser(description="VQ Feasibility Test on η_style")
    parser.add_argument('--max-frames', type=int, default=50000,
                        help='Max frames to process')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = BASE_DIR / 'outputs' / 'eta_validation'

    # 加载并计算 η_style
    s, eta, eta_style, phones, speaker_ids = load_and_compute_eta_style(args.max_frames)

    # 运行 VQ 可行性测试
    results, recommendation = test_vq_feasibility(
        eta_style, speaker_ids, phones, output_dir
    )

    return results, recommendation


if __name__ == '__main__':
    main()
