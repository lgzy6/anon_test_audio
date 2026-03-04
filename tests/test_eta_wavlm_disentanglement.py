#!/usr/bin/env python3
"""
步骤 3：验证 Eta-WavLM 解耦效果 (四路对比)

四路对比：
  1. 原始 s         → Phone probe + Speaker probe
  2. η (去speaker)   → Phone probe + Speaker probe
  3. h_style (P_orth去content) → Phone probe + Speaker probe
  4. η_style (去speaker+content)   → Phone probe + Speaker probe

复用已有组件：
  - GPULinearProbe (from test_v32_disentanglement)
  - ContentSubspaceProjectorV2 (from test_v32_disentanglement)
  - load_test_data (from test_v32_disentanglement)

使用方法：
    cd /root/autodl-tmp/anon_test
    python tests/test_eta_wavlm_disentanglement.py
    python tests/test_eta_wavlm_disentanglement.py --max-frames 30000
"""

import sys
import json
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import h5py
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_v32_disentanglement import (
    ContentSubspaceProjectorV2,
    GPULinearProbe,
    analyze_phone_distribution,
)
from models.phone_predictor.predictor import PhonePredictor
import yaml


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
CKPT_DIR = BASE_DIR / 'checkpoints'


def load_eta_components():
    """加载步骤 1 和 2 的产物"""
    # Eta-WavLM 投影矩阵
    eta_path = CKPT_DIR / 'eta_projection.pt'
    if not eta_path.exists():
        raise FileNotFoundError(
            f"{eta_path} not found. Run step2_compute_eta_projection.py first."
        )

    eta_ckpt = torch.load(eta_path, map_location='cpu')
    A_star = eta_ckpt['A_star']    # (P, Q)
    b_star = eta_ckpt['b_star']    # (Q,)

    # Speaker embeddings
    embeddings_pca = np.load(CKPT_DIR / 'speaker_embeddings_pca.npy')
    speaker_ids = np.load(CKPT_DIR / 'speaker_ids.npy')
    utt_indices = np.load(CKPT_DIR / 'utt_indices.npy')

    print(f"Eta-WavLM loaded:")
    print(f"  A_star: {A_star.shape}, b_star: {b_star.shape}")
    print(f"  Speaker embeddings: {embeddings_pca.shape}")
    print(f"  Condition number: {eta_ckpt['condition_number']:.2e}")
    print(f"  Total frames used: {eta_ckpt['n_frames']:,}")

    return A_star, b_star, embeddings_pca, speaker_ids, utt_indices


def compute_eta_for_frames(A_star, b_star, embeddings_pca,
                           utt_indices, max_frames=50000):
    """
    对帧级特征计算 η = s - A^T d - b

    返回: s_all, eta_all, speaker_ids_per_frame, phones_per_frame
    """
    with open(CACHE_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']

    # 话语索引 → PCA 索引
    utt_to_pca = {int(idx): i for i, idx in enumerate(utt_indices)}

    # 按说话人分组话语，确保采样覆盖所有说话人
    spk_to_utts = {}
    for utt_idx in utt_to_pca.keys():
        spk = utterances[utt_idx]['speaker_id']
        if spk not in spk_to_utts:
            spk_to_utts[spk] = []
        spk_to_utts[spk].append(utt_idx)

    # 打乱每个说话人的话语顺序
    for spk in spk_to_utts:
        np.random.shuffle(spk_to_utts[spk])

    # 轮流从每个说话人采样
    speakers = list(spk_to_utts.keys())
    utt_order = []
    max_len = max(len(utts) for utts in spk_to_utts.values())
    for i in range(max_len):
        for spk in speakers:
            if i < len(spk_to_utts[spk]):
                utt_order.append(spk_to_utts[spk][i])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A_star_d = A_star.to(device)
    b_star_d = b_star.to(device)

    # 加载 phone predictor
    phone_ckpt = str(CKPT_DIR / 'phone_decoder.pt')
    phone_predictor = None
    if Path(phone_ckpt).exists():
        phone_predictor = PhonePredictor.load(phone_ckpt, device=device)

    all_s = []
    all_eta = []
    all_speaker_ids = []
    all_phones = []
    total_frames = 0

    with h5py.File(CACHE_DIR / 'features.h5', 'r') as f:
        features_ds = f['features']

        for utt_meta_idx in tqdm(utt_order, desc="Computing η"):
            pca_idx = utt_to_pca[utt_meta_idx]
            utt = utterances[utt_meta_idx]
            start, end = utt['h5_start_idx'], utt['h5_end_idx']

            if end <= start:
                continue

            s_np = features_ds[start:end]  # (T, 1024)
            T = s_np.shape[0]

            if T == 0:
                continue

            s = torch.from_numpy(s_np).float().to(device)

            # speaker component = d @ A + b
            d_pca = torch.from_numpy(
                embeddings_pca[pca_idx]
            ).float().to(device)

            speaker_component = d_pca @ A_star_d + b_star_d  # (1024,)
            eta = s - speaker_component.unsqueeze(0)          # (T, 1024)

            # phone prediction
            if phone_predictor is not None:
                with torch.inference_mode():
                    phones_t = phone_predictor(s)  # (T,)
                all_phones.append(phones_t.cpu().numpy())
            else:
                all_phones.append(np.random.randint(0, 41, size=T))

            all_s.append(s.cpu().numpy())
            all_eta.append(eta.cpu().numpy())
            all_speaker_ids.extend([utt['speaker_id']] * T)

            total_frames += T
            if total_frames >= max_frames:
                break

    all_s = np.concatenate(all_s, axis=0)[:max_frames]
    all_eta = np.concatenate(all_eta, axis=0)[:max_frames]
    all_phones = np.concatenate(all_phones)[:max_frames]
    all_speaker_ids = np.array(all_speaker_ids[:max_frames])

    print(f"\nComputed η: {all_eta.shape}")
    print(f"  Speakers: {len(np.unique(all_speaker_ids))}")
    print(f"  Phone range: [{all_phones.min()}, {all_phones.max()}]")

    return all_s, all_eta, all_speaker_ids, all_phones


def run_probe(features, labels, n_classes, epochs=30, batch_size=2048):
    """运行单个 Linear Probe，返回 test accuracy"""
    n = len(features)
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)
    tr, te = idx[:n_train], idx[n_train:]

    probe = GPULinearProbe(features.shape[1], n_classes)
    probe.fit(features[tr], labels[tr], epochs=epochs, batch_size=batch_size)
    return probe.score(features[te], labels[te])


def run_four_way_probe(s, eta, phones, speaker_ids,
                       projector_on_s, projector_on_eta):
    """四路探针实验"""
    # 剔除 silence
    mask = phones != 0
    s_ns = s[mask]
    eta_ns = eta[mask]
    phones_ns = phones[mask]
    spk_ns = speaker_ids[mask]

    # 重新映射 phone ID 到连续整数 [0, n_phones-1]
    unique_phones = np.unique(phones_ns)
    n_phones = len(unique_phones)
    phone_to_int = {p: i for i, p in enumerate(unique_phones)}
    phones_ns = np.array([phone_to_int[p] for p in phones_ns])

    unique_spk = np.unique(spk_ns)
    n_speakers = len(unique_spk)

    # 编码 speaker_id 为整数
    spk_to_int = {s: i for i, s in enumerate(unique_spk)}
    spk_int = np.array([spk_to_int[s] for s in spk_ns])

    # 四种特征空间
    h_style_s = projector_on_s.project_to_style(s_ns)
    eta_style = projector_on_eta.project_to_style(eta_ns)

    feature_sets = {
        '1_original_s': s_ns,
        '2_eta_no_speaker': eta_ns,
        '3_h_style_no_content': h_style_s,
        '4_eta_style_full': eta_style,
    }

    results = {}

    print("\n" + "=" * 70)
    print(f"{'Feature Space':<30} {'Phone Acc':>10} {'Speaker Acc':>12}")
    print("-" * 70)

    for name, feats in feature_sets.items():
        phone_acc = run_probe(feats, phones_ns, n_phones)
        spk_acc = run_probe(feats, spk_int, n_speakers)

        results[name] = {'phone_acc': float(phone_acc), 'speaker_acc': float(spk_acc)}
        print(f"{name:<30} {phone_acc:>10.2%} {spk_acc:>12.2%}")

    print("-" * 70)
    print(f"{'Random baseline':<30} {1/n_phones:>10.2%} {1/n_speakers:>12.2%}")
    print("=" * 70)

    return results, feature_sets, phones_ns, spk_int


def visualize_four_way(feature_sets, phones, speaker_ids, output_dir):
    """四路 t-SNE 可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 采样 2000 帧
    n = len(phones)
    sample_idx = np.random.choice(n, min(2000, n), replace=False)

    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.suptitle('Four-Way Disentanglement Comparison', fontsize=16, y=0.98)

    for row, (name, feats) in enumerate(feature_sets.items()):
        feats_s = feats[sample_idx]
        phones_s = phones[sample_idx]
        spk_s = speaker_ids[sample_idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb_2d = tsne.fit_transform(feats_s)

        # By Speaker
        axes[row, 0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                             c=spk_s, cmap='tab20', alpha=0.5, s=10)
        axes[row, 0].set_title(f'{name}\nBy Speaker')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # By Phone
        axes[row, 1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                             c=phones_s, cmap='tab20', alpha=0.5, s=10)
        axes[row, 1].set_title(f'{name}\nBy Phone')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # KMeans k=8
        km = KMeans(n_clusters=8, random_state=42, n_init=10)
        cl = km.fit_predict(feats_s)
        sil = silhouette_score(feats_s, cl) if len(np.unique(cl)) > 1 else 0
        ari = adjusted_rand_score(spk_s, cl)

        axes[row, 2].scatter(emb_2d[:, 0], emb_2d[:, 1],
                             c=cl, cmap='tab10', alpha=0.5, s=10)
        axes[row, 2].set_title(
            f'{name}\nKMeans k=8 (Sil={sil:.3f}, ARI_spk={ari:.3f})')
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = output_dir / 'four_way_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Eta-WavLM Four-Way Disentanglement Test")
    parser.add_argument('--max-frames', type=int, default=50000,
                        help='Max frames to process')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE visualization')
    args = parser.parse_args()

    print("=" * 60)
    print("Eta-WavLM Disentanglement Validation (Four-Way)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = BASE_DIR / 'outputs' / 'eta_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 加载 Eta-WavLM 组件 ===
    A_star, b_star, embeddings_pca, spk_ids_utt, utt_indices = \
        load_eta_components()

    # === 计算帧级 η ===
    s, eta, speaker_ids_frame, phones = compute_eta_for_frames(
        A_star, b_star, embeddings_pca, utt_indices,
        max_frames=args.max_frames
    )

    # === 分析分布 ===
    analyze_phone_distribution(phones)

    # === 学习 content 子空间 ===
    print("\n" + "=" * 50)
    print("Learning content subspace on original s...")
    projector_on_s = ContentSubspaceProjectorV2(
        n_phones=41, feature_dim=1024, silence_id=0, use_lda=False
    )
    projector_on_s.fit(s, phones, max_per_class=2000)

    print("\nLearning content subspace on η (speaker-removed)...")
    projector_on_eta = ContentSubspaceProjectorV2(
        n_phones=41, feature_dim=1024, silence_id=0, use_lda=False
    )
    projector_on_eta.fit(eta, phones, max_per_class=2000)

    # === 四路探针 ===
    results, feature_sets, phones_ns, spk_int = run_four_way_probe(
        s, eta, phones, speaker_ids_frame,
        projector_on_s, projector_on_eta
    )

    # === Energy ratio 诊断 ===
    print("\n" + "=" * 50)
    print("Energy Ratio Diagnostic")
    print("=" * 50)
    s_energy = np.linalg.norm(s, axis=1).mean()
    eta_energy = np.linalg.norm(eta, axis=1).mean()
    speaker_energy = np.linalg.norm(s - eta, axis=1).mean()
    print(f"  ||s||  mean: {s_energy:.4f}")
    print(f"  ||η||  mean: {eta_energy:.4f}")
    print(f"  ||spk|| mean: {speaker_energy:.4f}")
    print(f"  η/s ratio: {eta_energy/s_energy:.4f}  (expect 0.85-0.99)")
    print(f"  spk/s ratio: {speaker_energy/s_energy:.4f}  (expect 0.01-0.15)")
    if eta_energy / s_energy < 0.7:
        print("  WARNING: Speaker component too large, check A_star/b_star")
    elif eta_energy / s_energy > 0.99:
        print("  WARNING: Speaker component negligible, check embedding quality")

    # === 可视化 ===
    if not args.skip_tsne:
        print("\nGenerating t-SNE visualizations...")
        visualize_four_way(feature_sets, phones_ns, spk_int, output_dir)

    # === 判定 ===
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    eta_spk = results['2_eta_no_speaker']['speaker_acc']
    orig_spk = results['1_original_s']['speaker_acc']
    eta_phone = results['2_eta_no_speaker']['phone_acc']
    orig_phone = results['1_original_s']['phone_acc']
    eta_style_spk = results['4_eta_style_full']['speaker_acc']
    eta_style_phone = results['4_eta_style_full']['phone_acc']

    print(f"\n[1] Speaker removal (η vs original):")
    print(f"  Speaker Acc: {orig_spk:.2%} -> {eta_spk:.2%} "
          f"(reduced {(orig_spk - eta_spk) / orig_spk:.1%})")
    print(f"  Phone Acc:   {orig_phone:.2%} -> {eta_phone:.2%} "
          f"(preserved: {eta_phone / orig_phone:.1%})")

    if eta_spk < orig_spk * 0.5:
        print("  -> Eta-WavLM speaker removal: EFFECTIVE")
    elif eta_spk < orig_spk * 0.7:
        print("  -> Eta-WavLM speaker removal: PARTIAL")
    else:
        print("  -> Eta-WavLM speaker removal: INSUFFICIENT")

    print(f"\n[2] Full disentanglement (η_style):")
    print(f"  Speaker Acc: {eta_style_spk:.2%}")
    print(f"  Phone Acc:   {eta_style_phone:.2%}")

    if eta_style_spk < 0.2 and eta_style_phone < 0.4:
        print("  -> Full disentanglement: SUCCESS")
        print("  -> Ready for VQ or continuous retrieval on η_style")
    elif eta_style_spk < 0.3:
        print("  -> Full disentanglement: PARTIAL")
        print("  -> Consider continuous α-weighted scheme")
    else:
        print("  -> Full disentanglement: NEEDS IMPROVEMENT")
        print("  -> Check data coverage or model capacity")

    # 保存结果
    with open(output_dir / 'probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 保存 P_orth 矩阵供后续使用
    if projector_on_eta.P_orth is not None:
        np.save(output_dir / 'P_orth_on_eta.npy', projector_on_eta.P_orth)
        print(f"\nSaved P_orth: {output_dir / 'P_orth_on_eta.npy'}")

    print(f"\nAll results: {output_dir}")
    return results


if __name__ == '__main__':
    main()
