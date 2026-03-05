#!/usr/bin/env python3
"""
步骤 2：在已有的 features.h5 上计算 Eta-WavLM 的 A_star 和 b_star

数学原理（与 abylouw/Eta-WavLM 完全一致）：
  s = A^T d + b + η
  η = s - A^T d - b

  正规方程：(D̃^T D̃) Ã = D̃^T S
  其中 D̃ = [d_pca, 1]

输入：
  - cache/features/wavlm/features.h5       (帧级 WavLM 特征)
  - cache/features/wavlm/metadata.json     (话语边界)
  - checkpoints/speaker_embeddings_pca.npy  (步骤1 的输出)
  - checkpoints/utt_indices.npy

输出：
  - checkpoints/eta_projection.pt  (A_star, b_star, 诊断信息)

使用方法：
    cd /root/autodl-tmp/anon_test
    python scripts/step2_compute_eta_projection.py
"""

import sys
import json
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute Eta-WavLM Projection")
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML path')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Frame chunk size for long utterances (default: 2000)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = Path(__file__).parent.parent

    # 从 config 或默认路径读取
    config_path = Path(args.config) if args.config else base_dir / 'configs' / 'base.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_split = config['offline'].get('train_split', 'train-clean-100')
        split_name = train_split.replace('-', '_')
        cache_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
        ckpt_dir = Path(config['paths']['checkpoints_dir'])
    else:
        cache_dir = base_dir / 'cache' / 'features' / 'wavlm'
        ckpt_dir = base_dir / 'checkpoints'

    # === 检查步骤 1 的输出 ===
    required_files = [
        ckpt_dir / 'speaker_embeddings_pca.npy',
        ckpt_dir / 'utt_indices.npy',
    ]
    for f in required_files:
        if not f.exists():
            print(f"ERROR: {f} not found. Run step1 first.")
            sys.exit(1)

    # === 加载步骤 1 的输出 ===
    embeddings_pca = np.load(ckpt_dir / 'speaker_embeddings_pca.npy')   # (N_utt, P)
    utt_indices = np.load(ckpt_dir / 'utt_indices.npy')

    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']

    P = embeddings_pca.shape[1]  # PCA dim
    Q = metadata['feature_dim']  # WavLM dim (1024)

    # 不做归一化，保持原始尺度
    embeddings_mean = np.zeros(P)
    embeddings_std = np.ones(P)

    # 自适应正则化：P 越大，正则化越强
    regularization = max(2e-4, 2e-3 * (P / 128))

    print(f"Speaker PCA dim (P): {P}")
    print(f"SSL feature dim (Q): {Q}")
    print(f"Utterances with embeddings: {len(utt_indices)}")
    print(f"Total utterances: {len(utterances)}")

    # === 初始化累加器（double 精度，与 Eta-WavLM 一致）===
    G = torch.zeros(P + 1, P + 1, dtype=torch.float64, device=device)
    H = torch.zeros(P + 1, Q, dtype=torch.float64, device=device)
    n_frames_total = 0
    n_utterances_processed = 0

    # === 构建话语索引到 PCA embedding 的映射 ===
    utt_to_pca_idx = {int(utt_idx): pca_idx
                      for pca_idx, utt_idx in enumerate(utt_indices)}

    # === 加载 phone predictor（用于过滤静音）===
    phone_ckpt = ckpt_dir / 'phone_decoder.pt'
    phone_predictor = None
    if phone_ckpt.exists():
        from models.phone_predictor.predictor import PhonePredictor
        phone_predictor = PhonePredictor.load(str(phone_ckpt), device=device)
        print("  Loaded phone predictor for silence filtering")

    # === 逐话语累积正规方程 ===
    print("\nAccumulating normal equations (filtering silence)...")

    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features_ds = f['features']

        for utt_meta_idx, utt in enumerate(tqdm(utterances, desc="Accumulating")):
            if utt_meta_idx not in utt_to_pca_idx:
                continue

            pca_idx = utt_to_pca_idx[utt_meta_idx]

            start_idx = utt['h5_start_idx']
            end_idx = utt['h5_end_idx']

            if end_idx <= start_idx:
                continue

            # 读取 WavLM 特征
            s_np = features_ds[start_idx:end_idx]  # (T, Q)
            T = s_np.shape[0]

            if T == 0:
                continue

            s = torch.from_numpy(s_np).to(device).to(torch.float64)  # (T, Q)

            # 过滤静音帧
            if phone_predictor is not None:
                with torch.no_grad():
                    phones = phone_predictor(s.float())  # (T,)
                    non_silence_mask = phones != 0
                    s = s[non_silence_mask]
                    T = s.shape[0]
                    if T == 0:
                        continue

            # 该话语的 PCA speaker embedding
            d = torch.from_numpy(
                embeddings_pca[pca_idx]
            ).to(device).to(torch.float64)  # (P,)

            # 分块累积（防止长话语 GPU OOM）
            CHUNK = args.chunk_size
            for chunk_start in range(0, T, CHUNK):
                chunk_end = min(chunk_start + CHUNK, T)
                s_chunk = s[chunk_start:chunk_end]
                T_chunk = s_chunk.shape[0]

                d_rep = d.unsqueeze(0).expand(T_chunk, -1)
                ones = torch.ones(T_chunk, 1, dtype=torch.float64, device=device)
                D_tilde = torch.cat([d_rep, ones], dim=1)

                G += D_tilde.T @ D_tilde
                H += D_tilde.T @ s_chunk

            n_frames_total += T
            n_utterances_processed += 1

            # 定期清理 GPU 缓存
            if n_utterances_processed % 200 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nAccumulation done:")
    print(f"  Utterances processed: {n_utterances_processed}")
    print(f"  Total frames: {n_frames_total:,}")

    if n_utterances_processed == 0:
        print("ERROR: No utterances processed!")
        sys.exit(1)

    # === Cholesky 求解（与 abylouw/Eta-WavLM 完全一致）===
    print("\nSolving normal equations...")

    G_reg = G + regularization * torch.eye(
        P + 1, dtype=torch.float64, device=device
    )

    try:
        L = torch.linalg.cholesky(G_reg)
        y = torch.linalg.solve_triangular(L, H, upper=False)
        A_tilde = torch.linalg.solve_triangular(L.T, y, upper=True)
        solver = 'cholesky'
        print("  Cholesky solver succeeded")
    except Exception as e:
        print(f"  Cholesky failed ({e}), falling back to lstsq")
        A_tilde = torch.linalg.lstsq(G_reg, H).solution
        solver = 'lstsq'

    # 提取 A_star 和 b_star
    A_star = A_tilde[:-1, :].to(torch.float32)   # (P, Q)
    b_star = A_tilde[-1, :].to(torch.float32)     # (Q,)

    # 诊断
    cond = torch.linalg.cond(G_reg).item()
    residual = torch.norm(G_reg @ A_tilde - H).item()

    print(f"\nResults:")
    print(f"  A_star: {A_star.shape}")
    print(f"  b_star: {b_star.shape}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Residual norm: {residual:.2e}")
    print(f"  Solver: {solver}")

    # === 快速验证：计算几条话语的 η 并检查统计量 ===
    print("\nQuick validation on sample utterances...")
    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features_ds = f['features']
        sample_etas = []

        for i, utt_meta_idx in enumerate(list(utt_to_pca_idx.keys())[:5]):
            pca_idx = utt_to_pca_idx[utt_meta_idx]
            utt = utterances[utt_meta_idx]
            s_np = features_ds[utt['h5_start_idx']:utt['h5_end_idx']]
            s = torch.from_numpy(s_np).float().to(device)

            d = torch.from_numpy(embeddings_pca[pca_idx]).float().to(device)
            speaker_component = d @ A_star.to(device) + b_star.to(device)
            eta = s - speaker_component.unsqueeze(0)

            recon_ratio = speaker_component.norm().item() / s[0].norm().item()
            eta_ratio = eta.norm(dim=-1).mean().item() / s.norm(dim=-1).mean().item()

            print(f"  Utt {utt['utt_id']}: "
                  f"speaker_comp/s = {recon_ratio:.4f}, "
                  f"η/s = {eta_ratio:.4f}")
            sample_etas.append(eta.cpu().numpy())

    # === 保存 ===
    save_path = ckpt_dir / 'eta_projection.pt'
    torch.save({
        'A_star': A_star.cpu(),
        'b_star': b_star.cpu(),
        'pca_components': P,
        'ssl_dim': Q,
        'n_frames': n_frames_total,
        'n_utterances': n_utterances_processed,
        'condition_number': cond,
        'residual_norm': residual,
        'regularization': regularization,
        'solver': solver,
        'embeddings_mean': torch.from_numpy(embeddings_mean).float(),
        'embeddings_std': torch.from_numpy(embeddings_std).float(),
    }, save_path)

    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
