#!/usr/bin/env python3
"""
Bank 构建 v2:
  L24 K-Means(8簇)，混淆熵用 L24 计算，动态 top-p% 过滤
  每个通过过滤的簇取最近 frames_per_cluster 帧做二次 K-Means(sub_clusters)
  保存二次聚类质心：L24 质心 + 对应子簇 L6 均值
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import gc
import json
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm

CKPT_DIR  = Path("/root/autodl-tmp/anon_test/checkpoints")
BANKS_DIR = CKPT_DIR / "banks_v2"


def _cluster_entropy(l24_frames: np.ndarray, spk_ids: np.ndarray, temperature: float = 5.0) -> float:
    unique_spks = np.unique(spk_ids)
    if len(unique_spks) < 2:
        return 0.0
    spk_embs = np.stack([l24_frames[spk_ids == s].mean(0) for s in unique_spks])
    spk_embs = spk_embs / (np.linalg.norm(spk_embs, axis=1, keepdims=True) + 1e-8)
    frames_norm = l24_frames / (np.linalg.norm(l24_frames, axis=1, keepdims=True) + 1e-8)
    sims = frames_norm @ spk_embs.T * temperature
    sims -= sims.max(axis=1, keepdims=True)
    exp_s = np.exp(sims)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    return float(ent.mean() / np.log(len(unique_spks)))


def build_bank(data_dir, pool_id, gender_tag, clusters=8, frames_per_cluster=20, sub_clusters=4,
               min_spk_diversity=3, entropy_top_p=0.5, chunk_size=200_000, silence_phones=(0,)):
    data_dir = Path(data_dir)
    BANKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BANKS_DIR / f"pool_{pool_id}_gender-{gender_tag}.pt"

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    total_frames = meta["total_frames"]
    utts = meta["utterances"]

    spk2id = {u["speaker_id"]: i for i, u in enumerate(
        {u["speaker_id"]: u for u in utts}.values())}
    frame_to_spk = np.full(total_frames, -1, dtype=np.int32)
    keep_mask = np.zeros(total_frames, dtype=bool)
    for u in utts:
        s, e = u["h5_start_idx"], u["h5_end_idx"]
        frame_to_spk[s:e] = spk2id[u["speaker_id"]]
        keep_mask[s:e] = True

    with h5py.File(data_dir / "phones.h5") as f:
        all_phones = f["phones"][:]
    for sil in silence_phones:
        keep_mask &= (all_phones != sil)

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"  有效帧: {keep_mask.sum()}, 音素数: {len(unique_phones)}")

    l6_buckets  = {ph: [] for ph in unique_phones}
    l24_buckets = {ph: [] for ph in unique_phones}
    spk_buckets = {ph: [] for ph in unique_phones}

    with h5py.File(data_dir / "layer_6.h5") as h6, \
         h5py.File(data_dir / "layer_24.h5") as h24:
        ds6, ds24 = h6["features"], h24["features"]
        for start in tqdm(range(0, total_frames, chunk_size), desc="分桶", leave=False):
            end = min(start + chunk_size, total_frames)
            mask = keep_mask[start:end]
            if not np.any(mask):
                continue
            l6  = ds6[start:end][mask]
            l24 = ds24[start:end][mask]
            phones_c = all_phones[start:end][mask]
            spks_c = frame_to_spk[start:end][mask]
            for ph in np.unique(phones_c):
                m = phones_c == ph
                l6_buckets[ph].append(l6[m])
                l24_buckets[ph].append(l24[m])
                spk_buckets[ph].append(spks_c[m])

    bank = {}
    for ph in tqdm(unique_phones, desc="聚类", leave=False):
        if not l24_buckets[ph]:
            continue
        l6_all  = np.concatenate(l6_buckets[ph])
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]

        n, k = len(l24_all), min(clusters, len(l24_all))
        if n < k:
            bank[ph] = {
                "l6":  torch.from_numpy(l6_all.mean(0, keepdims=True)).float(),
                "l24": torch.from_numpy(l24_all.mean(0, keepdims=True)).float(),
            }
            continue

        km1 = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
        km1.fit(l24_all)

        valid_clusters, entropies = [], []
        for c in range(k):
            cm = km1.labels_ == c
            if not np.any(cm):
                continue
            if len(np.unique(spk_all[cm])) < min_spk_diversity:
                continue
            entropies.append(_cluster_entropy(l24_all[cm], spk_all[cm]))
            valid_clusters.append(c)

        centroids_l6, centroids_l24 = [], []
        if valid_clusters:
            threshold = np.percentile(entropies, (1 - entropy_top_p) * 100)
            for c, ent in zip(valid_clusters, entropies):
                if ent < threshold:
                    continue
                cm = km1.labels_ == c
                dists = np.linalg.norm(l24_all[cm] - km1.cluster_centers_[c], axis=1)
                n_sel = min(frames_per_cluster, int(cm.sum()))
                top_local = dists.argsort()[:n_sel]
                cluster_global = np.where(cm)[0][top_local]
                cand_l6  = l6_all[cluster_global]
                cand_l24 = l24_all[cluster_global]

                k2 = min(sub_clusters, len(cand_l24))
                if len(cand_l24) <= k2:
                    centroids_l24.append(cand_l24)
                    centroids_l6.append(cand_l6)
                else:
                    km2 = KMeans(n_clusters=k2, random_state=42, n_init='auto')
                    km2.fit(cand_l24)
                    centroids_l24.append(km2.cluster_centers_.astype(np.float32))
                    centroids_l6.append(np.stack([
                        cand_l6[km2.labels_ == sc].mean(0)
                        for sc in range(k2) if np.any(km2.labels_ == sc)
                    ]))

        if centroids_l24:
            bank[ph] = {
                "l6":  torch.from_numpy(np.concatenate(centroids_l6)).float(),
                "l24": torch.from_numpy(np.concatenate(centroids_l24)).float(),
            }
        del l6_all, l24_all, spk_all

    del l6_buckets, l24_buckets, spk_buckets
    gc.collect()

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    print(f"  Bank 完成: {out_path.relative_to(CKPT_DIR)} | 音素:{len(bank)}, 质心总数:{total_f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--frames-per-cluster", type=int, default=20)
    parser.add_argument("--sub-clusters", type=int, default=4)
    parser.add_argument("--min-spk-diversity", type=int, default=3)
    parser.add_argument("--entropy-top-p", type=float, default=0.5)
    args = parser.parse_args()

    pool_ids = [args.pool] if args.pool is not None else range(4)
    for pid in pool_ids:
        print(f"\n=== Pool {pid} ===")
        for gender in ["m", "f"]:
            data_dir = CKPT_DIR / f"pool_{pid}" / gender
            if not (data_dir / "metadata.json").exists():
                print(f"  {gender}: 无数据，跳过")
                continue
            print(f"  [{gender}]")
            build_bank(data_dir, pid, gender, args.clusters, args.frames_per_cluster,
                       args.sub_clusters, args.min_spk_diversity, args.entropy_top_p)


if __name__ == "__main__":
    main()
