#!/usr/bin/env python3
"""
为每个 pool 的 m/f/mix 子目录构建 bank（基于 L6/L24，无 purifier）。
输出: checkpoints/pool_{i}/{gender}/pseudo_bank.pt
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import gc
import sys
import json
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

CKPT_DIR = Path("/root/autodl-tmp/anon_test/checkpoints")


def build_bank(data_dir, clusters=8, frames_per_cluster=20, sub_clusters=5,
               min_spk_diversity=3, chunk_size=200_000, silence_phones=(0,)):
    data_dir = Path(data_dir)
    out_path = data_dir / "pseudo_bank.pt"

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

    # 阶段1: L24 分桶
    l24_buckets = {ph: [] for ph in unique_phones}
    spk_buckets  = {ph: [] for ph in unique_phones}
    idx_buckets  = {ph: [] for ph in unique_phones}

    with h5py.File(data_dir / "layer_24.h5") as h5:
        ds = h5["features"]
        for start in tqdm(range(0, total_frames, chunk_size), desc="L24分桶", leave=False):
            end = min(start + chunk_size, total_frames)
            mask = keep_mask[start:end]
            if not np.any(mask):
                continue
            local_idx = np.where(mask)[0]
            global_idx = local_idx + start
            l24 = ds[start:end][mask]
            phones_c = all_phones[start:end][mask]
            spks_c = frame_to_spk[start:end][mask]
            for ph in np.unique(phones_c):
                m = phones_c == ph
                l24_buckets[ph].append(l24[m])
                spk_buckets[ph].append(spks_c[m])
                idx_buckets[ph].append(global_idx[m])

    # 阶段2: K-Means 选帧
    selected_indices = {}
    for ph in tqdm(unique_phones, desc="聚类", leave=False):
        if not l24_buckets[ph]:
            continue
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        idx_all = np.concatenate(idx_buckets[ph])
        del l24_buckets[ph], spk_buckets[ph], idx_buckets[ph]

        n, k = len(l24_all), min(clusters, len(l24_all))
        if n < k:
            selected_indices[ph] = idx_all
            continue

        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
        km.fit(l24_all)
        picked = []
        for c in range(k):
            cm = km.labels_ == c
            if not np.any(cm):
                continue
            if len(np.unique(spk_all[cm])) < min_spk_diversity:
                continue
            dists = np.linalg.norm(l24_all[cm] - km.cluster_centers_[c], axis=1)
            n_sel = min(frames_per_cluster, int(cm.sum()))
            ci = np.where(cm)[0][dists.argsort()[:n_sel]]
            picked.append(idx_all[ci])
        selected_indices[ph] = np.concatenate(picked) if picked else idx_all
        del l24_all, spk_all, idx_all

    gc.collect()

    # 阶段3: 读取选中帧的 L6/L24
    all_gidx = np.sort(np.concatenate(list(selected_indices.values())))
    g2row = {int(g): i for i, g in enumerate(all_gidx)}
    n_total = len(all_gidx)
    buf_l6  = np.empty((n_total, 1024), dtype=np.float32)
    buf_l24 = np.empty((n_total, 1024), dtype=np.float32)

    with h5py.File(data_dir / "layer_6.h5") as h6, \
         h5py.File(data_dir / "layer_24.h5") as h24:
        ds6, ds24 = h6["features"], h24["features"]
        bs = 0
        while bs < n_total:
            lo = int(all_gidx[bs])
            be = bs + 1
            while be < n_total and int(all_gidx[be]) - lo < chunk_size:
                be += 1
            hi = int(all_gidx[be - 1]) + 1
            c6, c24 = ds6[lo:hi], ds24[lo:hi]
            for j in range(bs, be):
                g = int(all_gidx[j])
                buf_l6[j]  = c6[g - lo]
                buf_l24[j] = c24[g - lo]
            bs = be

    bank = {}
    for ph, gidx in selected_indices.items():
        rows = np.array([g2row[int(g)] for g in gidx])
        bank[ph] = {
            "l6":  torch.from_numpy(buf_l6[rows].copy()).float(),
            "l24": torch.from_numpy(buf_l24[rows].copy()).float(),
        }

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    print(f"  Bank 完成: {out_path.relative_to(CKPT_DIR)} | 音素:{len(bank)}, 帧:{total_f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None, help="只处理指定 pool (0-3)")
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--frames-per-cluster", type=int, default=20)
    parser.add_argument("--min-spk-diversity", type=int, default=3)
    args = parser.parse_args()

    pool_ids = [args.pool] if args.pool is not None else range(4)

    for pid in pool_ids:
        print(f"\n=== Pool {pid} ===")
        for gender in ["m", "f", "mix"]:
            data_dir = CKPT_DIR / f"pool_{pid}" / gender
            if not (data_dir / "metadata.json").exists():
                print(f"  {gender}: 无数据，跳过")
                continue
            print(f"  [{gender}]")
            build_bank(data_dir, args.clusters, args.frames_per_cluster,
                       args.min_spk_diversity)


if __name__ == "__main__":
    main()
