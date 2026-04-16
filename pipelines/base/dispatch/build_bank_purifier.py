#!/usr/bin/env python3
"""
构建 Purifier Bank:
  - 用 purifier encoder 净化 L24 特征
  - 按音素分桶 + 净化后 L24 K-Means 聚类 (16簇)
  - Bank 存储: purified_l24 (检索用) + l6 (合成用)
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"

import gc
import sys
import argparse
import json
import h5py
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.train.purifier import FeaturePurifier

CKPT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
DATA_DIR = CKPT_DIR / "trainother500_200spk"


def load_purifier(device):
    cfg_path = CKPT_DIR / "purifier" / "train_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    ckpt = torch.load(CKPT_DIR / "purifier" / "best_purifier.pt", map_location=device)
    model = FeaturePurifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_phones=cfg["num_phones"],
        num_speakers=ckpt["num_speakers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["full_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def purify_chunk(encoder, l24_np, device, batch=8192):
    """分批净化，返回 numpy float32"""
    out = []
    t = torch.from_numpy(l24_np.astype(np.float32))
    for i in range(0, len(t), batch):
        out.append(encoder(t[i:i+batch].to(device)).cpu().numpy())
    return np.concatenate(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default=str(DATA_DIR))
    p.add_argument("--output",    default=None)
    p.add_argument("--gender",    default="all", help="m/f/mix/all")
    p.add_argument("--silence-phones", type=int, nargs="+", default=[0])
    p.add_argument("--clusters",  type=int, default=16)
    p.add_argument("--frames-per-cluster", type=int, default=20)
    p.add_argument("--min-spk-diversity", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def _build_one(args, gender):
    """构建单个 bank，gender=None 表示 mix"""
    args.gender = gender
    data_dir = args.data_dir
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.output:
        out_path = args.output
    elif gender:
        out_path = f"{data_dir}/pseudo_bank_purifier.gender-{gender}.pt"
    else:
        out_path = f"{data_dir}/pseudo_bank_purifier.pt"

    print("=" * 60)
    print(f"构建 Purifier Bank - 性别: {args.gender or 'mix'}, 聚类数: {args.clusters}")
    print("=" * 60)

    purifier = load_purifier(device)
    encoder = purifier.encoder

    with open(f"{data_dir}/metadata.json") as f:
        meta = json.load(f)
    total_frames = meta["total_frames"]

    selected_utts = [u for u in meta["utterances"]
                     if not args.gender or u.get("gender", "unknown") == args.gender]

    spk2id = {s: i for i, s in enumerate(set(u["speaker_id"] for u in selected_utts))}
    frame_to_spk = np.full(total_frames, -1, dtype=np.int32)
    keep_mask = np.zeros(total_frames, dtype=bool)
    for u in selected_utts:
        s, e = u["h5_start_idx"], u["h5_end_idx"]
        frame_to_spk[s:e] = spk2id[u["speaker_id"]]
        keep_mask[s:e] = True

    with h5py.File(f"{data_dir}/phones.h5") as f:
        all_phones = f["phones"][:]
    for sil in args.silence_phones:
        keep_mask &= (all_phones != sil)

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"有效帧: {keep_mask.sum()}, 音素数: {len(unique_phones)}")

    # --- 阶段1: 读 L24 -> 净化 -> 分桶 ---
    print("[1/3] 读取 L24 并净化分桶...")
    pur_buckets = {ph: [] for ph in unique_phones}
    spk_buckets = {ph: [] for ph in unique_phones}
    idx_buckets = {ph: [] for ph in unique_phones}

    with h5py.File(f"{data_dir}/layer_24.h5") as h5:
        ds = h5["features"]
        for start in tqdm(range(0, total_frames, args.chunk_size), desc="净化分桶"):
            end = min(start + args.chunk_size, total_frames)
            mask = keep_mask[start:end]
            if not np.any(mask):
                continue
            local_idx = np.where(mask)[0]
            global_idx = local_idx + start
            l24_raw = ds[start:end][mask]
            pur = purify_chunk(encoder, l24_raw, device)
            phones_c = all_phones[start:end][mask]
            spks_c = frame_to_spk[start:end][mask]
            for ph in np.unique(phones_c):
                m = phones_c == ph
                pur_buckets[ph].append(pur[m])
                spk_buckets[ph].append(spks_c[m])
                idx_buckets[ph].append(global_idx[m])

    # --- 阶段2: 聚类选帧 ---
    print(f"[2/3] 桶内 K-Means({args.clusters} 簇) 选帧...")
    selected_indices = {}
    for ph in tqdm(unique_phones, desc="聚类"):
        if not pur_buckets[ph]:
            continue
        pur_all = np.concatenate(pur_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        idx_all = np.concatenate(idx_buckets[ph])
        del pur_buckets[ph], spk_buckets[ph], idx_buckets[ph]

        n = len(pur_all)
        k = min(args.clusters, n)
        if n < k:
            selected_indices[ph] = idx_all
            continue

        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
        km.fit(pur_all)
        picked = []
        for c in range(k):
            cm = km.labels_ == c
            if not np.any(cm):
                continue
            if len(np.unique(spk_all[cm])) < args.min_spk_diversity:
                continue
            dists = np.linalg.norm(pur_all[cm] - km.cluster_centers_[c], axis=1)
            n_sel = min(args.frames_per_cluster, int(cm.sum()))
            ci = np.where(cm)[0][dists.argsort()[:n_sel]]
            picked.append(idx_all[ci])
        selected_indices[ph] = np.concatenate(picked) if picked else idx_all
        del pur_all, spk_all, idx_all

    del pur_buckets, spk_buckets, idx_buckets
    gc.collect()

    # --- 阶段3: 读选中帧的 L6 + L24(再净化) ---
    print("[3/3] 读取选中帧 L6 + 净化 L24...")
    all_gidx = np.sort(np.concatenate(list(selected_indices.values())))
    g2row = {int(g): i for i, g in enumerate(all_gidx)}
    n_total = len(all_gidx)

    buf_l6  = np.empty((n_total, 1024), dtype=np.float32)
    buf_pur = np.empty((n_total, 512),  dtype=np.float32)  # purified dim=512

    with h5py.File(f"{data_dir}/layer_6.h5") as h6, \
         h5py.File(f"{data_dir}/layer_24.h5") as h24:
        ds6, ds24 = h6["features"], h24["features"]
        bs = 0
        while bs < n_total:
            lo = int(all_gidx[bs])
            be = bs + 1
            while be < n_total and int(all_gidx[be]) - lo < args.chunk_size:
                be += 1
            hi = int(all_gidx[be - 1]) + 1
            c6  = ds6[lo:hi]
            c24 = ds24[lo:hi]
            for j in range(bs, be):
                g = int(all_gidx[j])
                buf_l6[j] = c6[g - lo]
            # 批量净化这段 L24
            raw24 = np.array([c24[int(all_gidx[j]) - lo] for j in range(bs, be)])
            pur24 = purify_chunk(encoder, raw24, device)
            buf_pur[bs:be] = pur24
            bs = be

    bank = {}
    for ph, gidx in selected_indices.items():
        rows = np.array([g2row[int(g)] for g in gidx])
        bank[ph] = {
            "l6":           torch.from_numpy(buf_l6[rows].copy()).float(),
            "purified_l24": torch.from_numpy(buf_pur[rows].copy()).float(),
        }

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    print(f"\nPurifier Bank 完成: {out_path}")
    print(f"  音素数: {len(bank)}, 总帧数: {total_f}")


def main():
    args = parse_args()
    genders = ["m", "f", None] if args.gender == "all" else [None if args.gender in (None, "none", "mix") else args.gender]
    for g in genders:
        _build_one(args, g)


if __name__ == "__main__":
    main()
