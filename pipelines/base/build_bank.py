#!/usr/bin/env python3
"""
构建伪风格 Bank v2 (Base 纯净版)：
- 移除所有 Identity/混合熵过滤
- 仅保留：性别过滤 + 静音帧(Silence)过滤
- 按音素分桶 -> 桶内 L24 K-Means 聚类提纯
- 两阶段读取控制内存
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import gc
import argparse
import json
import h5py
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build baseline pseudo bank (No Identity Filter)")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk")
    parser.add_argument('--output', default=None)
    parser.add_argument('--gender', default=None, help="m/f/none")
    parser.add_argument('--silence-phones', type=int, nargs='+', default=[0],
                        help="视为静音并过滤的音素 ID 列表 (默认: 0)")
    parser.add_argument('--l24-clusters', type=int, default=8,
                        help="桶内 L24 K-Means 聚类数")
    parser.add_argument('--frames-per-cluster', type=int, default=20,
                        help="每簇保留的帧数")
    parser.add_argument('--min-spk-diversity', type=int, default=3,
                        help="每簇最少说话人数")
    parser.add_argument('--chunk-size', type=int, default=200_000)
    return parser.parse_args()


def pick_output_path(data_dir, gender, output):
    if output:
        return output
    if not gender:
        return f"{data_dir}/pseudo_bank_base.pt"
    return f"{data_dir}/pseudo_bank_base.gender-{gender}.pt"


def _build_bank(args):
    data_dir = args.data_dir
    meta_path = f"{data_dir}/metadata.json"
    output_path = pick_output_path(data_dir, args.gender, args.output)

    print("=" * 60)
    print(f"构建纯净版 Bank (Base) - 性别: {args.gender or 'mix'}")
    print(f"  过滤静音音素 ID: {args.silence_phones}")
    print(f"  L24 聚类数: {args.l24_clusters}")
    print(f"  每簇帧数: {args.frames_per_cluster}")
    print("=" * 60)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    total_frames = meta['total_frames']

    selected_utts = [u for u in meta['utterances']
                     if not args.gender or u.get('gender', 'unknown') == args.gender]
    if not selected_utts:
        print("无匹配 utterances，退出。")
        return

    # 1. 构建说话人索引
    print("[1/5] 构建说话人索引...")
    speaker_list = list(set(u['speaker_id'] for u in selected_utts))
    spk2id = {spk: i for i, spk in enumerate(speaker_list)}
    frame_to_spk = np.full(total_frames, fill_value=-1, dtype=np.int32)

    # 动态生成纯净掩码 (仅激活当前性别的帧)
    keep_mask = np.zeros(total_frames, dtype=bool)

    for utt in selected_utts:
        start, end = utt['h5_start_idx'], utt['h5_end_idx']
        frame_to_spk[start:end] = spk2id[utt['speaker_id']]
        keep_mask[start:end] = True  # 标记为属于当前性别/选定集合

    # 2. 加载 Phones 并过滤静音帧
    print("[2/5] 加载 Phones 并执行静音过滤...")
    with h5py.File(f"{data_dir}/phones.h5", 'r') as h5:
        all_phones = h5['phones'][:]

    # 剔除静音音素
    for sil_ph in args.silence_phones:
        keep_mask &= (all_phones != sil_ph)

    kept_count = int(np.sum(keep_mask))
    print(f"  过滤后保留帧: {kept_count} / {total_frames} ({kept_count/total_frames:.1%})")

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"  有效音素数: {len(unique_phones)}")

    # ================================================================
    # 3. 第一阶段：只读 L24，按音素分桶 + 记录全局索引
    # ================================================================
    print("[3/5] 第一阶段：仅读取 L24 分桶...")
    l24_buckets = {ph: [] for ph in unique_phones}
    spk_buckets = {ph: [] for ph in unique_phones}
    idx_buckets = {ph: [] for ph in unique_phones}

    with h5py.File(f"{data_dir}/layer_24.h5", 'r') as h5_l24:
        ds_l24 = h5_l24['features']

        for start_idx in tqdm(range(0, total_frames, args.chunk_size), desc="L24 分桶"):
            end_idx = min(start_idx + args.chunk_size, total_frames)
            mask_chunk = keep_mask[start_idx:end_idx]

            if not np.any(mask_chunk):
                continue

            local_kept = np.where(mask_chunk)[0]
            global_kept = local_kept + start_idx

            l24_chunk = ds_l24[start_idx:end_idx][mask_chunk]
            phones_chunk = all_phones[start_idx:end_idx][mask_chunk]
            spks_chunk = frame_to_spk[start_idx:end_idx][mask_chunk]

            for ph in np.unique(phones_chunk):
                ph_mask = (phones_chunk == ph)
                l24_buckets[ph].append(l24_chunk[ph_mask])
                spk_buckets[ph].append(spks_chunk[ph_mask])
                idx_buckets[ph].append(global_kept[ph_mask])

    # ================================================================
    # 4. K-Means 聚类选帧，逐音素处理后立即释放
    # ================================================================
    print(f"[4/5] 桶内 K-Means({args.l24_clusters} 簇) 选帧...")
    selected_indices = {}

    for phone_id in tqdm(unique_phones, desc="聚类选帧"):
        if not l24_buckets[phone_id]:
            continue

        l24_all = np.concatenate(l24_buckets[phone_id])
        spk_all = np.concatenate(spk_buckets[phone_id])
        idx_all = np.concatenate(idx_buckets[phone_id])

        del l24_buckets[phone_id], spk_buckets[phone_id], idx_buckets[phone_id]

        n_frames = len(l24_all)
        n_clusters = args.l24_clusters

        if n_frames < n_clusters:
            selected_indices[phone_id] = idx_all
            del l24_all, spk_all, idx_all
            continue

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
        km.fit(l24_all)
        labels = km.labels_

        picked = []
        for c in range(n_clusters):
            cluster_mask = (labels == c)
            cluster_count = int(np.sum(cluster_mask))
            if cluster_count == 0:
                continue

            # 纯净版依然保留簇内多样性验证，防止取到极端同源数据
            if len(np.unique(spk_all[cluster_mask])) < args.min_spk_diversity:
                continue

            cluster_l24 = l24_all[cluster_mask]
            dists = np.linalg.norm(cluster_l24 - km.cluster_centers_[c], axis=1)
            n_select = min(args.frames_per_cluster, cluster_count)
            top_local = dists.argsort()[:n_select]

            cluster_global = np.where(cluster_mask)[0]
            picked.append(idx_all[cluster_global[top_local]])

        if picked:
            selected_indices[phone_id] = np.concatenate(picked)
        else:
            selected_indices[phone_id] = idx_all

        del l24_all, spk_all, idx_all, km, labels

    del l24_buckets, spk_buckets, idx_buckets
    gc.collect()

    total_selected = sum(len(v) for v in selected_indices.values())
    print(f"  选中总帧数: {total_selected} (覆盖 {len(selected_indices)} 个音素)")

    # ================================================================
    # 5. 第二阶段：只读选中帧的 L6/L12/L24
    # ================================================================
    print("[5/5] 第二阶段：按选中索引读取 L6/L12/L24...")

    all_global_idx = np.concatenate(list(selected_indices.values()))
    all_global_idx_sorted = np.sort(all_global_idx)

    sorted_to_original = {int(g): i for i, g in enumerate(all_global_idx_sorted)}

    feats = {}
    with h5py.File(f"{data_dir}/layer_6.h5",  'r') as h5_l6,  \
         h5py.File(f"{data_dir}/layer_12.h5", 'r') as h5_l12, \
         h5py.File(f"{data_dir}/layer_24.h5", 'r') as h5_l24:

        ds_l6 = h5_l6['features']
        ds_l12 = h5_l12['features']
        ds_l24 = h5_l24['features']

        n_total = len(all_global_idx_sorted)
        buf_l6  = np.empty((n_total, 1024), dtype=np.float32)
        buf_l12 = np.empty((n_total, 1024), dtype=np.float32)
        buf_l24 = np.empty((n_total, 1024), dtype=np.float32)

        batch_start = 0
        while batch_start < n_total:
            idx_lo = int(all_global_idx_sorted[batch_start])
            batch_end = batch_start + 1
            while batch_end < n_total and \
                  int(all_global_idx_sorted[batch_end]) - idx_lo < args.chunk_size:
                batch_end += 1
            idx_hi = int(all_global_idx_sorted[batch_end - 1]) + 1

            chunk_l6  = ds_l6[idx_lo:idx_hi]
            chunk_l12 = ds_l12[idx_lo:idx_hi]
            chunk_l24 = ds_l24[idx_lo:idx_hi]

            for j in range(batch_start, batch_end):
                g = int(all_global_idx_sorted[j])
                local = g - idx_lo
                buf_l6[j]  = chunk_l6[local]
                buf_l12[j] = chunk_l12[local]
                buf_l24[j] = chunk_l24[local]

            batch_start = batch_end

    pseudo_bank = {}
    for phone_id, gidx in selected_indices.items():
        rows = np.array([sorted_to_original[int(g)] for g in gidx])
        pseudo_bank[phone_id] = {
            'l6':  torch.from_numpy(buf_l6[rows].copy()).float(),
            'l12': torch.from_numpy(buf_l12[rows].copy()).float(),
            'l24': torch.from_numpy(buf_l24[rows].copy()).float(),
        }

    del buf_l6, buf_l12, buf_l24, selected_indices, all_global_idx, all_global_idx_sorted
    gc.collect()

    torch.save(pseudo_bank, output_path)

    total_bank_frames = sum(v['l6'].shape[0] for v in pseudo_bank.values())
    print("\n" + "=" * 60)
    print(f"Base Bank 构建完成: {output_path}")
    print(f"  涵盖音素数: {len(pseudo_bank)}")
    print(f"  总帧数: {total_bank_frames}")
    print("=" * 60)

def main():
    args = parse_args()
    args.gender = None if args.gender in (None, 'none') else args.gender
    _build_bank(args)

if __name__ == "__main__":
    main()