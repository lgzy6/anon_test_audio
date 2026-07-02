#!/usr/bin/env python3
"""基于混淆熵与聚类构建最终的 kNN-VC 伪风格 Bank (标签分组版)"""

import argparse
import json
import h5py
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pseudo bank")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_with_phones")
    parser.add_argument('--entropy', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--gender', default=None, help="m/f，不传则批量处理男女")
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--frames-per-cluster', type=int, default=20)
    parser.add_argument('--entropy-percentile', type=float, default=75)
    parser.add_argument('--min-spk-diversity', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    return parser.parse_args()


def pick_entropy_path(data_dir: str, gender: str | None, entropy: str | None) -> str:
    if entropy:
        return entropy
    if not gender:
        return f"{data_dir}/entropies.h5"
    return f"{data_dir}/entropies.gender-{gender}.h5"


def pick_output_path(data_dir: str, gender: str | None, output: str | None) -> str:
    if output:
        return output
    if not gender:
        return f"{data_dir}/pseudo_bank.pt"
    return f"{data_dir}/pseudo_bank.gender-{gender}.pt"


def main() -> None:
    args = parse_args()
    genders = ['m', 'f'] if args.gender is None else [args.gender]
    for gender in genders:
        args.gender = gender
        _build_bank(args)


def _build_bank(args) -> None:
    data_dir = args.data_dir
    meta_path = f"{data_dir}/metadata.json"
    entropy_path = pick_entropy_path(data_dir, args.gender, args.entropy)
    output_bank_path = pick_output_path(data_dir, args.gender, args.output)

    print("=" * 60)
    print(f"开始构建伪风格 Bank - 性别: {args.gender}")
    print("=" * 60)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    total_frames = meta['total_frames']

    def match_labels(utt: dict) -> bool:
        if args.gender and utt.get('gender', 'unknown') != args.gender:
            return False
        return True

    selected_utts = [u for u in meta['utterances'] if match_labels(u)]
    if not selected_utts:
        print("未找到满足标签条件的 utterances，退出。")
        return

    # 1. 构建内存说话人索引
    print("[1/4] 构建内存说话人索引...")
    speaker_list = list(set([u['speaker_id'] for u in selected_utts]))
    spk2id = {spk: i for i, spk in enumerate(speaker_list)}
    frame_to_spk = np.full(total_frames, fill_value=-1, dtype=np.int32)
    frame_keep_mask = np.zeros(total_frames, dtype=bool)

    for utt in selected_utts:
        start, end = utt['h5_start_idx'], utt['h5_end_idx']
        frame_to_spk[start:end] = spk2id[utt['speaker_id']]
        frame_keep_mask[start:end] = True

    # 2. 读取标签与混淆熵，预计算全局高熵掩码
    print("[2/4] 计算全局高熵掩码...")
    with h5py.File(f"{data_dir}/phones.h5", 'r') as h5_p, h5py.File(entropy_path, 'r') as h5_e:
        all_phones = h5_p['phones'][:]
        all_entropies = h5_e['entropies'][:]

    if not np.any(frame_keep_mask):
        print("未找到有效帧，退出。")
        return

    unique_phones = np.unique(all_phones[frame_keep_mask])
    global_keep_mask = np.zeros(total_frames, dtype=bool)

    for phone_id in unique_phones:
        ph_indices = np.where((all_phones == phone_id) & frame_keep_mask)[0]
        if len(ph_indices) == 0:
            continue

        ph_entropies = all_entropies[ph_indices]
        valid_mask = ~np.isnan(ph_entropies)
        if not np.any(valid_mask):
            continue

        ph_entropies_valid = ph_entropies[valid_mask]
        valid_indices = ph_indices[valid_mask]

        entropy_threshold = np.percentile(ph_entropies_valid, args.entropy_percentile)
        high_entropy_mask = ph_entropies_valid >= entropy_threshold

        selected_indices = valid_indices[high_entropy_mask]
        global_keep_mask[selected_indices] = True

    # 3. 顺序扫描 HDF5，将高熵帧全部吸入内存 (L6 + L12 + L24)
    print("[3/4] 顺序读取磁盘，将高熵特征装载至内存 (L6 + L12 + L24)...")
    l6_filtered_dict  = {ph: [] for ph in unique_phones}
    l12_filtered_dict = {ph: [] for ph in unique_phones}
    l24_filtered_dict = {ph: [] for ph in unique_phones}
    spk_filtered_dict = {ph: [] for ph in unique_phones}

    with h5py.File(f"{data_dir}/layer_6.h5",  'r') as h5_l6,  \
         h5py.File(f"{data_dir}/layer_12.h5", 'r') as h5_l12, \
         h5py.File(f"{data_dir}/layer_24.h5", 'r') as h5_l24:
        ds_l6  = h5_l6['features']
        ds_l12 = h5_l12['features']
        ds_l24 = h5_l24['features']

        for start_idx in tqdm(range(0, total_frames, args.chunk_size), desc="顺序块读取"):
            end_idx = min(start_idx + args.chunk_size, total_frames)
            mask_chunk = global_keep_mask[start_idx:end_idx]

            if not np.any(mask_chunk):
                continue

            l6_chunk_kept  = ds_l6[start_idx:end_idx][mask_chunk]
            l12_chunk_kept = ds_l12[start_idx:end_idx][mask_chunk]
            l24_chunk_kept = ds_l24[start_idx:end_idx][mask_chunk]
            phones_chunk_kept = all_phones[start_idx:end_idx][mask_chunk]
            spks_chunk_kept   = frame_to_spk[start_idx:end_idx][mask_chunk]

            for ph in np.unique(phones_chunk_kept):
                ph_mask = (phones_chunk_kept == ph)
                l6_filtered_dict[ph].append(l6_chunk_kept[ph_mask])
                l12_filtered_dict[ph].append(l12_chunk_kept[ph_mask])
                l24_filtered_dict[ph].append(l24_chunk_kept[ph_mask])
                spk_filtered_dict[ph].append(spks_chunk_kept[ph_mask])

    # 4. 在内存中逐音素执行 K-Means 二次聚类 (基于L24)，绑定 L6/L12/L24
    print("[4/4] 逐音素聚类并构建最终 Bank (L6 + L12 + L24 绑定)...")
    pseudo_bank_tensors = {}

    for phone_id in tqdm(unique_phones, desc="Processing Phones"):
        if not l6_filtered_dict[phone_id]:
            continue

        l6_filtered  = np.concatenate(l6_filtered_dict[phone_id])
        l12_filtered = np.concatenate(l12_filtered_dict[phone_id])
        l24_filtered = np.concatenate(l24_filtered_dict[phone_id])
        spks_filtered = np.concatenate(spk_filtered_dict[phone_id])

        if len(l6_filtered) < args.clusters:
            pseudo_bank_tensors[phone_id] = {
                'l6':  torch.from_numpy(l6_filtered).float(),
                'l12': torch.from_numpy(l12_filtered).float(),
                'l24': torch.from_numpy(l24_filtered).float(),
            }
            continue

        km = MiniBatchKMeans(n_clusters=args.clusters, random_state=42, batch_size=2048)
        km.fit(l24_filtered)
        labels = km.labels_

        selected_l6, selected_l12, selected_l24 = [], [], []
        for c in range(args.clusters):
            cluster_mask = labels == c
            cluster_spks = spks_filtered[cluster_mask]
            if len(np.unique(cluster_spks)) < args.min_spk_diversity:
                continue

            cluster_l24 = l24_filtered[cluster_mask]
            dists   = np.linalg.norm(cluster_l24 - km.cluster_centers_[c], axis=1)
            top_idx = dists.argsort()[:args.frames_per_cluster]
            selected_l6.append(l6_filtered[cluster_mask][top_idx])
            selected_l12.append(l12_filtered[cluster_mask][top_idx])
            selected_l24.append(cluster_l24[top_idx])

        if selected_l6:
            final_l6  = np.concatenate(selected_l6)
            final_l12 = np.concatenate(selected_l12)
            final_l24 = np.concatenate(selected_l24)
        else:
            final_l6, final_l12, final_l24 = l6_filtered, l12_filtered, l24_filtered

        pseudo_bank_tensors[phone_id] = {
            'l6':  torch.from_numpy(final_l6).float(),
            'l12': torch.from_numpy(final_l12).float(),
            'l24': torch.from_numpy(final_l24).float(),
        }

    torch.save(pseudo_bank_tensors, output_bank_path)

    print("\n" + "=" * 60)
    print(f"Bank 构建完成并已保存至: {output_bank_path}")
    print(f"涵盖音素数: {len(pseudo_bank_tensors)}")
    print(f"总帧数 (L6): {sum(t['l6'].shape[0] for t in pseudo_bank_tensors.values())}")
    print(f"存储格式: {{phone_id: {{'l6': tensor, 'l12': tensor, 'l24': tensor}}}}")
    print("=" * 60)


if __name__ == "__main__":
    main()
