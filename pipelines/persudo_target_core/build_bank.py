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
    parser = argparse.ArgumentParser(description="Build pseudo bank with label filtering")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_with_phones")
    parser.add_argument('--entropy', default=None, help="熵文件路径，默认按标签自动命名")
    parser.add_argument('--output', default=None, help="输出 bank 路径，默认按标签自动命名")
    parser.add_argument('--gender', default=None, help="按性别过滤 (e.g., m/f/unknown)")
    parser.add_argument('--emotion', default=None, help="按情绪过滤 (e.g., neutral/happy/unknown)")
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('--frames-per-cluster', type=int, default=20)
    parser.add_argument('--entropy-percentile', type=float, default=60)
    parser.add_argument('--min-spk-diversity', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    return parser.parse_args()


def pick_entropy_path(data_dir: str, gender: str | None, emotion: str | None, entropy: str | None) -> str:
    if entropy:
        return entropy
    if not gender and not emotion:
        return f"{data_dir}/entropies.h5"
    suffix = []
    if gender:
        suffix.append(f"gender-{gender}")
    if emotion:
        suffix.append(f"emotion-{emotion}")
    suffix_str = '.'.join(suffix)
    return f"{data_dir}/entropies.{suffix_str}.h5"


def pick_output_path(data_dir: str, gender: str | None, emotion: str | None, output: str | None) -> str:
    if output:
        return output
    if not gender and not emotion:
        return f"{data_dir}/pseudo_bank.pt"
    suffix = []
    if gender:
        suffix.append(f"gender-{gender}")
    if emotion:
        suffix.append(f"emotion-{emotion}")
    suffix_str = '.'.join(suffix)
    return f"{data_dir}/pseudo_bank.{suffix_str}.pt"


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    meta_path = f"{data_dir}/metadata.json"
    entropy_path = pick_entropy_path(data_dir, args.gender, args.emotion, args.entropy)
    output_bank_path = pick_output_path(data_dir, args.gender, args.emotion, args.output)

    print("=" * 60)
    print("开始构建最终伪风格 Bank (顺序读取优化版)...")
    print("=" * 60)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    total_frames = meta['total_frames']

    def match_labels(utt: dict) -> bool:
        if args.gender and utt.get('gender', 'unknown') != args.gender:
            return False
        if args.emotion and utt.get('emotion', 'unknown') != args.emotion:
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

    # 3. 顺序扫描 HDF5，将高熵帧全部吸入内存
    print("[3/4] 顺序读取磁盘，将高熵特征装载至内存 (极速 I/O)...")
    l6_filtered_dict = {ph: [] for ph in unique_phones}
    spk_filtered_dict = {ph: [] for ph in unique_phones}

    with h5py.File(f"{data_dir}/layer_6.h5", 'r') as h5_l6:
        ds_l6 = h5_l6['features']

        for start_idx in tqdm(range(0, total_frames, args.chunk_size), desc="顺序块读取 L6"):
            end_idx = min(start_idx + args.chunk_size, total_frames)
            mask_chunk = global_keep_mask[start_idx:end_idx]

            if not np.any(mask_chunk):
                continue

            l6_chunk = ds_l6[start_idx:end_idx]

            l6_chunk_kept = l6_chunk[mask_chunk]
            phones_chunk_kept = all_phones[start_idx:end_idx][mask_chunk]
            spks_chunk_kept = frame_to_spk[start_idx:end_idx][mask_chunk]

            for ph in np.unique(phones_chunk_kept):
                ph_mask = (phones_chunk_kept == ph)
                l6_filtered_dict[ph].append(l6_chunk_kept[ph_mask])
                spk_filtered_dict[ph].append(spks_chunk_kept[ph_mask])

    # 4. 在内存中逐音素执行 K-Means 二次聚类
    print("[4/4] 逐音素聚类并构建最终 Bank...")
    pseudo_bank_tensors = {}

    for phone_id in tqdm(unique_phones, desc="Processing Phones"):
        if not l6_filtered_dict[phone_id]:
            continue

        l6_filtered = np.concatenate(l6_filtered_dict[phone_id])
        spks_filtered = np.concatenate(spk_filtered_dict[phone_id])

        if len(l6_filtered) < args.clusters:
            pseudo_bank_tensors[phone_id] = torch.from_numpy(l6_filtered).float()
            continue

        km = MiniBatchKMeans(n_clusters=args.clusters, random_state=42, batch_size=2048)
        km.fit(l6_filtered)
        labels = km.labels_

        selected_frames = []
        for c in range(args.clusters):
            cluster_mask = labels == c
            cluster_l6 = l6_filtered[cluster_mask]
            cluster_spks = spks_filtered[cluster_mask]

            if len(np.unique(cluster_spks)) < args.min_spk_diversity:
                continue

            dists = np.linalg.norm(cluster_l6 - km.cluster_centers_[c], axis=1)
            top_idx = dists.argsort()[:args.frames_per_cluster]
            selected_frames.append(cluster_l6[top_idx])

        if selected_frames:
            final_l6_np = np.concatenate(selected_frames)
        else:
            final_l6_np = l6_filtered

        pseudo_bank_tensors[phone_id] = torch.from_numpy(final_l6_np).float()

    torch.save(pseudo_bank_tensors, output_bank_path)

    print("\n" + "=" * 60)
    print(f"Bank 构建完成并已保存至: {output_bank_path}")
    print(f"涵盖音素数: {len(pseudo_bank_tensors)}")
    print(f"总帧数: {sum(t.shape[0] for t in pseudo_bank_tensors.values())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
