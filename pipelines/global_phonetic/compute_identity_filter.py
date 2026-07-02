#!/usr/bin/env python3
"""
Identity 自相似度过滤：替代混淆熵
- 对每帧计算 sim_self = cos(L12_frame, L12_centroid_of_speaker)
- sim_self > τ → 该帧身份信息过强，标记为 drop
- 每个 phone 至少保留 min_speakers_per_phone 个说话人
"""

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Identity filter based on sim_self")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk")
    parser.add_argument('--centroids', default=None, help="说话人质心文件，默认 {data_dir}/speaker_centroids.npz")
    parser.add_argument('--output', default=None, help="输出掩码文件，默认 {data_dir}/identity_mask.h5")
    parser.add_argument('--gender', default=None, help="m/f，不传则批量处理男女")
    parser.add_argument('--tau', type=float, default=0.80,
                        help="sim_self 阈值，超过则 drop（建议 0.75~0.85）")
    parser.add_argument('--min-speakers-per-phone', type=int, default=3,
                        help="每个音素至少保留的说话人数")
    parser.add_argument('--chunk-size', type=int, default=500_000)
    return parser.parse_args()


def pick_output_path(data_dir, gender, output):
    if output:
        return output
    if not gender:
        return f"{data_dir}/identity_mask.h5"
    return f"{data_dir}/identity_mask.gender-{gender}.h5"


def compute_filter_for_gender(data_dir, gender, centroids_path, output_path, tau,
                               min_speakers_per_phone, chunk_size):
    meta_path = f"{data_dir}/metadata.json"
    l12_path = f"{data_dir}/layer_12.h5"
    phone_path = f"{data_dir}/phones.h5"

    print("=" * 60)
    print(f"Identity 过滤 - 性别: {gender or 'all'}, τ={tau}")
    print("=" * 60)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    utterances = meta['utterances']
    total_frames = meta['total_frames']

    if gender:
        utterances = [u for u in utterances if u.get('gender', 'unknown') == gender]

    if not utterances:
        print("无匹配 utterances，退出。")
        return

    # 加载说话人质心
    centroid_data = np.load(centroids_path)
    speaker_ids = list(set(u['speaker_id'] for u in utterances))
    print(f"加载了 {len(centroid_data.files)} 个说话人质心，当前性别有 {len(speaker_ids)} 个说话人")

    # Step 1: 构建逐帧的说话人映射 + sim_self 计算
    # 先建立 frame → speaker_id 的映射
    frame_spk = np.full(total_frames, '', dtype='U10')
    frame_gender_mask = np.zeros(total_frames, dtype=bool)  # 当前性别的帧

    for utt in utterances:
        start, end = utt['h5_start_idx'], utt['h5_end_idx']
        frame_spk[start:end] = utt['speaker_id']
        frame_gender_mask[start:end] = True

    # Step 2: 逐块计算 sim_self
    print("[Step 1/3] 逐块计算 sim_self...")
    sim_self_arr = np.full(total_frames, np.nan, dtype=np.float32)

    with h5py.File(l12_path, 'r') as h5:
        ds = h5['features']
        for start in tqdm(range(0, total_frames, chunk_size), desc="计算 sim_self"):
            end = min(start + chunk_size, total_frames)
            mask_chunk = frame_gender_mask[start:end]
            if not np.any(mask_chunk):
                continue

            l12_chunk = ds[start:end]  # [chunk, 1024]
            local_indices = np.where(mask_chunk)[0]

            for li in local_indices:
                global_idx = start + li
                spk = frame_spk[global_idx]
                if spk not in centroid_data:
                    continue
                centroid = centroid_data[spk]  # [1024], 已经 L2 归一化
                frame_vec = l12_chunk[li]
                frame_norm = frame_vec / (np.linalg.norm(frame_vec) + 1e-8)
                sim_self_arr[global_idx] = np.dot(frame_norm, centroid)

    # Step 3: 按阈值生成 keep mask（sim_self <= τ → keep）
    print(f"[Step 2/3] 按 τ={tau} 生成 keep mask...")
    keep_mask = np.zeros(total_frames, dtype=bool)

    # 基本规则：sim_self <= τ → keep
    valid_sim = ~np.isnan(sim_self_arr) & frame_gender_mask
    basic_keep = valid_sim & (sim_self_arr <= tau)
    keep_mask[basic_keep] = True

    # 统计
    total_valid = np.sum(valid_sim)
    total_kept = np.sum(keep_mask)
    total_dropped = total_valid - total_kept
    print(f"  有效帧: {total_valid}, 保留: {total_kept} ({total_kept/total_valid:.1%}), "
          f"丢弃: {total_dropped} ({total_dropped/total_valid:.1%})")

    # Step 4: 每个音素至少保留 min_speakers_per_phone 个说话人
    # 对于被丢弃过多导致某些音素说话人不足的情况，回捞一些帧
    print(f"[Step 3/3] 保证每个音素至少 {min_speakers_per_phone} 个说话人...")

    with h5py.File(phone_path, 'r') as h5:
        all_phones = h5['phones'][:]

    unique_phones = np.unique(all_phones[frame_gender_mask])
    rescued_count = 0

    for ph in unique_phones:
        ph_indices = np.where((all_phones == ph) & frame_gender_mask)[0]
        if len(ph_indices) == 0:
            continue

        # 当前保留帧中该音素涉及的说话人
        ph_kept = ph_indices[keep_mask[ph_indices]]
        kept_spks = set(frame_spk[ph_kept])

        if len(kept_spks) >= min_speakers_per_phone:
            continue

        # 需要回捞：从被丢弃的帧中，按 sim_self 从低到高排序，补充到满足条件
        ph_dropped = ph_indices[~keep_mask[ph_indices] & valid_sim[ph_indices]]
        if len(ph_dropped) == 0:
            continue

        drop_sims = sim_self_arr[ph_dropped]
        sorted_order = np.argsort(drop_sims)  # sim_self 最低的优先回捞

        for oi in sorted_order:
            idx = ph_dropped[oi]
            spk = frame_spk[idx]
            keep_mask[idx] = True
            kept_spks.add(spk)
            rescued_count += 1
            if len(kept_spks) >= min_speakers_per_phone:
                break

    print(f"  回捞了 {rescued_count} 帧以满足最低说话人数要求")

    # 保存
    with h5py.File(output_path, 'w') as h5:
        h5.create_dataset('keep_mask', data=keep_mask)
        h5.create_dataset('sim_self', data=sim_self_arr)
        h5.attrs['tau'] = tau
        h5.attrs['min_speakers_per_phone'] = min_speakers_per_phone
        h5.attrs['gender'] = gender or 'all'

    final_kept = np.sum(keep_mask)
    print(f"\n完成！掩码已写入: {output_path}")
    print(f"最终保留帧数: {final_kept} / {total_valid} ({final_kept/total_valid:.1%})")


def main():
    args = parse_args()
    centroids_path = args.centroids or f"{args.data_dir}/speaker_centroids.npz"

    gender = None if args.gender in (None, 'none') else args.gender
    output_path = pick_output_path(args.data_dir, gender, args.output)
    compute_filter_for_gender(
        args.data_dir, gender, centroids_path, output_path,
        args.tau, args.min_speakers_per_phone, args.chunk_size
    )


if __name__ == "__main__":
    main()