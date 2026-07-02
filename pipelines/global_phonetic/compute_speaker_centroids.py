#!/usr/bin/env python3
"""预计算每个说话人的 L12 全局质心（不分音素，整体均值）"""

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute per-speaker L12 centroids")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk")
    parser.add_argument('--output', default=None, help="输出路径，默认 {data_dir}/speaker_centroids.npz")
    parser.add_argument('--gender', default=None, help="m/f，不传则处理全部")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    meta_path = f"{data_dir}/metadata.json"
    l12_path = f"{data_dir}/layer_12.h5"
    output_path = args.output or f"{data_dir}/speaker_centroids.npz"

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    utterances = meta['utterances']

    if args.gender:
        utterances = [u for u in utterances if u.get('gender', 'unknown') == args.gender]

    print(f"计算说话人 L12 质心，共 {len(utterances)} 条 utterances")

    spk_accum = defaultdict(lambda: [np.zeros(1024, dtype=np.float64), 0])

    with h5py.File(l12_path, 'r') as h5:
        ds = h5['features']
        for utt in tqdm(utterances, desc="累积 L12"):
            start, end = utt['h5_start_idx'], utt['h5_end_idx']
            spk = utt['speaker_id']
            frames = ds[start:end]  # [T, 1024]
            spk_accum[spk][0] += frames.sum(axis=0).astype(np.float64)
            spk_accum[spk][1] += len(frames)

    centroids = {}
    for spk, (feat_sum, count) in spk_accum.items():
        centroid = (feat_sum / count).astype(np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-8)  # L2 归一化
        centroids[spk] = centroid

    np.savez(output_path, **centroids)
    print(f"已保存 {len(centroids)} 个说话人质心至: {output_path}")


if __name__ == "__main__":
    main()