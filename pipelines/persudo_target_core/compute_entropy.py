#!/usr/bin/env python3
"""基于 HDF5 的流式 Layer 12 混淆熵计算"""

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute confusion entropy for L12 features")
    parser.add_argument('--data-dir', default="/root/autodl-tmp/anon_test/checkpoints/trainother500_with_phones")
    parser.add_argument('--output', default=None, help="输出熵文件路径，默认按标签自动命名")
    parser.add_argument('--gender', default=None, help="按性别过滤 (e.g., m/f/unknown)")
    parser.add_argument('--emotion', default=None, help="按情绪过滤 (e.g., neutral/happy/unknown)")
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--min-speakers', type=int, default=3)
    return parser.parse_args()


def pick_entropy_path(data_dir: str, gender: str | None, emotion: str | None, output: str | None) -> str:
    if output:
        return output
    if not gender and not emotion:
        return f"{data_dir}/entropies.h5"
    suffix = []
    if gender:
        suffix.append(f"gender-{gender}")
    if emotion:
        suffix.append(f"emotion-{emotion}")
    suffix_str = '.'.join(suffix)
    return f"{data_dir}/entropies.{suffix_str}.h5"


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    meta_path = f"{data_dir}/metadata.json"
    l12_path = f"{data_dir}/layer_12.h5"
    phone_path = f"{data_dir}/phones.h5"
    output_entropy_path = pick_entropy_path(data_dir, args.gender, args.emotion, args.output)

    print("=" * 60)
    print("Step 1: 计算音素内说话人嵌入 (Centroids)")
    print("=" * 60)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    utterances = meta['utterances']

    def match_labels(utt: dict) -> bool:
        if args.gender and utt.get('gender', 'unknown') != args.gender:
            return False
        if args.emotion and utt.get('emotion', 'unknown') != args.emotion:
            return False
        return True

    selected_utts = [u for u in utterances if match_labels(u)]
    if not selected_utts:
        print("未找到满足标签条件的 utterances，退出。")
        return

    # 用于累加特征的字典: {phone_id: {spk_id: [sum_vector, count]}}
    phone_spk_stats = defaultdict(lambda: defaultdict(lambda: [np.zeros(1024), 0]))

    with h5py.File(l12_path, 'r') as h5_l12, h5py.File(phone_path, 'r') as h5_p:
        ds_l12 = h5_l12['features']
        ds_p = h5_p['phones']

        for utt in tqdm(selected_utts, desc="Accumulating L12"):
            start, end = utt['h5_start_idx'], utt['h5_end_idx']
            spk_id = utt['speaker_id']

            l12_frames = ds_l12[start:end]   # [T, 1024]
            phone_frames = ds_p[start:end]   # [T]

            for i in range(len(phone_frames)):
                ph = phone_frames[i]
                phone_spk_stats[ph][spk_id][0] += l12_frames[i]
                phone_spk_stats[ph][spk_id][1] += 1

    phone_spk_embs = {}
    for ph, spk_dict in phone_spk_stats.items():
        phone_spk_embs[ph] = {}
        for spk, (feat_sum, count) in spk_dict.items():
            if count >= 3:
                phone_spk_embs[ph][spk] = feat_sum / count

    valid_phones = {ph: embs for ph, embs in phone_spk_embs.items() if len(embs) >= args.min_speakers}
    print(f"有效音素数量: {len(valid_phones)} (要求每音素至少包含 {args.min_speakers} 个说话人)")

    print("\n" + "=" * 60)
    print("Step 2: 计算每帧的混淆熵并存入 HDF5")
    print("=" * 60)

    total_frames = meta['total_frames']
    h5_entropy = h5py.File(output_entropy_path, 'w')
    ds_ent = h5_entropy.create_dataset('entropies', shape=(total_frames,), dtype='float32', fillvalue=np.nan)

    precomputed_embs = {}
    for ph, spk_dict in valid_phones.items():
        emb_matrix = np.stack(list(spk_dict.values()))
        emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        precomputed_embs[ph] = emb_norms

    valid_count = 0

    with h5py.File(l12_path, 'r') as h5_l12, h5py.File(phone_path, 'r') as h5_p:
        ds_l12 = h5_l12['features']
        ds_p = h5_p['phones']

        for utt in tqdm(selected_utts, desc="Calculating Entropy"):
            start, end = utt['h5_start_idx'], utt['h5_end_idx']
            l12_frames = ds_l12[start:end]
            phone_frames = ds_p[start:end]

            entropies = np.full(end - start, np.nan, dtype=np.float32)

            unique_phones_in_utt = np.unique(phone_frames)
            for ph in unique_phones_in_utt:
                if ph not in precomputed_embs:
                    continue

                ph_mask = (phone_frames == ph)
                ph_l12 = l12_frames[ph_mask]  # [N, 1024]
                emb_norms = precomputed_embs[ph]  # [N_spk, 1024]
                num_spks = emb_norms.shape[0]

                ph_l12_norm = ph_l12 / (np.linalg.norm(ph_l12, axis=1, keepdims=True) + 1e-8)
                sims = ph_l12_norm @ emb_norms.T  # [N, N_spk]

                scaled_sims = sims * args.temperature
                max_sims = np.max(scaled_sims, axis=1, keepdims=True)
                exp_sims = np.exp(scaled_sims - max_sims)
                probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

                ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)
                max_entropy = np.log(num_spks)

                entropies[ph_mask] = ent / max_entropy
                valid_count += np.sum(ph_mask)

            ds_ent[start:end] = entropies

    h5_entropy.close()

    print("\n" + "=" * 60)
    print(f"完成！混淆熵已写入: {output_entropy_path}")
    print(f"总帧数: {total_frames}, 成功计算熵的帧数: {valid_count} ({valid_count/total_frames:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
