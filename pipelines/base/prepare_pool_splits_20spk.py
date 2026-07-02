#!/usr/bin/env python3
"""
从 train-other-500 随机选 800 人（男女各400），划分为 40 个 pool，每 pool 20 人（男10女10）。
输出: checkpoints/pool_splits_20spk.json
"""
import json
import random
import argparse
from pathlib import Path

SPEAKERS_TXT = "/root/autodl-tmp/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
AUDIO_DIR    = "/root/autodl-tmp/datasets/LibriTTS/train-other-500"
OUTPUT       = "/root/autodl-tmp/anon_test/checkpoints/pool_splits_20spk.json"


def load_speakers():
    male, female = [], []
    with open(SPEAKERS_TXT) as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[2] == 'train-other-500':
                spk_id = parts[0]
                if not (Path(AUDIO_DIR) / spk_id).exists():
                    continue
                if parts[1] == 'M':
                    male.append(spk_id)
                elif parts[1] == 'F':
                    female.append(spk_id)
    return male, female


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total', type=int, default=800)
    parser.add_argument('--per-pool-m', type=int, default=20)
    parser.add_argument('--per-pool-f', type=int, default=20)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)
    male, female = load_speakers()
    print(f"可用说话人: 男{len(male)}, 女{len(female)}")

    per_m = args.per_pool_m
    per_f = args.per_pool_f
    n_pools = args.total // (per_m + per_f)
    need_m = n_pools * per_m
    need_f = n_pools * per_f
    assert len(male) >= need_m, f"男性说话人不足: {len(male)} < {need_m}"
    assert len(female) >= need_f, f"女性说话人不足: {len(female)} < {need_f}"

    selected_m = random.sample(male, need_m)
    selected_f = random.sample(female, need_f)

    pools = []
    for i in range(n_pools):
        pool_m = selected_m[i*per_m:(i+1)*per_m]
        pool_f = selected_f[i*per_f:(i+1)*per_f]
        pools.append({
            "pool_id": i,
            "male":    pool_m,
            "female":  pool_f,
            "mix":     pool_m + pool_f,
        })

    all_speakers = selected_m + selected_f
    result = {
        "n_pools":         n_pools,
        "per_pool_male":   per_m,
        "per_pool_female": per_f,
        "total_speakers":  len(all_speakers),
        "all_speakers":    all_speakers,
        "pools":           pools,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"划分完成: {n_pools} pools x (男{per_m}+女{per_f}), "
          f"总人数 {len(all_speakers)}, 保存至: {args.output}")


if __name__ == "__main__":
    main()
