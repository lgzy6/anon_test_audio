#!/usr/bin/env python3
"""
在现有 pool_splits.json 基础上追加新 pool，从未使用的说话人中采样。
用法: python extend_pool_splits.py --add 4
"""
import json
import random
import argparse
from pathlib import Path

SPEAKERS_TXT = "/root/autodl-tmp/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
AUDIO_DIR    = "/root/autodl-tmp/datasets/LibriTTS/train-other-500"
OUTPUT       = "/root/autodl-tmp/anon_test/checkpoints/pool_splits.json"


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
    parser.add_argument('--add', type=int, required=True, help='要新增的 pool 数量')
    parser.add_argument('--per-pool-m', type=int, default=None, help='每个新 pool 的男性数，默认与原来相同')
    parser.add_argument('--per-pool-f', type=int, default=None, help='每个新 pool 的女性数，默认与原来相同')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()

    with open(args.output) as f:
        data = json.load(f)

    per_m = args.per_pool_m or data['per_pool_male']
    per_f = args.per_pool_f or data['per_pool_female']

    used = set(data['all_speakers'])
    all_male, all_female = load_speakers()

    avail_m = [s for s in all_male   if s not in used]
    avail_f = [s for s in all_female if s not in used]
    print(f"可用未使用说话人: 男{len(avail_m)}, 女{len(avail_f)}")

    need_m = args.add * per_m
    need_f = args.add * per_f
    assert len(avail_m) >= need_m, f"男性不足: {len(avail_m)} < {need_m}"
    assert len(avail_f) >= need_f, f"女性不足: {len(avail_f)} < {need_f}"

    random.seed(args.seed)
    new_m = random.sample(avail_m, need_m)
    new_f = random.sample(avail_f, need_f)

    start_id = data['n_pools']
    for i in range(args.add):
        pool_m = new_m[i*per_m:(i+1)*per_m]
        pool_f = new_f[i*per_f:(i+1)*per_f]
        data['pools'].append({
            "pool_id": start_id + i,
            "male":    pool_m,
            "female":  pool_f,
            "mix":     pool_m + pool_f,
        })
        print(f"Pool {start_id + i}: 男{len(pool_m)}, 女{len(pool_f)}, mix{len(pool_m+pool_f)}")

    data['n_pools'] += args.add
    data['all_speakers'] = data['all_speakers'] + new_m + new_f
    data['total_speakers'] = len(data['all_speakers'])

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n已追加 {args.add} 个 pool，总 pool 数: {data['n_pools']}，保存至: {args.output}")


if __name__ == "__main__":
    main()
