#!/usr/bin/env python3
"""
从 train-other-500 随机选 900 人（男450女450），划分为 4 个大池（每池男50女50）。
输出: checkpoints/pool_splits.json
"""
import json
import random
import argparse
from pathlib import Path

SPEAKERS_TXT = "/root/autodl-tmp/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
AUDIO_DIR    = "/root/autodl-tmp/datasets/LibriTTS/train-other-500"
OUTPUT       = "/root/autodl-tmp/anon_test/checkpoints/pool_splits.json"

N_POOLS      = 4
PER_POOL_M   = 50
PER_POOL_F   = 50


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
                # 只保留音频目录中实际存在的说话人
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
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)

    male, female = load_speakers()
    print(f"可用说话人: 男{len(male)}, 女{len(female)}")

    need_m = N_POOLS * PER_POOL_M  # 200
    need_f = N_POOLS * PER_POOL_F  # 200
    assert len(male) >= need_m, f"男性说话人不足: {len(male)} < {need_m}"
    assert len(female) >= need_f, f"女性说话人不足: {len(female)} < {need_f}"

    selected_m = random.sample(male, need_m)
    selected_f = random.sample(female, need_f)

    pools = []
    for i in range(N_POOLS):
        pool_m = selected_m[i*PER_POOL_M:(i+1)*PER_POOL_M]
        pool_f = selected_f[i*PER_POOL_F:(i+1)*PER_POOL_F]
        pools.append({
            "pool_id": i,
            "male":    pool_m,
            "female":  pool_f,
            "mix":     pool_m + pool_f,
        })
        print(f"Pool {i}: 男{len(pool_m)}, 女{len(pool_f)}, mix{len(pool_m+pool_f)}")

    all_speakers = selected_m + selected_f
    result = {
        "n_pools": N_POOLS,
        "per_pool_male": PER_POOL_M,
        "per_pool_female": PER_POOL_F,
        "total_speakers": len(all_speakers),
        "all_speakers": all_speakers,
        "pools": pools,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n划分完成，保存至: {args.output}")


if __name__ == "__main__":
    main()
