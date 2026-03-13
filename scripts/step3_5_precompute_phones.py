#!/usr/bin/env python3
"""
步骤 3.5：预计算音素序列（避免重复推理）

输入：
  - cache/features/wavlm/{split}/features.h5
  - cache/features/wavlm/{split}/metadata.json
  - checkpoints/phone_decoder.pt

输出：
  - checkpoints/phones.npy (稠密存储，每帧一个 phone_id)
  - checkpoints/phones_metadata.json (索引信息)
"""

import sys
import json
import numpy as np
import torch
import h5py
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor

BASE_DIR = Path(__file__).parent.parent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='Sample ratio (0.0-1.0)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取配置
    config_path = Path(args.config) if args.config else BASE_DIR / 'configs' / 'base.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_split = config['offline'].get('train_split', 'train-clean-100')
        split_name = train_split.replace('-', '_')
        cache_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
        ckpt_dir = Path(config['paths']['checkpoints_dir'])
    else:
        cache_dir = BASE_DIR / 'cache' / 'features' / 'wavlm'
        ckpt_dir = BASE_DIR / 'checkpoints'

    print("="*60)
    print("Step 3.5: 预计算音素序列")
    print("="*60)

    # 加载数据
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']

    # 采样
    if args.sample_ratio < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(utterances) * args.sample_ratio)
        sampled_indices = set(random.sample(range(len(utterances)), sample_size))
        print(f"Sampled {len(sampled_indices)}/{len(utterances)} utterances")
    else:
        sampled_indices = None

    # 加载音素预测器
    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

    print(f"\n处理话语...")

    all_phones = []
    utt_boundaries = []

    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features = f['features']

        for idx, utt in enumerate(tqdm(utterances, desc="预测音素")):
            if sampled_indices is not None and idx not in sampled_indices:
                continue
            start, end = utt['h5_start_idx'], utt['h5_end_idx']
            feats = torch.from_numpy(features[start:end]).float().to(device)

            with torch.no_grad():
                phones = phone_predictor(feats).cpu().numpy().astype(np.int8)

            all_phones.append(phones)
            utt_boundaries.append([len(all_phones) - 1, 0, len(phones)])

    # 拼接并保存
    all_phones = np.concatenate(all_phones)
    utt_boundaries = np.array(utt_boundaries)

    np.save(ckpt_dir / 'phones.npy', all_phones)
    np.save(ckpt_dir / 'phones_boundaries.npy', utt_boundaries)

    print(f"\n✓ 保存: {ckpt_dir / 'phones.npy'}")
    print(f"  总帧数: {len(all_phones):,}")
    print(f"  文件大小: {len(all_phones) / 1024 / 1024:.1f} MB")
    print("="*60)


if __name__ == '__main__':
    main()
