"""音素预计算"""

import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def precompute_phones(features_h5_path, metadata, phone_predictor, ckpt_dir,
                     device='cuda', sample_ratio=1.0):
    """预计算所有帧的音素标签"""
    utterances = metadata['utterances']

    if sample_ratio < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(utterances) * sample_ratio)
        sampled_indices = set(random.sample(range(len(utterances)), sample_size))
        print(f"Sampled {len(sampled_indices)}/{len(utterances)}")
    else:
        sampled_indices = None

    all_phones = []

    with h5py.File(features_h5_path, 'r') as f:
        features = f['features']
        for idx, utt in enumerate(tqdm(utterances, desc="Precomputing phones")):
            if sampled_indices is not None and idx not in sampled_indices:
                continue
            feats = torch.from_numpy(features[utt['h5_start_idx']:utt['h5_end_idx']]).float().to(device)
            with torch.no_grad():
                phones = phone_predictor(feats).cpu().numpy().astype(np.int8)
            all_phones.append(phones)

    all_phones = np.concatenate(all_phones)
    ckpt_dir = Path(ckpt_dir)
    np.save(ckpt_dir / 'phones.npy', all_phones)
    print(f"Saved: {ckpt_dir / 'phones.npy'} ({len(all_phones):,} frames)")
    return all_phones
