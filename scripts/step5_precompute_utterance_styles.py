#!/usr/bin/env python3
"""
步骤 5：为训练集所有话语预计算风格向量

输入：
  - cache/features/wavlm/{split}/features.h5
  - cache/features/wavlm/{split}/metadata.json
  - checkpoints/eta_projection.pt (Step 2)
  - checkpoints/speaker_embeddings_pca.npy (Step 1)
  - checkpoints/style_extractor.pkl (Step 4)

输出：
  - checkpoints/utterance_styles.npz
    {
      'styles': (N_utt, 64),
      'speaker_ids': (N_utt,),
      'genders': (N_utt,),
      'utt_boundaries': (N_utt, 2)
    }
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取配置
    config_path = BASE_DIR / 'configs' / 'base.yaml'
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

    print(f"Cache dir: {cache_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")

    # 加载模型和数据
    from scripts.step4_build_style_extractor import SegmentStyleExtractor
    extractor = SegmentStyleExtractor.load(ckpt_dir / 'style_extractor.pkl')

    eta_ckpt = torch.load(ckpt_dir / 'eta_projection.pt', map_location=device)
    A_star = eta_ckpt['A_star'].to(device)
    b_star = eta_ckpt['b_star'].to(device)

    embeddings_pca = np.load(ckpt_dir / 'speaker_embeddings_pca.npy')
    utt_indices = np.load(ckpt_dir / 'utt_indices.npy')
    utt_to_pca = {int(idx): i for i, idx in enumerate(utt_indices)}

    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    utterances = metadata['utterances']

    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

    print(f"\nProcessing {len(utterances)} utterances...")

    styles = []
    speaker_ids = []
    genders = []
    utt_boundaries = []

    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features_ds = f['features']

        for utt_idx, utt in enumerate(tqdm(utterances, desc="Computing styles")):
            if utt_idx not in utt_to_pca:
                continue

            pca_idx = utt_to_pca[utt_idx]
            s_np = features_ds[utt['h5_start_idx']:utt['h5_end_idx']]

            if len(s_np) == 0:
                continue

            s = torch.from_numpy(s_np).float().to(device)
            d = torch.from_numpy(embeddings_pca[pca_idx]).float().to(device)

            with torch.no_grad():
                eta = (s - (d @ A_star + b_star).unsqueeze(0)).cpu().numpy()
                phones = phone_predictor(s).cpu().numpy()

            style_vec = extractor.extract_utterance_style(eta, phones)

            styles.append(style_vec)
            speaker_ids.append(utt['speaker_id'])
            genders.append(utt.get('gender', 'unknown'))
            utt_boundaries.append([utt['h5_start_idx'], utt['h5_end_idx']])

    styles = np.stack(styles)
    speaker_ids = np.array(speaker_ids)
    genders = np.array(genders)
    utt_boundaries = np.array(utt_boundaries)

    # 保存
    save_path = ckpt_dir / 'utterance_styles.npz'
    np.savez(save_path,
             styles=styles,
             speaker_ids=speaker_ids,
             genders=genders,
             utt_boundaries=utt_boundaries)

    print(f"\nSaved: {save_path}")
    print(f"  Utterances: {len(styles)}")
    print(f"  Speakers: {len(np.unique(speaker_ids))}")
    print(f"  Style dim: {styles.shape[1]}")


if __name__ == '__main__':
    main()
