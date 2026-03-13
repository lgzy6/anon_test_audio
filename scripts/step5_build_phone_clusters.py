#!/usr/bin/env python3
"""
Step 5: Phone-level 聚类预处理 (借鉴 spkanon)

为每个目标话语的每个音素构建 K-Means 簇中心，加速检索
"""

import sys
import argparse
import numpy as np
import torch
import h5py
import pickle
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent


def load_phone_predictor(ckpt_dir, device='cuda'):
    """加载音素预测器"""
    from models.phone_predictor.predictor import PhonePredictor
    ckpt_path = ckpt_dir / 'phone_decoder.pt'
    return PhonePredictor.load(str(ckpt_path), device=device)


def build_phone_clusters(features_path, utterances, phone_predictor, n_clusters=10, device='cuda', use_cached_phones=True, ckpt_dir=None):
    """为每个话语构建 phone-level 聚类（优化版）"""
    print(f"\n构建 phone-level 聚类 (k={n_clusters})...")

    # 尝试加载缓存的 phones
    cached_phones = None
    if use_cached_phones and ckpt_dir:
        phones_path = ckpt_dir / 'phones.npy'
        if phones_path.exists():
            print(f"  加载缓存的 phones: {phones_path}")
            cached_phones = np.load(phones_path)

    phone_clusters = []

    with h5py.File(features_path, 'r') as f:
        features = f['features']

        for utt_idx, utt_info in enumerate(tqdm(utterances, desc="聚类")):
            start, end = utt_info['h5_start_idx'], utt_info['h5_end_idx']

            # 优化 C：特征留在 CPU
            feats_np = features[start:end].astype(np.float32)

            # 获取 phones
            if cached_phones is not None:
                phones = cached_phones[start:end]
            else:
                feats_gpu = torch.from_numpy(feats_np).to(device)
                with torch.no_grad():
                    phones = phone_predictor(feats_gpu).cpu().numpy()

            # 按 phone 分组并聚类
            utt_clusters = {}
            for phone_id in np.unique(phones):
                # 优化 E：跳过 silence/blank
                if phone_id == 0:
                    continue

                phone_mask = (phones == phone_id)
                phone_feats = feats_np[phone_mask]

                if len(phone_feats) < n_clusters:
                    utt_clusters[int(phone_id)] = phone_feats.astype(np.float32)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
                    kmeans.fit(phone_feats)
                    utt_clusters[int(phone_id)] = kmeans.cluster_centers_.astype(np.float32)

            phone_clusters.append(utt_clusters)

    print(f"✓ 完成 {len(phone_clusters)} 个话语的聚类")
    return phone_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--n-clusters', type=int, default=10, help='每个 phone 的簇数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='Sample ratio (0.0-1.0)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Step 5: Phone-level 聚类")
    print("="*60)

    # 读取配置
    import yaml
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

    # 加载数据
    features_path = cache_dir / 'features.h5'

    import json
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    utterances = metadata['utterances']

    # 采样
    if args.sample_ratio < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(utterances) * args.sample_ratio)
        sampled_indices = set(random.sample(range(len(utterances)), sample_size))
        utterances = [utt for i, utt in enumerate(utterances) if i in sampled_indices]
        print(f"Sampled {len(utterances)}/{len(metadata['utterances'])} utterances")

    # 加载音素预测器
    phone_predictor = load_phone_predictor(ckpt_dir, device)

    # 构建聚类
    phone_clusters = build_phone_clusters(
        features_path, utterances, phone_predictor,
        n_clusters=args.n_clusters, device=device,
        use_cached_phones=True, ckpt_dir=ckpt_dir
    )

    # 保存
    output_path = ckpt_dir / 'phone_clusters.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'phone_clusters': phone_clusters,
            'n_clusters': args.n_clusters,
            'metadata': metadata
        }, f)

    print(f"\n✓ 保存到 {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
