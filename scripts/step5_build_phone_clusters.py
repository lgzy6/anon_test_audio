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


def load_phone_predictor(device='cuda'):
    """加载音素预测器"""
    from models.phone_predictor.predictor import PhonePredictor
    ckpt_path = BASE_DIR / 'checkpoints' / 'phone_decoder.pt'
    return PhonePredictor.load(str(ckpt_path), device=device)


def build_phone_clusters(features_path, metadata, phone_predictor, n_clusters=10, device='cuda'):
    """为每个话语构建 phone-level 聚类"""
    print(f"\n构建 phone-level 聚类 (k={n_clusters})...")

    phone_clusters = []

    with h5py.File(features_path, 'r') as f:
        features = f['features']

        for utt_idx, utt_info in enumerate(tqdm(metadata, desc="聚类")):
            start, end = utt_info['start_idx'], utt_info['end_idx']
            feats = torch.from_numpy(features[start:end]).float().to(device)

            # 预测音素
            with torch.no_grad():
                phones = phone_predictor(feats).cpu().numpy()

            # 按 phone 分组并聚类
            utt_clusters = {}
            for phone_id in np.unique(phones):
                phone_mask = (phones == phone_id)
                phone_feats = feats[phone_mask].cpu().numpy()

                if len(phone_feats) < n_clusters:
                    # 不足 k 个，直接使用原始帧
                    utt_clusters[int(phone_id)] = phone_feats
                else:
                    # K-Means 聚类
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
                    kmeans.fit(phone_feats)
                    utt_clusters[int(phone_id)] = kmeans.cluster_centers_

            phone_clusters.append(utt_clusters)

    print(f"✓ 完成 {len(phone_clusters)} 个话语的聚类")
    return phone_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clusters', type=int, default=10, help='每个 phone 的簇数')
    parser.add_argument('--output', type=str, default='checkpoints/phone_clusters.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Step 5: Phone-level 聚类")
    print("="*60)

    # 加载数据
    cache_dir = BASE_DIR / 'cache' / 'features' / 'wavlm' / 'train_clean_360'
    features_path = cache_dir / 'features.h5'

    import json
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # 加载音素预测器
    phone_predictor = load_phone_predictor(device)

    # 构建聚类
    phone_clusters = build_phone_clusters(
        features_path, metadata, phone_predictor,
        n_clusters=args.n_clusters, device=device
    )

    # 保存
    output_path = BASE_DIR / args.output
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
