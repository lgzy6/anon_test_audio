"""Phone-level聚类"""

import numpy as np
import h5py
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


def build_phone_clusters(features_path, utterances, phone_predictor, n_clusters=10,
                        device='cuda', use_cached_phones=True, ckpt_dir=None):
    """为每个话语构建phone-level聚类"""
    print(f"Building phone clusters (k={n_clusters})...")

    cached_phones = None
    if use_cached_phones and ckpt_dir:
        phones_path = Path(ckpt_dir) / 'phones.npy'
        if phones_path.exists():
            print(f"Loading cached phones: {phones_path}")
            cached_phones = np.load(phones_path)

    phone_clusters = []

    with h5py.File(features_path, 'r') as f:
        features = f['features']
        for utt_idx, utt_info in enumerate(tqdm(utterances, desc="Clustering")):
            start, end = utt_info['h5_start_idx'], utt_info['h5_end_idx']
            feats_np = features[start:end].astype(np.float32)

            if cached_phones is not None:
                phones = cached_phones[start:end]
            else:
                import torch
                feats_gpu = torch.from_numpy(feats_np).to(device)
                with torch.no_grad():
                    phones = phone_predictor(feats_gpu).cpu().numpy()

            utt_clusters = {}
            for phone_id in np.unique(phones):
                if phone_id == 0:
                    continue
                phone_feats = feats_np[phones == phone_id]
                if len(phone_feats) < n_clusters:
                    utt_clusters[int(phone_id)] = phone_feats.astype(np.float32)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
                    utt_clusters[int(phone_id)] = kmeans.fit(phone_feats).cluster_centers_.astype(np.float32)
            phone_clusters.append(utt_clusters)

    print(f"Completed {len(phone_clusters)} utterances")
    return phone_clusters
