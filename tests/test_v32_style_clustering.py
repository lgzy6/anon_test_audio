#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 Test B: 风格聚类可视化 (V3 优化版)

验证目标:
1. 使用 ExpandedSubspaceProjector 提取 H_style
2. 提取句子级风格 Embedding
3. 使用 t-SNE 可视化聚类结构
4. 验证风格与说话人身份的解耦程度

成功标准:
- 能看到清晰的聚类结构
- 聚类不应与说话人身份强相关 (解耦验证)
"""

import sys
import numpy as np
import torch
import h5py
import json
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_v32_disentanglement_v3 import ExpandedSubspaceProjector, load_test_data


def extract_utterance_embeddings(features_h5_path: str,
                                 metadata_path: str,
                                 projector: ExpandedSubspaceProjector,
                                 max_utterances: int = 500):
    """
    提取句子级风格 Embedding

    对每个 utterance:
    1. 投影到风格子空间
    2. 取均值作为句子 embedding
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    utterances = metadata['utterances'][:max_utterances]
    print(f"Processing {len(utterances)} utterances...")

    embeddings = []
    speaker_ids = []
    genders = []

    with h5py.File(features_h5_path, 'r') as f:
        features = f['features']

        for utt in utterances:
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            h = features[s:e][:]

            # 投影到风格子空间
            h_style = projector.project_to_style(h)

            # 句子级 embedding (均值池化)
            emb = h_style.mean(axis=0)
            embeddings.append(emb)
            speaker_ids.append(utt['speaker_id'])
            genders.append(utt.get('gender', 'unknown'))

    return np.array(embeddings), speaker_ids, genders


def visualize_style_clusters(embeddings: np.ndarray,
                            speaker_ids: list,
                            genders: list,
                            n_clusters: int = 8,
                            save_path: str = None):
    """可视化风格聚类"""

    # t-SNE 降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # KMeans 聚类
    print(f"Running KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    # 绘图 (3个子图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: 按聚类着色
    scatter1 = axes[0].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=cluster_ids, cmap='tab10', alpha=0.6, s=20
    )
    axes[0].set_title(f'Style Clusters (K={n_clusters})')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')

    # 图2: 按说话人着色
    unique_speakers = list(set(speaker_ids))
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}
    speaker_ids_num = [speaker_to_id[s] for s in speaker_ids]

    scatter2 = axes[1].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=speaker_ids_num, cmap='tab20', alpha=0.6, s=20
    )
    axes[1].set_title(f'Colored by Speaker ({len(unique_speakers)} speakers)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    # 图3: 按性别着色
    gender_map = {'m': 0, 'f': 1, 'male': 0, 'female': 1, 'unknown': 2}
    gender_ids = [gender_map.get(g.lower(), 2) for g in genders]

    scatter3 = axes[2].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=gender_ids, cmap='coolwarm', alpha=0.6, s=20
    )
    axes[2].set_title('Colored by Gender')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return cluster_ids, embeddings_2d


def compute_cluster_metrics(cluster_ids, speaker_ids, genders):
    """计算聚类质量指标"""
    # 说话人纯度
    unique_speakers = list(set(speaker_ids))
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}
    speaker_ids_num = np.array([speaker_to_id[s] for s in speaker_ids])

    # 性别映射
    gender_map = {'m': 0, 'f': 1, 'male': 0, 'female': 1, 'unknown': 2}
    gender_ids = np.array([gender_map.get(g.lower(), 2) for g in genders])

    # ARI/NMI vs speaker (低 = 好，说明解耦)
    ari_speaker = adjusted_rand_score(speaker_ids_num, cluster_ids)
    nmi_speaker = normalized_mutual_info_score(speaker_ids_num, cluster_ids)

    # ARI/NMI vs gender
    ari_gender = adjusted_rand_score(gender_ids, cluster_ids)
    nmi_gender = normalized_mutual_info_score(gender_ids, cluster_ids)

    return {
        'ari_speaker': ari_speaker,
        'nmi_speaker': nmi_speaker,
        'ari_gender': ari_gender,
        'nmi_gender': nmi_gender,
    }


def main():
    """主测试流程"""
    print("="*60)
    print("DS-SAMM-Anon v3.2 Test B: Style Clustering (V3)")
    print("="*60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 配置路径
    config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'
    cache_dir = Path(__file__).parent.parent / 'cache'
    features_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'
    metadata_json = cache_dir / 'features' / 'wavlm' / 'metadata.json'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'v32_tests'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据文件
    if not features_h5.exists():
        raise FileNotFoundError(f"Features not found: {features_h5}")
    if not metadata_json.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_json}")

    # Step 1: 加载数据并训练投影器
    print("\n[Step 1] Loading data and training projector...")
    features, phones = load_test_data(str(config_path), max_samples=100000)

    # 使用 ExpandedSubspaceProjector (dim=100, 效果最佳)
    projector = ExpandedSubspaceProjector(
        n_phones=41,
        feature_dim=features.shape[1],
        n_components=100
    )
    projector.fit(features, phones, max_per_class=2000)

    # Step 2: 提取句子级风格 embedding
    print("\n[Step 2] Extracting utterance-level style embeddings...")
    embeddings, speaker_ids, genders = extract_utterance_embeddings(
        str(features_h5),
        str(metadata_json),
        projector,
        max_utterances=500
    )
    print(f"  Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # Step 3: 可视化聚类
    print("\n[Step 3] Visualizing clusters...")
    n_clusters = 8
    cluster_ids, embeddings_2d = visualize_style_clusters(
        embeddings, speaker_ids, genders,
        n_clusters=n_clusters,
        save_path=str(output_dir / 'style_clusters_v3.png')
    )

    # Step 4: 计算聚类质量指标
    print("\n[Step 4] Computing cluster metrics...")
    metrics = compute_cluster_metrics(cluster_ids, speaker_ids, genders)

    # 总结
    print("\n" + "="*60)
    print("SUMMARY: Style Clustering Results")
    print("="*60)
    print(f"  Utterances: {len(embeddings)}")
    print(f"  Speakers: {len(set(speaker_ids))}")
    print(f"  Clusters: {n_clusters}")
    print()
    print("Disentanglement Metrics (lower = better decoupling):")
    print(f"  ARI vs Speaker: {metrics['ari_speaker']:.3f}")
    print(f"  NMI vs Speaker: {metrics['nmi_speaker']:.3f}")
    print(f"  ARI vs Gender:  {metrics['ari_gender']:.3f}")
    print(f"  NMI vs Gender:  {metrics['nmi_gender']:.3f}")
    print()

    if metrics['ari_speaker'] < 0.1:
        print("✓ SUCCESS: Style clusters are well decoupled from speaker identity")
    elif metrics['ari_speaker'] < 0.3:
        print("◐ PARTIAL: Some decoupling achieved")
    else:
        print("✗ WARNING: Style clusters correlate with speaker identity")

    print(f"\nVisualization saved to: {output_dir / 'style_clusters_v3.png'}")

    return metrics


if __name__ == '__main__':
    main()
