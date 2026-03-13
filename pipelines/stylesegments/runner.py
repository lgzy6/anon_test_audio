#!/usr/bin/env python3
"""统一的风格分割管线运行器"""

import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.stylesegments.feature_extraction import run_feature_extraction
from pipelines.stylesegments.speaker_embeddings import extract_speaker_embeddings
from pipelines.stylesegments.eta_projection import compute_eta_projection
from pipelines.stylesegments.phone_precompute import precompute_phones
from pipelines.stylesegments.style_extractor import build_style_extractor
from pipelines.stylesegments.phone_clusters import build_phone_clusters
from models.phone_predictor.predictor import PhonePredictor


def run_pipeline(config_path, steps='all', sample_ratio=1.0):
    """运行完整管线"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_split = config['offline'].get('train_split', 'train-clean-100')
    split_name = train_split.replace('-', '_')

    cache_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
    ckpt_dir = Path(config['paths']['checkpoints_dir'])
    ckpt_dir.mkdir(exist_ok=True)

    steps_list = steps.split(',') if steps != 'all' else ['1', '2', '3', '4', '5', '6']

    # Step 1: WavLM特征提取（多层）
    if '1' in steps_list:
        print("\n" + "="*60)
        print("Step 1: WavLM Feature Extraction (Multi-layer)")
        print("="*60)
        metadata = run_feature_extraction(config)
    else:
        with open(cache_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

    # Step 2: 说话人嵌入提取
    if '2' in steps_list:
        print("\n" + "="*60)
        print("Step 2: Speaker Embeddings + PCA")
        print("="*60)
        librispeech_roots = [config['paths']['librispeech_root']]
        embeddings_pca, speaker_ids, utt_indices = extract_speaker_embeddings(
            metadata, librispeech_roots, ckpt_dir, device,
            pca_dim=128, sample_ratio=sample_ratio)
    else:
        embeddings_pca = np.load(ckpt_dir / 'speaker_embeddings_pca.npy')
        utt_indices = np.load(ckpt_dir / 'utt_indices.npy')

    # Step 3: 音素预计算
    if '3' in steps_list:
        print("\n" + "="*60)
        print("Step 3: Precompute Phones")
        print("="*60)
        phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

        # 为每层预计算
        layers = config['ssl'].get('layers', [6, 24])
        for layer in layers:
            layer_h5 = cache_dir / f"layer_{layer}" / "features.h5"
            if layer_h5.exists():
                print(f"\nLayer {layer}:")
                precompute_phones(layer_h5, metadata, phone_predictor,
                                ckpt_dir / f"layer_{layer}", device, sample_ratio)

    # Step 4: Eta投影计算
    if '4' in steps_list:
        print("\n" + "="*60)
        print("Step 4: Eta-WavLM Projection")
        print("="*60)
        phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

        layers = config['ssl'].get('layers', [6, 24])
        for layer in layers:
            layer_h5 = cache_dir / f"layer_{layer}" / "features.h5"
            if layer_h5.exists():
                print(f"\nLayer {layer}:")
                compute_eta_projection(
                    layer_h5, metadata, embeddings_pca, utt_indices,
                    ckpt_dir / f"layer_{layer}", device, chunk_size=2000,
                    sample_ratio=sample_ratio, phone_predictor=phone_predictor)

    # Step 5: 风格提取器构建
    if '5' in steps_list:
        print("\n" + "="*60)
        print("Step 5: Style Extractor")
        print("="*60)
        phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

        layers = config['ssl'].get('layers', [6, 24])
        for layer in layers:
            layer_dir = ckpt_dir / f"layer_{layer}"
            layer_h5 = cache_dir / f"layer_{layer}" / "features.h5"

            if not layer_h5.exists():
                continue

            print(f"\nLayer {layer}:")
            eta_ckpt = torch.load(layer_dir / 'eta_projection.pt', map_location=device)
            A_star = eta_ckpt['A_star'].to(device)
            b_star = eta_ckpt['b_star'].to(device)

            utt_to_pca = {int(idx): i for i, idx in enumerate(utt_indices)}

            def eta_generator():
                import h5py
                with h5py.File(layer_h5, 'r') as f:
                    features_ds = f['features']
                    for utt_idx, utt in enumerate(metadata['utterances']):
                        if utt_idx not in utt_to_pca:
                            continue
                        s_np = features_ds[utt['h5_start_idx']:utt['h5_end_idx']]
                        if len(s_np) < 5:
                            continue
                        s = torch.from_numpy(s_np).float().to(device)
                        d = torch.from_numpy(embeddings_pca[utt_to_pca[utt_idx]]).float().to(device)
                        with torch.no_grad():
                            eta = (s - (d @ A_star + b_star).unsqueeze(0)).cpu().numpy()
                            phones = phone_predictor(s).cpu().numpy()
                        yield eta, phones, utt['speaker_id']

            build_style_extractor(eta_generator(), layer_dir, pca_dim=64)

    # Step 6: Phone聚类
    if '6' in steps_list:
        print("\n" + "="*60)
        print("Step 6: Phone Clusters")
        print("="*60)
        phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)

        layers = config['ssl'].get('layers', [6, 24])
        for layer in layers:
            layer_h5 = cache_dir / f"layer_{layer}" / "features.h5"
            layer_dir = ckpt_dir / f"layer_{layer}"

            if not layer_h5.exists():
                continue

            print(f"\nLayer {layer}:")
            phone_clusters = build_phone_clusters(
                layer_h5, metadata['utterances'], phone_predictor,
                n_clusters=10, device=device, use_cached_phones=True,
                ckpt_dir=layer_dir)

            import pickle
            with open(layer_dir / 'phone_clusters.pkl', 'wb') as f:
                pickle.dump({
                    'phone_clusters': phone_clusters,
                    'n_clusters': 10,
                    'metadata': metadata
                }, f)
            print(f"Saved: {layer_dir / 'phone_clusters.pkl'}")

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--steps', type=str, default='all',
                       help='Steps to run: all or comma-separated (e.g., 1,2,3)')
    parser.add_argument('--sample-ratio', type=float, default=1.0)
    args = parser.parse_args()

    run_pipeline(args.config, args.steps, args.sample_ratio)
