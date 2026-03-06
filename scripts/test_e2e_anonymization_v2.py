#!/usr/bin/env python3
"""
端到端匿名化测试 (优化版)

结合 spkanon 的 phone-level 聚类 + Top-k 平均

流程：
1. 风格检索：选择风格最相似的目标话语
2. Phone-level kNN：使用预聚类的簇中心加速匹配
3. Top-k 平均：保持音频平滑
"""

import sys
import argparse
import numpy as np
import torch
import torchaudio
import json
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent


def load_models(device='cuda'):
    """加载所有必要的模型"""
    print("加载模型...")
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    wavlm = WavLMSSLExtractor(
        ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
        layer=15,
        device=device
    )

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/ecapa",
        run_opts={"device": device}
    )

    import pickle
    with open(ckpt_dir / 'speaker_pca_model.pkl', 'rb') as f:
        speaker_pca = pickle.load(f)

    eta_ckpt = torch.load(ckpt_dir / 'eta_projection.pt', map_location=device)
    A_star = eta_ckpt['A_star'].to(device)
    b_star = eta_ckpt['b_star'].to(device)

    from models.phone_predictor.predictor import PhonePredictor
    phone_predictor = PhonePredictor.load(
        str(ckpt_dir / 'phone_decoder.pt'),
        device=device
    )

    from scripts.step4_build_style_extractor import SegmentStyleExtractor
    style_extractor = SegmentStyleExtractor.load(
        ckpt_dir / 'style_extractor.pkl'
    )

    from models.vocoder.hifigan import HiFiGAN
    vocoder = HiFiGAN.load(
        checkpoint_path=str(ckpt_dir / 'hifigan.pt'),
        device=device
    )

    print("✓ 所有模型加载完成")

    return {
        'wavlm': wavlm,
        'ecapa': ecapa,
        'speaker_pca': speaker_pca,
        'A_star': A_star,
        'b_star': b_star,
        'phone_predictor': phone_predictor,
        'style_extractor': style_extractor,
        'vocoder': vocoder
    }


def load_target_pool(device='cuda'):
    """加载目标池数据"""
    print("\n加载目标池...")

    ckpt_dir = BASE_DIR / 'checkpoints'
    cache_dir = BASE_DIR / 'cache' / 'features' / 'wavlm' / 'train_clean_360'

    styles_data = np.load(ckpt_dir / 'utterance_styles.npz')

    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    pool = {
        'styles': styles_data['styles'],
        'speaker_ids': styles_data['speaker_ids'],
        'genders': styles_data['genders'],
        'boundaries': styles_data['utt_boundaries'],
        'metadata': metadata,
        'features_path': cache_dir / 'features.h5',
        'phone_clusters': None
    }

    # 尝试加载 phone-level 聚类
    phone_clusters_path = ckpt_dir / 'phone_clusters.pkl'
    if phone_clusters_path.exists():
        with open(phone_clusters_path, 'rb') as f:
            data = pickle.load(f)
            pool['phone_clusters'] = data['phone_clusters']
        print(f"✓ 目标池: {len(styles_data['styles'])} 话语 (含 phone 聚类)")
    else:
        print(f"✓ 目标池: {len(styles_data['styles'])} 话语 (无 phone 聚类)")

    return pool


def extract_source_features(audio_path, models, device='cuda'):
    """提取源音频特征"""
    print(f"\n提取源音频特征: {audio_path}")

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    with torch.no_grad():
        wavlm_feats = models['wavlm'](waveform).squeeze(0).cpu().numpy()

    spk_emb = models['ecapa'].encode_batch(waveform).squeeze().cpu().numpy()
    spk_emb_pca = models['speaker_pca'].transform(spk_emb.reshape(1, -1))[0]

    s = torch.from_numpy(wavlm_feats).float().to(device)
    d = torch.from_numpy(spk_emb_pca).float().to(device)
    eta = (s - (d @ models['A_star'] + models['b_star']).unsqueeze(0)).cpu().numpy()

    with torch.no_grad():
        phones = models['phone_predictor'](s).cpu().numpy()

    style_vec = models['style_extractor'].extract_utterance_style(eta, phones)

    print(f"✓ WavLM: {wavlm_feats.shape}, style: {style_vec.shape}")

    return {
        'wavlm': wavlm_feats,
        'phones': phones,
        'style': style_vec
    }


def select_target_utterance(source_style, source_gender, target_pool):
    """选择目标话语"""
    print(f"\n选择目标话语...")

    valid_idx = [i for i, g in enumerate(target_pool['genders']) if g == source_gender]
    if not valid_idx:
        valid_idx = list(range(len(target_pool['styles'])))

    valid_styles = target_pool['styles'][valid_idx]
    sims = np.dot(valid_styles, source_style) / (
        np.linalg.norm(valid_styles, axis=1) * np.linalg.norm(source_style) + 1e-8
    )
    best_local_idx = np.argmax(sims)
    target_idx = valid_idx[best_local_idx]

    print(f"✓ 选择话语 {target_idx}, 相似度: {sims[best_local_idx]:.3f}")

    return target_idx


def anonymize_with_phone_clusters(source_wavlm, source_phones, target_idx, target_pool, device='cuda', k=4):
    """使用 phone-level 聚类加速 kNN"""
    print(f"\nkNN 检索 (phone-clustered, k={k})...")

    src_feats = torch.from_numpy(source_wavlm).float().to(device)
    phone_clusters = target_pool['phone_clusters'][target_idx]

    h_anon = torch.zeros_like(src_feats)
    unique_phones = np.unique(source_phones)

    for phone_id in tqdm(unique_phones, desc="kNN"):
        src_mask = torch.from_numpy(source_phones == phone_id).to(device)

        if phone_id == 0 or int(phone_id) not in phone_clusters:
            h_anon[src_mask] = src_feats[src_mask]
            continue

        tgt_clusters = torch.from_numpy(phone_clusters[int(phone_id)]).float().to(device)

        src_batch = torch.nn.functional.normalize(src_feats[src_mask], dim=-1)
        tgt_batch = torch.nn.functional.normalize(tgt_clusters, dim=-1)

        cos_sim = torch.mm(src_batch, tgt_batch.T)
        actual_k = min(k, tgt_clusters.shape[0])
        topk_indices = cos_sim.topk(k=actual_k, dim=1)[1]

        selected = tgt_clusters[topk_indices]
        h_anon[src_mask] = selected.mean(dim=1)

    print(f"✓ 匿名化完成: {h_anon.shape}")
    return h_anon.cpu().numpy()


def synthesize_audio(h_anon, vocoder, output_path):
    """合成音频"""
    print(f"\n合成音频: {output_path}")

    h_anon_tensor = torch.from_numpy(h_anon).unsqueeze(0).to(vocoder.device)

    with torch.no_grad():
        waveform = vocoder(h_anon_tensor).squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(output_path), waveform.cpu(), 16000)
    print(f"✓ 保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gender', type=str, default='f', choices=['f', 'm'])
    parser.add_argument('--k', type=int, default=4, help='Top-k 平均')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("端到端匿名化 (优化版)")
    print("="*60)

    models = load_models(device)
    target_pool = load_target_pool(device)

    if target_pool['phone_clusters'] is None:
        print("\n警告: 未找到 phone_clusters.pkl，请先运行:")
        print("  python scripts/step5_build_phone_clusters.py")
        return

    source = extract_source_features(args.audio, models, device)
    target_idx = select_target_utterance(source['style'], args.gender, target_pool)

    h_anon = anonymize_with_phone_clusters(
        source['wavlm'], source['phones'], target_idx, target_pool, device, k=args.k
    )

    synthesize_audio(h_anon, models['vocoder'], args.output)

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)


if __name__ == '__main__':
    main()

