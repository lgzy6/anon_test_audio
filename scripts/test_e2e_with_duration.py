#!/usr/bin/env python3
"""
端到端匿名化测试 (带 Duration 适配)

关键改进：
1. 加载 duration predictor
2. 按音素分段并预测时长
3. 通过插值调整帧数后再进行 kNN 匹配
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


def load_models(device='cuda', dur_weight=0.5):
    """加载所有必要的模型"""
    print("加载模型...")
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    wavlm = WavLMSSLExtractor(
        ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
        layer=[6, 24],
        device=device
    )

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/ecapa",
        run_opts={"device": device}
    )

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

    # 加载 duration predictor
    duration_predictor = PhonePredictor.load(
        str(ckpt_dir / 'duration_decoder.pt'),
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

    print("✓ 所有模型加载完成 (含 duration predictor)")

    return {
        'wavlm': wavlm,
        'ecapa': ecapa,
        'speaker_pca': speaker_pca,
        'A_star': A_star,
        'b_star': b_star,
        'phone_predictor': phone_predictor,
        'duration_predictor': duration_predictor,
        'style_extractor': style_extractor,
        'vocoder': vocoder,
        'dur_weight': dur_weight
    }


def load_target_pool(device='cuda'):
    """加载目标池数据"""
    print("\n加载目标池...")
    ckpt_dir = BASE_DIR / 'checkpoints' / 'large'
    cache_dir = BASE_DIR / 'cache' / 'large' / 'features' / 'wavlm' / 'train_clean_360'

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
        wavlm_feats = models['wavlm'](waveform)  # [2, 1, T, D]
        wavlm_vc_feats = wavlm_feats[0].squeeze(0).cpu().numpy()
        wavlm_phone_feats = wavlm_feats[1].squeeze(0)

    spk_emb = models['ecapa'].encode_batch(waveform).squeeze().cpu().numpy()
    spk_emb_pca = models['speaker_pca'].transform(spk_emb.reshape(1, -1))[0]

    s = torch.from_numpy(wavlm_vc_feats).float().to(device)
    d = torch.from_numpy(spk_emb_pca).float().to(device)
    eta = (s - (d @ models['A_star'] + models['b_star']).unsqueeze(0)).cpu().numpy()

    with torch.no_grad():
        phones = models['phone_predictor'](wavlm_phone_feats).cpu().numpy()

    style_vec = models['style_extractor'].extract_utterance_style(eta, phones)
    print(f"✓ WavLM: {wavlm_vc_feats.shape}, style: {style_vec.shape}")

    return {
        'wavlm': wavlm_vc_feats,
        'phones': phones,
        'style': style_vec
    }


def select_target_utterance(source_style, source_gender, target_pool, enforce_same_gender=True):
    """选择目标话语"""
    print(f"\n选择目标话语...")

    if enforce_same_gender:
        valid_idx = [i for i, g in enumerate(target_pool['genders']) if g == source_gender]
        if not valid_idx:
            print(f"警告: 未找到同性别目标，使用所有目标")
            valid_idx = list(range(len(target_pool['styles'])))
    else:
        valid_idx = list(range(len(target_pool['styles'])))

    valid_styles = target_pool['styles'][valid_idx]
    sims = np.dot(valid_styles, source_style) / (
        np.linalg.norm(valid_styles, axis=1) * np.linalg.norm(source_style) + 1e-8
    )
    best_local_idx = np.argmax(sims)
    target_idx = valid_idx[best_local_idx]

    target_gender = target_pool['genders'][target_idx]
    print(f"✓ 选择话语 {target_idx}, 性别: {target_gender}, 相似度: {sims[best_local_idx]:.3f}")
    return target_idx


def anonymize_with_duration(source_wavlm, source_phones, target_idx, target_pool, models, device='cuda', k=4):
    """使用 duration 适配的 kNN 匹配"""
    print(f"\nkNN 检索 (with duration, k={k})...")

    src_feats = torch.from_numpy(source_wavlm).float().to(device)
    phone_clusters = target_pool['phone_clusters'][target_idx]
    dur_weight = models['dur_weight']

    # 1. 获取连续音素段及其实际时长
    unique_phones, phone_durations = [], []
    current_phone = source_phones[0]
    current_count = 1

    for i in range(1, len(source_phones)):
        if source_phones[i] == current_phone:
            current_count += 1
        else:
            unique_phones.append(current_phone)
            phone_durations.append(current_count)
            current_phone = source_phones[i]
            current_count = 1
    unique_phones.append(current_phone)
    phone_durations.append(current_count)

    phones_tensor = torch.tensor(unique_phones, dtype=torch.long, device=device).unsqueeze(0)
    durations_tensor = torch.tensor(phone_durations, dtype=torch.float32, device=device).unsqueeze(0)

    # 2. 预测时长并插值
    with torch.no_grad():
        pred_durations = models['duration_predictor'].model(phones_tensor).squeeze(0)

    interpolated_durations = dur_weight * pred_durations + (1 - dur_weight) * durations_tensor.squeeze(0)
    n_frames = torch.round(interpolated_durations).clamp(min=1).long()

    # 3. 调整源特征帧数
    adjusted_feats = []
    adjusted_phones = []
    feat_idx_start = 0

    for seg_idx, (phone_id, orig_dur, new_dur) in enumerate(zip(unique_phones, phone_durations, n_frames)):
        feat_idx_end = feat_idx_start + orig_dur - 1
        indices = torch.linspace(feat_idx_start, feat_idx_end, new_dur.item(), dtype=torch.long, device=device)
        adjusted_feats.append(src_feats[indices])
        adjusted_phones.append(torch.full((new_dur.item(),), phone_id, dtype=torch.long, device=device))
        feat_idx_start = feat_idx_end + 1

    src_feats_adj = torch.cat(adjusted_feats, dim=0)
    src_phones_adj = torch.cat(adjusted_phones, dim=0)

    # 4. Phone-level kNN 匹配
    h_anon = torch.zeros_like(src_feats_adj)
    unique_phone_ids = torch.unique(src_phones_adj)

    for phone_id in tqdm(unique_phone_ids, desc="kNN"):
        if phone_id == 0 or int(phone_id) not in phone_clusters:
            mask = src_phones_adj == phone_id
            h_anon[mask] = src_feats_adj[mask]
            continue

        tgt_clusters = torch.from_numpy(phone_clusters[int(phone_id)]).float().to(device)
        mask = src_phones_adj == phone_id

        src_batch = torch.nn.functional.normalize(src_feats_adj[mask], dim=-1)
        tgt_batch = torch.nn.functional.normalize(tgt_clusters, dim=-1)

        cos_sim = torch.mm(src_batch, tgt_batch.T)
        max_indices = torch.argmax(cos_sim, dim=1)

        h_anon[mask] = tgt_clusters[max_indices]

    print(f"✓ 匿名化完成: {src_feats.shape} -> {h_anon.shape}")
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
    parser.add_argument('--dur_weight', type=float, default=0.5, help='Duration 预测权重 (0-1)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("端到端匿名化 (带 Duration 适配)")
    print("="*60)

    models = load_models(device, dur_weight=args.dur_weight)
    target_pool = load_target_pool(device)

    if target_pool['phone_clusters'] is None:
        print("\n错误: 未找到 phone_clusters.pkl")
        return

    source = extract_source_features(args.audio, models, device)
    target_idx = select_target_utterance(source['style'], args.gender, target_pool)

    h_anon = anonymize_with_duration(
        source['wavlm'], source['phones'], target_idx, target_pool, models, device, k=args.k
    )

    synthesize_audio(h_anon, models['vocoder'], args.output)

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)


if __name__ == '__main__':
    main()
