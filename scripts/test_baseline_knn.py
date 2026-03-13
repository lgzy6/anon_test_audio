#!/usr/bin/env python3
"""
验证实验：Private kNN-VC 基线
- 跳过风格检索
- 使用目标说话人的全部原始 WavLM 帧
- k=1 最近邻（不做平均）
"""

import sys
import argparse
import numpy as np
import torch
import torchaudio
import json
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_DIR = Path(__file__).parent.parent


def load_models(device='cuda'):
    """加载必要模型"""
    print("加载模型...")
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    wavlm = WavLMSSLExtractor(
        ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
        layer=15,
        device=device
    )

    from models.phone_predictor.predictor import PhonePredictor
    phone_predictor = PhonePredictor.load(
        str(ckpt_dir / 'phone_decoder.pt'),
        device=device
    )

    from models.vocoder.hifigan import HiFiGAN
    vocoder = HiFiGAN.load(
        checkpoint_path=str(ckpt_dir / 'hifigan.pt'),
        device=device
    )

    print("✓ 模型加载完成")
    return {'wavlm': wavlm, 'phone_predictor': phone_predictor, 'vocoder': vocoder}


def load_target_speaker(speaker_id, device='cuda'):
    """加载目标说话人的全部帧"""
    print(f"\n加载目标说话人 {speaker_id}...")

    cache_dir = BASE_DIR / 'cache' / 'large' / 'features' / 'wavlm' / 'train_clean_360'

    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # 筛选该说话人的所有话语（确保类型匹配）
    speaker_utts = [u for u in metadata['utterances'] if str(u['speaker_id']) == str(speaker_id)]

    if not speaker_utts:
        available = sorted(set(str(u['speaker_id']) for u in metadata['utterances']))[:10]
        raise ValueError(f"说话人 {speaker_id} 不存在。可用示例: {available}")

    # 加载所有帧和音素
    all_frames = []
    all_phones = []

    with h5py.File(cache_dir / 'features.h5', 'r') as f:
        features = f['features']
        phones_data = np.load(BASE_DIR / 'checkpoints' / 'large' / 'phones.npy')

        for utt in speaker_utts:
            start, end = utt['h5_start_idx'], utt['h5_end_idx']
            all_frames.append(features[start:end])
            all_phones.append(phones_data[start:end])

    frames = torch.from_numpy(np.concatenate(all_frames)).float().to(device)
    phones = torch.from_numpy(np.concatenate(all_phones)).long().to(device)

    print(f"✓ 加载 {len(speaker_utts)} 条话语, 共 {len(frames)} 帧")
    return frames, phones


def knn_anonymize(src_feats, src_phones, tgt_feats, tgt_phones, k=1):
    """k=1 最近邻匹配（按音素约束）"""
    print(f"\nkNN 匹配 (k={k})...")

    h_anon = torch.zeros_like(src_feats)
    unique_phones = src_phones.unique()

    for phone_id in tqdm(unique_phones, desc="kNN"):
        src_mask = (src_phones == phone_id)
        tgt_mask = (tgt_phones == phone_id)

        if tgt_mask.sum() == 0:
            h_anon[src_mask] = src_feats[src_mask]
            continue

        src_batch = torch.nn.functional.normalize(src_feats[src_mask], dim=-1)
        tgt_batch = torch.nn.functional.normalize(tgt_feats[tgt_mask], dim=-1)

        cos_sim = torch.mm(src_batch, tgt_batch.T)
        nearest_idx = cos_sim.argmax(dim=1)
        h_anon[src_mask] = tgt_feats[tgt_mask][nearest_idx]

    return h_anon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target-speaker', type=str, default='1272')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Private kNN-VC 基线验证")
    print("="*60)

    models = load_models(device)
    tgt_feats, tgt_phones = load_target_speaker(args.target_speaker, device)

    # 提取源音频特征
    print(f"\n提取源音频: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    with torch.no_grad():
        src_feats = models['wavlm'](waveform).squeeze(0)
        src_phones = models['phone_predictor'](src_feats)

    print(f"✓ 源特征: {src_feats.shape}")

    # kNN 匹配
    h_anon = knn_anonymize(src_feats, src_phones, tgt_feats, tgt_phones, k=args.k)

    # 合成
    print(f"\n合成音频: {args.output}")
    with torch.no_grad():
        waveform_anon = models['vocoder'](h_anon.unsqueeze(0)).squeeze(0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, waveform_anon.unsqueeze(0).cpu(), 16000)

    print(f"✓ 保存: {args.output}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

