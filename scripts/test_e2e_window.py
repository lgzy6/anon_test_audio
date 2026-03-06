#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-based E2E Anonymization Test

Key differences from segment-based approach:
1. Window-level style extraction (fixed-length overlapping windows)
2. α-weighted style-guided retrieval (continuous prosody control)
3. Phone bucket + soft blending for local controllability

Usage:
    python scripts/test_e2e_window.py --audio input.wav --output output.wav --alpha 0.2
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import json
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent

# Import WindowConfig and WindowStyleExtractor
exec(open(Path(__file__).parent / "test.py").read(), globals())


def load_models(device='cuda'):
    """Load all required models"""
    print("Loading models...")
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

    style_extractor = WindowStyleExtractor.load(
        ckpt_dir / 'style_extractor_window.pkl'
    )

    from models.vocoder.hifigan import HiFiGAN
    vocoder = HiFiGAN.load(
        checkpoint_path=str(ckpt_dir / 'hifigan.pt'),
        device=device
    )

    print("✓ Models loaded")

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
    """Load target pool with window-level styles"""
    print("\nLoading target pool...")

    ckpt_dir = BASE_DIR / 'checkpoints'
    cache_dir = BASE_DIR / 'cache' / 'features' / 'wavlm' / 'train_clean_360'

    pool_data = np.load(ckpt_dir / 'target_pool_window_styles.npz', allow_pickle=True)

    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"✓ Target pool: {len(pool_data['utt_ids'])} utterances")

    return {
        'pool_data': pool_data,
        'metadata': metadata,
        'features_path': cache_dir / 'features.h5'
    }


def extract_source_features(audio_path, models, wcfg, device='cuda'):
    """Extract source audio features with window-level style"""
    print(f"\nExtracting source features: {audio_path}")

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

    # Extract window-level style trajectory
    style_windows, win_starts, win_ends = models['style_extractor'].transform_windows(
        eta, phones, wcfg
    )

    print(f"✓ WavLM: {wavlm_feats.shape}, Windows: {len(style_windows)}")

    return {
        'wavlm': wavlm_feats,
        'eta': eta,
        'phones': phones,
        'style_windows': style_windows,
        'win_starts': win_starts,
        'win_ends': win_ends
    }


def select_target_utterance(source_style_windows, source_gender, target_pool, mode='preserve'):
    """Select target utterance based on average window style"""
    print(f"\nSelecting target (mode={mode})...")

    pool_data = target_pool['pool_data']

    # Average source style
    src_style_avg = source_style_windows.mean(axis=0)

    # Filter by gender
    valid_idx = []
    for i, gender in enumerate(pool_data['genders']):
        if gender == source_gender:
            valid_idx.append(i)

    if not valid_idx:
        valid_idx = list(range(len(pool_data['utt_ids'])))

    if mode == 'random':
        return np.random.choice(valid_idx)

    # Compute style similarity (average of all windows)
    best_sim = -1
    best_idx = valid_idx[0]

    for idx in valid_idx:
        tgt_styles = pool_data['window_styles'][idx]
        tgt_avg = tgt_styles.mean(axis=0)
        sim = float(np.dot(src_style_avg, tgt_avg))
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    print(f"✓ Selected utt {best_idx}, style_sim={best_sim:.3f}")
    return best_idx


def anonymize_window_guided(source, target_idx, target_pool, phone_predictor, alpha=0.2, device='cuda'):
    """
    α-weighted window-level style-guided anonymization

    For each source window:
      1. Find candidate target windows (phone bucket)
      2. Compute: sim = (1-α)*content_sim + α*style_sim
      3. Select best match and blend features
    """
    print(f"\nWindow-guided retrieval (α={alpha})...")

    pool_data = target_pool['pool_data']
    tgt_utt_idx = int(target_idx)

    # Load target utterance features
    tgt_meta_idx = pool_data['utt_ids'][tgt_utt_idx]
    tgt_meta = target_pool['metadata']['utterances'][tgt_meta_idx]

    with h5py.File(target_pool['features_path'], 'r') as f:
        tgt_feats = f['features'][tgt_meta['h5_start_idx']:tgt_meta['h5_end_idx']]

    tgt_feats_t = torch.from_numpy(tgt_feats).float().to(device)

    # Predict target phones
    with torch.no_grad():
        tgt_phones = phone_predictor(tgt_feats_t).cpu().numpy()

    # Get target windows
    tgt_styles = pool_data['window_styles'][tgt_utt_idx]
    tgt_starts = pool_data['window_starts'][tgt_utt_idx]
    tgt_ends = pool_data['window_ends'][tgt_utt_idx]

    # Build phone buckets
    phone_buckets = {}
    for i, (s, e) in enumerate(zip(tgt_starts, tgt_ends)):
        ph = tgt_phones[s:e]
        dom_ph = int(ph[ph != 0].mode()[0]) if (ph != 0).any() else 0
        if dom_ph not in phone_buckets:
            phone_buckets[dom_ph] = []
        phone_buckets[dom_ph].append(i)

    # Match source windows
    src_wavlm = torch.from_numpy(source['wavlm']).float().to(device)
    h_anon = torch.zeros_like(src_wavlm)

    for i, (src_style, s, e) in enumerate(zip(
        source['style_windows'], source['win_starts'], source['win_ends']
    )):
        src_ph = source['phones'][s:e]
        dom_ph = int(src_ph[src_ph != 0].mode()[0]) if (src_ph != 0).any() else 0

        # Get candidates
        if dom_ph in phone_buckets:
            cands = phone_buckets[dom_ph]
        else:
            cands = list(range(len(tgt_styles)))

        if not cands:
            h_anon[s:e] = src_wavlm[s:e]
            continue

        # Compute α-weighted similarity
        best_sim = -1
        best_idx = cands[0]

        for c in cands:
            tgt_style = tgt_styles[c]
            style_sim = float(np.dot(src_style, tgt_style))

            ts, te = tgt_starts[c], tgt_ends[c]
            tph = tgt_phones[ts:te]
            content_sim = float((src_ph[:, None] == tph[None, :]).any(axis=1).mean())

            combined = (1 - alpha) * content_sim + alpha * style_sim

            if combined > best_sim:
                best_sim = combined
                best_idx = c

        # Copy matched features
        ts, te = tgt_starts[best_idx], tgt_ends[best_idx]
        h_anon[s:e] = tgt_feats_t[ts:te]

    print(f"✓ Anonymization complete")
    return h_anon.cpu().numpy()


def synthesize_audio(h_anon, vocoder, output_path):
    """Synthesize audio from anonymized features"""
    print(f"\nSynthesizing: {output_path}")

    h_tensor = torch.from_numpy(h_anon).unsqueeze(0).to(vocoder.device)

    with torch.no_grad():
        waveform = vocoder(h_tensor).squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(output_path), waveform.cpu(), 16000)
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--mode', type=str, default='preserve', choices=['preserve', 'random'])
    parser.add_argument('--gender', type=str, default='f', choices=['f', 'm'])
    parser.add_argument('--alpha', type=float, default=0.2, help='Style weight (0=content, 1=style)')
    parser.add_argument('--win-ms', type=float, default=400.0)
    parser.add_argument('--hop-ms', type=float, default=200.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Window-based E2E Anonymization")
    print("="*60)

    models = load_models(device)
    target_pool = load_target_pool(device)

    wcfg = WindowConfig(
        win_frames=int(args.win_ms / 20),
        hop_frames=int(args.hop_ms / 20),
        min_voiced_ratio=0.4,
        silence_id=0
    )

    source = extract_source_features(args.audio, models, wcfg, device)
    target_idx = select_target_utterance(source['style_windows'], args.gender, target_pool, args.mode)
    h_anon = anonymize_window_guided(source, target_idx, target_pool, models['phone_predictor'], args.alpha, device)
    synthesize_audio(h_anon, models['vocoder'], args.output)

    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

