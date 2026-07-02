#!/usr/bin/env python3
"""
测试静音帧跳过对 vocoder 输出的影响
对比:
  A) 全部帧参与检索 (无跳过)
  B) 静音帧→零向量 (当前 VPC pipeline 做法)
  C) 静音帧→保留源 L6 特征

用法: cd /root/autodl-tmp/anon_test && python pipelines/base/test_silence_skip.py
"""

import sys
import random
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR  = Path(__file__).parent.parent.parent
CKPT_DIR  = BASE_DIR / 'checkpoints'
BANKS_DIR = CKPT_DIR / 'banks_c12f32s8'
N_POOLS   = 4

# 与 pipeline_top3Random_l6l24unit.py 保持一致
SILENCE_PHONES = {0, 1}

PHONE_ALPHA_MAP = {
    2: 0.4, 4: 0.4, 5: 0.35, 6: 0.4, 11: 0.4, 13: 0.35,
    16: 0.4, 18: 0.4, 20: 0.4, 23: 0.4, 32: 0.4, 35: 0.4, 38: 0.4,
    8: 0.25, 15: 0.25, 19: 0.25, 21: 0.25, 24: 0.25, 28: 0.25, 29: 0.25,
    7: 0.15, 12: 0.15, 30: 0.15, 37: 0.15,
    3: 0.05, 9: 0.05, 10: 0.05, 14: 0.05, 17: 0.05,
    22: 0.05, 25: 0.05, 26: 0.05, 27: 0.05, 31: 0.05,
    33: 0.05, 34: 0.05, 36: 0.05, 39: 0.05, 40: 0.05,
    41: 0.05,
}

TEST_AUDIO = '/root/autodl-tmp/Voice-Privacy-Challenge-2024/data/libri_dev/wav/84-121123-0000/84-121123-0000.wav'
OUTPUT_DIR = BASE_DIR / 'test_outputs' / 'silence_skip_test'


def _load_bank(path, device):
    raw = torch.load(str(path), map_location=device)
    bank = {}
    for ph, d in raw.items():
        bank[ph] = {
            'l6':  d['l6'].to(device),
            'l24': d['l24'].to(device),
        }
    fallback = {
        'l6':  torch.cat([v['l6'] for v in bank.values()], dim=0),
        'l24': torch.cat([v['l24'] for v in bank.values()], dim=0),
    }
    return bank, fallback


def load_models(device='cuda'):
    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor
    from models.vocoder.hifigan import HiFiGAN

    return {
        'wavlm': WavLMSSLExtractor(ckpt_path=str(CKPT_DIR / 'WavLM-Large.pt'), layer=6, device=device),
        'phone_predictor': PhonePredictor.load(str(CKPT_DIR / 'phone_decoder.pt'), device=device),
        'vocoder': HiFiGAN.load(checkpoint_path=str(CKPT_DIR / 'hifigan.pt'), device=device),
    }


def extract_features(audio_path, models, device):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0, keepdim=True).to(device)
    with torch.no_grad():
        multi = models['wavlm'].forward_multi_layer(wav, layers=[6, 24])
        phones = models['phone_predictor'](multi[24].squeeze(0)).cpu().numpy()
    return {
        'l6':  multi[6].squeeze(0),
        'l24': multi[24].squeeze(0),
        'phones': phones,
    }


def retrieve(source, bank, fallback, device, silence_mode='skip_zero', alpha=0.2):
    """
    silence_mode:
      'no_skip'    - 全部帧参与检索 (A)
      'skip_zero'  - 静音帧→零向量 (B, 当前VPC做法)
      'keep_src'   - 静音帧→保留源L6 (C)
    """
    query_l24 = source['l24'].to(device)
    query_l6 = source['l6'].to(device)
    phones_t = torch.tensor(source['phones'], dtype=torch.long, device=device)
    T = query_l24.shape[0]
    h_anon = torch.zeros(T, 1024, device=device)

    q_l6_norm = query_l6 / (query_l6.norm(dim=-1, keepdim=True) + 1e-8)
    q_l24_norm = query_l24 / (query_l24.norm(dim=-1, keepdim=True) + 1e-8)

    silence_count = 0
    for phone_id in torch.unique(phones_t):
        ph = int(phone_id.item())
        mask = phones_t == phone_id
        N_q = mask.sum().item()

        # 静音帧处理逻辑
        if ph in SILENCE_PHONES:
            silence_count += N_q
            if silence_mode == 'skip_zero':
                continue  # h_anon[mask] 保持零向量
            elif silence_mode == 'keep_src':
                h_anon[mask] = query_l6[mask]
                continue
            # 'no_skip' 则继续正常检索

        tgt_l24 = bank[ph]['l24'] if ph in bank else fallback['l24']
        tgt_l6 = bank[ph]['l6'] if ph in bank else fallback['l6']

        N_t = tgt_l24.shape[0]
        if N_t == 0:
            continue
        if N_t == 1:
            h_anon[mask] = tgt_l6[0].expand(N_q, -1)
            continue

        tgt_l6_norm = tgt_l6 / (tgt_l6.norm(dim=-1, keepdim=True) + 1e-8)
        tgt_l24_norm = tgt_l24 / (tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)

        dist_l6 = torch.cdist(q_l6_norm[mask], tgt_l6_norm)
        dist_l24 = torch.cdist(q_l24_norm[mask], tgt_l24_norm)

        ph_alpha = PHONE_ALPHA_MAP.get(ph, alpha)
        dist_mix = ph_alpha * dist_l6 + (1 - ph_alpha) * dist_l24

        # Top-3 加权融合 (与 synth_multipool_v2 保持一致)
        k = min(3, N_t)
        topk_dists, topk_idx = dist_mix.topk(k, dim=-1, largest=False)
        weights = 1.0 / (topk_dists + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        topk_l6 = tgt_l6[topk_idx]
        h_anon[mask] = (topk_l6 * weights.unsqueeze(-1)).sum(dim=1)

    return h_anon, silence_count


def vocoder_batch_synth(h_list, models, device, voc_batch=8):
    """批量 vocoder 合成 (与 VPC pipeline 209-226 一致)"""
    h_lengths = [h.shape[0] for h in h_list]
    wav_list = []
    with torch.no_grad():
        for start in range(0, len(h_list), voc_batch):
            end = min(start + voc_batch, len(h_list))
            sub_h = h_list[start:end]
            max_T = max(h.shape[0] for h in sub_h)
            h_padded = torch.zeros(end - start, max_T, 1024, device=device)
            for k, h in enumerate(sub_h):
                h_padded[k, :h.shape[0]] = h
            wav_batch_out = models['vocoder'](h_padded)
            for k in range(end - start):
                out_len = h_lengths[start + k] * 320
                wav_out = wav_batch_out[k, :out_len]
                wav_list.append(wav_out.cpu())
    return wav_list


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("静音帧跳过测试")
    print("=" * 60)

    print("\n[1] 加载模型...")
    models = load_models(device)

    # 随机选一个 pool
    pool_id = random.randint(0, N_POOLS - 1)
    gender = 'm'
    bank_path = BANKS_DIR / f'pool_{pool_id}_gender-{gender}.pt'
    if not bank_path.exists():
        # fallback to banks_v2
        bank_path = CKPT_DIR / 'banks_v2' / f'pool_{pool_id}_gender-{gender}.pt'
    print(f"  Bank: {bank_path.relative_to(CKPT_DIR)}")
    bank, fallback = _load_bank(bank_path, device)

    print(f"\n[2] 提取特征: {Path(TEST_AUDIO).name}")
    source = extract_features(TEST_AUDIO, models, device)
    total_frames = source['phones'].shape[0]

    phones_arr = source['phones']
    sil_frames = sum(1 for p in phones_arr if p in SILENCE_PHONES)
    print(f"  总帧数: {total_frames}, 静音帧: {sil_frames} ({100*sil_frames/total_frames:.1f}%)")

    # 三种策略
    modes = [
        ('no_skip',    'A_全部检索'),
        ('skip_zero',  'B_静音零向量'),
        ('keep_src',   'C_静音保留源L6'),
    ]

    h_list = []
    labels = []
    print(f"\n[3] 执行检索...")
    for mode, label in modes:
        h_anon, sil_cnt = retrieve(source, bank, fallback, device, silence_mode=mode)
        h_list.append(h_anon)
        labels.append(label)
        print(f"  {label}: done (跳过帧={sil_cnt if mode != 'no_skip' else 'N/A'})")

    print(f"\n[4] Vocoder 合成...")
    wav_list = vocoder_batch_synth(h_list, models, device)

    print(f"\n[5] 保存结果...")
    for wav_out, label in zip(wav_list, labels):
        out_path = OUTPUT_DIR / f"{label}.wav"
        torchaudio.save(str(out_path), wav_out.unsqueeze(0), 16000)
        print(f"  {out_path.name}: RMS={wav_out.abs().mean():.6f}, max={wav_out.abs().max():.4f}")

    # 静音段局部对比
    print(f"\n[6] 静音段波形对比...")
    phones_t = torch.tensor(source['phones'], dtype=torch.long)
    sil_mask_frames = torch.zeros(total_frames, dtype=torch.bool)
    for ph in SILENCE_PHONES:
        sil_mask_frames |= (phones_t == ph)

    # 找到第一段连续静音对应的波形区间
    sil_indices = sil_mask_frames.nonzero().squeeze()
    if len(sil_indices) > 0:
        first_sil_start = sil_indices[0].item()
        # 找连续段
        seg_end = first_sil_start
        for idx in sil_indices:
            if idx.item() == seg_end:
                seg_end += 1
            else:
                break
        wav_start = first_sil_start * 320
        wav_end = seg_end * 320
        print(f"  第一段静音: frame[{first_sil_start}:{seg_end}] → sample[{wav_start}:{wav_end}]")
        for wav_out, label in zip(wav_list, labels):
            seg = wav_out[wav_start:min(wav_end, len(wav_out))]
            print(f"    {label}: seg_RMS={seg.abs().mean():.6f}, seg_max={seg.abs().max():.6f}")

    print(f"\n{'=' * 60}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"对比方法: 听 A/B/C 三个文件")
    print(f"  - 如果 B 在静音处有噪声/click → 零向量策略有问题")
    print(f"  - 如果 C 比 B 更自然 → 应改为保留源L6")
    print(f"  - 如果 A ≈ C → 静音帧检索本身无害，跳过也无害")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
