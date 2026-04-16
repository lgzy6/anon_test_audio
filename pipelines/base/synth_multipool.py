#!/usr/bin/env python3
"""
Multi-Pool kNN 匿名化合成:
  - 随机选取 pool (0-3)
  - 按 same/cross/mix 策略选对应 bank
  - top1 检索，用 L24 计算距离，L6 合成
"""

import sys
import random
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR  = Path(__file__).parent.parent.parent
CKPT_DIR  = BASE_DIR / 'checkpoints'
BANKS_DIR = CKPT_DIR / 'banks'
N_POOLS   = 4


def _load_bank(path, device):
    bank = torch.load(str(path), map_location=device)
    fallback = {
        'l6':  torch.cat([v['l6']  for v in bank.values()], dim=0),
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


def load_pool_banks(pool_id, device='cuda'):
    """加载指定 pool 的 m/f 两个 bank。"""
    banks = {}
    for gender in ['m', 'f']:
        path = BANKS_DIR / f'pool_{pool_id}_gender-{gender}.pt'
        if path.exists():
            banks[gender] = _load_bank(path, device)
    return banks


def select_bank(pool_banks, src_gender, mode):
    if mode == 'same':
        key = src_gender
    else:  # cross
        key = 'f' if src_gender == 'm' else 'm'
    return pool_banks[key]  # (bank, fallback)


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


def anonymize(source, pool_banks, src_gender, mode, device):
    bank, fallback = select_bank(pool_banks, src_gender, mode)
    query = source['l24'].to(device)
    phones_t = torch.tensor(source['phones'], dtype=torch.long, device=device)
    T = query.shape[0]
    h_anon = torch.zeros(T, 1024, device=device)

    for phone_id in torch.unique(phones_t):
        ph = int(phone_id.item())
        mask = phones_t == phone_id
        N_q = mask.sum().item()

        tgt_l24 = bank[ph]['l24'].to(device) if ph in bank else fallback['l24'].to(device)
        tgt_l6  = bank[ph]['l6'].to(device)  if ph in bank else fallback['l6'].to(device)

        if tgt_l24.shape[0] == 0:
            continue
        if tgt_l24.shape[0] == 1:
            h_anon[mask] = tgt_l6[0].expand(N_q, -1)
            continue

        # top20 随机选5帧取均值
        dists = torch.cdist(query[mask], tgt_l24)
        k = min(20, tgt_l24.shape[0])
        top20 = dists.topk(k, dim=-1, largest=False).indices  # [N_q, k]
        n_pick = min(5, k)
        # 对每个查询帧随机选 n_pick 列
        rand_cols = torch.stack([torch.randperm(k, device=device)[:n_pick] for _ in range(top20.shape[0])])  # [N_q, n_pick]
        picked_idx = top20[torch.arange(top20.shape[0], device=device).unsqueeze(1), rand_cols]  # [N_q, n_pick]
        h_anon[mask] = tgt_l6[picked_idx].mean(dim=1)

    return h_anon.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio',      default=str('/root/autodl-tmp/Voice-Privacy-Challenge-2024/data/libri_dev/wav/84-121123-0000/84-121123-0000.wav'))
    parser.add_argument('--output-dir', default=str(BASE_DIR / 'outputs' / 'multipool'))
    parser.add_argument('--src-gender', default='m', choices=['m', 'f'])
    parser.add_argument('--mode',       default='all', choices=['same', 'cross', 'all'])
    parser.add_argument('--pool',       default=None, help="指定 pool id，或 'all' 遍历所有池子（默认随机）")
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    modes = ['same', 'cross'] if args.mode == 'all' else [args.mode]

    if args.pool == 'all':
        pool_ids = list(range(N_POOLS))
    elif args.pool is not None:
        pool_ids = [int(args.pool)]
    else:
        pool_ids = [random.randint(0, N_POOLS - 1)]

    models = load_models(device)
    source = extract_features(args.audio, models, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pool_id in pool_ids:
        pool_banks = load_pool_banks(pool_id, device)
        for mode in modes:
            print(f"Mode: {mode}, Pool: {pool_id}")
            h_anon = anonymize(source, pool_banks, args.src_gender, mode, device)
            with torch.no_grad():
                wav_out = models['vocoder'](h_anon.unsqueeze(0).to(device)).squeeze()
            out_path = out_dir / f"anon_{args.src_gender}_{mode}_pool{pool_id}.wav"
            torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)
            print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
