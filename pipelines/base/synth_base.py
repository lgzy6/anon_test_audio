#!/usr/bin/env python3
"""
Global Phonetic 匿名化合成 (Base 纯净基线版)

核心逻辑 (绝对减法)：
  1. WavLM 提取 L6, L12, L24 和 音素序列 (不改变时长/不预测时长)
  2. 根据指定的 --retrieval-layer (l6, l12, l24) 在 Bank 中计算欧氏距离
  3. 获取最近的 Top-K 索引
  4. 永远使用这组索引从 Bank 中抽取 L6 特征
  5. 简单平均后送入 HiFi-GAN 合成
"""

import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent

# 注意这里指向我们刚生成的纯净版 Bank
DATA_DIR = BASE_DIR / 'checkpoints' / 'trainother500_200spk'
# DEFAULT_AUDIO = str(BASE_DIR.parent / 'datasets/LibriSpeech/test-clean/61/70968/61-70968-0007.flac')
DEFAULT_AUDIO = str('/root/autodl-tmp/datasets/Emotion Speech Dataset/0011/Sad/0011_001052.wav')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'outputs' / 'global_phonetic_base')

# ============================================================
# Model & Bank Loading
# ============================================================

def _load_bank(path, device):
    bank = torch.load(str(path), map_location=device)
    fallback = {
        'l6':  torch.cat([v['l6']  for v in bank.values()], dim=0),
        'l12': torch.cat([v['l12'] for v in bank.values()], dim=0),
        'l24': torch.cat([v['l24'] for v in bank.values()], dim=0),
    }
    return bank, fallback

def load_models(modes, device='cuda'):
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor
    from models.vocoder.hifigan import HiFiGAN

    # 移除了 duration_predictor，回归最纯粹的帧级替换
    models = {
        'wavlm': WavLMSSLExtractor(ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'), layer=6, device=device),
        'phone_predictor': PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device),
        'vocoder': HiFiGAN.load(checkpoint_path=str(ckpt_dir / 'hifigan.pt'), device=device)
    }

    print("加载纯净版 Base Bank...")
    if any(m in modes for m in ('same', 'cross')):
        b_m, fb_m = _load_bank(DATA_DIR / 'pseudo_bank_base.gender-m.pt', device)
        b_f, fb_f = _load_bank(DATA_DIR / 'pseudo_bank_base.gender-f.pt', device)
        models.update({'bank_m': b_m, 'fallback_m': fb_m, 'bank_f': b_f, 'fallback_f': fb_f})
    
    if 'mix' in modes:
        b_mix, fb_mix = _load_bank(DATA_DIR / 'pseudo_bank_base.pt', device)
        models.update({'bank_mix': b_mix, 'fallback_mix': fb_mix})

    return models

def select_bank(models, src_gender, mode):
    if mode == 'same':
        key = src_gender
    elif mode == 'cross':
        key = 'f' if src_gender == 'm' else 'm'
    else:
        key = 'mix'
    return models[f'bank_{key}'], models[f'fallback_{key}']

# ============================================================
# Feature Extraction
# ============================================================

def extract_source_features(audio_path, models, device='cuda'):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0, keepdim=True).to(device)

    with torch.no_grad():
        multi_feats = models['wavlm'].forward_multi_layer(waveform, layers=[6, 12, 24])
        phones = models['phone_predictor'](multi_feats[24].squeeze(0)).cpu().numpy()

    return {
        'l6': multi_feats[6].squeeze(0),
        'l12': multi_feats[12].squeeze(0),
        'l24': multi_feats[24].squeeze(0),
        'phones': phones
    }

# ============================================================
# Core: Pure kNN Anonymization (极简检索)
# ============================================================

def anonymize_pure_knn(source, models, src_gender, mode, 
                       retrieval_layer='l24', top_k=4, device='cuda'):
    """
    极简 kNN 检索：
    使用指定的 retrieval_layer 计算 L2 距离，
    获取最近的 Top-K 索引后，统一下发 bank 中的 L6 特征进行合成。
    """
    bank, fallback = select_bank(models, src_gender, mode)
    
    # 提取查询层和音素序列
    query_feats = source[retrieval_layer].to(device)
    phones_t = torch.tensor(source['phones'], dtype=torch.long, device=device).squeeze()

    T_out = query_feats.shape[0]
    h_anon = torch.zeros(T_out, 1024, device=device)

    pool_label = {'same': src_gender, 'cross': ('f' if src_gender == 'm' else 'm'), 'mix': 'mix'}[mode]

    for phone_id in tqdm(torch.unique(phones_t), desc=f"kNN [{retrieval_layer}] [{mode}→{pool_label}]"):
        ph = int(phone_id.item())
        mask = (phones_t == phone_id)
        N_q = mask.sum().item()

        # 1. 确定搜索目标 (Target Search Feats) 和 输出目标 (Target Output L6)
        if ph in bank:
            tgt_search = bank[ph][retrieval_layer].to(device)
            tgt_l6_out = bank[ph]['l6'].to(device)
        else:
            tgt_search = fallback[retrieval_layer].to(device)
            tgt_l6_out = fallback['l6'].to(device)

        N_t = tgt_search.shape[0]
        if N_t == 0:
            continue

        if N_t <= top_k:
            h_anon[mask] = tgt_l6_out.mean(dim=0).expand(N_q, -1)
            continue

        # 2. 在指定的 retrieval_layer 上计算 L2 距离
        d_matrix = torch.cdist(query_feats[mask], tgt_search)
        _, topk_idx = d_matrix.topk(top_k, largest=False)

        # 3. 永远使用检索到的索引去提取 L6
        topk_l6 = tgt_l6_out[topk_idx]  # 维度: [N_q, K, 1024]
        
        # 4. 简单平均 (抛弃复杂的 softmax 温度加权)
        h_anon[mask] = topk_l6.mean(dim=1)

    return h_anon.cpu()

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pure kNN Baseline Synthesis")
    parser.add_argument('--audio',       default=DEFAULT_AUDIO)
    parser.add_argument('--output-dir',  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--src-gender',  default='m', choices=['m', 'f'])
    parser.add_argument('--mode',        default='same', choices=['same', 'cross', 'mix', 'all'])
    
    # 核心控制变量
    parser.add_argument('--retrieval-layer', default='l24', choices=['l6', 'l12', 'l24'],
                        help="用于计算距离的特征层 (默认: l24)")
    parser.add_argument('--top-k',       type=int, default=4,
                        help="检索邻居数量")
    parser.add_argument('--device',      default='cuda')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    modes = ['same', 'cross', 'mix'] if args.mode == 'all' else [args.mode]

    print("=" * 60)
    print("  Global Phonetic Anonymization [PURE BASELINE]")
    print("=" * 60)
    print(f"  Retrieval Layer: {args.retrieval_layer.upper()} (Synthesis with L6)")
    print(f"  Top-K:           {args.top_k}")
    
    models = load_models(modes, device=device)
    source = extract_source_features(args.audio, models, device)

    for mode in modes:
        h_anon = anonymize_pure_knn(
            source, models, args.src_gender, mode, 
            retrieval_layer=args.retrieval_layer, 
            top_k=args.top_k, device=device
        )

        with torch.no_grad():
            wav_anon = models['vocoder'](h_anon.unsqueeze(0).to(device)).squeeze()

        out_path = out_dir / f"anon_{args.src_gender}_{mode}_layer-{args.retrieval_layer}_top{args.top_k}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), wav_anon.unsqueeze(0).cpu(), 16000)
        print(f"  [+] Saved: {out_path}")

if __name__ == '__main__':
    main()