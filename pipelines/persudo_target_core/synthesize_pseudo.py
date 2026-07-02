#!/usr/bin/env python3
"""
基于伪风格 Bank 的端到端匿名化合成
检索策略: L24 query -> L6 value (音素约束 kNN)
时长匿名化: w * d_pred + (1-w) * d_true, w=0.3
"""

import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / 'checkpoints' / 'trainother500_with_phones'
DEFAULT_AUDIO = str(BASE_DIR.parent / 'datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'outputs' / 'test_anonymization_v2')
DEFAULT_SRC_GENDER = 'm'  # speaker 61 is male


def load_models(src_gender, device='cuda', dur_weight=1):
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
    from models.vocoder.hifigan import HiFiGAN

    wavlm = WavLMSSLExtractor(ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'), layer=6, device=device)
    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)
    duration_predictor = DurationPredictor.load(str(ckpt_dir / 'duration_decoder.pt'), device=device)
    vocoder = HiFiGAN.load(checkpoint_path=str(ckpt_dir / 'hifigan.pt'), device=device)

    bank_m = torch.load(str(DATA_DIR / 'pseudo_bank.gender-m.pt'), map_location=device)
    bank_f = torch.load(str(DATA_DIR / 'pseudo_bank.gender-f.pt'), map_location=device)

    def build_fallback(bank):
        return {
            'l6':  torch.cat([v['l6']  for v in bank.values()], dim=0),
            'l24': torch.cat([v['l24'] for v in bank.values()], dim=0),
        }

    print(f"Bank loaded (M: {len(bank_m)} phones, F: {len(bank_f)} phones)")
    return {
        'wavlm': wavlm,
        'phone_predictor': phone_predictor,
        'duration_predictor': duration_predictor,
        'vocoder': vocoder,
        'bank_m': bank_m, 'bank_f': bank_f,
        'fallback_m': build_fallback(bank_m),
        'fallback_f': build_fallback(bank_f),
        'dur_weight': dur_weight,
    }


def extract_source_features(audio_path, models, device='cuda'):
    """提取 L6 + L12 + L24，音素预测使用 L24"""
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0, keepdim=True).to(device)

    with torch.no_grad():
        multi_feats = models['wavlm'].forward_multi_layer(waveform, layers=[6, 12, 24])
        l6  = multi_feats[6].squeeze(0)
        l12 = multi_feats[12].squeeze(0)
        l24 = multi_feats[24].squeeze(0)
        phones = models['phone_predictor'](l24).cpu().numpy()

    return {'l6': l6, 'l12': l12, 'l24': l24, 'phones': phones}


def anonymize_v2(source, models, tgt_gender, device='cuda', k=4, drop_ratio=0.3):
    """L24 query -> L6 value kNN，时长匿名化 w=dur_weight
    Stage 1: top-50 语义锁定
    Stage 3: 音色去身份化（非对称 Dirichlet）
    """
    l24 = source['l24'].to(device)
    l12 = source['l12'].to(device)
    src_phones = source['phones']
    dur_weight = models['dur_weight']

    bank     = models['bank_m']     if tgt_gender == 'm' else models['bank_f']
    fallback = models['fallback_m'] if tgt_gender == 'm' else models['fallback_f']


    # 预计算源说话人 L6 质心（整句均值，只算一次）
    e_src = source['l6'].mean(dim=0).to(device)  # [1024]

    # 连续音素段 + 真实时长
    unique_phones, phone_durations = [], []
    cur, cnt = src_phones[0], 1
    for i in range(1, len(src_phones)):
        if src_phones[i] == cur:
            cnt += 1
        else:
            unique_phones.append(cur); phone_durations.append(cnt)
            cur, cnt = src_phones[i], 1
    unique_phones.append(cur); phone_durations.append(cnt)

    phones_t = torch.tensor(unique_phones, dtype=torch.long, device=device)
    dur_true = torch.tensor(phone_durations, dtype=torch.float32, device=device)

    # 时长匿名化
    with torch.no_grad():
        dur_pred = models['duration_predictor'](phones_t).squeeze(0)
    dur_anon = (dur_weight * dur_pred + (1 - dur_weight) * dur_true).clamp(min=1).round().long()

    # 按匿名化时长插值对齐 L24
    adj_l24, adj_l12, adj_phones = [], [], []
    idx = 0
    for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
        end = idx + orig
        new_len = new.item()
        

        if new_len == 0:
            idx = end
            continue
    
        # 用浮点插值而非整数截断
        if orig == 1:
            # 只有一帧，直接复制
            adj_l24.append(l24[idx:idx+1].expand(new_len, -1))
            adj_l12.append(l12[idx:idx+1].expand(new_len, -1))
        else:
            # 浮点索引 + 线性插值
            t = torch.linspace(0, 1, new_len, device=device)  # [0,1]归一化
            src_indices_f = t * (orig - 1)  # 映射到 [0, orig-1]
            idx_low = src_indices_f.long().clamp(max=orig - 2)
            idx_high = idx_low + 1
            w = (src_indices_f - idx_low.float()).unsqueeze(-1)  # 插值权重

            feat_low = l24[idx + idx_low]
            feat_high = l24[idx + idx_high]
            interpolated = (1 - w) * feat_low + w * feat_high
            adj_l24.append(interpolated)

            feat_low_12 = l12[idx + idx_low]
            feat_high_12 = l12[idx + idx_high]
            interpolated_12 = (1 - w) * feat_low_12 + w * feat_high_12
            adj_l12.append(interpolated_12)
        
        adj_phones.append(torch.full((new_len,), ph, dtype=torch.long, device=device))
        idx = end

    l24_adj    = torch.cat(adj_l24, dim=0)
    l12_adj    = torch.cat(adj_l12, dim=0)
    phones_adj = torch.cat(adj_phones, dim=0)

    h_anon = torch.zeros(l24_adj.shape[0], 1024, device=device)

    for phone_id in tqdm(torch.unique(phones_adj), desc=f"kNN [{tgt_gender}]"):
        ph = int(phone_id.item())
        mask = phones_adj == phone_id

        query_l24 = l24_adj[mask]
        query_l12 = l12_adj[mask]

        if ph in bank:
            tgt_l6  = bank[ph]['l6']
            tgt_l12 = bank[ph]['l12']
            tgt_l24 = bank[ph]['l24']
        else:
            tgt_l6  = fallback['l6']
            tgt_l12 = fallback['l12']
            tgt_l24 = fallback['l24']

        N_q, N_t = query_l24.shape[0], tgt_l24.shape[0]

        if N_t < k:
            h_anon[mask] = tgt_l6.mean(dim=0).expand(N_q, -1)
            continue

        # Stage 1: L24 语义锁定，top-50（桶不足100帧时取 top-30%）
        pool_size = min(100, N_t)
        dists_l24 = torch.cdist(query_l24, tgt_l24)
        top_l24_idx = dists_l24.topk(pool_size, largest=False).indices  # [N_q, pool_size]

        result = torch.zeros(N_q, 1024, device=device)

        for i in range(N_q):
            # 获取第一级筛选出的 100 个候选帧
            cand_l6  = tgt_l6[top_l24_idx[i]]   # [pool_size, 1024]
            cand_l12 = tgt_l12[top_l24_idx[i]]  # [pool_size, 1024]

            # 第二级：L6 隐私防火墙 (剔除源说话人残留区)
            sims_l6 = torch.cosine_similarity(cand_l6, e_src.unsqueeze(0), dim=-1)
            
            # 策略：按降序排列(最相似在前面)，切掉最相似的 drop_ratio (默认 30%)
            drop_count = max(int(pool_size * drop_ratio), 1)
            sorted_sims_idx = sims_l6.argsort(descending=True) 
            safe_idx = sorted_sims_idx[drop_count:]  # 截取后部的作为绝对安全池

            safe_cand_l6  = cand_l6[safe_idx]
            safe_cand_l12 = cand_l12[safe_idx]

            # 第三级：L12 情感优选 (在安全区内找能量与韵律起伏最吻合的)
            # 计算源帧 L12 与安全候选帧 L12 的欧氏距离
            dist_l12 = torch.cdist(query_l12[i].unsqueeze(0), safe_cand_l12).squeeze(0)
            
            # 选取情感距离最近的 k 帧
            actual_k = min(k, safe_cand_l12.shape[0])
            best_l12_idx = dist_l12.topk(actual_k, largest=False).indices

            selected_l6 = safe_cand_l6[best_l12_idx]

            # 合成：使用纯均值平滑替换 Dirichlet 抖动，彻底修复微观断音问题
            result[i] = selected_l6.mean(dim=0)

        h_anon[mask] = result

    return h_anon.cpu()


def anonymize(source, models, tgt_gender, device='cuda', k=4):
    """L24 query -> L6 value kNN，时长匿名化 w=dur_weight"""
    l24 = source['l24'].to(device)
    src_phones = source['phones']
    dur_weight = models['dur_weight']

    bank     = models['bank_m']     if tgt_gender == 'm' else models['bank_f']
    fallback = models['fallback_m'] if tgt_gender == 'm' else models['fallback_f']

    unique_phones, phone_durations = [], []
    cur, cnt = src_phones[0], 1
    for i in range(1, len(src_phones)):
        if src_phones[i] == cur:
            cnt += 1
        else:
            unique_phones.append(cur); phone_durations.append(cnt)
            cur, cnt = src_phones[i], 1
    unique_phones.append(cur); phone_durations.append(cnt)

    phones_t = torch.tensor(unique_phones, dtype=torch.long, device=device)
    dur_true = torch.tensor(phone_durations, dtype=torch.float32, device=device)

    with torch.no_grad():
        dur_pred = models['duration_predictor'](phones_t).squeeze(0)
    dur_anon = (dur_weight * dur_pred + (1 - dur_weight) * dur_true).clamp(min=1).round().long()

    adj_l24, adj_phones = [], []
    idx = 0
    for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
        end = idx + orig
        new_len = new.item()

        if new_len == 0:
            idx = end
            continue

        # 浮点插值避免截断
        if orig == 1:
            adj_l24.append(l24[idx:idx+1].expand(new_len, -1))
        else:
            t = torch.linspace(0, 1, new_len, device=device)
            src_indices_f = t * (orig - 1)
            idx_low = src_indices_f.long().clamp(max=orig - 2)
            idx_high = idx_low + 1
            w = (src_indices_f - idx_low.float()).unsqueeze(-1)

            feat_low = l24[idx + idx_low]
            feat_high = l24[idx + idx_high]
            interpolated = (1 - w) * feat_low + w * feat_high
            adj_l24.append(interpolated)

        adj_phones.append(torch.full((new_len,), ph, dtype=torch.long, device=device))
        idx = end

    l24_adj    = torch.cat(adj_l24, dim=0)
    phones_adj = torch.cat(adj_phones, dim=0)

    h_anon = torch.zeros(l24_adj.shape[0], 1024, device=device)
    pool_size = 20

    for phone_id in tqdm(torch.unique(phones_adj), desc=f"kNN [{tgt_gender}]"):
        ph = int(phone_id.item())
        mask = phones_adj == phone_id
        query = l24_adj[mask]

        if ph in bank:
            tgt_l6  = bank[ph]['l6']
            tgt_l24 = bank[ph]['l24']
        else:
            tgt_l6  = fallback['l6']
            tgt_l24 = fallback['l24']

        dists = torch.cdist(query, tgt_l24)
        N_q, N_t = query.shape[0], tgt_l24.shape[0]

        if N_t >= pool_size:
            top_idx = dists.topk(pool_size, largest=False).indices
            rand_idx = torch.randint(0, pool_size, (N_q, k), device=device)
            sampled = torch.gather(top_idx, 1, rand_idx)
            frames = tgt_l6[sampled]
            weights = torch.distributions.Dirichlet(torch.ones(k, device=device)).sample((N_q,)).unsqueeze(-1)
            h_anon[mask] = (frames * weights).sum(dim=1)
        else:
            actual_k = min(k, N_t)
            top_idx = dists.topk(actual_k, largest=False).indices
            h_anon[mask] = tgt_l6[top_idx].mean(dim=1)

    return h_anon.cpu()


def save_wav(tensor, path, sr=16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(path), tensor.cpu(), sr)
    print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio',      default=DEFAULT_AUDIO)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--src-gender', default=DEFAULT_SRC_GENDER, choices=['m', 'f'])
    parser.add_argument('--k',          type=int,   default=4)
    parser.add_argument('--dur-weight', type=float, default=0.3)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)

    # 保存原始音频
    wav, sr = torchaudio.load(args.audio)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    save_wav(wav.mean(dim=0), out_dir / 'original.wav')

    models = load_models(args.src_gender, device, args.dur_weight)
    source = extract_source_features(args.audio, models, device)

    # cross-gender
    tgt_cross = 'f' if args.src_gender == 'm' else 'm'
    h_cross = anonymize_v2(source, models, tgt_cross, device, args.k)
    with torch.no_grad():
        wav_cross = models['vocoder'](h_cross.unsqueeze(0).to(device)).squeeze()
    save_wav(wav_cross, out_dir / f'anon_cross_{tgt_cross}.wav')

    # same-gender
    h_same = anonymize_v2(source, models, args.src_gender, device, args.k)
    with torch.no_grad():
        wav_same = models['vocoder'](h_same.unsqueeze(0).to(device)).squeeze()
    save_wav(wav_same, out_dir / f'anon_same_{args.src_gender}.wav')

    print(f"\nDone. Output: {out_dir}")


if __name__ == '__main__':
    main()
