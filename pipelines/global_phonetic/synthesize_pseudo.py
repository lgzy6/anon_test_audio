#!/usr/bin/env python3
"""
Global Phonetic 匿名化合成

支持三种匿名模式:
  same:  同性别池 (src=m → bank_m, src=f → bank_f)
  cross: 跨性别池 (src=m → bank_f, src=f → bank_m)
  mix:   混合池   (不区分性别)

支持两个版本:
  v1 (原版): 确定性联合打分检索
  v2 (隐私增强): P0 随机子池采样 + P1 查询量化 + P2 韵律强化

检索策略 (v1):
  1) WavLM → L24_q, L12_q
  2) 按 phone 找 bank bucket
  3) L24 距离 Top-N 快速召回
  4) 联合打分: score = d_L24 + λ · sim_L12(src→cand)
  5) Top-K 最低分 → Temperature Softmax 加权
  6) 取对应 L6 加权均值 → HiFi-GAN → 匿名音频

检索策略 (v2 - 隐私增强):
  1) WavLM → L24_q, L12_q
  2) [P2] 强化 duration 匿名 (dur_weight=0.7) + 帧级噪声扰动
  3) [P1] 查询 L24 通过 bank 侧 K-Means 量化，擦除源说话人精细指纹
  4) [P0] per-utterance 随机采样 bank 子集，打破确定性映射
  5) 纯 L24 距离检索 (移除 L12 身份锚点)
  6) 高 Temperature Softmax 加权 → HiFi-GAN → 匿名音频
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / 'checkpoints' / 'trainother500_200spk'
DEFAULT_AUDIO = str(BASE_DIR.parent / 'datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'outputs' / 'global_phonetic_test')


# ============================================================
# Model Loading
# ============================================================

def _load_bank(path, device):
    bank = torch.load(str(path), map_location=device)
    fallback = {
        'l6':  torch.cat([v['l6']  for v in bank.values()], dim=0),
        'l12': torch.cat([v['l12'] for v in bank.values()], dim=0),
        'l24': torch.cat([v['l24'] for v in bank.values()], dim=0),
    }
    n_phones = len(bank)
    n_frames = sum(v['l6'].shape[0] for v in bank.values())
    return bank, fallback, n_phones, n_frames


def load_models(modes, device='cuda', dur_weight=0.3):
    ckpt_dir = BASE_DIR / 'checkpoints'

    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
    from models.vocoder.hifigan import HiFiGAN

    wavlm = WavLMSSLExtractor(ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'), layer=6, device=device)
    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)
    duration_predictor = DurationPredictor.load(str(ckpt_dir / 'duration_decoder.pt'), device=device)
    vocoder = HiFiGAN.load(checkpoint_path=str(ckpt_dir / 'hifigan.pt'), device=device)

    need_gendered = any(m in modes for m in ('same', 'cross'))
    need_mix = 'mix' in modes

    result = {
        'wavlm': wavlm,
        'phone_predictor': phone_predictor,
        'duration_predictor': duration_predictor,
        'vocoder': vocoder,
        'dur_weight': dur_weight,
    }

    print("Bank loaded:")
    if need_gendered:
        bank_m, fb_m, np_m, nf_m = _load_bank(DATA_DIR / 'pseudo_bank_v2.gender-m.pt', device)
        bank_f, fb_f, np_f, nf_f = _load_bank(DATA_DIR / 'pseudo_bank_v2.gender-f.pt', device)
        result.update({'bank_m': bank_m, 'fallback_m': fb_m,
                       'bank_f': bank_f, 'fallback_f': fb_f})
        print(f"  M:   {np_m} phones, {nf_m} frames")
        print(f"  F:   {np_f} phones, {nf_f} frames")

    if need_mix:
        bank_mix, fb_mix, np_mix, nf_mix = _load_bank(DATA_DIR / 'pseudo_bank_v2.pt', device)
        result.update({'bank_mix': bank_mix, 'fallback_mix': fb_mix})
        print(f"  Mix: {np_mix} phones, {nf_mix} frames")

    return result


def select_bank(models, src_gender, mode):
    """根据源性别和模式选择 bank"""
    if mode == 'same':
        key = src_gender
    elif mode == 'cross':
        key = 'f' if src_gender == 'm' else 'm'
    else:
        key = 'mix'
    return models[f'bank_{key}'], models[f'fallback_{key}']


# ============================================================
# Source Feature Extraction
# ============================================================

def extract_source_features(audio_path, models, device='cuda'):
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


# ============================================================
# Duration Anonymization + Frame Interpolation
# ============================================================

def anonymize_duration(source, models, device='cuda'):
    l24 = source['l24']
    l12 = source['l12']
    src_phones = source['phones']
    dur_weight = models['dur_weight']

    unique_phones, phone_durations = [], []
    cur, cnt = src_phones[0], 1
    for i in range(1, len(src_phones)):
        if src_phones[i] == cur:
            cnt += 1
        else:
            unique_phones.append(cur)
            phone_durations.append(cnt)
            cur, cnt = src_phones[i], 1
    unique_phones.append(cur)
    phone_durations.append(cnt)

    phones_t = torch.tensor(unique_phones, dtype=torch.long, device=device)
    dur_true = torch.tensor(phone_durations, dtype=torch.float32, device=device)

    with torch.no_grad():
        dur_pred = models['duration_predictor'](phones_t).squeeze(0)
    dur_anon = (dur_weight * dur_pred + (1 - dur_weight) * dur_true).clamp(min=1).round().long()

    adj_l24, adj_l12, adj_phones = [], [], []
    idx = 0
    for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
        end = idx + orig
        new_len = new.item()

        if new_len == 0:
            idx = end
            continue

        if orig == 1:
            adj_l24.append(l24[idx:idx+1].expand(new_len, -1))
            adj_l12.append(l12[idx:idx+1].expand(new_len, -1))
        else:
            t = torch.linspace(0, 1, new_len, device=device)
            src_idx_f = t * (orig - 1)
            idx_low = src_idx_f.long().clamp(max=orig - 2)
            idx_high = idx_low + 1
            w = (src_idx_f - idx_low.float()).unsqueeze(-1)
            adj_l24.append((1 - w) * l24[idx + idx_low] + w * l24[idx + idx_high])
            adj_l12.append((1 - w) * l12[idx + idx_low] + w * l12[idx + idx_high])

        adj_phones.append(torch.full((new_len,), ph, dtype=torch.long, device=device))
        idx = end

    return {
        'l24': torch.cat(adj_l24, dim=0),
        'l12': torch.cat(adj_l12, dim=0),
        'phones': torch.cat(adj_phones, dim=0),
    }


# ============================================================
# Core V0: Pure kNN Baseline (最原始基线)
# ============================================================

def anonymize_knn_baseline(source, models, src_gender, mode,
                           device='cuda', top_k=4,
                           temperature=0.1):
    """
    [V0] 纯 kNN-VC 匿名基线

    最简单的检索策略：
      1) WavLM → L24 (查询) + phone 分桶
      2) 每帧在 bank 对应音素桶中找 Top-K 最近 L24 邻居
      3) Top-K 的 L6 简单平均 → HiFi-GAN

    无 L12 联合打分、无身份惩罚、无 duration 匿名
    等价于原始 kNN-VC 在全局音素桶上的直接应用
    """
    l24 = source['l24'].to(device)
    phones = source['phones']
    phones_t = torch.tensor(phones, dtype=torch.long, device=device).squeeze()

    bank, fallback = select_bank(models, src_gender, mode)

    T_out = l24.shape[0]
    h_anon = torch.zeros(T_out, 1024, device=device)

    pool_label = {'same': src_gender, 'cross': ('f' if src_gender == 'm' else 'm'), 'mix': 'mix'}[mode]

    for phone_id in tqdm(torch.unique(phones_t), desc=f"v0 [{mode}→{pool_label}]"):
        ph = int(phone_id.item())
        mask = (phones_t == phone_id)
        N_q = mask.sum().item()

        query_l24 = l24[mask]

        if ph in bank:
            tgt_l6  = bank[ph]['l6'].to(device)
            tgt_l24 = bank[ph]['l24'].to(device)
        else:
            tgt_l6  = fallback['l6'].to(device)
            tgt_l24 = fallback['l24'].to(device)

        N_t = tgt_l24.shape[0]
        if N_t == 0:
            continue
        if N_t <= top_k:
            h_anon[mask] = tgt_l6.mean(dim=0).expand(N_q, -1)
            continue

        actual_k = min(top_k, N_t)
        d_l24 = torch.cdist(query_l24, tgt_l24)
        topk_vals, topk_idx = d_l24.topk(actual_k, largest=False)

        topk_l6 = tgt_l6[topk_idx]  # [N_q, K, 1024]

        # 简单平均 (kNN-VC 原始方式) 或 temperature softmax
        weights = F.softmax(-topk_vals / temperature, dim=-1)
        h_anon[mask] = (topk_l6 * weights.unsqueeze(-1)).sum(dim=1)

    return h_anon.cpu()


# ============================================================
# Core V1: Joint-Score Retrieval (联合打分)
# ============================================================

def anonymize_joint_score(source, models, src_gender, mode,
                          device='cuda', top_n=100, top_k=8,
                          temperature=0.1, lambda_id=0.5):
    """
    [V1] 联合打分匿名化检索

    相对于 V0 的改进：
      + duration 匿名 (dur_weight 控制时长变化)
      + L12 联合打分: score = d_L24 + λ·sim_L12 (推远源身份)
      + Top-N 召回 + Top-K 重排序 (两阶段检索)

    取 score 最低的 Top-K → Temperature Softmax 加权 L6 → HiFi-GAN
    """
    dur_result = anonymize_duration(source, models, device)
    l24_adj = dur_result['l24'].to(device)
    l12_adj = dur_result['l12'].to(device)
    phones_adj = dur_result['phones'].to(device)

    bank, fallback = select_bank(models, src_gender, mode)

    e_src = F.normalize(source['l12'].mean(dim=0, keepdim=True), dim=-1).to(device)

    T_out = l24_adj.shape[0]
    h_anon = torch.zeros(T_out, 1024, device=device)

    pool_label = {'same': src_gender, 'cross': ('f' if src_gender == 'm' else 'm'), 'mix': 'mix'}[mode]

    for phone_id in tqdm(torch.unique(phones_adj), desc=f"Retrieval [{mode}→{pool_label}]"):
        ph = int(phone_id.item())
        mask = (phones_adj == phone_id)
        N_q = mask.sum().item()

        query_l24 = l24_adj[mask]
        query_l12 = l12_adj[mask]

        if ph in bank:
            tgt_l6  = bank[ph]['l6'].to(device)
            tgt_l12 = bank[ph]['l12'].to(device)
            tgt_l24 = bank[ph]['l24'].to(device)
        else:
            tgt_l6  = fallback['l6'].to(device)
            tgt_l12 = fallback['l12'].to(device)
            tgt_l24 = fallback['l24'].to(device)

        N_t = tgt_l24.shape[0]
        if N_t == 0:
            continue
        if N_t <= top_k:
            h_anon[mask] = tgt_l6.mean(dim=0).expand(N_q, -1)
            continue

        actual_n = min(top_n, N_t)
        d_l24_all = torch.cdist(query_l24, tgt_l24)
        topn_vals, topn_idx = d_l24_all.topk(actual_n, largest=False)

        cand_l12 = tgt_l12[topn_idx]
        cand_l6  = tgt_l6[topn_idx]

        cand_l12_norm = F.normalize(cand_l12, dim=-1)
        sim_l12 = (cand_l12_norm * e_src.unsqueeze(1)).sum(dim=-1)

        d_l24_normed = topn_vals / (topn_vals.max(dim=-1, keepdim=True).values + 1e-8)
        sim_l12_shifted = (sim_l12 + 1.0) / 2.0

        score = d_l24_normed + lambda_id * sim_l12_shifted

        actual_k = min(top_k, actual_n)
        topk_scores, topk_local_idx = score.topk(actual_k, largest=False)

        topk_l6 = torch.gather(
            cand_l6, dim=1,
            index=topk_local_idx.unsqueeze(-1).expand(-1, -1, 1024)
        )

        weights = F.softmax(-topk_scores / temperature, dim=-1)
        h_anon[mask] = (topk_l6 * weights.unsqueeze(-1)).sum(dim=1)

    return h_anon.cpu()


# ============================================================
# Core V2: Privacy-Enhanced Anonymization (隐私增强版)
# ============================================================

def _quantize_query_l24(query_l24, bank_l24, n_quant=8):
    """
    [P1] 查询 L24 通过 bank 侧 K-Means 聚类中心量化
    将源说话人的精细 L24 映射到 bank 的离散表示空间，
    切断 source → query 的精细身份映射。
    """
    N_bank = bank_l24.shape[0]
    actual_k = min(n_quant, N_bank)
    if actual_k <= 1:
        return bank_l24.mean(dim=0, keepdim=True).expand(query_l24.shape[0], -1)

    # 在 bank 侧做 K-Means（GPU 加速版，避免依赖 sklearn）
    # 用 K-Means++ 风格初始化 + 迭代
    device = query_l24.device
    perm = torch.randperm(N_bank, device=device)[:actual_k]
    centers = bank_l24[perm].clone()  # [K, D]

    for _ in range(10):  # 10 轮迭代足够收敛
        # assign
        dists = torch.cdist(bank_l24, centers)  # [N, K]
        labels = dists.argmin(dim=-1)            # [N]
        # update
        new_centers = torch.zeros_like(centers)
        for k in range(actual_k):
            members = bank_l24[labels == k]
            if len(members) > 0:
                new_centers[k] = members.mean(dim=0)
            else:
                new_centers[k] = centers[k]
        centers = new_centers

    # 将查询帧映射到最近的聚类中心
    q_dists = torch.cdist(query_l24, centers)  # [N_q, K]
    q_labels = q_dists.argmin(dim=-1)           # [N_q]
    return centers[q_labels]                    # [N_q, D]


def anonymize_joint_score_v2(source, models, src_gender, mode,
                              device='cuda', top_n=100, top_k=8,
                              temperature=0.1, lambda_id=0.5,
                              enable_p0=True, pool_sample_ratio=0.6,
                              enable_p1=True, query_quant_k=8,
                              enable_p2=True, noise_scale=0.05,
                              utt_seed=None):
    """
    [V2] 隐私增强匿名化检索（各模块可独立开关）

    Base 逻辑与 V1 完全一致：
      - 同样的 L12 联合打分: score = d_L24 + λ·sim_L12
      - 同样的 dur_weight / temperature / lambda_id 默认值
      - 全关 P0/P1/P2 时输出与 V1 一致

    三个独立增量模块：
      [P0] 随机子池采样 → 打破确定性映射
      [P1] 查询 L24 量化 → 擦除源精细指纹
      [P2] 帧级高斯噪声 → 破坏韵律指纹
    """
    # P0: 设置 per-utterance 随机种子
    if enable_p0 and utt_seed is not None:
        torch.manual_seed(utt_seed)

    # Duration 匿名 (dur_weight 通过 --dur-weight 独立控制)
    dur_result = anonymize_duration(source, models, device)
    l24_adj = dur_result['l24'].to(device)
    l12_adj = dur_result['l12'].to(device)
    phones_adj = dur_result['phones'].to(device)

    # [P2] 帧级高斯噪声扰动
    if enable_p2 and noise_scale > 0:
        l24_adj = l24_adj + torch.randn_like(l24_adj) * noise_scale

    bank, fallback = select_bank(models, src_gender, mode)

    # L12 源说话人质心 (与 V1 一致)
    e_src = F.normalize(source['l12'].mean(dim=0, keepdim=True), dim=-1).to(device)

    T_out = l24_adj.shape[0]
    h_anon = torch.zeros(T_out, 1024, device=device)

    active = [x for x, on in [('P0', enable_p0), ('P1', enable_p1), ('P2', enable_p2)] if on]
    tag = '+'.join(active) if active else 'base'
    pool_label = {'same': src_gender, 'cross': ('f' if src_gender == 'm' else 'm'), 'mix': 'mix'}[mode]

    for phone_id in tqdm(torch.unique(phones_adj), desc=f"v2[{tag}] [{mode}→{pool_label}]"):
        ph = int(phone_id.item())
        mask = (phones_adj == phone_id)
        N_q = mask.sum().item()

        query_l24 = l24_adj[mask]

        if ph in bank:
            tgt_l6  = bank[ph]['l6'].to(device)
            tgt_l12 = bank[ph]['l12'].to(device)
            tgt_l24 = bank[ph]['l24'].to(device)
        else:
            tgt_l6  = fallback['l6'].to(device)
            tgt_l12 = fallback['l12'].to(device)
            tgt_l24 = fallback['l24'].to(device)

        N_t = tgt_l24.shape[0]
        if N_t == 0:
            continue
        if N_t <= top_k:
            h_anon[mask] = tgt_l6.mean(dim=0).expand(N_q, -1)
            continue

        # [P0] 随机子池采样
        if enable_p0:
            sample_size = max(top_k + 1, int(N_t * pool_sample_ratio))
            sample_size = min(sample_size, N_t)
            perm = torch.randperm(N_t, device=device)[:sample_size]
            tgt_l24_use = tgt_l24[perm]
            tgt_l12_use = tgt_l12[perm]
            tgt_l6_use  = tgt_l6[perm]
        else:
            tgt_l24_use = tgt_l24
            tgt_l12_use = tgt_l12
            tgt_l6_use  = tgt_l6

        # [P1] 查询 L24 量化
        if enable_p1:
            query_l24_use = _quantize_query_l24(query_l24, tgt_l24_use, n_quant=query_quant_k)
        else:
            query_l24_use = query_l24

        # Top-N 召回 (与 V1 一致的 L24 距离)
        N_use = tgt_l24_use.shape[0]
        actual_n = min(top_n, N_use)
        d_l24 = torch.cdist(query_l24_use, tgt_l24_use)
        topn_vals, topn_idx = d_l24.topk(actual_n, largest=False)

        cand_l12 = tgt_l12_use[topn_idx]
        cand_l6  = tgt_l6_use[topn_idx]

        # 联合打分 (与 V1 一致): score = d_L24_norm + λ·sim_L12_shift
        cand_l12_norm = F.normalize(cand_l12, dim=-1)
        sim_l12 = (cand_l12_norm * e_src.unsqueeze(1)).sum(dim=-1)

        d_l24_normed = topn_vals / (topn_vals.max(dim=-1, keepdim=True).values + 1e-8)
        sim_l12_shifted = (sim_l12 + 1.0) / 2.0

        score = d_l24_normed + lambda_id * sim_l12_shifted

        actual_k = min(top_k, actual_n)
        topk_scores, topk_local_idx = score.topk(actual_k, largest=False)

        topk_l6 = torch.gather(
            cand_l6, dim=1,
            index=topk_local_idx.unsqueeze(-1).expand(-1, -1, 1024)
        )

        weights = F.softmax(-topk_scores / temperature, dim=-1)
        h_anon[mask] = (topk_l6 * weights.unsqueeze(-1)).sum(dim=1)

    return h_anon.cpu()


# ============================================================
# Utilities
# ============================================================

def save_wav(tensor, path, sr=16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(path), tensor.cpu(), sr)
    print(f"  saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Global Phonetic Anonymization")
    parser.add_argument('--audio',       default=DEFAULT_AUDIO)
    parser.add_argument('--output-dir',  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--src-gender',  default='m', choices=['m', 'f'],
                        help="源说话人性别")
    parser.add_argument('--mode',        default='all', choices=['same', 'cross', 'mix', 'all'],
                        help="匿名模式: same=同性别池, cross=跨性别池, mix=混合池, all=三种全跑")
    parser.add_argument('--device',      default='cuda')

    # 版本选择
    parser.add_argument('--version',     default='v2', choices=['v0', 'v1', 'v2'],
                        help="v0=纯 kNN 基线, v1=联合打分, v2=隐私增强")

    # 通用参数
    parser.add_argument('--top-n',       type=int,   default=100)
    parser.add_argument('--top-k',       type=int,   default=8)

    # 通用参数 (V1/V2 默认值一致)
    parser.add_argument('--dur-weight',  type=float, default=0.3,
                        help="duration 匿名权重 (默认 0.3)")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="Softmax 温度 (默认 0.1)")
    parser.add_argument('--lambda-id',   type=float, default=0.5,
                        help="L12 身份惩罚权重")

    # V2 模块开关 (消融实验用)
    parser.add_argument('--enable-p0',   action='store_true', default=True,
                        help="[P0] 启用随机子池采样")
    parser.add_argument('--disable-p0',  action='store_true',
                        help="[P0] 禁用随机子池采样")
    parser.add_argument('--enable-p1',   action='store_true', default=True,
                        help="[P1] 启用查询 L24 量化")
    parser.add_argument('--disable-p1',  action='store_true',
                        help="[P1] 禁用查询 L24 量化")
    parser.add_argument('--enable-p2',   action='store_true', default=True,
                        help="[P2] 启用帧级噪声")
    parser.add_argument('--disable-p2',  action='store_true',
                        help="[P2] 禁用帧级噪声")

    # V2 参数
    parser.add_argument('--pool-sample-ratio', type=float, default=0.6,
                        help="[P0] bank 子池采样比例 (0~1)")
    parser.add_argument('--query-quant-k',     type=int,   default=8,
                        help="[P1] 查询 L24 量化聚类数")
    parser.add_argument('--noise-scale',       type=float, default=0.05,
                        help="[P2] 帧级高斯噪声标准差")
    parser.add_argument('--utt-seed',          type=int,   default=None,
                        help="[P0] per-utterance 随机种子 (None=每次不同)")

    args = parser.parse_args()

    # 解析模块开关 (--disable-pX 优先于 --enable-pX)
    args.enable_p0 = not args.disable_p0
    args.enable_p1 = not args.disable_p1
    args.enable_p2 = not args.disable_p2

    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)

    modes = ['same', 'cross', 'mix'] if args.mode == 'all' else [args.mode]

    print("=" * 60)
    print(f"  Global Phonetic Anonymization [{args.version.upper()}]")
    print("=" * 60)
    print(f"  Audio:       {args.audio}")
    print(f"  Src gender:  {args.src_gender}")
    print(f"  Mode(s):     {', '.join(modes)}")
    print(f"  Version:     {args.version}")
    print(f"  dur_weight:  {args.dur_weight}")
    print(f"  Top-N={args.top_n}  Top-K={args.top_k}  T={args.temperature}")
    if args.version == 'v0':
        print(f"  (纯kNN基线, 无L12打分, 无duration匿名)")
    elif args.version == 'v1':
        print(f"  λ_id={args.lambda_id}")
    else:
        p0_s = '✅' if args.enable_p0 else '❌'
        p1_s = '✅' if args.enable_p1 else '❌'
        p2_s = '✅' if args.enable_p2 else '❌'
        print(f"  [P0] 随机子池: {p0_s}  ratio={args.pool_sample_ratio}  seed={args.utt_seed}")
        print(f"  [P1] 查询量化: {p1_s}  quant_k={args.query_quant_k}")
        print(f"  [P2] 帧级噪声: {p2_s}  noise={args.noise_scale}")

    wav, sr = torchaudio.load(args.audio)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    save_wav(wav.mean(dim=0), out_dir / 'original.wav')

    models = load_models(modes, device=device, dur_weight=args.dur_weight)
    source = extract_source_features(args.audio, models, device)
    print(f"  Source: L24 {source['l24'].shape}, "
          f"phones unique={len(set(source['phones'].flatten().tolist()))}")

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode} (src={args.src_gender}) [{args.version}]")
        print(f"{'='*60}")

        if args.version == 'v0':
            h_anon = anonymize_knn_baseline(
                source, models, args.src_gender, mode, device,
                top_k=args.top_k, temperature=args.temperature,
            )
        elif args.version == 'v1':
            h_anon = anonymize_joint_score(
                source, models, args.src_gender, mode, device,
                top_n=args.top_n, top_k=args.top_k,
                temperature=args.temperature, lambda_id=args.lambda_id,
            )
        else:
            h_anon = anonymize_joint_score_v2(
                source, models, args.src_gender, mode, device,
                top_n=args.top_n, top_k=args.top_k,
                temperature=args.temperature, lambda_id=args.lambda_id,
                enable_p0=args.enable_p0, pool_sample_ratio=args.pool_sample_ratio,
                enable_p1=args.enable_p1, query_quant_k=args.query_quant_k,
                enable_p2=args.enable_p2, noise_scale=args.noise_scale,
                utt_seed=args.utt_seed,
            )

        with torch.no_grad():
            wav_anon = models['vocoder'](h_anon.unsqueeze(0).to(device)).squeeze()

        # 文件名带版本+模块标记
        if args.version == 'v2':
            flags = []
            if args.enable_p0: flags.append('P0')
            if args.enable_p1: flags.append('P1')
            if args.enable_p2: flags.append('P2')
            tag = '+'.join(flags) if flags else 'base'
            suffix = f'{args.src_gender}_{mode}_v2_{tag}'
        else:
            suffix = f'{args.src_gender}_{mode}_{args.version}'
        save_wav(wav_anon, out_dir / f'anon_{suffix}.wav')

    print(f"\nDone. Output: {out_dir}")


if __name__ == '__main__':
    main()