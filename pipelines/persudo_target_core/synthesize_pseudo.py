#!/usr/bin/env python3
"""
基于伪风格 Bank 的端到端匿名化合成 (带 Duration 适配)
"""

import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent


def load_models(bank_path, device='cuda', dur_weight=0.5):
    """加载模型与全局伪风格 Bank (大幅精简)"""
    print("加载模型与 Pseudo-Bank...")
    ckpt_dir = BASE_DIR / 'checkpoints'

    # 1. WavLM 提取器
    from models.ssl.wrappers import WavLMSSLExtractor
    wavlm = WavLMSSLExtractor(
        ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
        layer=6,
        device=device
    )

    # 2. 音素与时长预测器
    from models.phone_predictor.predictor import PhonePredictor
    phone_predictor = PhonePredictor.load(str(ckpt_dir / 'phone_decoder.pt'), device=device)
    duration_predictor = PhonePredictor.load(str(ckpt_dir / 'duration_decoder.pt'), device=device)

    # 3. HiFi-GAN 神经声码器
    from models.vocoder.hifigan import HiFiGAN
    vocoder = HiFiGAN.load(checkpoint_path=str(ckpt_dir / 'hifigan.pt'), device=device)

    # 4. 加载我们构建的 pseudo_bank
    pseudo_bank = torch.load(bank_path, map_location=device)
    # 构建一个全局后备池，防止遇到极罕见音素
    global_fallback_bank = torch.cat(list(pseudo_bank.values()), dim=0)

    print(f"✓ 模型与 Bank 加载完成 (包含 {len(pseudo_bank)} 个有效音素簇)")

    return {
        'wavlm': wavlm,
        'phone_predictor': phone_predictor,
        'duration_predictor': duration_predictor,
        'vocoder': vocoder,
        'pseudo_bank': pseudo_bank,
        'global_fallback': global_fallback_bank,
        'dur_weight': dur_weight
    }


def extract_source_features(audio_path, models, device='cuda'):
    """提取源音频特征 (仅需 L6 和 Phones)"""
    print(f"\n提取源音频特征: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # 强制转为单声道
    waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        waveform = waveform.to(device)

        # 使用 forward_multi_layer 提取多层
        multi_feats = models['wavlm'].forward_multi_layer(waveform, layers=[6, 24])

        # 兼容字典返回格式
        wavlm_vc_feats = multi_feats[6].squeeze(0).cpu().numpy()
        wavlm_phone_feats = multi_feats[24].squeeze(0)

        phones = models['phone_predictor'](wavlm_phone_feats).cpu().numpy()

    return {
        'wavlm': wavlm_vc_feats,
        'phones': phones
    }


def anonymize_with_duration(source_wavlm, source_phones, models, device='cuda', k=4):
    """基于 Pseudo-Bank 和 Duration 的 kNN 匹配"""
    print(f"\n音素约束 kNN 检索 (k={k}, dur_weight={models['dur_weight']})...")

    src_feats = torch.from_numpy(source_wavlm).float().to(device)
    dur_weight = models['dur_weight']
    pseudo_bank = models['pseudo_bank']

    # --- 1. 获取连续音素段及其实际时长 ---
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

    # --- 2. 预测时长并插值 ---
    with torch.no_grad():
        pred_durations = models['duration_predictor'].model(phones_tensor).squeeze(0)

    interpolated_durations = dur_weight * pred_durations + (1 - dur_weight) * durations_tensor.squeeze(0)
    n_frames = torch.round(interpolated_durations).clamp(min=1).long()

    # --- 3. 调整源特征帧数 (插值对齐) ---
    adjusted_feats, adjusted_phones = [], []
    feat_idx_start = 0

    for seg_idx, (phone_id, orig_dur, new_dur) in enumerate(zip(unique_phones, phone_durations, n_frames)):
        feat_idx_end = feat_idx_start + orig_dur - 1
        indices = torch.linspace(feat_idx_start, feat_idx_end, new_dur.item(), dtype=torch.long, device=device)
        adjusted_feats.append(src_feats[indices])
        adjusted_phones.append(torch.full((new_dur.item(),), phone_id, dtype=torch.long, device=device))
        feat_idx_start = feat_idx_end + 1

    src_feats_adj = torch.cat(adjusted_feats, dim=0)
    src_phones_adj = torch.cat(adjusted_phones, dim=0)

    # --- 4. Pseudo-Bank Phone-level kNN 匹配 (排斥 Top-3 + 狄利克雷混合) ---
    h_anon = torch.zeros_like(src_feats_adj)
    unique_phone_ids = torch.unique(src_phones_adj)

    for phone_id in tqdm(unique_phone_ids, desc="kNN"):
        ph_int = int(phone_id.item())
        mask = src_phones_adj == phone_id
        query_batch = src_feats_adj[mask]

        # 路由策略：如果音素在 Bank 中，使用对应的桶；否则使用全局后备池
        if ph_int in pseudo_bank:
            tgt_candidates = pseudo_bank[ph_int]
        else:
            tgt_candidates = models['global_fallback']

        # 计算欧氏距离
        dists = torch.cdist(query_batch, tgt_candidates) # [N_query, N_tgt]

        # --- 排斥 Top-3 + Top-20 随机采样 + 狄利克雷混合 ---
        N_query = query_batch.shape[0]
        N_tgt = tgt_candidates.shape[0]
        skip_top = 3
        pool_size = 20
        sample_k = k
        actual_N = min(skip_top + pool_size, N_tgt)

        if actual_N >= skip_top + sample_k:
            top_n_indices = dists.topk(actual_N, largest=False).indices
            candidate_indices = top_n_indices[:, skip_top:]
            rand_idx = torch.randint(0, candidate_indices.shape[1], (N_query, sample_k), device=device)
            sampled_tgt_indices = torch.gather(candidate_indices, 1, rand_idx)
            matched_k_frames = tgt_candidates[sampled_tgt_indices]

            alphas = torch.ones(sample_k, device=device)
            dirichlet_dist = torch.distributions.dirichlet.Dirichlet(alphas)
            weights = dirichlet_dist.sample((N_query,)).unsqueeze(-1)
            h_anon[mask] = (matched_k_frames * weights).sum(dim=1)
        else:
            actual_k = min(sample_k, N_tgt)
            topk_indices = dists.topk(actual_k, largest=False).indices
            matched_k_frames = tgt_candidates[topk_indices]
            h_anon[mask] = matched_k_frames.mean(dim=1)

    print(f"✓ 匿名化完成: 源帧数 {src_feats.shape[0]} -> 目标帧数 {h_anon.shape[0]}")
    return h_anon.cpu().numpy()


def synthesize_audio(h_anon, vocoder, output_path):
    """合成音频"""
    print(f"\n合成音频: {output_path}")
    h_anon_tensor = torch.from_numpy(h_anon).unsqueeze(0).to(vocoder.device)



    with torch.no_grad():
        waveform = vocoder(h_anon_tensor).squeeze()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(output_path), waveform.cpu(), 16000)
    print(f"✓ 保存成功: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help="输入音频路径")
    parser.add_argument('--output', type=str, required=True, help="输出音频路径")
    parser.add_argument('--bank', type=str, default="checkpoints/pseudo_bank.pt", help="伪风格 Bank 路径")
    parser.add_argument('--k', type=int, default=4, help='Top-k 平均检索')
    parser.add_argument('--dur_weight', type=float, default=0.5, help='时长预测权重 (0-1)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_comparison', action='store_true', help='保存对比音频（原始+匿名化）')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("端到端匿名化 (基于 Pseudo-Bank & Duration 适配)")
    print("="*60)

    models = load_models(args.bank, device, dur_weight=args.dur_weight)

    # 提取源特征
    source = extract_source_features(args.audio, models, device)

    # 执行 kNN 检索与时长插值
    h_anon = anonymize_with_duration(
        source['wavlm'], source['phones'], models, device, k=args.k
    )

    # 声码器合成
    synthesize_audio(h_anon, models['vocoder'], args.output)

    # 保存对比音频
    if args.save_comparison:
        output_path = Path(args.output)
        comparison_path = output_path.parent / f"{output_path.stem}_comparison.wav"

        print(f"\n生成对比音频: {comparison_path}")
        original_wav, sr = torchaudio.load(args.audio)
        if sr != 16000:
            original_wav = torchaudio.functional.resample(original_wav, sr, 16000)

        # 强制转为单声道，保证与 HiFi-GAN 输出匹配
        original_wav = original_wav.mean(dim=0, keepdim=True)

        anon_wav, _ = torchaudio.load(args.output)

        # 拼接：原始 + 0.5秒静音 + 匿名化
        silence = torch.zeros(1, 8000)
        comparison = torch.cat([original_wav, silence, anon_wav], dim=1)
        torchaudio.save(str(comparison_path), comparison, 16000)
        print(f"✓ 对比音频已保存: {comparison_path}")

    print("\n" + "="*60)
    print("✓ 全部流程执行完毕！")
    print("="*60)


if __name__ == '__main__':
    main()