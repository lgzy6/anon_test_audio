#!/usr/bin/env python3
"""
端到端匿名化测试脚本 (修复版)

正确流程：
1. 提取源音频的 WavLM 特征 s 和 speaker embedding d
2. 计算 η = s - (d @ A + b) 用于风格提取
3. 从 η 提取风格向量，选择目标话语
4. 使用原始 WavLM 特征 s 做 phone-constrained kNN
5. HiFi-GAN 合成音频

关键：kNN 和 vocoder 都使用原始 WavLM 特征，不是 η
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
    """加载所有必要的模型"""
    print("加载模型...")
    ckpt_dir = BASE_DIR / 'checkpoints'

    # 1. WavLM
    from models.ssl.wrappers import WavLMSSLExtractor
    wavlm = WavLMSSLExtractor(
        ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
        layer=15,
        device=device
    )

    # 2. ECAPA-TDNN
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/ecapa",
        run_opts={"device": device}
    )

    # 3. Speaker PCA
    import pickle
    with open(ckpt_dir / 'speaker_pca_model.pkl', 'rb') as f:
        speaker_pca = pickle.load(f)

    # 4. Eta projection
    eta_ckpt = torch.load(ckpt_dir / 'eta_projection.pt', map_location=device)
    A_star = eta_ckpt['A_star'].to(device)
    b_star = eta_ckpt['b_star'].to(device)

    # 5. Phone predictor
    from models.phone_predictor.predictor import PhonePredictor
    phone_predictor = PhonePredictor.load(
        str(ckpt_dir / 'phone_decoder.pt'),
        device=device
    )

    # 6. Style extractor
    from scripts.step4_build_style_extractor import SegmentStyleExtractor
    style_extractor = SegmentStyleExtractor.load(
        ckpt_dir / 'style_extractor.pkl'
    )

    # 7. HiFi-GAN
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

    # 加载预计算的风格
    styles_data = np.load(ckpt_dir / 'utterance_styles.npz')

    # 加载 metadata
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"✓ 目标池: {len(styles_data['styles'])} 话语")

    return {
        'styles': styles_data['styles'],
        'speaker_ids': styles_data['speaker_ids'],
        'genders': styles_data['genders'],
        'boundaries': styles_data['utt_boundaries'],
        'metadata': metadata,
        'features_path': cache_dir / 'features.h5'
    }


def extract_source_features(audio_path, models, device='cuda'):
    """提取源音频特征"""
    print(f"\n提取源音频特征: {audio_path}")

    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # 1. WavLM 特征 (原始)
    with torch.no_grad():
        wavlm_feats = models['wavlm'](waveform).squeeze(0).cpu().numpy()  # (T, 1024)

    # 2. Speaker embedding
    spk_emb = models['ecapa'].encode_batch(waveform).squeeze().cpu().numpy()  # (192,)
    spk_emb_pca = models['speaker_pca'].transform(spk_emb.reshape(1, -1))[0]  # (128,)

    # 3. 计算 η (仅用于风格提取)
    s = torch.from_numpy(wavlm_feats).float().to(device)
    d = torch.from_numpy(spk_emb_pca).float().to(device)
    eta = (s - (d @ models['A_star'] + models['b_star']).unsqueeze(0)).cpu().numpy()

    # 4. Phone 预测
    with torch.no_grad():
        phones = models['phone_predictor'](s).cpu().numpy()

    # 5. 风格提取 (从 η)
    style_vec = models['style_extractor'].extract_utterance_style(eta, phones)

    print(f"✓ WavLM: {wavlm_feats.shape}, η: {eta.shape}, style: {style_vec.shape}")

    return {
        'wavlm': wavlm_feats,  # 原始特征，用于 kNN
        'eta': eta,            # 仅用于风格提取
        'phones': phones,
        'style': style_vec,
        'speaker_emb': spk_emb_pca
    }


def select_target_utterance(source_style, source_gender, target_pool, mode='preserve'):
    """选择目标话语"""
    print(f"\n选择目标话语 (mode={mode})...")

    # 性别过滤
    valid_idx = []
    for i, gender in enumerate(target_pool['genders']):
        if gender == source_gender:
            valid_idx.append(i)

    if not valid_idx:
        print("  警告: 无同性别话语，使用全部")
        valid_idx = list(range(len(target_pool['styles'])))

    if mode == 'random':
        target_idx = np.random.choice(valid_idx)
    else:
        # 风格相似度
        valid_styles = target_pool['styles'][valid_idx]
        sims = np.dot(valid_styles, source_style) / (
            np.linalg.norm(valid_styles, axis=1) * np.linalg.norm(source_style) + 1e-8
        )
        best_local_idx = np.argmax(sims)
        target_idx = valid_idx[best_local_idx]
        print(f"  风格相似度: {sims[best_local_idx]:.3f}")

    target_spk = target_pool['speaker_ids'][target_idx]
    print(f"✓ 选择话语 {target_idx}, 说话人: {target_spk}")

    return target_idx


def anonymize_with_knn(source_wavlm, source_phones, target_idx, target_pool, phone_predictor, device='cuda', k=4, use_phone_constraint=True):
    """
    Phone-constrained kNN 匿名化 (改进版)

    关键改进：
    1. Top-k 平均：选择 k 个最相似帧并平均（自然平滑）
    2. 可选的 phone 约束
    3. 正确的余弦相似度计算

    Args:
        source_wavlm: 源音频的原始 WavLM 特征
        source_phones: 源音频的音素序列
        target_idx: 目标话语索引
        target_pool: 目标池数据
        phone_predictor: 音素预测器
        device: 设备
        k: Top-k 平均的 k 值（默认 4）
        use_phone_constraint: 是否使用 phone 约束
    """
    print(f"\nkNN 检索 (k={k}, phone_constraint={use_phone_constraint})...")

    # 1. 加载目标话语的原始 WavLM 特征
    start, end = target_pool['boundaries'][target_idx]

    with h5py.File(target_pool['features_path'], 'r') as f:
        target_feats = f['features'][start:end]

    # 2. 转换为 tensor
    src_feats = torch.from_numpy(source_wavlm).float().to(device)
    tgt_feats = torch.from_numpy(target_feats).float().to(device)

    if use_phone_constraint:
        # 预测目标音素
        print("  预测目标音素...")
        with torch.no_grad():
            target_phones = phone_predictor(tgt_feats).cpu().numpy()

        h_anon = torch.zeros_like(src_feats)
        unique_phones = np.unique(source_phones)

        # 按 phone 分别检索
        for phone_id in tqdm(unique_phones, desc="kNN"):
            if phone_id == 0:  # silence
                src_mask = source_phones == 0
                h_anon[src_mask] = src_feats[src_mask]
                continue

            src_mask = torch.from_numpy(source_phones == phone_id).to(device)
            tgt_mask = (target_phones == phone_id)

            if tgt_mask.sum() == 0:
                h_anon[src_mask] = src_feats[src_mask]
                continue

            # Top-k 平均
            src_batch = torch.nn.functional.normalize(src_feats[src_mask], dim=-1)
            tgt_batch = torch.nn.functional.normalize(tgt_feats[tgt_mask], dim=-1)

            cos_sim = torch.mm(src_batch, tgt_batch.T)
            actual_k = min(k, tgt_batch.shape[0])
            topk_values, topk_indices = cos_sim.topk(k=actual_k, dim=1)

            # 平均 top-k 帧
            tgt_phone_feats = tgt_feats[tgt_mask]
            selected = tgt_phone_feats[topk_indices]  # (n_src, k, 1024)
            h_anon[src_mask] = selected.mean(dim=1)

    else:
        # 无 phone 约束，全局检索
        print("  全局 kNN 检索...")
        src_norm = torch.nn.functional.normalize(src_feats, dim=-1)
        tgt_norm = torch.nn.functional.normalize(tgt_feats, dim=-1)

        cos_sim = torch.mm(src_norm, tgt_norm.T)
        actual_k = min(k, tgt_feats.shape[0])
        topk_values, topk_indices = cos_sim.topk(k=actual_k, dim=1)

        # 平均 top-k 帧
        selected = tgt_feats[topk_indices]  # (T, k, 1024)
        h_anon = selected.mean(dim=1)

    print(f"✓ 匿名化完成: {h_anon.shape}")

    return h_anon.cpu().numpy()


def synthesize_audio(h_anon, vocoder, output_path):
    """合成音频"""
    print(f"\n合成音频: {output_path}")

    # 转换为 tensor
    h_anon_tensor = torch.from_numpy(h_anon).unsqueeze(0).to(vocoder.device)

    # 生成波形
    with torch.no_grad():
        waveform = vocoder(h_anon_tensor).squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # torchaudio.save 需要 2D tensor
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(output_path), waveform.cpu(), 16000)

    print(f"✓ 保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='源音频路径')
    parser.add_argument('--output', type=str, required=True, help='输出音频路径')
    parser.add_argument('--mode', type=str, default='preserve',
                        choices=['preserve', 'random'], help='检索模式')
    parser.add_argument('--gender', type=str, default='f',
                        choices=['f', 'm'], help='源音频性别')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("端到端匿名化测试 (修复版)")
    print("="*60)

    # 1. 加载模型
    models = load_models(device)

    # 2. 加载目标池
    target_pool = load_target_pool(device)

    # 3. 提取源特征
    source = extract_source_features(args.audio, models, device)

    # 4. 选择目标话语
    target_idx = select_target_utterance(
        source['style'], args.gender, target_pool, args.mode
    )

    # 5. 匿名化 (Top-k 平均，自然平滑)
    h_anon = anonymize_with_knn(
        source['wavlm'],
        source['phones'],
        target_idx,
        target_pool,
        models['phone_predictor'],
        device,
        k=4,  # Top-4 平均
        use_phone_constraint=True  # 使用 phone 约束
    )

    # 6. 合成音频
    synthesize_audio(h_anon, models['vocoder'], args.output)

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)


if __name__ == '__main__':
    main()
