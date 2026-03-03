#!/usr/bin/env python3
"""
SAMM-Anon 完整测试流程: 解耦 → 风格聚类 → 匿名化生成 → 效果评估

使用方法:
    python -m tests.run_full_pipeline_test
    python -m tests.run_full_pipeline_test --audio /path/to/audio.flac
    python -m tests.run_full_pipeline_test --skip-disentangle  # 跳过解耦测试
"""

import sys
import os
import argparse
import numpy as np
import torch
import torchaudio
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 全局配置
# ============================================================

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / 'cache'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
FEATURES_H5 = CACHE_DIR / 'features' / 'wavlm' / 'features.h5'
METADATA_JSON = CACHE_DIR / 'features' / 'wavlm' / 'metadata.json'
TARGET_POOL_DIR = BASE_DIR / 'data' / 'samm_anon' / 'target_pool_v52'
LIBRISPEECH_DIR = Path('/root/autodl-tmp/datasets/LibriSpeech/test-clean')


def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================
# Part 1: 解耦测试 (Disentanglement)
# ============================================================

def run_disentanglement_test(max_samples=50000):
    """
    解耦验证: 投影后特征是否成功移除了音素(内容)信息

    方法: 在 H_original 和 H_style 上分别训练 Linear Probe 分类音素,
          H_style 上准确率应显著下降
    """
    print("\n" + "=" * 70)
    print("PART 1: Disentanglement Test (解耦验证)")
    print("=" * 70)

    import h5py
    from models.phone_predictor.predictor import PhonePredictor

    # 1. 加载数据
    print("\n[1.1] Loading WavLM features...")
    with h5py.File(str(FEATURES_H5), 'r') as f:
        total = f['features'].shape[0]
        n = min(max_samples, total)
        if total > n:
            idx = np.sort(np.random.choice(total, n, replace=False))
            features = f['features'][idx]
        else:
            features = f['features'][:]
    print(f"  Loaded {len(features)} frames, shape={features.shape}")

    # 2. 预测音素
    print("[1.2] Predicting phones...")
    phone_ckpt = str(CHECKPOINTS_DIR / 'phone_decoder.pt')
    if Path(phone_ckpt).exists():
        predictor = PhonePredictor.load(phone_ckpt, device='cuda')
        feat_t = torch.from_numpy(features).float().cuda()
        phones_list = []
        for i in range(0, len(feat_t), 10000):
            phones_list.append(predictor(feat_t[i:i+10000]).cpu().numpy())
        phones = np.concatenate(phones_list)
        del feat_t
        torch.cuda.empty_cache()
    else:
        print("  WARNING: phone_decoder.pt not found, using random labels")
        phones = np.random.randint(0, 41, len(features))
    print(f"  Phone range: [{phones.min()}, {phones.max()}], unique: {len(np.unique(phones))}")

    # 3. 剔除 silence (phone=0)
    mask = phones != 0
    feat_ns = features[mask]
    ph_ns = phones[mask]
    print(f"  After removing silence: {len(feat_ns)} frames")

    # 4. 学习内容子空间 (ExpandedSubspace)
    print("\n[1.3] Learning content subspace (ExpandedSubspace, dim=100)...")
    from tests.test_v32_disentanglement_v3 import ExpandedSubspaceProjector
    projector = ExpandedSubspaceProjector(
        n_phones=41, feature_dim=features.shape[1], n_components=100
    )
    projector.fit(features, phones, max_per_class=2000)

    # 5. 投影
    feat_style = projector.project_to_style(feat_ns)
    feat_content = projector.get_content_component(feat_ns)

    # 6. Linear Probe 评估
    print("\n[1.4] Running Linear Probe...")
    from tests.test_v32_disentanglement_v3 import GPULinearProbe

    n_samples = len(feat_ns)
    idx = np.random.permutation(n_samples)
    split = int(n_samples * 0.8)
    tr, te = idx[:split], idx[split:]

    results = {}

    for name, feat in [("H_original", feat_ns), ("H_style", feat_style), ("H_content", feat_content)]:
        print(f"\n  --- Probing: {name} ---")
        probe = GPULinearProbe(feat.shape[1], 41)
        probe.fit(feat[tr], ph_ns[tr], epochs=50, batch_size=2048)
        acc = probe.score(feat[te], ph_ns[te])
        results[name] = acc
        print(f"    Test Accuracy: {acc:.2%}")

    # 7. 汇总
    print("\n" + "-" * 50)
    print("Disentanglement Results:")
    print(f"  H_original accuracy:  {results['H_original']:.2%}")
    print(f"  H_style accuracy:     {results['H_style']:.2%}  (should be LOW)")
    print(f"  H_content accuracy:   {results['H_content']:.2%}  (should be HIGH)")
    print(f"  Random baseline:      {1/41:.2%}")

    reduction = (1 - results['H_style'] / results['H_original']) * 100
    print(f"  Content reduction:    {reduction:.1f}%")

    if results['H_style'] < 0.40:
        verdict = "SUCCESS"
    elif results['H_style'] < results['H_original'] * 0.6:
        verdict = "PARTIAL"
    else:
        verdict = "NEEDS IMPROVEMENT"
    print(f"  Verdict: {verdict}")

    return results, projector


# ============================================================
# Part 2: 风格聚类 (Style Clustering)
# ============================================================

def run_style_clustering_test(projector, output_dir, n_clusters=8, max_utterances=500):
    """
    风格聚类: 在 H_style 上提取句子级 embedding 并聚类

    验证: 聚类结构是否与说话人身份解耦 (ARI/NMI 应低)
    """
    print("\n" + "=" * 70)
    print("PART 2: Style Clustering Test (风格聚类)")
    print("=" * 70)

    import h5py
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. 加载 utterance 元数据
    print("\n[2.1] Loading utterance metadata...")
    with open(str(METADATA_JSON), 'r') as f:
        metadata = json.load(f)

    utterances = metadata['utterances'][:max_utterances]
    print(f"  Processing {len(utterances)} utterances...")

    # 2. 提取句子级风格 embedding
    print("[2.2] Extracting utterance-level style embeddings...")
    embeddings = []
    speaker_ids = []
    genders = []

    with h5py.File(str(FEATURES_H5), 'r') as f:
        features_ds = f['features']
        for utt in utterances:
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            h = features_ds[s:e][:]
            h_style = projector.project_to_style(h)
            emb = h_style.mean(axis=0)
            embeddings.append(emb)
            speaker_ids.append(utt['speaker_id'])
            genders.append(utt.get('gender', 'unknown'))

    embeddings = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Speakers: {len(set(speaker_ids))}")

    # 3. KMeans 聚类
    print(f"\n[2.3] Running KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    # 4. 计算聚类指标
    print("[2.4] Computing cluster metrics...")
    unique_speakers = list(set(speaker_ids))
    spk_to_id = {s: i for i, s in enumerate(unique_speakers)}
    spk_ids_num = np.array([spk_to_id[s] for s in speaker_ids])

    gender_map = {'m': 0, 'f': 1, 'male': 0, 'female': 1, 'unknown': 2}
    gender_ids = np.array([gender_map.get(g.lower(), 2) for g in genders])

    ari_spk = adjusted_rand_score(spk_ids_num, cluster_ids)
    nmi_spk = normalized_mutual_info_score(spk_ids_num, cluster_ids)
    ari_gen = adjusted_rand_score(gender_ids, cluster_ids)
    nmi_gen = normalized_mutual_info_score(gender_ids, cluster_ids)

    sil_score = silhouette_score(embeddings, cluster_ids) if n_clusters > 1 else 0

    # 5. t-SNE 可视化
    print("[2.5] Running t-SNE visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: 按聚类着色
    sc1 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_ids, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title(f'Style Clusters (K={n_clusters})')
    plt.colorbar(sc1, ax=axes[0])

    # 图2: 按说话人着色
    sc2 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=spk_ids_num, cmap='tab20', alpha=0.6, s=20)
    axes[1].set_title(f'By Speaker ({len(unique_speakers)} speakers)')

    # 图3: 按性别着色
    sc3 = axes[2].scatter(emb_2d[:, 0], emb_2d[:, 1], c=gender_ids, cmap='coolwarm', alpha=0.6, s=20)
    axes[2].set_title('By Gender')

    plt.tight_layout()
    plot_path = output_dir / 'style_clusters.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 汇总
    print("\n" + "-" * 50)
    print("Style Clustering Results:")
    print(f"  Utterances: {len(embeddings)}, Speakers: {len(unique_speakers)}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Silhouette Score:   {sil_score:.3f}  (higher = better cluster structure)")
    print(f"  ARI vs Speaker:     {ari_spk:.3f}  (lower = better decoupling)")
    print(f"  NMI vs Speaker:     {nmi_spk:.3f}  (lower = better decoupling)")
    print(f"  ARI vs Gender:      {ari_gen:.3f}")
    print(f"  NMI vs Gender:      {nmi_gen:.3f}")

    if ari_spk < 0.1:
        verdict = "SUCCESS - well decoupled from speaker"
    elif ari_spk < 0.3:
        verdict = "PARTIAL - some decoupling"
    else:
        verdict = "WARNING - clusters correlate with speaker"
    print(f"  Verdict: {verdict}")
    print(f"  Visualization: {plot_path}")

    return {
        'ari_speaker': ari_spk, 'nmi_speaker': nmi_spk,
        'ari_gender': ari_gen, 'nmi_gender': nmi_gen,
        'silhouette': sil_score,
    }


# ============================================================
# Part 3: 匿名化音频生成 (Simplified Pipeline)
# ============================================================

def run_anonymization(audio_path, output_dir, source_gender='M'):
    """
    简化匿名化流程 (跳过缺失的 SAMM/Eta-WavLM 组件):
      WavLM → Phone Prediction → kNN Retrieval (target pool v52) → HiFi-GAN
    """
    print("\n" + "=" * 70)
    print("PART 3: Anonymization (匿名化生成)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 0. 加载原始音频
    print(f"\n[3.0] Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    duration_sec = waveform.shape[1] / 16000
    print(f"  Duration: {duration_sec:.2f}s, Sample rate: 16000 Hz")

    # 保存原始音频 (16k mono)
    original_path = output_dir / 'original.wav'
    torchaudio.save(str(original_path), waveform, 16000)
    print(f"  Saved original: {original_path}")

    # 1. WavLM 特征提取
    print("\n[3.1] Extracting WavLM features...")
    from models.ssl.wrappers import WavLMSSLExtractor
    ssl = WavLMSSLExtractor(
        ckpt_path=str(CHECKPOINTS_DIR / 'WavLM-Large.pt'),
        layer=15, device=device
    )
    wav_input = waveform.squeeze(0).to(device)
    with torch.inference_mode():
        h_ssl = ssl(wav_input.unsqueeze(0)).squeeze(0)  # [T, 1024]
    print(f"  SSL features: {h_ssl.shape}")

    # 2. Phone Prediction
    print("\n[3.2] Predicting phones...")
    from models.phone_predictor.predictor import PhonePredictor
    phone_predictor = PhonePredictor.load(
        str(CHECKPOINTS_DIR / 'phone_decoder.pt'), device=device
    )
    with torch.inference_mode():
        phones = phone_predictor(h_ssl)  # [T]
    print(f"  Phones: {phones.shape}, unique: {len(torch.unique(phones))}")

    # 3. kNN Retrieval from Target Pool v52
    print("\n[3.3] kNN Retrieval from target pool...")
    h_anon = knn_retrieve_from_pool(h_ssl, phones, source_gender, device)
    print(f"  Anonymized features: {h_anon.shape}")

    # 4. HiFi-GAN Vocoder
    print("\n[3.4] Synthesizing with HiFi-GAN...")
    from models.vocoder.hifigan import HiFiGAN
    vocoder = HiFiGAN.load(
        str(CHECKPOINTS_DIR / 'hifigan.pt'), device=device
    )
    with torch.inference_mode():
        waveform_anon = vocoder(h_anon)  # [L]
    print(f"  Synthesized waveform: {waveform_anon.shape}, "
          f"duration: {len(waveform_anon)/16000:.2f}s")

    # 保存匿名音频
    anon_path = output_dir / 'anonymized.wav'
    torchaudio.save(str(anon_path), waveform_anon.unsqueeze(0).cpu(), 16000)
    print(f"  Saved anonymized: {anon_path}")

    # 清理 GPU 内存
    del ssl, vocoder, h_ssl, h_anon, waveform_anon
    torch.cuda.empty_cache()

    return str(original_path), str(anon_path)


def knn_retrieve_from_pool(h_ssl, phones, source_gender, device):
    """
    从 target_pool_v52 进行 kNN 检索

    使用 cosine similarity Top-1 策略，带 phone 约束
    """
    # 加载 pool
    pool_features = torch.from_numpy(
        np.load(str(TARGET_POOL_DIR / 'features.npy'))
    ).float().to(device)
    pool_genders = torch.from_numpy(
        np.load(str(TARGET_POOL_DIR / 'genders.npy'))
    ).long().to(device)

    # 确定目标性别 (same gender)
    target_gender = 0 if source_gender == 'M' else 1

    # 性别约束
    gender_mask = (pool_genders == target_gender)
    if gender_mask.sum() == 0:
        print("  WARNING: No samples for target gender, using all")
        gender_mask = torch.ones(len(pool_features), dtype=torch.bool, device=device)

    candidates = pool_features[gender_mask]
    print(f"  Pool: {len(pool_features)} total, {len(candidates)} with gender={target_gender}")

    # 预计算 candidates 的归一化
    candidates_norm = candidates / (candidates.norm(dim=-1, keepdim=True) + 1e-8)

    # 逐帧 cosine Top-1 检索
    T = h_ssl.shape[0]
    h_anon = torch.zeros_like(h_ssl)

    # 分批处理以节省显存
    batch_size = 512
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        h_batch = h_ssl[start:end]  # [B, D]
        h_norm = h_batch / (h_batch.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity: [B, N_candidates]
        sim = torch.mm(h_norm, candidates_norm.T)
        nearest_idx = sim.argmax(dim=-1)  # [B]
        h_anon[start:end] = candidates[nearest_idx]

    # 释放 pool
    del pool_features, pool_genders, candidates, candidates_norm
    torch.cuda.empty_cache()

    return h_anon


# ============================================================
# Part 4: 效果评估
# ============================================================

def run_evaluation(original_path, anonymized_path, output_dir):
    """
    评估匿名化效果:
    1. 特征级诊断 (帧连续性、分布对比)
    2. ASR 语义保留 (Whisper WER)
    """
    print("\n" + "=" * 70)
    print("PART 4: Evaluation (效果评估)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # --- 4A: 特征级诊断 ---
    print("\n[4A] Feature-level diagnosis...")

    waveform_orig, sr = torchaudio.load(original_path)
    waveform_anon, sr2 = torchaudio.load(anonymized_path)
    if sr != 16000:
        waveform_orig = torchaudio.transforms.Resample(sr, 16000)(waveform_orig)
    if sr2 != 16000:
        waveform_anon = torchaudio.transforms.Resample(sr2, 16000)(waveform_anon)

    from models.ssl.wrappers import WavLMSSLExtractor
    ssl = WavLMSSLExtractor(
        ckpt_path=str(CHECKPOINTS_DIR / 'WavLM-Large.pt'),
        layer=15, device=device
    )

    with torch.inference_mode():
        h_orig = ssl(waveform_orig.squeeze(0).to(device).unsqueeze(0)).squeeze(0)
        h_anon = ssl(waveform_anon.squeeze(0).to(device).unsqueeze(0)).squeeze(0)

    # 帧连续性
    def frame_continuity(h, name):
        h_norm = torch.nn.functional.normalize(h, dim=-1)
        cos = (h_norm[:-1] * h_norm[1:]).sum(dim=-1)
        mean_cos = cos.mean().item()
        low_ratio = (cos < 0.5).float().mean().item()
        print(f"  {name}: mean_cos={mean_cos:.4f}, low_ratio={low_ratio:.1%}")
        return mean_cos

    print("  Frame continuity:")
    cont_orig = frame_continuity(h_orig, "Original")
    cont_anon = frame_continuity(h_anon, "Anonymized")
    results['continuity_original'] = cont_orig
    results['continuity_anonymized'] = cont_anon

    # 说话人相似度 (句子级 embedding cosine)
    emb_orig = h_orig.mean(dim=0)
    emb_anon = h_anon.mean(dim=0)
    spk_sim = torch.nn.functional.cosine_similarity(
        emb_orig.unsqueeze(0), emb_anon.unsqueeze(0)
    ).item()
    print(f"\n  Speaker similarity (lower=better anonymization): {spk_sim:.4f}")
    results['speaker_similarity'] = spk_sim

    del ssl, h_orig, h_anon
    torch.cuda.empty_cache()

    # --- 4B: Whisper ASR 评估 ---
    print("\n[4B] Whisper ASR evaluation...")
    try:
        import whisper
    except ImportError:
        print("  Installing openai-whisper...")
        os.system("pip install -q openai-whisper")
        import whisper

    whisper_model = whisper.load_model("base")

    def transcribe(audio_path):
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.mean(0).numpy().astype(np.float32)
        result = whisper_model.transcribe(wav, language="en", verbose=False)
        return result["text"].strip()

    print("  Transcribing original...")
    text_orig = transcribe(original_path)
    print(f"    Original: \"{text_orig}\"")

    print("  Transcribing anonymized...")
    text_anon = transcribe(anonymized_path)
    print(f"    Anonymized: \"{text_anon}\"")

    # WER
    wer, edit_dist = calculate_wer(text_orig, text_anon)
    jaccard = calculate_jaccard(text_orig, text_anon)

    results['text_original'] = text_orig
    results['text_anonymized'] = text_anon
    results['wer'] = wer
    results['edit_distance'] = edit_dist
    results['jaccard_similarity'] = jaccard

    # 音频时长对比
    dur_orig = torchaudio.load(original_path)[0].shape[1] / 16000
    dur_anon = torchaudio.load(anonymized_path)[0].shape[1] / 16000
    results['duration_original'] = dur_orig
    results['duration_anonymized'] = dur_anon

    # 汇总
    print("\n" + "-" * 50)
    print("Evaluation Results:")
    print(f"  Transcription:")
    print(f"    Original:   \"{text_orig}\"")
    print(f"    Anonymized: \"{text_anon}\"")
    print(f"  WER:                {wer:.2%}")
    print(f"  Edit Distance:      {edit_dist}")
    print(f"  Jaccard Similarity: {jaccard:.2%}")
    print(f"  Duration:           {dur_orig:.2f}s -> {dur_anon:.2f}s")
    print(f"  Speaker Similarity: {spk_sim:.4f} (lower = better privacy)")
    print(f"  Frame Continuity:   {cont_orig:.4f} -> {cont_anon:.4f}")

    # 评级
    if wer < 0.10:
        grade = "Excellent"
    elif wer < 0.30:
        grade = "Good"
    elif wer < 0.50:
        grade = "Fair"
    else:
        grade = "Poor"
    results['grade'] = grade
    print(f"\n  Semantic Preservation Grade: {grade} (WER={wer:.2%})")

    if spk_sim < 0.5:
        print(f"  Privacy Grade: Good (speaker similarity low)")
    elif spk_sim < 0.8:
        print(f"  Privacy Grade: Moderate")
    else:
        print(f"  Privacy Grade: Poor (speaker still recognizable)")

    # 保存
    result_file = output_dir / 'evaluation_results.json'
    with open(str(result_file), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {result_file}")

    return results


def calculate_wer(ref, hyp):
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0, len(hyp_words)

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    ed = d[len(ref_words)][len(hyp_words)]
    return ed / len(ref_words), ed


def calculate_jaccard(text1, text2):
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if len(w1 | w2) == 0:
        return 1.0
    return len(w1 & w2) / len(w1 | w2)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SAMM-Anon Full Pipeline Test")
    parser.add_argument('--audio', type=str, default=None,
                        help='Input audio path (default: random from LibriSpeech)')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'F'],
                        help='Source speaker gender')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip-disentangle', action='store_true',
                        help='Skip disentanglement test')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='Skip style clustering test')
    parser.add_argument('--skip-anonymize', action='store_true',
                        help='Skip anonymization')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--max-samples', type=int, default=50000,
                        help='Max frames for disentanglement test')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    timestamp = get_timestamp()
    output_dir = Path(args.output) if args.output else BASE_DIR / 'outputs' / f'full_test_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAMM-Anon Full Pipeline Test")
    print(f"  Timestamp: {timestamp}")
    print(f"  Output:    {output_dir}")
    print("=" * 70)

    # 选择测试音频
    if args.audio:
        audio_path = args.audio
    else:
        flac_files = list(LIBRISPEECH_DIR.glob('**/*.flac'))
        if not flac_files:
            print("ERROR: No audio files found in LibriSpeech test-clean")
            return
        import random
        random.seed(42)
        audio_path = str(random.choice(flac_files))
    print(f"  Audio:     {audio_path}")
    print(f"  Gender:    {args.gender}")

    all_results = {}
    projector = None

    # ---- Part 1: 解耦 ----
    if not args.skip_disentangle:
        disentangle_results, projector = run_disentanglement_test(args.max_samples)
        all_results['disentanglement'] = disentangle_results
    else:
        print("\n[SKIPPED] Part 1: Disentanglement")

    # ---- Part 2: 聚类 ----
    if not args.skip_clustering:
        if projector is None:
            # 需要先训练 projector
            print("\n[INFO] Training projector for clustering...")
            import h5py
            from models.phone_predictor.predictor import PhonePredictor
            from tests.test_v32_disentanglement_v3 import ExpandedSubspaceProjector

            with h5py.File(str(FEATURES_H5), 'r') as f:
                features = f['features'][:50000]
            predictor = PhonePredictor.load(str(CHECKPOINTS_DIR / 'phone_decoder.pt'), device='cuda')
            feat_t = torch.from_numpy(features).float().cuda()
            phones = []
            for i in range(0, len(feat_t), 10000):
                phones.append(predictor(feat_t[i:i+10000]).cpu().numpy())
            phones = np.concatenate(phones)
            del feat_t; torch.cuda.empty_cache()

            projector = ExpandedSubspaceProjector(n_phones=41, feature_dim=1024, n_components=100)
            projector.fit(features, phones, max_per_class=2000)

        clustering_results = run_style_clustering_test(projector, output_dir)
        all_results['clustering'] = clustering_results
    else:
        print("\n[SKIPPED] Part 2: Style Clustering")

    # ---- Part 3: 匿名化 ----
    original_path, anonymized_path = None, None
    if not args.skip_anonymize:
        original_path, anonymized_path = run_anonymization(
            audio_path, output_dir, source_gender=args.gender
        )
    else:
        print("\n[SKIPPED] Part 3: Anonymization")

    # ---- Part 4: 评估 ----
    if not args.skip_eval and original_path and anonymized_path:
        eval_results = run_evaluation(original_path, anonymized_path, output_dir)
        all_results['evaluation'] = eval_results
    else:
        print("\n[SKIPPED] Part 4: Evaluation")

    # ---- 最终汇总 ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if 'disentanglement' in all_results:
        d = all_results['disentanglement']
        reduction = (1 - d['H_style'] / d['H_original']) * 100
        print(f"  Disentanglement:  content reduction {reduction:.1f}%")

    if 'clustering' in all_results:
        c = all_results['clustering']
        print(f"  Style Clustering: ARI={c['ari_speaker']:.3f}, silhouette={c['silhouette']:.3f}")

    if 'evaluation' in all_results:
        e = all_results['evaluation']
        print(f"  Semantic:         WER={e['wer']:.2%} ({e['grade']})")
        print(f"  Privacy:          speaker_sim={e['speaker_similarity']:.4f}")

    # 保存总结果
    summary_path = output_dir / 'full_results.json'
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()}
    with open(str(summary_path), 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n  All results saved to: {output_dir}")
    print("=" * 70)
    print("Done!")


if __name__ == '__main__':
    main()
