#!/usr/bin/env python3
"""
步骤 1：从 metadata.json 对应的 LibriSpeech 音频中提取 ECAPA-TDNN speaker embedding
并拟合 PCA 降维到 128 维

输入：
  - cache/features/wavlm/metadata.json（含 utt_id, speaker_id, gender）
  - LibriSpeech 音频文件（通过 utt_id 定位）

输出：
  - checkpoints/speaker_embeddings_pca.npy  (N_utterances, 128)
  - checkpoints/speaker_pca_model.pkl
  - checkpoints/speaker_ids.npy             (N_utterances,)
  - checkpoints/utt_indices.npy             (N_utterances,)

使用方法：
    cd /root/autodl-tmp/anon_test
    python scripts/step1_extract_speaker_embeddings.py
    python scripts/step1_extract_speaker_embeddings.py --librispeech /path/to/LibriSpeech
"""

import sys
import json
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_audio_for_utterance(utt_id, librispeech_roots):
    """
    根据 utt_id (如 "100-121669-0000") 定位 LibriSpeech 音频文件

    LibriSpeech 路径格式: <root>/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    utt_id 格式: <speaker>-<chapter>-<utt>
    """
    parts = utt_id.split('-')
    if len(parts) != 3:
        return None

    speaker, chapter, utt_num = parts

    for root in librispeech_roots:
        root = Path(root)
        if not root.exists():
            continue

        # 搜索所有 split 子目录
        for split_dir in root.iterdir():
            if not split_dir.is_dir():
                continue
            audio_path = split_dir / speaker / chapter / f"{utt_id}.flac"
            if audio_path.exists():
                return str(audio_path)

        # 也可能直接在 root 下 (如 root 已经是 split)
        audio_path = root / speaker / chapter / f"{utt_id}.flac"
        if audio_path.exists():
            return str(audio_path)

    return None


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract Speaker Embeddings + PCA")
    parser.add_argument('--librispeech', type=str, nargs='+',
                        default=['/root/autodl-tmp/datasets/LibriSpeech'],
                        help='LibriSpeech root directories')
    parser.add_argument('--pca-dim', type=int, default=128,
                        help='PCA output dimension (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / 'cache' / 'features' / 'wavlm'
    ckpt_dir = base_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # === 加载 metadata ===
    metadata_path = cache_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    utterances = metadata['utterances']
    print(f"Metadata: {len(utterances)} utterances, {metadata['total_frames']} frames")

    # === 加载 ECAPA-TDNN ===
    print("\nLoading ECAPA-TDNN speaker encoder...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(base_dir / "cache" / "models" / "speaker")
    )
    print("  ECAPA-TDNN loaded")

    # === 逐话语提取 embedding ===
    all_embeddings = []
    all_speaker_ids = []
    all_utt_indices = []
    skipped = 0

    for utt_idx, utt in enumerate(tqdm(utterances, desc="Extracting speaker embeddings")):
        utt_id = utt['utt_id']
        speaker_id = utt['speaker_id']

        # 定位音频文件
        audio_path = find_audio_for_utterance(utt_id, args.librispeech)

        if audio_path is None:
            skipped += 1
            if skipped <= 5:
                tqdm.write(f"  Skip {utt_id}: audio not found")
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)

            # 重采样到 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # 单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(device)

            with torch.no_grad():
                emb = speaker_encoder.encode_batch(waveform)
                if emb.dim() == 3:
                    emb = emb.squeeze(1)
                all_embeddings.append(emb[0].cpu().numpy())
                all_speaker_ids.append(speaker_id)
                all_utt_indices.append(utt_idx)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                tqdm.write(f"  Skip {utt_id}: {e}")
            continue

    if not all_embeddings:
        print("\nERROR: No embeddings extracted!")
        print("Check LibriSpeech paths:")
        for p in args.librispeech:
            print(f"  {p} -> exists: {Path(p).exists()}")
        sys.exit(1)

    all_embeddings = np.stack(all_embeddings)   # (N, 192)
    all_speaker_ids = np.array(all_speaker_ids)
    all_utt_indices = np.array(all_utt_indices)

    print(f"\nExtracted: {len(all_embeddings)}, skipped: {skipped}")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Unique speakers: {len(np.unique(all_speaker_ids))}")

    # === PCA 降维 ===
    from sklearn.decomposition import PCA

    pca_dim = min(args.pca_dim, all_embeddings.shape[1], len(all_embeddings) - 1)
    pca_model = PCA(n_components=pca_dim, random_state=42)
    embeddings_pca = pca_model.fit_transform(all_embeddings)

    explained = pca_model.explained_variance_ratio_.sum()
    print(f"\nPCA: {all_embeddings.shape[1]} -> {pca_dim}")
    print(f"  Explained variance: {explained:.2%}")

    # === 保存 ===
    np.save(ckpt_dir / 'speaker_embeddings_pca.npy', embeddings_pca)
    np.save(ckpt_dir / 'speaker_ids.npy', all_speaker_ids)
    np.save(ckpt_dir / 'utt_indices.npy', all_utt_indices)
    np.save(ckpt_dir / 'speaker_embeddings_raw.npy', all_embeddings)

    with open(ckpt_dir / 'speaker_pca_model.pkl', 'wb') as f:
        pickle.dump(pca_model, f)

    print(f"\nSaved to {ckpt_dir}:")
    print(f"  speaker_embeddings_pca.npy: {embeddings_pca.shape}")
    print(f"  speaker_embeddings_raw.npy: {all_embeddings.shape}")
    print(f"  speaker_pca_model.pkl")
    print(f"  speaker_ids.npy: {all_speaker_ids.shape}")
    print(f"  utt_indices.npy: {all_utt_indices.shape}")


if __name__ == '__main__':
    main()
