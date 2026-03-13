#!/usr/bin/env python3
"""优化版特征提取：同时提取 L6/L12/L24，但只保存 L6/L12 + phone 标签"""

import sys
import json
import random
import torch
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.ssl.wrappers import WavLMSSLExtractor
from models.phone_predictor.predictor import PhonePredictor
from data.datasets.librispeech import LibriSpeechDataset

# 配置
LIBRISPEECH_ROOT = "/root/autodl-tmp/datasets/LibriSpeech"
WAVLM_CKPT = "/root/autodl-tmp/spkanon/checkpoints/WavLM-Large.pt"
PHONE_CKPT = "./checkpoints/phone_decoder.pt"
OUTPUT_DIR = "/root/autodl-tmp/anon_test/checkpoints/train360_with_phones"
SPEAKER_RATIO = 0.5
DEVICE = "cuda"
BATCH_SIZE = 8

print("="*60)
print("优化版特征提取 (L6/L12 + Phone 标签)")
print("="*60)

# 1. 加载数据集并采样说话人
print("\n[1/4] 加载数据集...")
dataset = LibriSpeechDataset(root=LIBRISPEECH_ROOT, split="train-clean-360")
speaker_to_indices = defaultdict(list)
for idx in range(len(dataset)):
    speaker_to_indices[dataset[idx]['speaker_id']].append(idx)

random.seed(42)
selected_speakers = random.sample(list(speaker_to_indices.keys()),
                                 len(speaker_to_indices) // 2)
selected_indices = []
for spk in selected_speakers:
    selected_indices.extend(speaker_to_indices[spk])

# 按音频时长排序，减少 batch padding
print("\n按时长排序样本...")
durations = [(idx, len(dataset[idx]['waveform'])) for idx in tqdm(selected_indices, desc="读取时长")]
selected_indices = [idx for idx, _ in sorted(durations, key=lambda x: x[1])]

print(f"选中 {len(selected_speakers)} 说话人, {len(selected_indices)} 样本")

# 2. 初始化模型
print("\n[2/4] 加载模型...")
extractor = WavLMSSLExtractor(WAVLM_CKPT, layer=6, device=DEVICE)
phone_predictor = PhonePredictor.load(PHONE_CKPT, device=DEVICE)

# 3. 准备输出
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
h5_l6 = h5py.File(f"{OUTPUT_DIR}/layer_6.h5", 'w')
h5_l12 = h5py.File(f"{OUTPUT_DIR}/layer_12.h5", 'w')
h5_phones = h5py.File(f"{OUTPUT_DIR}/phones.h5", 'w')

ds_l6 = h5_l6.create_dataset('features', shape=(0, 1024), maxshape=(None, 1024),
                             dtype='float32', chunks=(1000, 1024))
ds_l12 = h5_l12.create_dataset('features', shape=(0, 1024), maxshape=(None, 1024),
                               dtype='float32', chunks=(1000, 1024))
ds_phones = h5_phones.create_dataset('phones', shape=(0,), maxshape=(None,),
                                     dtype='int16', chunks=(10000,))

metadata = []
current_frame = 0

# 4. 批量提取
print("\n[3/4] 提取特征...")
batch_wavs, batch_infos = [], []

for idx in tqdm(selected_indices):
    item = dataset[idx]
    batch_wavs.append(item['waveform'])
    batch_infos.append({
        'utt_id': item['utt_id'],
        'speaker_id': item['speaker_id'],
        'gender': item.get('gender', 'unknown')
    })

    if len(batch_wavs) >= BATCH_SIZE or idx == selected_indices[-1]:
        # 单次前向提取多层（高效）
        with torch.inference_mode():
            # 批量处理
            max_len = max(len(w) for w in batch_wavs)
            batch_tensor = torch.zeros(len(batch_wavs), max_len)
            for i, w in enumerate(batch_wavs):
                w_tensor = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
                batch_tensor[i, :len(w)] = w_tensor

            batch_tensor = batch_tensor.to(DEVICE)
            multi_feats = extractor.forward_multi_layer(batch_tensor, layers=[6, 12, 24])

            feats_l6 = [multi_feats[6][i] for i in range(len(batch_wavs))]
            feats_l12 = [multi_feats[12][i] for i in range(len(batch_wavs))]
            feats_l24 = [multi_feats[24][i] for i in range(len(batch_wavs))]

        # 预测 phone 并立即释放 L24
        for i, info in enumerate(batch_infos):
            l6 = feats_l6[i].cpu().numpy()
            l12 = feats_l12[i].cpu().numpy()
            phones = phone_predictor(feats_l24[i].unsqueeze(0)).squeeze(0).cpu().numpy()

            num_frames = len(l6)
            start, end = current_frame, current_frame + num_frames

            # 保存 L6/L12/phones
            ds_l6.resize((end, 1024))
            ds_l12.resize((end, 1024))
            ds_phones.resize((end,))

            ds_l6[start:end] = l6
            ds_l12[start:end] = l12
            ds_phones[start:end] = phones

            metadata.append({
                'utt_id': info['utt_id'],
                'speaker_id': info['speaker_id'],
                'gender': info['gender'],
                'num_frames': num_frames,
                'h5_start_idx': start,
                'h5_end_idx': end
            })

            current_frame = end

        # 释放 L24 内存
        del feats_l24
        torch.cuda.empty_cache()

        batch_wavs, batch_infos = [], []

h5_l6.close()
h5_l12.close()
h5_phones.close()

# 5. 保存元数据
print("\n[4/4] 保存元数据...")
with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
    json.dump({
        'total_frames': current_frame,
        'total_utterances': len(metadata),
        'utterances': metadata
    }, f, indent=2)

print("\n" + "="*60)
print("完成!")
print(f"输出: {OUTPUT_DIR}")
print(f"总帧数: {current_frame}")
print(f"总样本: {len(metadata)}")
print(f"存储: L6 + L12 + Phones (节省 ~33% 空间)")
print("="*60)
