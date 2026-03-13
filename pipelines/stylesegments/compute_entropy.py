#!/usr/bin/env python3
"""基于 HDF5 的流式 Layer 12 混淆熵计算"""

import json
import h5py
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from collections import defaultdict

# 配置路径（与你上一步提取的路径对应）
DATA_DIR = "/root/autodl-tmp/anon_test/checkpoints/train360_with_phones"
META_PATH = f"{DATA_DIR}/metadata.json"
L12_PATH = f"{DATA_DIR}/layer_12.h5"
PHONE_PATH = f"{DATA_DIR}/phones.h5"
OUTPUT_ENTROPY_PATH = f"{DATA_DIR}/entropies.h5"

TEMPERATURE = 10.0
MIN_SPEAKERS = 3

print("="*60)
print("Step 1: 计算音素内说话人嵌入 (Centroids)")
print("="*60)

# 读取元数据
with open(META_PATH, 'r') as f:
    meta = json.load(f)
utterances = meta['utterances']

# 用于累加特征的字典: {phone_id: {spk_id: [sum_vector, count]}}
phone_spk_stats = defaultdict(lambda: defaultdict(lambda: [np.zeros(1024), 0]))

with h5py.File(L12_PATH, 'r') as h5_l12, h5py.File(PHONE_PATH, 'r') as h5_p:
    ds_l12 = h5_l12['features']
    ds_p = h5_p['phones']
    
    # 遍历所有音频片段
    for utt in tqdm(utterances, desc="Accumulating L12"):
        start, end = utt['h5_start_idx'], utt['h5_end_idx']
        spk_id = utt['speaker_id']
        
        # 块读取，避免内存溢出
        l12_frames = ds_l12[start:end]   # [T, 1024]
        phone_frames = ds_p[start:end]   # [T]
        
        for i in range(len(phone_frames)):
            ph = phone_frames[i]
            phone_spk_stats[ph][spk_id][0] += l12_frames[i]
            phone_spk_stats[ph][spk_id][1] += 1

# 计算均值，得到最终的 Embedding
phone_spk_embs = {}
for ph, spk_dict in phone_spk_stats.items():
    phone_spk_embs[ph] = {}
    for spk, (feat_sum, count) in spk_dict.items():
        if count >= 3:  # 至少3帧才算有效代表
            phone_spk_embs[ph][spk] = feat_sum / count

# 过滤说话人数不足的音素
valid_phones = {ph: embs for ph, embs in phone_spk_embs.items() if len(embs) >= MIN_SPEAKERS}
print(f"有效音素数量: {len(valid_phones)} (要求每音素至少包含 {MIN_SPEAKERS} 个说话人)")


print("\n" + "="*60)
print("Step 2: 计算每帧的混淆熵并存入 HDF5")
print("="*60)

total_frames = meta['total_frames']
h5_entropy = h5py.File(OUTPUT_ENTROPY_PATH, 'w')
# 创建与特征等长的 float32 数据集，默认填 NaN
ds_ent = h5_entropy.create_dataset('entropies', shape=(total_frames,), dtype='float32', fillvalue=np.nan)

# 预计算：为了加速，把每个音素的说话人矩阵预先 Normalize 备用
# precomputed_embs[ph] = normalized_matrix [N_spk, 1024]
precomputed_embs = {}
for ph, spk_dict in valid_phones.items():
    emb_matrix = np.stack(list(spk_dict.values()))
    # L2 归一化
    emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
    precomputed_embs[ph] = emb_norms

valid_count = 0

with h5py.File(L12_PATH, 'r') as h5_l12, h5py.File(PHONE_PATH, 'r') as h5_p:
    ds_l12 = h5_l12['features']
    ds_p = h5_p['phones']
    
    for utt in tqdm(utterances, desc="Calculating Entropy"):
        start, end = utt['h5_start_idx'], utt['h5_end_idx']
        l12_frames = ds_l12[start:end]
        phone_frames = ds_p[start:end]

        entropies = np.full(end - start, np.nan, dtype=np.float32)

        # 向量化：按音素批量计算
        unique_phones_in_utt = np.unique(phone_frames)
        for ph in unique_phones_in_utt:
            if ph not in precomputed_embs:
                continue

            # 找到当前音频中所有属于该音素的帧
            ph_mask = (phone_frames == ph)
            ph_l12 = l12_frames[ph_mask]  # [N, 1024]
            emb_norms = precomputed_embs[ph]  # [N_spk, 1024]
            num_spks = emb_norms.shape[0]

            # 批量计算余弦相似度
            ph_l12_norm = ph_l12 / (np.linalg.norm(ph_l12, axis=1, keepdims=True) + 1e-8)
            sims = ph_l12_norm @ emb_norms.T  # [N, N_spk]

            # 批量计算 Softmax (数值稳定版)
            scaled_sims = sims * TEMPERATURE
            max_sims = np.max(scaled_sims, axis=1, keepdims=True)
            exp_sims = np.exp(scaled_sims - max_sims)
            probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

            # 批量计算 Entropy
            ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)
            max_entropy = np.log(num_spks)

            # 赋值回原数组
            entropies[ph_mask] = ent / max_entropy
            valid_count += np.sum(ph_mask)

        # 写入 HDF5
        ds_ent[start:end] = entropies

h5_entropy.close()

print("\n" + "="*60)
print(f"完成！混淆熵已写入: {OUTPUT_ENTROPY_PATH}")
print(f"总帧数: {total_frames}, 成功计算熵的帧数: {valid_count} ({valid_count/total_frames:.1%})")
print("="*60)