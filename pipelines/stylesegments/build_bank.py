#!/usr/bin/env python3
"""基于混淆熵与聚类构建最终的 kNN-VC 伪风格 Bank (120GB 内存优化版)"""

import json
import h5py
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

DATA_DIR = "/root/autodl-tmp/anon_test/checkpoints/train360_with_phones"
OUTPUT_BANK_PATH = "/root/autodl-tmp/anon_test/checkpoints/pseudo_bank.pt"

N_PSEUDO_CLUSTERS = 50
FRAMES_PER_CLUSTER = 20
ENTROPY_PERCENTILE = 60
MIN_SPK_DIVERSITY = 3
CHUNK_SIZE = 1000000  # 每次顺序读取 100 万帧 (极大优化磁盘 I/O)

print("="*60)
print("开始构建最终伪风格 Bank (顺序读取优化版)...")
print("="*60)

with open(f"{DATA_DIR}/metadata.json", 'r') as f:
    meta = json.load(f)
total_frames = meta['total_frames']

# 1. 构建内存说话人索引
print("[1/4] 构建内存说话人索引...")
speaker_list = list(set([u['speaker_id'] for u in meta['utterances']]))
spk2id = {spk: i for i, spk in enumerate(speaker_list)}
frame_to_spk = np.zeros(total_frames, dtype=np.uint16)

for utt in meta['utterances']:
    start, end = utt['h5_start_idx'], utt['h5_end_idx']
    frame_to_spk[start:end] = spk2id[utt['speaker_id']]

# 2. 读取标签与混淆熵，预计算全局高熵掩码
print("[2/4] 计算全局高熵掩码...")
with h5py.File(f"{DATA_DIR}/phones.h5", 'r') as h5_p, h5py.File(f"{DATA_DIR}/entropies.h5", 'r') as h5_e:
    all_phones = h5_p['phones'][:]
    all_entropies = h5_e['entropies'][:]

unique_phones = np.unique(all_phones)
global_keep_mask = np.zeros(total_frames, dtype=bool)

for phone_id in unique_phones:
    ph_indices = np.where(all_phones == phone_id)[0]
    if len(ph_indices) == 0: continue
        
    ph_entropies = all_entropies[ph_indices]
    valid_mask = ~np.isnan(ph_entropies)
    if not np.any(valid_mask): continue
        
    ph_entropies_valid = ph_entropies[valid_mask]
    valid_indices = ph_indices[valid_mask]
    
    # 获取高熵帧的索引
    entropy_threshold = np.percentile(ph_entropies_valid, ENTROPY_PERCENTILE)
    high_entropy_mask = ph_entropies_valid >= entropy_threshold
    
    # 标记在全局掩码中
    selected_indices = valid_indices[high_entropy_mask]
    global_keep_mask[selected_indices] = True

# 3. 顺序扫描 HDF5，将高熵帧全部吸入内存 (发挥 120GB 内存的威力)
print("[3/4] 顺序读取磁盘，将高熵特征装载至内存 (极速 I/O)...")
l6_filtered_dict = {ph: [] for ph in unique_phones}
spk_filtered_dict = {ph: [] for ph in unique_phones}

with h5py.File(f"{DATA_DIR}/layer_6.h5", 'r') as h5_l6:
    ds_l6 = h5_l6['features']
    
    # 每次顺序读取 100 万帧，绝不在硬盘上来回跳跃
    for start_idx in tqdm(range(0, total_frames, CHUNK_SIZE), desc="顺序块读取 L6"):
        end_idx = min(start_idx + CHUNK_SIZE, total_frames)
        mask_chunk = global_keep_mask[start_idx:end_idx]
        
        if not np.any(mask_chunk):
            continue
            
        # 顺序读取一大块到内存，速度极快
        l6_chunk = ds_l6[start_idx:end_idx]
        
        # 在内存中使用 NumPy 进行极速掩码过滤
        l6_chunk_kept = l6_chunk[mask_chunk]
        phones_chunk_kept = all_phones[start_idx:end_idx][mask_chunk]
        spks_chunk_kept = frame_to_spk[start_idx:end_idx][mask_chunk]
        
        # 分发到对应的音素桶中
        for ph in np.unique(phones_chunk_kept):
            ph_mask = (phones_chunk_kept == ph)
            l6_filtered_dict[ph].append(l6_chunk_kept[ph_mask])
            spk_filtered_dict[ph].append(spks_chunk_kept[ph_mask])

# 4. 在内存中逐音素执行 K-Means 二次聚类
print("[4/4] 逐音素聚类并构建最终 Bank...")
pseudo_bank_tensors = {}

for phone_id in tqdm(unique_phones, desc="Processing Phones"):
    if not l6_filtered_dict[phone_id]:
        continue
        
    # 把各个 chunk 收集到的特征拼起来
    l6_filtered = np.concatenate(l6_filtered_dict[phone_id])
    spks_filtered = np.concatenate(spk_filtered_dict[phone_id])

    if len(l6_filtered) < N_PSEUDO_CLUSTERS:
        pseudo_bank_tensors[phone_id] = torch.from_numpy(l6_filtered).float()
        continue

    # 聚类与筛选
    km = MiniBatchKMeans(n_clusters=N_PSEUDO_CLUSTERS, random_state=42, batch_size=2048)
    km.fit(l6_filtered)
    labels = km.labels_
    
    selected_frames = []
    for c in range(N_PSEUDO_CLUSTERS):
        cluster_mask = labels == c
        cluster_l6 = l6_filtered[cluster_mask]
        cluster_spks = spks_filtered[cluster_mask]
        
        if len(np.unique(cluster_spks)) < MIN_SPK_DIVERSITY:
            continue
            
        dists = np.linalg.norm(cluster_l6 - km.cluster_centers_[c], axis=1)
        top_idx = dists.argsort()[:FRAMES_PER_CLUSTER]
        selected_frames.append(cluster_l6[top_idx])
        
    if selected_frames:
        final_l6_np = np.concatenate(selected_frames)
    else:
        final_l6_np = l6_filtered
        
    pseudo_bank_tensors[phone_id] = torch.from_numpy(final_l6_np).float()

# 5. 保存结果
torch.save(pseudo_bank_tensors, OUTPUT_BANK_PATH)

print("\n" + "="*60)
print(f"Bank 构建完成并已保存至: {OUTPUT_BANK_PATH}")
print(f"涵盖音素数: {len(pseudo_bank_tensors)}")
print(f"总帧数: {sum(t.shape[0] for t in pseudo_bank_tensors.values())}")
print("="*60)