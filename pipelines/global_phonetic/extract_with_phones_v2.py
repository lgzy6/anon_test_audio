#!/usr/bin/env python3
"""
重构版特征提取：
1. 从 train-other-500 随机选择 200 个说话人（男女各100）
2. 提取 L6/L12/L24 三层特征 + phone 标签
"""

import sys
import json
import random
import shutil
import torch
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.ssl.wrappers import WavLMSSLExtractor
from models.phone_predictor.predictor import PhonePredictor
from data.datasets.librispeech import LibriSpeechDataset

# ==================== 配置 ====================
LIBRISPEECH_ROOT = "/root/autodl-tmp/datasets/LibriTTS"
LIBRISPEECH_SPLIT = "train-other-500"

WAVLM_CKPT = "/root/autodl-tmp/anon_test/checkpoints/WavLM-Large.pt"
PHONE_CKPT = "/root/autodl-tmp/anon_test/checkpoints/phone_decoder.pt"
OUTPUT_DIR = "/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk"

# 说话人采样配置
NUM_SPEAKERS = 200  # 总共选择 200 个说话人
NUM_MALE = 100      # 男性说话人数量
NUM_FEMALE = 100    # 女性说话人数量
UTT_RATIO = 1.0     # 每个说话人保留的 utterance 比例（1.0=全量保存）

# 模型配置
DEVICE = "cuda"
BATCH_SIZE = 1  # 设为1避免batch padding导致的自注意力污染
LAYERS = [6, 12, 24]  # 提取的层

print("=" * 70)
print("重构版特征提取 (200 说话人: 男女各100)")
print("=" * 70)


def sample_speakers_by_gender(dataset, num_male: int, num_female: int, seed: int = 42):
    """按性别采样说话人（仅从实际有音频的说话人中采样）"""
    random.seed(seed)

    actual_speakers = set(u['speaker_id'] for u in dataset.utterances)

    male_speakers = []
    female_speakers = []

    for spk_id, info in dataset.speakers_info.items():
        if spk_id not in actual_speakers:
            continue
        gender = info.get('gender', 'unknown')
        if gender == 'm':
            male_speakers.append(spk_id)
        elif gender == 'f':
            female_speakers.append(spk_id)

    print(f"\n数据集统计 (仅 {dataset.split} 中有音频的说话人):")
    print(f"  男性说话人: {len(male_speakers)}")
    print(f"  女性说话人: {len(female_speakers)}")

    # 随机采样
    if len(male_speakers) < num_male:
        print(f"警告: 男性说话人不足 {num_male}，实际只有 {len(male_speakers)}")
        selected_male = male_speakers
    else:
        selected_male = random.sample(male_speakers, num_male)

    if len(female_speakers) < num_female:
        print(f"警告: 女性说话人不足 {num_female}，实际只有 {len(female_speakers)}")
        selected_female = female_speakers
    else:
        selected_female = random.sample(female_speakers, num_female)

    selected_speakers = set(selected_male + selected_female)

    print(f"\n采样结果:")
    print(f"  选中男性: {len(selected_male)}")
    print(f"  选中女性: {len(selected_female)}")
    print(f"  总计: {len(selected_speakers)}")

    return selected_speakers


def collect_utterances_from_speakers(dataset, selected_speakers: set, utt_ratio: float):
    """从选中的说话人中收集 utterances（仅读取元数据）"""
    speaker_to_indices = defaultdict(list)

    for idx, item in enumerate(dataset.utterances):
        spk_id = item['speaker_id']
        if spk_id in selected_speakers:
            speaker_to_indices[spk_id].append(idx)

    # 对每个说话人采样 utterances
    random.seed(42)
    selected_indices = []

    for spk_id, indices in speaker_to_indices.items():
        if not indices:
            continue
        target_utts = max(1, int(len(indices) * utt_ratio))
        selected_indices.extend(random.sample(indices, target_utts))

    print(f"\n收集到 {len(selected_indices)} 条 utterances")
    return selected_indices




def get_audio_duration_fast(audio_path: str) -> float:
    """快速读取音频时长（仅读取头部元数据，不加载waveform）"""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        # 如果失败，返回一个默认值
        return 0.0


def sort_by_duration(dataset, indices: List[int]):
    """按音频时长排序（仅读取元数据，不加载waveform）"""
    print("\n按时长排序样本（仅读取元数据）...")
    durations = []
    for idx in tqdm(indices, desc="读取时长"):
        # 只读取音频路径，不加载waveform
        utt_info = dataset.utterances[idx]
        audio_path = utt_info['audio_path']
        duration = get_audio_duration_fast(audio_path)
        durations.append((idx, duration))

    sorted_indices = [idx for idx, _ in sorted(durations, key=lambda x: x[1])]
    return sorted_indices


def extract_features(datasets_with_indices: List[tuple], output_dir: str):
    """
    提取特征

    Args:
        datasets_with_indices: [(dataset, indices, dataset_name), ...]
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    print("\n[1/3] 加载模型...")
    extractor = WavLMSSLExtractor(WAVLM_CKPT, layer=6, device=DEVICE)
    phone_predictor = PhonePredictor.load(PHONE_CKPT, device=DEVICE)

    # 准备 HDF5 文件
    print("\n[2/3] 准备输出文件...")
    h5_files = {}
    datasets_h5 = {}

    for layer in LAYERS:
        h5_files[f'l{layer}'] = h5py.File(f"{output_dir}/layer_{layer}.h5", 'w')
        datasets_h5[f'l{layer}'] = h5_files[f'l{layer}'].create_dataset(
            'features',
            shape=(0, 1024),
            maxshape=(None, 1024),
            dtype='float32',
            chunks=(1000, 1024)
        )

    h5_phones = h5py.File(f"{output_dir}/phones.h5", 'w')
    ds_phones = h5_phones.create_dataset(
        'phones',
        shape=(0,),
        maxshape=(None,),
        dtype='int16',
        chunks=(10000,)
    )

    metadata = []
    current_frame = 0

    # 批量提取
    print("\n[3/3] 提取特征...")

    for dataset, indices, dataset_name in datasets_with_indices:
        print(f"\n处理数据集: {dataset_name} ({len(indices)} utterances)")

        batch_wavs, batch_infos = [], []

        for idx in tqdm(indices, desc=f"提取 {dataset_name}"):
            item = dataset[idx]
            waveform = item['waveform']

            batch_wavs.append(waveform)
            batch_infos.append({
                'utt_id': item['utt_id'],
                'speaker_id': item['speaker_id'],
                'gender': item.get('gender', 'unknown'),
                'emotion': item.get('emotion', 'neutral'),
                'dataset': dataset_name
            })

            # 批次处理
            if len(batch_wavs) >= BATCH_SIZE or idx == indices[-1]:
                with torch.inference_mode():
                    # 构建批次（BATCH_SIZE=1时无padding，避免自注意力污染）
                    if BATCH_SIZE == 1:
                        # 单样本处理，无需padding
                        batch_tensor = []
                        for w in batch_wavs:
                            w_tensor = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
                            if w_tensor.dim() > 1:
                                w_tensor = w_tensor.mean(dim=0)
                            batch_tensor.append(w_tensor.unsqueeze(0).to(DEVICE))
                    else:
                        # 多样本批处理（会有padding）
                        max_len = max(len(w) for w in batch_wavs)
                        batch_tensor = torch.zeros(len(batch_wavs), max_len)

                        for i, w in enumerate(batch_wavs):
                            w_tensor = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
                            if w_tensor.dim() > 1:
                                w_tensor = w_tensor.mean(dim=0)
                            batch_tensor[i, :len(w_tensor)] = w_tensor

                        batch_tensor = [batch_tensor.to(DEVICE)]

                    # 逐样本提取多层特征（避免padding污染）
                    all_multi_feats = []
                    for single_batch in batch_tensor:
                        multi_feats = extractor.forward_multi_layer(single_batch, layers=LAYERS)
                        all_multi_feats.append(multi_feats)

                # 处理每个样本
                for i, info in enumerate(batch_infos):
                    # 获取对应的multi_feats
                    if BATCH_SIZE == 1:
                        multi_feats = all_multi_feats[i]
                        sample_idx = 0  # 单样本batch中的索引总是0
                    else:
                        multi_feats = all_multi_feats[0]
                        sample_idx = i

                    # 提取各层特征
                    layer_features = {}
                    for layer in LAYERS:
                        feat = multi_feats[layer][sample_idx].cpu().numpy()
                        layer_features[f'l{layer}'] = feat

                    # 预测 phone（使用 L24）
                    l24_tensor = multi_feats[24][sample_idx].unsqueeze(0)
                    phones = phone_predictor(l24_tensor).squeeze(0).cpu().numpy()

                    num_frames = len(layer_features['l6'])
                    start, end = current_frame, current_frame + num_frames

                    # 保存到 HDF5
                    for layer in LAYERS:
                        datasets_h5[f'l{layer}'].resize((end, 1024))
                        datasets_h5[f'l{layer}'][start:end] = layer_features[f'l{layer}']

                    ds_phones.resize((end,))
                    ds_phones[start:end] = phones

                    # 记录元数据
                    metadata.append({
                        'utt_id': info['utt_id'],
                        'speaker_id': info['speaker_id'],
                        'gender': info['gender'],
                        'emotion': info['emotion'],
                        'dataset': info['dataset'],
                        'num_frames': num_frames,
                        'h5_start_idx': start,
                        'h5_end_idx': end
                    })

                    current_frame = end

                # 清理内存
                del all_multi_feats
                torch.cuda.empty_cache()

                batch_wavs, batch_infos = [], []

    # 关闭文件
    for h5_file in h5_files.values():
        h5_file.close()
    h5_phones.close()

    # 保存元数据
    print("\n保存元数据...")

    # 统计信息
    stats = {
        'librispeech': {'male': 0, 'female': 0, 'utterances': 0},
        'iemocap': {'male': 0, 'female': 0, 'utterances': 0}
    }

    for meta in metadata:
        dataset_key = meta['dataset']
        if dataset_key not in stats:
            stats[dataset_key] = {'male': 0, 'female': 0, 'utterances': 0}

        stats[dataset_key]['utterances'] += 1
        if meta['gender'] == 'm':
            stats[dataset_key]['male'] += 1
        elif meta['gender'] == 'f':
            stats[dataset_key]['female'] += 1

    metadata_dict = {
        'total_frames': current_frame,
        'total_utterances': len(metadata),
        'layers': LAYERS,
        'statistics': stats,
        'utterances': metadata
    }

    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    return metadata_dict, stats


def main():
    """主函数"""

    # 1. 加载 LibriSpeech 数据集
    print("\n[步骤 1/4] 加载 LibriSpeech 数据集...")
    libri_dataset = LibriSpeechDataset(
        root=LIBRISPEECH_ROOT,
        split=LIBRISPEECH_SPLIT
    )

    # 2. 按性别采样说话人
    print("\n[步骤 2/4] 采样说话人...")
    selected_speakers = sample_speakers_by_gender(
        libri_dataset,
        num_male=NUM_MALE,
        num_female=NUM_FEMALE
    )

    # 3. 收集 utterances
    print("\n[步骤 3/4] 收集 utterances...")
    libri_indices = collect_utterances_from_speakers(
        libri_dataset,
        selected_speakers,
        UTT_RATIO
    )
    libri_indices = sort_by_duration(libri_dataset, libri_indices)

    datasets_with_indices = [
        (libri_dataset, libri_indices, 'librispeech')
    ]

    # 4. 提取特征
    print("\n[步骤 4/4] 提取特征...")
    metadata_dict, stats = extract_features(datasets_with_indices, OUTPUT_DIR)

    # 打印结果
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总帧数: {metadata_dict['total_frames']}")
    print(f"总样本: {metadata_dict['total_utterances']}")
    print(f"提取层: {LAYERS}")
    print("\n数据集统计:")
    for dataset_name, stat in stats.items():
        print(f"  {dataset_name}:")
        print(f"    男性: {stat['male']}, 女性: {stat['female']}")
        print(f"    总 utterances: {stat['utterances']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
