#!/usr/bin/env python3
"""按说话人采样提取 train-clean-360 特征"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.stylesegments.feature_extraction import FeatureExtractor
from data.datasets.librispeech import LibriSpeechDataset

# 配置
LIBRISPEECH_ROOT = "/root/autodl-tmp/datasets/LibriSpeech"
WAVLM_CKPT = "/root/autodl-tmp/spkanon/checkpoints/WavLM-Large.pt"
OUTPUT_DIR = "/root/autodl-tmp/anon_test/checkpoints/train360_features"
LAYERS = [6, 12]
SPEAKER_RATIO = 0.5
DEVICE = "cuda"
BATCH_SIZE = 8

print("="*60)
print("Train-360 特征提取 (按说话人采样)")
print("="*60)

# 1. 加载数据集
print("\n加载数据集...")
dataset = LibriSpeechDataset(root=LIBRISPEECH_ROOT, split="train-clean-360")
print(f"总样本数: {len(dataset)}")

# 2. 按说话人分组
print("\n按说话人分组...")
speaker_to_indices = defaultdict(list)
for idx in range(len(dataset)):
    item = dataset[idx]
    speaker_to_indices[item['speaker_id']].append(idx)

all_speakers = list(speaker_to_indices.keys())
print(f"总说话人数: {len(all_speakers)}")

# 3. 随机选择一半说话人
random.seed(42)
selected_speakers = random.sample(all_speakers, len(all_speakers) // 2)
print(f"选择说话人数: {len(selected_speakers)}")

# 4. 获取选中说话人的所有样本索引
selected_indices = []
for spk in selected_speakers:
    selected_indices.extend(speaker_to_indices[spk])
selected_indices.sort()
print(f"选中样本数: {len(selected_indices)}")

# 5. 创建子数据集
class SubsetDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

subset_dataset = SubsetDataset(dataset, selected_indices)

# 6. 提取特征
print(f"\n初始化特征提取器 (layers={LAYERS})...")
extractor = FeatureExtractor(
    wavlm_ckpt=WAVLM_CKPT,
    layers=LAYERS,
    device=DEVICE,
    batch_size=BATCH_SIZE
)

print(f"\n开始提取特征到: {OUTPUT_DIR}")
metadata = extractor.extract_dataset(
    dataset=subset_dataset,
    output_dir=OUTPUT_DIR,
    save_interval=1000,
    resume=True,
    sample_ratio=1.0
)

print("\n" + "="*60)
print("提取完成!")
print(f"输出目录: {OUTPUT_DIR}")
print(f"总帧数: {metadata['total_frames']}")
print(f"总样本数: {metadata['total_utterances']}")
print("="*60)
