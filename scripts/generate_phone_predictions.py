#!/usr/bin/env python3
"""
生成 Phone 预测文件
修复 Target Pool 中 phones 缺失的问题
"""

import sys
import json
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phone_predictor.predictor import PhonePredictor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 路径配置
    cache_dir = Path("data/samm_anon/cache/features/cleaned")
    features_path = cache_dir / "features.h5"
    metadata_path = cache_dir / "metadata.json"
    output_path = cache_dir / "phone_predictions.h5"

    # 加载 Phone Predictor
    print("\n[1/3] 加载 Phone Predictor...")
    predictor = PhonePredictor.load("checkpoints/phone_decoder.pt", device=device)
    print("  ✓ Phone Predictor 加载完成")

    # 加载元数据
    print("\n[2/3] 加载元数据...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    utterances = metadata['utterances']
    print(f"  ✓ 共 {len(utterances)} 个语句")

    # 生成预测
    print("\n[3/3] 生成 Phone 预测...")

    with h5py.File(features_path, 'r') as f_in:
        features = f_in['features']
        total_frames = features.shape[0]
        print(f"  总帧数: {total_frames:,}")

        with h5py.File(output_path, 'w') as f_out:
            for utt in tqdm(utterances, desc="Processing"):
                utt_id = utt['utt_id']
                start = utt['h5_start_idx']
                end = utt['h5_end_idx']

                # 提取特征
                utt_features = torch.from_numpy(
                    features[start:end][:]
                ).float().to(device)

                # 预测 phones
                with torch.no_grad():
                    phones = predictor(utt_features)

                # 保存
                f_out.create_dataset(
                    utt_id,
                    data=phones.cpu().numpy(),
                    compression='gzip'
                )

    print(f"\n✓ Phone 预测已保存到: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
