#!/usr/bin/env python3
"""
快速查看所有 Target Pool 版本信息
"""

import json
import numpy as np
from pathlib import Path


def analyze_pool(pool_path: Path):
    """分析单个 pool"""
    print(f"\n{'='*70}")
    print(f"Target Pool: {pool_path.name}")
    print(f"{'='*70}")
    print(f"路径: {pool_path}")

    if not pool_path.exists():
        print("✗ 不存在")
        return

    # Metadata
    metadata_path = pool_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\n📄 Metadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    else:
        print(f"\n⚠️  无 metadata.json")

    # Features
    features_path = pool_path / "features.npy"
    if not features_path.exists():
        features_path = pool_path / "prototypes.npy"

    if features_path.exists():
        features = np.load(features_path, mmap_mode='r')
        print(f"\n🎯 Features:")
        print(f"  - 形状: {features.shape}")
        print(f"  - 数量: {len(features):,}")
        print(f"  - 维度: {features.shape[1]}")

        # 采样统计
        sample = features[:min(1000, len(features))]
        print(f"  - 均值: {sample.mean():.6f}")
        print(f"  - 标准差: {sample.std():.6f}")
        print(f"  - 范围: [{sample.min():.6f}, {sample.max():.6f}]")
    else:
        print(f"\n✗ 无 features 文件")

    # Phone Clusters
    phone_clusters_path = pool_path / "phone_clusters.pt"
    if phone_clusters_path.exists():
        import torch
        clusters = torch.load(phone_clusters_path, map_location='cpu')
        print(f"\n📱 Phone Clusters:")
        print(f"  - ✓ 存在")
        print(f"  - 数量: {len(clusters)}")
        if clusters:
            first_key = list(clusters.keys())[0]
            print(f"  - 示例: {first_key} → {clusters[first_key].shape}")
    else:
        print(f"\n📱 Phone Clusters:")
        print(f"  - ✗ 不存在")

    # Phones
    phones_path = pool_path / "phones.npy"
    if phones_path.exists():
        phones = np.load(phones_path)
        print(f"\n🔤 Phones:")
        print(f"  - ✓ 存在 ({len(phones):,} 个)")
        unique_phones = np.unique(phones)
        print(f"  - 唯一音素数: {len(unique_phones)}")
    else:
        print(f"\n🔤 Phones: ✗ 不存在")

    # Symbols
    symbols_path = pool_path / "symbols.npy"
    if symbols_path.exists():
        symbols = np.load(symbols_path)
        print(f"\n🔣 Symbols:")
        print(f"  - ✓ 存在 ({len(symbols):,} 个)")
        unique_symbols = np.unique(symbols)
        print(f"  - 唯一符号数: {len(unique_symbols)}")
    else:
        print(f"\n🔣 Symbols: ✗ 不存在")

    # Genders
    genders_path = pool_path / "genders.npy"
    if genders_path.exists():
        genders = np.load(genders_path)
        print(f"\n⚧ Genders:")
        print(f"  - ✓ 存在 ({len(genders):,} 个)")
        unique, counts = np.unique(genders, return_counts=True)
        for g, c in zip(unique, counts):
            gender_name = 'Male' if g == 0 else 'Female'
            print(f"  - {gender_name} (ID={g}): {c:,} ({c/len(genders)*100:.1f}%)")
    else:
        print(f"\n⚧ Genders: ✗ 不存在")

    # 大小
    total_size = sum(f.stat().st_size for f in pool_path.glob("*") if f.is_file())
    print(f"\n💾 总大小: {total_size / 1024 / 1024:.1f} MB")

    # 文件列表
    print(f"\n📁 文件列表:")
    for f in sorted(pool_path.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name}: {size_mb:.1f} MB")


def main():
    """主函数"""
    print("="*70)
    print("Target Pool 版本信息查看")
    print("="*70)

    # 查找所有版本
    base_dirs = [
        Path("data/samm_anon/checkpoints"),
        Path("checkpoints"),
    ]

    pools_found = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pool_dir in base_dir.glob("target_pool*"):
            if pool_dir.is_symlink():
                target = pool_dir.resolve()
                print(f"\n软链接: {pool_dir.name} → {target.name}")
                continue
            pools_found.append(pool_dir)

    # 分析每个版本
    for pool_path in sorted(pools_found, key=lambda p: p.name):
        analyze_pool(pool_path)

    print(f"\n{'='*70}")
    print(f"总共找到 {len(pools_found)} 个 Target Pool 版本")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
