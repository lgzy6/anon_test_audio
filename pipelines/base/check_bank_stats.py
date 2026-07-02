#!/usr/bin/env python3
"""检查 bank 各文件的质心数量和分散度（平均成对余弦距离）"""
import torch
import numpy as np
from pathlib import Path

BANK_DIR = Path("/root/autodl-tmp/anon_test/checkpoints/banks_c8f20s4_e05_d3")

def centroid_stats(centroids: torch.Tensor):
    """返回 (数量, 平均成对余弦距离)"""
    n = centroids.shape[0]
    if n < 2:
        return n, 0.0
    v = centroids.float()
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    sim = v @ v.T  # [n, n]
    # 取上三角（不含对角）
    idx = torch.triu_indices(n, n, offset=1)
    avg_sim = sim[idx[0], idx[1]].mean().item()
    return n, 1.0 - avg_sim  # 余弦距离 = 1 - 余弦相似度

print(f"{'文件':<35} {'音素数':>6} {'总质心数':>8} {'均质心/音素':>11} {'l6分散度':>10} {'l24分散度':>10}")
print("-" * 85)

for pt_file in sorted(BANK_DIR.glob("*.pt")):
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    phone_count = len(data)
    all_l6, all_l24 = [], []
    for v in data.values():
        all_l6.append(v["l6"])
        all_l24.append(v["l24"])
    all_l6 = torch.cat(all_l6, dim=0)
    all_l24 = torch.cat(all_l24, dim=0)
    total = all_l6.shape[0]
    _, disp_l6  = centroid_stats(all_l6)
    _, disp_l24 = centroid_stats(all_l24)
    print(f"{pt_file.name:<35} {phone_count:>6} {total:>8} {total/phone_count:>11.2f} {disp_l6:>10.4f} {disp_l24:>10.4f}")
