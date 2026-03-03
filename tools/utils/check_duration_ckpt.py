#!/usr/bin/env python3
"""检查 duration_decoder checkpoint 结构"""
import torch
import sys

ckpt_path = "checkpoints/duration_decoder.pt"
print(f"检查 checkpoint: {ckpt_path}\n")

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print("=" * 60)
    print("Checkpoint 结构:")
    print("=" * 60)

    if isinstance(ckpt, dict):
        print(f"\n顶层键: {list(ckpt.keys())}\n")

        # 如果有 'model' 键
        if 'model' in ckpt:
            print("模型权重键 (前20个):")
            for i, key in enumerate(list(ckpt['model'].keys())[:20]):
                print(f"  {i+1}. {key}")
        else:
            print("模型权重键 (前20个):")
            for i, key in enumerate(list(ckpt.keys())[:20]):
                print(f"  {i+1}. {key}")

        # 检查 config
        if 'config' in ckpt:
            print(f"\n配置: {ckpt['config']}")
    else:
        print(f"Checkpoint 类型: {type(ckpt)}")

except Exception as e:
    print(f"错误: {e}")
    sys.exit(1)
