#!/usr/bin/env python3
"""
使用 v6 Pattern 模型构建 Target Pool

v6 特点:
- Sinkhorn-Knopp 强制均匀分配
- 固定温度退火策略
- Pattern 死亡检测与重新初始化
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import faiss
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接从 v6 测试脚本导入模型定义
from tests.test_v32_style_clustering_v6 import DiverseSAMMEncoder


PHONE_LIST = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
]


def load_v6_model(model_path: str, device: str = 'cuda'):
    """加载 v6 训练好的模型"""
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt['config']

    model = DiverseSAMMEncoder(
        input_dim=cfg['input_dim'],
        d_model=cfg['d_model'],
        n_symbols=cfg['n_symbols'],
        n_layers=cfg['n_layers'],
        n_patterns=cfg['n_patterns'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    print(f"Loaded v6 model: {cfg}")
    print(f"Results: {ckpt.get('results', {})}")
    return model, cfg


class V6PoolBuilder:
    """使用 v6 模型构建 Target Pool"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.phone_to_idx = {p: i for i, p in enumerate(PHONE_LIST)}
        self.pattern_model, self.model_cfg = load_v6_model(
            config['model_path'], self.device
        )
        self.n_patterns = self.model_cfg['n_patterns']

    @torch.no_grad()
    def get_pattern_assignments(self, features: np.ndarray, batch_size: int = 64):
        """批量获取 pattern 分配"""
        N = len(features)
        patterns = []

        for i in tqdm(range(0, N, batch_size), desc="Pattern assignment"):
            batch = features[i:i+batch_size]
            x = torch.from_numpy(batch).float().unsqueeze(0).to(self.device)
            p = self.pattern_model.get_pattern_assignments(x)
            patterns.append(p.squeeze(0).cpu().numpy())

        return np.concatenate(patterns)

    def build(self):
        """构建 Target Pool"""
        print("=" * 60)
        print("Building Target Pool with v6 Pattern Model")
        print("=" * 60)

        cache_dir = Path(self.config['cache_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载特征和元数据
        print("\n[1/4] Loading features...")
        h5_path = cache_dir / 'features.h5'
        meta_path = cache_dir / 'metadata.json'

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]
        print(f"  Features shape: {features.shape}")

        # 2. 获取 Pattern 分配
        print("\n[2/4] Getting pattern assignments...")
        patterns = self.get_pattern_assignments(features)
        print(f"  Pattern distribution: {np.bincount(patterns, minlength=self.n_patterns)}")

        # 3. 构建性别信息
        print("\n[3/4] Building gender array...")
        genders = self._build_gender_array(metadata, features.shape[0])
        print(f"  Gender distribution: M={np.sum(genders==0)}, F={np.sum(genders==1)}")

        # 4. 保存 Pool
        print("\n[4/4] Saving pool...")
        self._save_pool(output_dir, features, patterns, genders)

        print("\n" + "=" * 60)
        print("Pool building complete!")
        print("=" * 60)

    def _build_gender_array(self, metadata, n_frames):
        """构建性别数组"""
        genders = np.zeros(n_frames, dtype=np.int32)
        for utt in metadata.get('utterances', []):
            start = utt['h5_start_idx']
            end = utt['h5_end_idx']
            gender = 0 if utt.get('gender', 'M').upper() == 'M' else 1
            genders[start:end] = gender
        return genders

    def _save_pool(self, output_dir, features, patterns, genders):
        """保存 Pool 数据"""
        # 保存特征和标签
        np.save(output_dir / "features.npy", features)
        np.save(output_dir / "patterns.npy", patterns)
        np.save(output_dir / "genders.npy", genders)

        # 构建 FAISS 索引
        print("  Building FAISS index...")
        features_norm = features.copy()
        faiss.normalize_L2(features_norm)
        dim = features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(features_norm)
        faiss.write_index(index, str(output_dir / "faiss.index"))

        # 保存元数据
        meta = {
            'n_frames': len(features),
            'feature_dim': features.shape[1],
            'n_patterns': self.n_patterns,
            'pattern_distribution': np.bincount(patterns, minlength=self.n_patterns).tolist(),
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved to: {output_dir}")


def main():
    """主函数"""
    config = {
        'model_path': 'outputs/v6_tests/samm_v6_model.pt',
        'cache_dir': 'cache/features/wavlm',
        'output_dir': 'data/samm_anon/target_pool_v6',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    builder = V6PoolBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()