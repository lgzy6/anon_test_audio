#!/usr/bin/env python3
"""
使用 v5.2 Pattern 模型构建 Target Pool 并进行语音合成测试

流程:
1. 加载 v5.2 训练好的 Pattern 模型
2. 对所有特征进行 Pattern 分配
3. 构建量化 Target Pool
4. 运行语音合成测试
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
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# v5.2 模型定义 (与训练时一致)
# ============================================================

class SymbolizationLayer(nn.Module):
    def __init__(self, input_dim: int, n_symbols: int = 64, codebook_dim: int = 256):
        super().__init__()
        self.n_symbols = n_symbols
        self.proj = nn.Linear(input_dim, codebook_dim)
        self.codebook = nn.Parameter(torch.randn(n_symbols, codebook_dim) * 0.1)
        self.temperature = 0.5

    def forward(self, x: torch.Tensor, hard: bool = False):
        h = self.proj(x)
        h_norm = F.normalize(h, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        logits = torch.matmul(h_norm, cb_norm.T) / self.temperature
        if hard:
            indices = logits.argmax(dim=-1)
            quantized = self.codebook[indices]
        else:
            soft = F.gumbel_softmax(logits, tau=0.5, hard=False, dim=-1)
            quantized = torch.matmul(soft, self.codebook)
            indices = logits.argmax(dim=-1)
        return quantized, indices


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = res + x
        x = x + self.ffn(self.norm2(x))
        return x


class PatternMatrixV52(nn.Module):
    """v5.2 版本的 Pattern Matrix"""
    def __init__(self, d_model: int = 256, n_patterns: int = 32):
        super().__init__()
        self.n_patterns = n_patterns
        self.d_model = d_model

        patterns_init = torch.randn(d_model, n_patterns)
        Q, R = torch.linalg.qr(patterns_init)
        self.patterns = nn.Parameter(Q.T.contiguous() * 0.5)

        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.register_buffer('pattern_usage', torch.zeros(n_patterns))
        self.register_buffer('usage_momentum', torch.tensor(0.99))

    def get_hard_assignments(self, x: torch.Tensor):
        Q = self.query_proj(x)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(self.patterns, dim=-1)
        logits = torch.matmul(Q, K.T)
        return logits.argmax(dim=-1)


class PrototypeSAMMEncoder(nn.Module):
    """v5.2 Encoder"""
    def __init__(self, input_dim=1024, d_model=256, n_symbols=64,
                 n_heads=4, n_layers=2, n_patterns=32):
        super().__init__()
        self.d_model = d_model
        self.symbol = SymbolizationLayer(input_dim, n_symbols, d_model)
        self.layers = nn.ModuleList([
            MaskedSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pattern = PatternMatrixV52(d_model, n_patterns)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

    def _encode_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, _ = self.symbol(x, hard=False)
        h = h + self.pos_enc[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        return h

    def get_pattern_assignments(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._encode_to_hidden(x)
            return self.pattern.get_hard_assignments(h)


PHONE_LIST = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
]


def load_v52_model(model_path: str, device: str = 'cuda'):
    """加载 v5.2 训练好的模型"""
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt['config']

    model = PrototypeSAMMEncoder(
        input_dim=cfg['input_dim'],
        d_model=cfg['d_model'],
        n_symbols=cfg['n_symbols'],
        n_layers=cfg['n_layers'],
        n_patterns=cfg['n_patterns'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    print(f"Loaded v5.2 model: {cfg}")
    print(f"Results: {ckpt['results']}")
    return model, cfg


class V52PoolBuilder:
    """使用 v5.2 模型构建 Target Pool"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.phone_to_idx = {p: i for i, p in enumerate(PHONE_LIST)}
        self.pattern_model, self.model_cfg = load_v52_model(
            config['v52_model_path'], self.device
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
        print("Building Target Pool with v5.2 Pattern Model")
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

        # 3. 构建说话人索引和性别信息
        print("\n[3/4] Building speaker index...")
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
        'v52_model_path': 'outputs/v52_tests/samm_v5.2_model.pt',
        'cache_dir': 'cache/features/wavlm',
        'output_dir': 'data/samm_anon/target_pool_v52',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    builder = V52PoolBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()
