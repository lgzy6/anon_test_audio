# pipelines/offline/pattern_learning.py
"""
Step 5: Pattern Matrix 学习（GPU 适配版）
"""

import torch
import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict


class PatternMatrixLearner:
    """
    Pattern Matrix 学习器

    统计符号序列的 bigram 转移概率
    （量化在 GPU，统计在 CPU）
    """

    def __init__(
        self,
        codebook_size: int = 512,
        smoothing_alpha: float = 0.1,
        device: str = "cuda",
        max_gpu_batch: int = 200_000,
    ):
        self.K = codebook_size
        self.alpha = smoothing_alpha
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.max_gpu_batch = max_gpu_batch

        self.M = None
        self.marginal = None

    # ------------------------------------------------------------------
    # GPU 量化（核心优化）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _quantize_gpu(
        self,
        features: np.ndarray,
        codebook: torch.Tensor,
    ) -> np.ndarray:
        """
        Args:
            features: [T, D] numpy
            codebook: [K, D] torch (on GPU)
        Returns:
            z: [T] numpy
        """
        T = features.shape[0]
        z = np.empty(T, dtype=np.int32)

        for i in range(0, T, self.max_gpu_batch):
            end = min(i + self.max_gpu_batch, T)

            x = torch.from_numpy(features[i:end]).to(self.device)

            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2x·c
            x_norm = (x ** 2).sum(dim=1, keepdim=True)          # [B,1]
            c_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)   # [1,K]

            dist = x_norm + c_norm - 2.0 * (x @ codebook.T)
            z[i:end] = torch.argmin(dist, dim=1).cpu().numpy()

        return z

    # ------------------------------------------------------------------
    # 主学习流程
    # ------------------------------------------------------------------
    def learn(
        self,
        features_h5_path: str,
        metadata_path: str,
        codebook_path: str,
    ):
        # 1. 加载 Codebook（GPU）
        ckpt = torch.load(codebook_path, map_location="cpu")
        codebook = ckpt["codebook"].float().to(self.device)
        codebook.requires_grad_(False)

        # 2. 加载元数据
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        utterances = metadata["utterances"]
        print(f"Learning pattern matrix from {len(utterances)} utterances...")
        print(f"Using device: {self.device}")

        # 3. 初始化统计矩阵
        counts = np.zeros((self.K, self.K), dtype=np.float64)

        # 4. 读取特征并逐 utterance 处理
        with h5py.File(features_h5_path, "r") as h5f:
            features = h5f["features"]

            for utt in tqdm(utterances, desc="Processing utterances"):
                s = utt["h5_start_idx"]
                e = utt["h5_end_idx"]

                utt_features = features[s:e]
                if len(utt_features) < 2:
                    continue

                # ---- GPU 量化 ----
                z = self._quantize_gpu(utt_features, codebook)

                # ---- 向量化 bigram 统计 ----
                prev = z[:-1]
                nxt = z[1:]
                np.add.at(counts, (prev, nxt), 1)

        # 5. Laplace 平滑 + 归一化
        counts_smooth = counts + self.alpha
        row_sums = counts_smooth.sum(axis=1, keepdims=True)
        self.M = counts_smooth / (row_sums + 1e-10)

        # 6. 边缘分布
        total = counts.sum()
        if total > 0:
            self.marginal = counts.sum(axis=0) / total
        else:
            self.marginal = np.ones(self.K) / self.K

        print(f"Pattern matrix learned: {self.M.shape}")
        self._print_statistics(counts)

    # ------------------------------------------------------------------
    # 统计信息（保持可解释性）
    # ------------------------------------------------------------------
    def _print_statistics(self, counts: np.ndarray):
        nonzero = np.count_nonzero(counts)
        sparsity = 1.0 - nonzero / (self.K * self.K)

        print("Pattern matrix statistics:")
        print(f"  - Non-zero transitions: {nonzero}")
        print(f"  - Sparsity: {sparsity:.2%}")

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "M": torch.from_numpy(self.M).float(),
                "marginal": torch.from_numpy(self.marginal).float(),
                "codebook_size": self.K,
                "smoothing_alpha": self.alpha,
            },
            path,
        )
        print(f"Pattern matrix saved to {path}")


# ----------------------------------------------------------------------
# Runner（接口不变）
# ----------------------------------------------------------------------
def run_pattern_learning(config: Dict):
    cache_dir = Path(config["paths"]["cache_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])

    learner = PatternMatrixLearner(
        codebook_size=config["samm"]["codebook_size"],
        smoothing_alpha=config["pattern"]["smoothing_alpha"],
        device=config["pattern"].get("device", "cuda"),
        max_gpu_batch=config["pattern"].get("gpu_batch", 200_000),
    )

    learner.learn(
        features_h5_path=str(
            cache_dir / "features" / "cleaned" / "features.h5"
        ),
        metadata_path=str(
            cache_dir / "features" / "cleaned" / "metadata.json"
        ),
        codebook_path=str(checkpoint_dir / "codebook.pt"),
    )

    learner.save(str(checkpoint_dir / "pattern_matrix.pt"))
