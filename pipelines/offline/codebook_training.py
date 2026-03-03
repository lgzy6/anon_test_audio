"""Step 4: Codebook 训练（优化版：Streaming + 顺序段采样）"""

import torch
import h5py
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Optional, List, Tuple


class CodebookTrainer:
    """
    Codebook 训练器（优化版）

    - 顺序段采样（HDF5 友好）
    - MiniBatchKMeans.partial_fit 流式训练
    - 不再一次性加载全部特征
    """

    def __init__(
        self,
        codebook_size: int = 512,
        max_samples: Optional[int] = None,
        kmeans_batch_size: int = 4096,
        kmeans_max_iter: int = 30,
        random_state: int = 42,
        sampling_method: str = "segmented",  # "segmented" | "uniform"
        num_segments: int = 10,
    ):
        self.K = codebook_size
        self.max_samples = max_samples
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_max_iter = kmeans_max_iter
        self.random_state = random_state
        self.sampling_method = sampling_method
        self.num_segments = num_segments

        self.codebook = None

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def _build_segments(
        self,
        total_frames: int,
    ) -> List[Tuple[int, int]]:
        """
        构造顺序读取的 (start, length) 段
        """
        if self.max_samples is None or self.max_samples >= total_frames:
            return [(0, total_frames)]

        rng = np.random.RandomState(self.random_state)

        if self.sampling_method == "uniform":
            # 均匀切分为若干段，保证顺序读取
            seg_len = self.max_samples // self.num_segments
            starts = np.linspace(
                0, total_frames - seg_len, self.num_segments, dtype=int
            )
            return [(int(s), seg_len) for s in starts]

        elif self.sampling_method == "segmented":
            # 随机选取若干连续段
            seg_len = self.max_samples // self.num_segments
            segments = []

            for _ in range(self.num_segments):
                max_start = max(0, total_frames - seg_len)
                start = rng.randint(0, max_start + 1)
                segments.append((start, seg_len))

            # 按 start 排序，保证 HDF5 顺序读
            segments.sort(key=lambda x: x[0])
            return segments

        else:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        features_h5_path: str,
        metadata_path: str,
    ) -> np.ndarray:
        """
        训练 Codebook（流式）
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        total_frames = metadata["total_frames"]
        feature_dim = metadata["feature_dim"]

        print(f"[Codebook] Total frames: {total_frames}")
        print(f"[Codebook] Feature dim: {feature_dim}")

        segments = self._build_segments(total_frames)

        total_used = sum(length for _, length in segments)
        print(f"[Codebook] Using {total_used} frames in {len(segments)} segments")

        kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            batch_size=self.kmeans_batch_size,
            max_iter=self.kmeans_max_iter,
            random_state=self.random_state,
            compute_labels=False,  # 不需要 labels
            verbose=1,
        )

        with h5py.File(features_h5_path, "r") as h5f:
            feats = h5f["features"]

            for seg_id, (start, length) in enumerate(segments, 1):
                end = start + length
                print(
                    f"[Codebook] Segment {seg_id}/{len(segments)} "
                    f"frames [{start}:{end}]"
                )

                # 在 segment 内再按 batch_size 切
                for i in range(start, end, self.kmeans_batch_size):
                    batch = feats[i : min(i + self.kmeans_batch_size, end)]
                    kmeans.partial_fit(batch)

        self.codebook = kmeans.cluster_centers_
        print("[Codebook] Training finished")

        self._compute_quality_metrics(kmeans)
        return self.codebook

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_quality_metrics(self, kmeans: MiniBatchKMeans):
        """
        Codebook 基本质量指标（不依赖全量 labels）
        """
        centers = kmeans.cluster_centers_
        norms = np.linalg.norm(centers, axis=1)

        print("[Codebook] Quality metrics:")
        print(f"  - Codebook size: {centers.shape[0]}")
        print(f"  - Feature dim: {centers.shape[1]}")
        print(f"  - Center norm mean/std: {norms.mean():.3f} / {norms.std():.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "codebook": torch.from_numpy(self.codebook).float(),
                "codebook_size": self.K,
                "feature_dim": self.codebook.shape[1],
            },
            path,
        )
        print(f"[Codebook] Saved to {path}")


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_codebook_training(config: Dict):
    cache_dir = Path(config["paths"]["cache_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])

    features_h5 = cache_dir / "features" / "cleaned" / "features.h5"
    metadata_path = cache_dir / "features" / "cleaned" / "metadata.json"

    trainer = CodebookTrainer(
        codebook_size=config["samm"]["codebook_size"],
        max_samples=config["offline"]["codebook_training"]["max_samples"],
        kmeans_batch_size=config["offline"]["codebook_training"]["kmeans_batch_size"],
        kmeans_max_iter=config["offline"]["codebook_training"]["kmeans_max_iter"],
        sampling_method=config["offline"]["codebook_training"].get(
            "sampling_method", "segmented"
        ),
        num_segments=config["offline"]["codebook_training"].get(
            "num_segments", 10
        ),
    )

    trainer.train(
        features_h5_path=str(features_h5),
        metadata_path=str(metadata_path),
    )

    output_path = checkpoint_dir / "codebook.pt"
    trainer.save(str(output_path))
