# pipelines/offline/pool_building_v3.py
"""
Step 6: Target Pool Construction (v3.0)
基于 Pattern 分区的目标池构建

v3.0 核心改进:
1. 按说话人 Pattern 聚类，构建分区池
2. 移除 symbols.npy 的符号约束依赖
3. 支持 TargetSelector 的 Pattern-based 目标选择

输出结构:
target_pool/
├── metadata.json
├── pattern_centroids.pt          # Pattern 聚类中心
├── pattern_assignments.json      # 说话人 -> Pattern 映射
├── pool_pattern_0/               # Pattern 0 专属池
│   ├── features.npy
│   ├── phones.npy
│   ├── genders.npy
│   └── phone_clusters.pt
├── pool_pattern_1/
│   └── ...
└── ...
"""

import json
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from sklearn.cluster import KMeans
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class PoolConfigV3:
    """Target Pool v3.0 配置"""
    # Pattern 聚类
    n_patterns: int = 8                    # Pattern 数量
    pattern_repr_dim: int = 512            # Pattern 表示维度
    use_transition_matrix: bool = True     # 是否使用转移矩阵作为 Pattern 特征

    # Phone Clusters
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8

    # IO 参数
    io_batch_size: int = 100000
    max_samples_per_pattern: int = 10000000  # 每个 Pattern 最大帧数

    # 功能开关
    build_legacy_pool: bool = True         # 是否同时构建旧格式池 (兼容)


class TargetPoolBuilderV3:
    """
    Target Pool 构建器 v3.0

    核心改进:
    1. 基于 Pattern 分区构建
    2. 不再生成用于约束检索的 symbols.npy
    3. 支持 TargetSelector 的 Pattern-based 选择
    """

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: Dict, pool_config: Optional[PoolConfigV3] = None):
        self.config = config
        self.pool_config = pool_config or self._load_pool_config()

        self.cache_dir = Path(config['paths']['cache_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoints_dir'])
        self.pool_dir = self.checkpoint_dir / 'target_pool'
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.feature_dim = config.get('ssl', {}).get('hidden_dim', 1024)
        self.phone_to_idx = {p: i for i, p in enumerate(self.PHONE_LIST)}
        self.idx_to_phone = {i: p for p, i in self.phone_to_idx.items()}
        self.num_phones = len(self.PHONE_LIST)

        # 加载 codebook (用于 Pattern 计算)
        self.codebook = self._load_codebook()

    def _load_pool_config(self) -> PoolConfigV3:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        knn_cfg = self.config.get('knn_vc', {})

        return PoolConfigV3(
            n_patterns=cfg.get('n_patterns', 8),
            pattern_repr_dim=cfg.get('pattern_repr_dim', 512),
            use_transition_matrix=cfg.get('use_transition_matrix', True),
            build_phone_clusters=cfg.get('build_phone_clusters', True),
            n_phone_clusters=knn_cfg.get('num_clusters', 8),
            io_batch_size=cfg.get('io_batch_size', 100000),
            max_samples_per_pattern=cfg.get('max_samples_per_pattern', 10000000),
            build_legacy_pool=cfg.get('build_legacy_pool', True),
        )

    def _load_codebook(self) -> Optional[torch.Tensor]:
        """加载 SAMM codebook"""
        codebook_path = self.checkpoint_dir / 'codebook.pt'

        if not codebook_path.exists():
            logger.warning("Codebook not found, Pattern clustering will be disabled")
            return None

        data = torch.load(codebook_path, map_location='cpu')
        if isinstance(data, dict):
            codebook = data.get('codebook', data.get('centers'))
        else:
            codebook = data

        if isinstance(codebook, np.ndarray):
            codebook = torch.from_numpy(codebook)

        logger.info(f"Loaded codebook: {codebook.shape}")
        return codebook.float()

    def build(self) -> Dict:
        """构建 v3.0 Target Pool"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Target Pool Construction (v3.0)")
        logger.info("=" * 60)

        # 验证输入
        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 1. 构建基础索引映射
        logger.info("\n[1/5] 构建基础索引映射...")
        index_map = self._build_index_map(features_path, metadata)

        # 2. 计算说话人 Pattern 表示并聚类
        logger.info("\n[2/5] 计算说话人 Pattern 并聚类...")
        pattern_assignments, pattern_centroids = self._cluster_speakers_by_pattern(
            features_path, index_map, metadata
        )

        # 3. 构建分区 Pattern 池
        logger.info("\n[3/5] 构建分区 Pattern 池...")
        self._build_partitioned_pools(
            features_path, index_map, metadata, pattern_assignments
        )

        # 4. 保存 Pattern 元数据
        logger.info("\n[4/5] 保存 Pattern 元数据...")
        self._save_pattern_metadata(pattern_assignments, pattern_centroids)

        # 5. (可选) 构建兼容旧格式的统一池
        if self.pool_config.build_legacy_pool:
            logger.info("\n[5/5] 构建兼容旧格式池...")
            self._build_legacy_pool(features_path, index_map, metadata)

        logger.info("\n" + "=" * 60)
        logger.info("Target Pool v3.0 构建完成!")
        logger.info("=" * 60)

        return {
            'pool_dir': str(self.pool_dir),
            'n_patterns': self.pool_config.n_patterns,
            'total_frames': index_map['total_frames'],
            'pattern_assignments': pattern_assignments,
        }

    def _build_index_map(self, features_path: Path, metadata: Dict) -> Dict:
        """构建帧级索引映射"""
        utterances = metadata['utterances']
        total_frames = metadata['total_frames']

        frame_to_utt = np.zeros(total_frames, dtype=np.int32)
        frame_to_phone = np.full(total_frames, self.phone_to_idx['SIL'], dtype=np.int32)
        frame_to_gender = np.zeros(total_frames, dtype=np.int8)
        frame_to_spk = np.zeros(total_frames, dtype=np.int32)  # 新增: 说话人索引

        # 构建说话人索引
        spk_to_idx = {}
        spk_idx = 0

        # 加载 phone 预测
        phone_predictions = self._load_phone_predictions()

        for utt_idx, utt in enumerate(tqdm(utterances, desc="Building index map")):
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            n_frames = e - s

            frame_to_utt[s:e] = utt_idx

            # 性别
            gender = utt.get('gender', 'M')
            if gender not in ['M', 'F']:
                gender = 'M'
            frame_to_gender[s:e] = 0 if gender == 'M' else 1

            # 说话人
            spk_id = utt.get('speaker_id', utt.get('spk_id', f"spk_{utt_idx}"))
            if spk_id not in spk_to_idx:
                spk_to_idx[spk_id] = spk_idx
                spk_idx += 1
            frame_to_spk[s:e] = spk_to_idx[spk_id]

            # 音素
            utt_id = utt['utt_id']
            if phone_predictions is not None and utt_id in phone_predictions:
                phones = phone_predictions[utt_id]
                phones = phones[:n_frames] if len(phones) >= n_frames else \
                    np.pad(phones, (0, n_frames - len(phones)),
                           constant_values=self.phone_to_idx['SIL'])
                frame_to_phone[s:e] = phones

        logger.info(f"  Total frames: {total_frames:,}")
        logger.info(f"  Utterances: {len(utterances):,}")
        logger.info(f"  Speakers: {len(spk_to_idx):,}")

        return {
            'total_frames': total_frames,
            'utterances': utterances,
            'frame_to_utt': frame_to_utt,
            'frame_to_phone': frame_to_phone,
            'frame_to_gender': frame_to_gender,
            'frame_to_spk': frame_to_spk,
            'spk_to_idx': spk_to_idx,
            'idx_to_spk': {v: k for k, v in spk_to_idx.items()},
        }

    def _load_phone_predictions(self) -> Optional[Dict]:
        """加载 phone 预测"""
        paths = [
            self.cache_dir / 'features' / 'cleaned' / 'phone_predictions.h5',
            self.checkpoint_dir / 'target_pool' / 'phones.npy',
        ]

        for path in paths:
            if path.suffix == '.h5' and path.exists():
                predictions = {}
                with h5py.File(path, 'r') as f:
                    for utt_id in f.keys():
                        predictions[utt_id] = f[utt_id][:]
                if predictions:
                    logger.info(f"  Loaded phone predictions: {len(predictions)} utts")
                    return predictions

        logger.warning("  No phone predictions found")
        return None

    def _cluster_speakers_by_pattern(
        self,
        features_path: Path,
        index_map: Dict,
        metadata: Dict,
    ) -> Tuple[Dict[str, int], np.ndarray]:
        """
        基于 Pattern 对说话人进行聚类

        Returns:
            pattern_assignments: {speaker_id: pattern_id}
            pattern_centroids: [n_patterns, repr_dim]
        """
        if self.codebook is None:
            # 没有 codebook，随机分配
            logger.warning("  No codebook, using random pattern assignment")
            spk_to_idx = index_map['spk_to_idx']
            n_spk = len(spk_to_idx)
            n_patterns = self.pool_config.n_patterns

            assignments = {}
            for spk_id, idx in spk_to_idx.items():
                assignments[spk_id] = idx % n_patterns

            centroids = np.zeros((n_patterns, self.feature_dim), dtype=np.float32)
            return assignments, centroids

        # 计算每个说话人的 Pattern 表示
        logger.info("  Computing speaker pattern representations...")
        spk_representations = self._compute_speaker_patterns(
            features_path, index_map
        )

        # K-Means 聚类
        logger.info(f"  Clustering speakers into {self.pool_config.n_patterns} patterns...")
        spk_ids = list(spk_representations.keys())
        repr_matrix = np.stack([spk_representations[s] for s in spk_ids])

        kmeans = KMeans(
            n_clusters=self.pool_config.n_patterns,
            n_init='auto',
            random_state=42
        )
        labels = kmeans.fit_predict(repr_matrix)
        centroids = kmeans.cluster_centers_

        # 构建分配映射
        assignments = {spk_ids[i]: int(labels[i]) for i in range(len(spk_ids))}

        # 统计
        pattern_counts = np.bincount(labels, minlength=self.pool_config.n_patterns)
        logger.info(f"  Pattern distribution: {pattern_counts.tolist()}")

        return assignments, centroids

    def _compute_speaker_patterns(
        self,
        features_path: Path,
        index_map: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        计算每个说话人的 Pattern 表示

        Pattern 表示 = [符号分布直方图, 转移矩阵 (可选)]
        """
        K = self.codebook.shape[0]
        spk_to_idx = index_map['spk_to_idx']
        idx_to_spk = index_map['idx_to_spk']
        frame_to_spk = index_map['frame_to_spk']

        # 初始化计数器
        spk_symbol_counts = {s: np.zeros(K, dtype=np.float32) for s in spk_to_idx}
        spk_transition_counts = {s: np.zeros((K, K), dtype=np.float32) for s in spk_to_idx}
        spk_frame_counts = {s: 0 for s in spk_to_idx}

        # 批量处理特征
        batch_size = self.pool_config.io_batch_size
        total_frames = index_map['total_frames']

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            prev_symbols = {}  # 上一帧的符号

            for start in tqdm(range(0, total_frames, batch_size), desc="Computing patterns"):
                end = min(start + batch_size, total_frames)
                batch_feats = torch.from_numpy(feats[start:end]).float()

                # 量化到 codebook
                dist = torch.cdist(batch_feats, self.codebook)
                symbols = dist.argmin(dim=-1).numpy()

                # 统计
                batch_spk = frame_to_spk[start:end]

                for i, (sym, spk_idx) in enumerate(zip(symbols, batch_spk)):
                    spk_id = idx_to_spk[spk_idx]
                    spk_symbol_counts[spk_id][sym] += 1
                    spk_frame_counts[spk_id] += 1

                    # 转移计数
                    if self.pool_config.use_transition_matrix:
                        if spk_id in prev_symbols:
                            prev_sym = prev_symbols[spk_id]
                            spk_transition_counts[spk_id][prev_sym, sym] += 1
                        prev_symbols[spk_id] = sym

        # 归一化并构建表示
        representations = {}
        for spk_id in spk_to_idx:
            count = max(spk_frame_counts[spk_id], 1)

            # 符号分布
            hist = spk_symbol_counts[spk_id] / count

            if self.pool_config.use_transition_matrix:
                # 转移矩阵归一化
                trans = spk_transition_counts[spk_id]
                row_sums = trans.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                trans = trans / row_sums

                # 拼接
                representations[spk_id] = np.concatenate([hist, trans.flatten()])
            else:
                representations[spk_id] = hist

        return representations

    def _build_partitioned_pools(
        self,
        features_path: Path,
        index_map: Dict,
        metadata: Dict,
        pattern_assignments: Dict[str, int],
    ):
        """构建分区 Pattern 池"""
        n_patterns = self.pool_config.n_patterns
        frame_to_spk = index_map['frame_to_spk']
        idx_to_spk = index_map['idx_to_spk']

        # 为每个 Pattern 创建目录
        for p in range(n_patterns):
            (self.pool_dir / f'pool_pattern_{p}').mkdir(exist_ok=True)

        # 按 Pattern 分组帧索引
        pattern_frames = {p: [] for p in range(n_patterns)}

        for frame_idx in range(index_map['total_frames']):
            spk_idx = frame_to_spk[frame_idx]
            spk_id = idx_to_spk[spk_idx]
            pattern_id = pattern_assignments.get(spk_id, 0)
            pattern_frames[pattern_id].append(frame_idx)

        # 构建每个 Pattern 的池
        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for pattern_id in tqdm(range(n_patterns), desc="Building pattern pools"):
                frames = pattern_frames[pattern_id]

                if len(frames) == 0:
                    logger.warning(f"  Pattern {pattern_id}: no frames, skipping")
                    continue

                # 限制最大帧数
                if len(frames) > self.pool_config.max_samples_per_pattern:
                    frames = np.random.choice(
                        frames,
                        self.pool_config.max_samples_per_pattern,
                        replace=False
                    ).tolist()

                frames = sorted(frames)
                n_frames = len(frames)

                logger.info(f"  Pattern {pattern_id}: {n_frames:,} frames")

                # 提取特征
                pattern_features = self._load_features_by_indices(feats, frames)
                pattern_phones = index_map['frame_to_phone'][frames]
                pattern_genders = index_map['frame_to_gender'][frames]

                # 保存
                pool_path = self.pool_dir / f'pool_pattern_{pattern_id}'
                np.save(pool_path / 'features.npy', pattern_features)
                np.save(pool_path / 'phones.npy', pattern_phones)
                np.save(pool_path / 'genders.npy', pattern_genders)

                # 构建 Phone Clusters
                if self.pool_config.build_phone_clusters:
                    phone_clusters = self._build_phone_clusters_for_pattern(
                        pattern_features, pattern_phones, pattern_genders
                    )
                    if phone_clusters:
                        torch.save(phone_clusters, pool_path / 'phone_clusters.pt')

                # 保存元数据
                meta = {
                    'n_frames': n_frames,
                    'pattern_id': pattern_id,
                    'feature_dim': self.feature_dim,
                }
                with open(pool_path / 'metadata.json', 'w') as mf:
                    json.dump(meta, mf, indent=2)

    def _load_features_by_indices(
        self,
        h5_dataset,
        indices: List[int],
        batch_size: int = 50000,
    ) -> np.ndarray:
        """批量加载特征"""
        n_samples = len(indices)
        features_list = []

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_features = h5_dataset[batch_indices].astype(np.float32)
            features_list.append(batch_features)

        return np.vstack(features_list)

    def _build_phone_clusters_for_pattern(
        self,
        features: np.ndarray,
        phones: np.ndarray,
        genders: np.ndarray,
    ) -> Dict:
        """为单个 Pattern 构建 Phone Clusters"""
        phone_clusters = {}
        n_clusters = self.pool_config.n_phone_clusters

        for phone_id in range(self.num_phones):
            for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                mask = (phones == phone_id) & (genders == gender_id)
                indices = np.where(mask)[0]

                if len(indices) < n_clusters:
                    continue

                # 采样
                if len(indices) > 50000:
                    indices = np.random.choice(indices, 50000, replace=False)

                phone_features = features[indices]

                # K-Means
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                kmeans.fit(phone_features)

                key = f"{phone_id}_{gender_name}"
                phone_clusters[key] = torch.from_numpy(kmeans.cluster_centers_).float()

            # 不区分性别的版本
            phone_indices = np.where(phones == phone_id)[0]
            if len(phone_indices) >= n_clusters:
                if len(phone_indices) > 100000:
                    phone_indices = np.random.choice(phone_indices, 100000, replace=False)

                phone_features = features[phone_indices]
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                kmeans.fit(phone_features)

                phone_clusters[str(phone_id)] = torch.from_numpy(kmeans.cluster_centers_).float()

        return phone_clusters

    def _save_pattern_metadata(
        self,
        pattern_assignments: Dict[str, int],
        pattern_centroids: np.ndarray,
    ):
        """保存 Pattern 元数据"""
        # Pattern 分配
        with open(self.pool_dir / 'pattern_assignments.json', 'w') as f:
            json.dump(pattern_assignments, f, indent=2)

        # Pattern 中心 (用于 TargetSelector)
        torch.save({
            'centroids': torch.from_numpy(pattern_centroids).float(),
            'n_patterns': self.pool_config.n_patterns,
        }, self.pool_dir / 'pattern_centroids.pt')

        logger.info(f"  Saved pattern metadata to {self.pool_dir}")

    def _build_legacy_pool(
        self,
        features_path: Path,
        index_map: Dict,
        metadata: Dict,
    ):
        """构建兼容旧格式的统一池"""
        # 创建软链接或复制必要文件
        logger.info("  Building legacy pool for backward compatibility...")

        # phones.npy 和 genders.npy
        np.save(self.pool_dir / 'phones.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'genders.npy', index_map['frame_to_gender'])

        # 创建 features.h5 软链接
        features_src = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        features_dst = self.pool_dir / 'features.h5'
        if not features_dst.exists() and features_src.exists():
            try:
                features_dst.symlink_to(features_src)
            except OSError:
                pass

        # 合并所有 Pattern 的 phone_clusters
        merged_clusters = {}
        for p in range(self.pool_config.n_patterns):
            cluster_path = self.pool_dir / f'pool_pattern_{p}' / 'phone_clusters.pt'
            if cluster_path.exists():
                clusters = torch.load(cluster_path, map_location='cpu')
                for key, value in clusters.items():
                    if key not in merged_clusters:
                        merged_clusters[key] = value

        if merged_clusters:
            torch.save(merged_clusters, self.pool_dir / 'phone_clusters.pt')

        # 元数据
        pool_meta = {
            'version': '3.0',
            'total_frames': index_map['total_frames'],
            'num_utterances': len(index_map['utterances']),
            'feature_dim': self.feature_dim,
            'n_patterns': self.pool_config.n_patterns,
            'phone_to_idx': self.phone_to_idx,
            'n_phone_clusters': self.pool_config.n_phone_clusters,
        }
        with open(self.pool_dir / 'metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        logger.info("  Legacy pool built successfully")


def run_pool_building_v3(config: Dict) -> Dict:
    """Step 6 入口 (v3.0)"""
    builder = TargetPoolBuilderV3(config)
    return builder.build()
