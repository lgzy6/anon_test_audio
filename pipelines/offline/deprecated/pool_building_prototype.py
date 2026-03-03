# pipelines/offline/pool_building_prototype.py
"""
Step 6: Speaker Prototype Pool Construction

研究导向的优化方案：
- 从 Frame-level Pool (55M) → Speaker Prototype Pool (~100K)
- 压缩比: 500x+
- 核心思想: 说话人身份编码在低维流形上，不需要帧级密度

理论依据:
1. 同一说话人的帧高度相似（冗余）
2. 匿名化需要的是说话人多样性，不是帧密度
3. 每个说话人用 K 个 prototype 即可覆盖其 timbre manifold
"""

import json
import numpy as np
import torch
import h5py
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
import logging
import pickle
import gc
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PrototypePoolConfig:
    """Speaker Prototype Pool 配置"""

    # Prototype 参数
    n_prototypes_per_speaker: int = 50      # 每个说话人的 prototype 数量
    n_prototypes_per_phone: int = 8         # 每个音素的 prototype 数量 (分层模式)
    use_phone_stratified: bool = True       # 是否按音素分层采样
    min_frames_per_speaker: int = 100       # 说话人最少帧数

    # FAISS 参数
    nlist: int = 256                        # IVF 聚类数 (pool 小，不需要太多)
    nprobe: int = 32                        # 搜索时探测数

    # 功能开关
    use_gpu: bool = True
    gpu_id: int = 0
    build_phone_indices: bool = True
    build_gender_indices: bool = True

    # Phone clusters (兼容 pknnvc)
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8


class SpeakerPrototypePoolBuilder:
    """
    Speaker Prototype Pool 构建器

    核心思想:
    - 每个说话人提取 K 个代表性 prototype
    - 可选: 按音素分层，确保覆盖不同发音
    - 最终 pool 规模: n_speakers × n_prototypes ≈ 100K

    相比 Frame-level Pool:
    - 规模: 55M → 100K (550x 压缩)
    - 构建时间: 6h → 10min
    - 搜索速度: 大幅提升
    """

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: Dict, pool_config: Optional[PrototypePoolConfig] = None):
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

        # GPU
        self.gpu_res = None
        if self.pool_config.use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_res = faiss.StandardGpuResources()

    def _load_pool_config(self) -> PrototypePoolConfig:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        prototype_cfg = cfg.get('prototype', {})

        return PrototypePoolConfig(
            n_prototypes_per_speaker=prototype_cfg.get('n_prototypes_per_speaker', 50),
            n_prototypes_per_phone=prototype_cfg.get('n_prototypes_per_phone', 8),
            use_phone_stratified=prototype_cfg.get('use_phone_stratified', True),
            min_frames_per_speaker=prototype_cfg.get('min_frames_per_speaker', 100),
            nlist=cfg.get('nlist', 256),
            nprobe=cfg.get('nprobe', 32),
            use_gpu=cfg.get('use_gpu', True),
            gpu_id=cfg.get('gpu_id', 0),
            build_phone_indices=cfg.get('build_phone_indices', True),
            build_gender_indices=cfg.get('build_gender_indices', True),
            build_phone_clusters=cfg.get('build_phone_clusters', True),
        )

    def build(self) -> Dict:
        """构建 Speaker Prototype Pool"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: Speaker Prototype Pool Construction")
        logger.info("=" * 70)
        logger.info(f"  Prototypes per speaker: {self.pool_config.n_prototypes_per_speaker}")
        logger.info(f"  Phone stratified: {self.pool_config.use_phone_stratified}")

        # 验证输入
        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 1. 构建说话人索引
        logger.info("\n[1/5] 构建说话人索引...")
        speaker_index = self._build_speaker_index(metadata)

        # 2. 加载 phone 预测 (可选)
        phone_predictions = None
        if self.pool_config.use_phone_stratified:
            logger.info("\n[2/5] 加载 Phone 预测...")
            phone_predictions = self._load_phone_predictions()

        # 3. 构建 Speaker Prototypes
        logger.info("\n[3/5] 构建 Speaker Prototypes...")
        prototype_data = self._build_speaker_prototypes(
            features_path, metadata, speaker_index, phone_predictions
        )

        # 4. 构建 FAISS 索引
        logger.info("\n[4/5] 构建 FAISS 索引...")
        global_index = self._build_faiss_index(prototype_data)

        # 5. 保存
        logger.info("\n[5/5] 保存 Prototype Pool...")
        self._save_pool(global_index, prototype_data, metadata)

        return {
            'pool_dir': str(self.pool_dir),
            'n_speakers': len(speaker_index),
            'n_prototypes': len(prototype_data['prototypes']),
            'compression_ratio': metadata['total_frames'] / len(prototype_data['prototypes']),
        }

    def _build_speaker_index(self, metadata: Dict) -> Dict[str, Dict]:
        """
        构建说话人索引

        Returns:
            {speaker_id: {'utterances': [...], 'gender': 'M/F', 'frame_ranges': [(s,e), ...]}}
        """
        speaker_index = defaultdict(lambda: {
            'utterances': [],
            'gender': 'M',
            'frame_ranges': [],
            'total_frames': 0
        })

        for utt in metadata['utterances']:
            spk_id = utt.get('speaker_id', utt.get('utt_id', '').split('-')[0])
            speaker_index[spk_id]['utterances'].append(utt)
            speaker_index[spk_id]['gender'] = utt.get('gender', 'M')
            speaker_index[spk_id]['frame_ranges'].append(
                (utt['h5_start_idx'], utt['h5_end_idx'])
            )
            speaker_index[spk_id]['total_frames'] += utt['h5_end_idx'] - utt['h5_start_idx']

        # 过滤帧数太少的说话人
        min_frames = self.pool_config.min_frames_per_speaker
        filtered = {
            spk: data for spk, data in speaker_index.items()
            if data['total_frames'] >= min_frames
        }

        logger.info(f"  Total speakers: {len(speaker_index)}")
        logger.info(f"  After filtering (>={min_frames} frames): {len(filtered)}")

        return dict(filtered)

    def _load_phone_predictions(self) -> Optional[Dict]:
        """加载 phone 预测"""
        phone_path = self.cache_dir / 'features' / 'cleaned' / 'phone_predictions.h5'

        if not phone_path.exists():
            logger.warning("  Phone predictions not found, using uniform sampling")
            return None

        predictions = {}
        with h5py.File(phone_path, 'r') as f:
            for utt_id in f.keys():
                predictions[utt_id] = f[utt_id][:]

        logger.info(f"  Loaded phone predictions: {len(predictions)} utterances")
        return predictions

    def _build_speaker_prototypes(
        self,
        features_path: Path,
        metadata: Dict,
        speaker_index: Dict,
        phone_predictions: Optional[Dict]
    ) -> Dict:
        """
        构建每个说话人的 prototypes

        两种模式:
        1. 均匀 K-Means: 对说话人所有帧聚类
        2. 音素分层: 对每个音素分别聚类，确保覆盖
        """
        all_prototypes = []
        prototype_speaker_ids = []
        prototype_genders = []
        prototype_phones = []  # 每个 prototype 的主要音素

        n_proto_per_spk = self.pool_config.n_prototypes_per_speaker

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for spk_id, spk_data in tqdm(speaker_index.items(), desc="Building prototypes"):
                # 收集该说话人的所有帧
                speaker_frames = []
                speaker_phones = []

                for utt, (start, end) in zip(spk_data['utterances'], spk_data['frame_ranges']):
                    utt_frames = feats[start:end]
                    speaker_frames.append(utt_frames)

                    # 获取 phone 标签
                    if phone_predictions and utt['utt_id'] in phone_predictions:
                        phones = phone_predictions[utt['utt_id']]
                        if len(phones) >= len(utt_frames):
                            phones = phones[:len(utt_frames)]
                        else:
                            phones = np.pad(phones, (0, len(utt_frames) - len(phones)),
                                          constant_values=self.phone_to_idx['SIL'])
                        speaker_phones.append(phones)
                    else:
                        speaker_phones.append(np.full(len(utt_frames), -1))

                speaker_frames = np.vstack(speaker_frames).astype(np.float32)
                speaker_phones = np.concatenate(speaker_phones)

                # 归一化
                faiss.normalize_L2(speaker_frames)

                # 提取 prototypes
                if self.pool_config.use_phone_stratified and phone_predictions:
                    prototypes, proto_phones = self._extract_phone_stratified_prototypes(
                        speaker_frames, speaker_phones, n_proto_per_spk
                    )
                else:
                    prototypes, proto_phones = self._extract_kmeans_prototypes(
                        speaker_frames, n_proto_per_spk
                    )

                # 记录
                all_prototypes.append(prototypes)
                prototype_speaker_ids.extend([spk_id] * len(prototypes))
                prototype_genders.extend([spk_data['gender']] * len(prototypes))
                prototype_phones.extend(proto_phones)

        all_prototypes = np.vstack(all_prototypes).astype(np.float32)

        logger.info(f"  Total prototypes: {len(all_prototypes):,}")
        logger.info(f"  Compression ratio: {metadata['total_frames'] / len(all_prototypes):.1f}x")

        return {
            'prototypes': all_prototypes,
            'speaker_ids': prototype_speaker_ids,
            'genders': prototype_genders,
            'phones': prototype_phones,
        }

    def _extract_kmeans_prototypes(
        self,
        frames: np.ndarray,
        n_prototypes: int
    ) -> Tuple[np.ndarray, List[int]]:
        """使用 K-Means 提取 prototypes"""
        n_samples = len(frames)

        if n_samples <= n_prototypes:
            # 样本太少，直接返回
            return frames, [-1] * len(frames)

        # Mini-batch K-Means (更快)
        kmeans = MiniBatchKMeans(
            n_clusters=n_prototypes,
            batch_size=min(1000, n_samples),
            n_init=3,
            random_state=42
        )
        kmeans.fit(frames)

        prototypes = kmeans.cluster_centers_.astype(np.float32)
        faiss.normalize_L2(prototypes)

        return prototypes, [-1] * n_prototypes

    def _extract_phone_stratified_prototypes(
        self,
        frames: np.ndarray,
        phones: np.ndarray,
        n_prototypes: int
    ) -> Tuple[np.ndarray, List[int]]:
        """
        按音素分层提取 prototypes

        确保每个音素都有代表性样本，避免被高频音素主导
        """
        prototypes = []
        proto_phones = []

        # 统计每个音素的帧数
        unique_phones, phone_counts = np.unique(phones[phones >= 0], return_counts=True)

        if len(unique_phones) == 0:
            # 没有 phone 标签，回退到 kmeans
            return self._extract_kmeans_prototypes(frames, n_prototypes)

        # 按比例分配 prototype 数量
        total_frames = phone_counts.sum()
        proto_per_phone = {}
        remaining = n_prototypes

        for phone, count in zip(unique_phones, phone_counts):
            # 至少 1 个，按比例分配
            n = max(1, int(n_prototypes * count / total_frames))
            proto_per_phone[phone] = min(n, remaining)
            remaining -= proto_per_phone[phone]

        # 剩余的分给最大的几个
        if remaining > 0:
            sorted_phones = sorted(unique_phones, key=lambda p: -phone_counts[unique_phones == p][0])
            for phone in sorted_phones:
                if remaining <= 0:
                    break
                proto_per_phone[phone] += 1
                remaining -= 1

        # 对每个音素提取 prototypes
        for phone in unique_phones:
            n_proto = proto_per_phone.get(phone, 0)
            if n_proto == 0:
                continue

            phone_mask = phones == phone
            phone_frames = frames[phone_mask]

            if len(phone_frames) <= n_proto:
                # 样本太少，直接使用
                prototypes.append(phone_frames)
                proto_phones.extend([int(phone)] * len(phone_frames))
            else:
                # K-Means 聚类
                kmeans = MiniBatchKMeans(
                    n_clusters=n_proto,
                    batch_size=min(500, len(phone_frames)),
                    n_init=3,
                    random_state=42
                )
                kmeans.fit(phone_frames)
                phone_protos = kmeans.cluster_centers_.astype(np.float32)
                faiss.normalize_L2(phone_protos)
                prototypes.append(phone_protos)
                proto_phones.extend([int(phone)] * n_proto)

        if len(prototypes) == 0:
            return self._extract_kmeans_prototypes(frames, n_prototypes)

        prototypes = np.vstack(prototypes).astype(np.float32)
        return prototypes, proto_phones

    def _build_faiss_index(self, prototype_data: Dict) -> faiss.Index:
        """构建 FAISS 索引"""
        prototypes = prototype_data['prototypes']
        n_protos = len(prototypes)

        # 对于小规模 pool，使用简单的 IVF 或 Flat
        if n_protos < 10000:
            logger.info(f"  Using IndexFlatIP (pool size < 10K)")
            index = faiss.IndexFlatIP(self.feature_dim)
        else:
            nlist = min(self.pool_config.nlist, int(np.sqrt(n_protos)))
            logger.info(f"  Using IndexIVFFlat: nlist={nlist}")

            quantizer = faiss.IndexFlatIP(self.feature_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.feature_dim, nlist,
                faiss.METRIC_INNER_PRODUCT
            )

            # GPU 训练
            if self.gpu_res:
                gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_res, self.pool_config.gpu_id, index
                )
                gpu_index.train(prototypes)
                index = faiss.index_gpu_to_cpu(gpu_index)
            else:
                index.train(prototypes)

            index.nprobe = self.pool_config.nprobe

        # 添加向量
        index.add(prototypes)
        logger.info(f"  Index built: {index.ntotal:,} vectors")

        return index

    def _save_pool(self, global_index: faiss.Index, prototype_data: Dict, metadata: Dict):
        """保存 Prototype Pool"""

        # 1. FAISS 索引
        faiss.write_index(global_index, str(self.pool_dir / 'faiss.index'))

        # 兼容命名
        faiss_legacy = self.pool_dir / 'faiss_trained.index'
        if not faiss_legacy.exists():
            try:
                faiss_legacy.symlink_to('faiss.index')
            except OSError:
                faiss.write_index(global_index, str(faiss_legacy))

        # 2. Prototype 元数据
        np.save(self.pool_dir / 'prototypes.npy', prototype_data['prototypes'])

        # Speaker IDs
        speaker_ids = prototype_data['speaker_ids']
        np.save(self.pool_dir / 'prototype_speaker_ids.npy',
                np.array(speaker_ids, dtype=object))

        # Gender (转换为数值)
        genders = [0 if g == 'M' else 1 for g in prototype_data['genders']]
        np.save(self.pool_dir / 'genders.npy', np.array(genders, dtype=np.int8))

        # Phones
        phones = np.array(prototype_data['phones'], dtype=np.int32)
        np.save(self.pool_dir / 'phones.npy', phones)

        # 3. 兼容性文件 (用于现有 retriever)
        # symbols 使用占位符
        np.save(self.pool_dir / 'symbols.npy',
                np.zeros(len(prototype_data['prototypes']), dtype=np.int32))

        # 4. 元数据
        pool_meta = {
            'pool_type': 'speaker_prototype',
            'n_prototypes': len(prototype_data['prototypes']),
            'n_speakers': len(set(prototype_data['speaker_ids'])),
            'n_prototypes_per_speaker': self.pool_config.n_prototypes_per_speaker,
            'use_phone_stratified': self.pool_config.use_phone_stratified,
            'feature_dim': self.feature_dim,
            'original_total_frames': metadata['total_frames'],
            'compression_ratio': metadata['total_frames'] / len(prototype_data['prototypes']),
            'phone_to_idx': self.phone_to_idx,
        }

        with open(self.pool_dir / 'metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        with open(self.pool_dir / 'pool_metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        # 5. 符号映射
        with open(self.pool_dir / 'symbol_index.pkl', 'wb') as f:
            pickle.dump(self.phone_to_idx, f)

        # 6. 特征文件链接 (保持兼容性)
        features_src = self.pool_dir / 'prototypes.npy'
        features_dst = self.pool_dir / 'features.npy'
        if not features_dst.exists():
            try:
                features_dst.symlink_to('prototypes.npy')
            except OSError:
                pass

        logger.info(f"\n  ✓ Saved to: {self.pool_dir}")
        logger.info(f"  Pool type: Speaker Prototype")
        logger.info(f"  Compression: {pool_meta['compression_ratio']:.1f}x")


def run_pool_building_prototype(config: Dict) -> Dict:
    """Step 6 入口 (Prototype 版本)"""
    builder = SpeakerPrototypePoolBuilder(config)
    return builder.build()
