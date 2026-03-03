# pipelines/offline/pool_building_sampled.py
"""
Step 6: Target Pool Construction (Sampled Version)

从完整数据集中均匀采样构建索引:
- 保留所有框架功能 (FAISS索引, Phone/Gender子索引, SAMM符号)
- 通过采样控制内存使用
- 保持 phone/gender/speaker 分布均匀

适用场景: 120GB 内存 + 5500万帧数据
目标: 采样 ~1500万帧，内存峰值 ~60GB
"""

import json
import numpy as np
import torch
import h5py
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import logging
import pickle
import gc

logger = logging.getLogger(__name__)


@dataclass
class SampledPoolConfig:
    """采样池配置"""
    # 采样参数
    target_frames: int = 15000000  # 目标采样帧数 (1500万)
    sample_strategy: str = "stratified"  # stratified | uniform | per_speaker

    # 每个phone/gender的最小/最大样本数
    min_samples_per_phone: int = 1000
    max_samples_per_phone: int = 1000000

    # FAISS 参数
    nlist: int = 1024
    nprobe: int = 64
    train_size: int = 200000

    # 批处理
    io_batch_size: int = 50000
    gpu_add_batch: int = 100000

    # GPU
    use_gpu: bool = True
    gpu_id: int = 0

    # 子索引
    build_phone_indices: bool = True
    build_gender_indices: bool = True
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8


class SampledTargetPoolBuilder:
    """
    采样式 Target Pool 构建器

    核心思想:
    1. 先扫描数据统计 phone/gender/speaker 分布
    2. 按分层采样策略选择帧
    3. 只对采样帧构建索引
    4. 保存采样到原始数据的映射
    """

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: Dict, pool_config: Optional[SampledPoolConfig] = None):
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

        # GPU 资源
        self.gpu_res = None
        if self.pool_config.use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_res = faiss.StandardGpuResources()
            self.gpu_res.setTempMemory(1024 * 1024 * 1024)

    def _load_pool_config(self) -> SampledPoolConfig:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        knn_cfg = self.config.get('knn_vc', {})

        return SampledPoolConfig(
            target_frames=cfg.get('target_frames', 15000000),
            sample_strategy=cfg.get('sample_strategy', 'stratified'),
            nlist=cfg.get('nlist', 1024),
            nprobe=cfg.get('nprobe', 64),
            train_size=cfg.get('train_size', 200000),
            io_batch_size=cfg.get('io_batch_size', 50000),
            gpu_add_batch=cfg.get('gpu_add_batch', 100000),
            use_gpu=cfg.get('use_gpu', True),
            gpu_id=cfg.get('gpu_id', 0),
            build_phone_indices=cfg.get('build_phone_indices', True),
            build_gender_indices=cfg.get('build_gender_indices', True),
            build_phone_clusters=cfg.get('build_phone_clusters', True),
            n_phone_clusters=knn_cfg.get('num_clusters', 8),
        )

    def build(self) -> Dict:
        """构建采样 Target Pool"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Target Pool Construction (Sampled Version)")
        logger.info("=" * 60)

        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        total_frames = metadata['total_frames']
        target_frames = self.pool_config.target_frames

        logger.info(f"  Original frames: {total_frames:,}")
        logger.info(f"  Target frames: {target_frames:,}")
        logger.info(f"  Sample ratio: {target_frames/total_frames:.1%}")

        # 1. 构建帧级映射
        logger.info("\n[1/7] 构建帧级映射...")
        frame_maps = self._build_frame_maps(features_path, metadata)

        # 2. 分层采样
        logger.info("\n[2/7] 分层采样...")
        sampled_indices = self._stratified_sample(frame_maps, target_frames)
        logger.info(f"  Sampled {len(sampled_indices):,} frames")

        # 3. 为采样帧分配 SAMM 符号
        logger.info("\n[3/7] 分配 SAMM 符号...")
        sampled_symbols = self._assign_symbols(features_path, sampled_indices)

        # 4. 构建全局 FAISS 索引
        logger.info("\n[4/7] 构建全局 FAISS 索引...")
        self._build_global_index(features_path, sampled_indices)
        gc.collect()

        # 5. 构建 Phone 子索引
        if self.pool_config.build_phone_indices:
            logger.info("\n[5/7] 构建 Phone 子索引...")
            self._build_phone_indices(features_path, sampled_indices, frame_maps)
            gc.collect()

        # 6. 构建 Gender 子索引
        if self.pool_config.build_gender_indices:
            logger.info("\n[6/7] 构建 Gender 子索引...")
            self._build_gender_indices(features_path, sampled_indices, frame_maps)
            gc.collect()

        # 7. 保存元数据和映射
        logger.info("\n[7/7] 保存元数据...")
        self._save_all(
            sampled_indices, sampled_symbols, frame_maps, metadata
        )

        return {
            'pool_dir': str(self.pool_dir),
            'original_frames': total_frames,
            'sampled_frames': len(sampled_indices),
            'sample_ratio': len(sampled_indices) / total_frames,
        }

    def _build_frame_maps(self, features_path: Path, metadata: Dict) -> Dict:
        """构建帧级映射（全量，用于采样决策）"""
        utterances = metadata['utterances']
        total_frames = metadata['total_frames']

        frame_to_phone = np.full(total_frames, self.phone_to_idx['SIL'], dtype=np.int32)
        frame_to_gender = np.zeros(total_frames, dtype=np.int8)
        frame_to_utt = np.zeros(total_frames, dtype=np.int32)
        frame_to_speaker = np.zeros(total_frames, dtype=np.int32)

        # 加载 phone 预测
        phone_predictions = self._load_phone_predictions()

        # 构建 speaker ID 映射
        speaker_to_idx = {}
        speaker_idx = 0

        for utt_idx, utt in enumerate(tqdm(utterances, desc="Building frame maps")):
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            n_frames = e - s

            frame_to_utt[s:e] = utt_idx

            # Gender - 注意：metadata中可能是小写 'm'/'f'
            gender = utt.get('gender', 'M').upper()  # 转为大写
            frame_to_gender[s:e] = 0 if gender == 'M' else 1

            # Speaker
            spk_id = utt.get('speaker_id', utt['utt_id'].split('-')[0])
            if spk_id not in speaker_to_idx:
                speaker_to_idx[spk_id] = speaker_idx
                speaker_idx += 1
            frame_to_speaker[s:e] = speaker_to_idx[spk_id]

            # Phone
            utt_id = utt['utt_id']
            if phone_predictions and utt_id in phone_predictions:
                phones = phone_predictions[utt_id]
                if len(phones) >= n_frames:
                    frame_to_phone[s:e] = phones[:n_frames]
                else:
                    frame_to_phone[s:s+len(phones)] = phones

        # 统计
        phone_counts = np.bincount(frame_to_phone, minlength=self.num_phones)
        gender_counts = np.bincount(frame_to_gender, minlength=2)
        n_speakers = len(speaker_to_idx)

        logger.info(f"  Total frames: {total_frames:,}")
        logger.info(f"  Speakers: {n_speakers}")
        logger.info(f"  Gender: M={gender_counts[0]:,}, F={gender_counts[1]:,}")

        return {
            'total_frames': total_frames,
            'utterances': utterances,
            'frame_to_phone': frame_to_phone,
            'frame_to_gender': frame_to_gender,
            'frame_to_utt': frame_to_utt,
            'frame_to_speaker': frame_to_speaker,
            'phone_counts': phone_counts,
            'gender_counts': gender_counts,
            'speaker_to_idx': speaker_to_idx,
            'n_speakers': n_speakers,
        }

    def _load_phone_predictions(self) -> Optional[Dict]:
        """加载 phone 预测"""
        path = self.cache_dir / 'features' / 'cleaned' / 'phone_predictions.h5'
        if path.exists():
            predictions = {}
            with h5py.File(path, 'r') as f:
                for utt_id in f.keys():
                    phones = f[utt_id][:]
                    if len(phones) > 0:
                        predictions[utt_id] = phones
            logger.info(f"  Loaded phone predictions: {len(predictions)} utterances")
            return predictions
        return None

    def _stratified_sample(self, frame_maps: Dict, target_frames: int) -> np.ndarray:
        """
        分层采样：保持 phone × gender 分布

        策略：
        1. 计算每个 (phone, gender) 组合的目标采样数
        2. 在每个组合内均匀采样
        3. 确保稀有组合不被忽略
        """
        frame_to_phone = frame_maps['frame_to_phone']
        frame_to_gender = frame_maps['frame_to_gender']
        total_frames = frame_maps['total_frames']

        # 计算每个 (phone, gender) 组合的帧数
        strata_counts = {}
        for phone_id in range(self.num_phones):
            for gender_id in [0, 1]:
                mask = (frame_to_phone == phone_id) & (frame_to_gender == gender_id)
                count = np.sum(mask)
                if count > 0:
                    strata_counts[(phone_id, gender_id)] = count

        logger.info(f"  Found {len(strata_counts)} strata (phone × gender combinations)")

        # 计算采样比例
        overall_ratio = target_frames / total_frames

        # 分层采样
        sampled_indices = []
        scan_batch = 2000000  # 200万帧一批扫描

        for (phone_id, gender_id), count in tqdm(
            strata_counts.items(), desc="Stratified sampling"
        ):
            # 目标采样数：按比例，但确保最小采样数
            target_count = max(
                self.pool_config.min_samples_per_phone,
                min(
                    int(count * overall_ratio * 1.1),  # 稍微多采一点
                    self.pool_config.max_samples_per_phone,
                    count
                )
            )

            # 分块扫描收集该 stratum 的索引
            stratum_indices = []
            for scan_start in range(0, total_frames, scan_batch):
                scan_end = min(scan_start + scan_batch, total_frames)
                chunk_phone = frame_to_phone[scan_start:scan_end]
                chunk_gender = frame_to_gender[scan_start:scan_end]
                mask = (chunk_phone == phone_id) & (chunk_gender == gender_id)
                chunk_indices = np.where(mask)[0] + scan_start
                stratum_indices.append(chunk_indices)

            stratum_indices = np.concatenate(stratum_indices)

            # 采样
            if len(stratum_indices) > target_count:
                sampled = np.random.choice(stratum_indices, target_count, replace=False)
            else:
                sampled = stratum_indices

            sampled_indices.append(sampled)

        # 合并并排序
        sampled_indices = np.concatenate(sampled_indices)
        sampled_indices = np.unique(sampled_indices)  # 去重并排序

        # 如果超出目标，再随机裁剪
        if len(sampled_indices) > target_frames:
            sampled_indices = np.random.choice(
                sampled_indices, target_frames, replace=False
            )

        # 最终排序（HDF5访问需要）
        sampled_indices = np.sort(sampled_indices)

        return sampled_indices

    def _assign_symbols(self, features_path: Path, sampled_indices: np.ndarray) -> np.ndarray:
        """为采样帧分配 SAMM 符号"""
        codebook_path = self.checkpoint_dir / 'codebook.pt'

        n_samples = len(sampled_indices)
        symbols = np.zeros(n_samples, dtype=np.int32)

        if not codebook_path.exists():
            logger.warning("  No codebook found, using placeholder symbols")
            return symbols

        try:
            codebook_data = torch.load(codebook_path, map_location='cpu')
            if isinstance(codebook_data, dict) and 'codebook' in codebook_data:
                codebook = codebook_data['codebook']
            else:
                codebook = codebook_data
            codebook = torch.from_numpy(codebook).float() if isinstance(codebook, np.ndarray) else codebook

            device = torch.device(f'cuda:{self.pool_config.gpu_id}' if torch.cuda.is_available() else 'cpu')
            codebook = codebook.to(device)

            batch_size = 50000

            # HDF5 需要排序的索引，创建排序映射
            sort_order = np.argsort(sampled_indices)
            sorted_indices = sampled_indices[sort_order]
            inverse_order = np.argsort(sort_order)  # 用于恢复原始顺序

            sorted_symbols = np.zeros(n_samples, dtype=np.int32)

            with h5py.File(features_path, 'r') as f:
                feats = f['features']

                for i in tqdm(range(0, n_samples, batch_size), desc="Assigning symbols"):
                    batch_sorted_idx = sorted_indices[i:i+batch_size]
                    batch_feats = torch.from_numpy(feats[batch_sorted_idx].astype(np.float32)).to(device)
                    dist = torch.cdist(batch_feats, codebook)
                    batch_symbols = dist.argmin(dim=-1).cpu().numpy()
                    sorted_symbols[i:i+batch_size] = batch_symbols

            # 恢复原始顺序
            symbols = sorted_symbols[inverse_order]

            logger.info(f"  Assigned symbols to {n_samples:,} frames")
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"  Failed to assign symbols: {e}")

        return symbols

    def _build_global_index(self, features_path: Path, sampled_indices: np.ndarray):
        """构建全局 FAISS 索引"""
        import time

        n_samples = len(sampled_indices)
        nlist = min(self.pool_config.nlist, int(np.sqrt(n_samples)))

        logger.info(f"  Building IVF: nlist={nlist}, samples={n_samples:,}")

        quantizer = faiss.IndexFlatIP(self.feature_dim)
        index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)

        # 训练 - sampled_indices 已经是排序的
        train_size = min(self.pool_config.train_size, n_samples)
        train_indices = sampled_indices[:train_size]  # 取前 train_size 个（已排序）

        logger.info(f"  Loading training data ({len(train_indices):,} vectors)...")
        with h5py.File(features_path, 'r') as f:
            train_data = f['features'][train_indices].astype(np.float32)
        faiss.normalize_L2(train_data)

        logger.info(f"  Training IVF index...")
        start_time = time.time()

        if self.gpu_res:
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.pool_config.gpu_id, index)
            gpu_index.train(train_data)
            index = faiss.index_gpu_to_cpu(gpu_index)
            del gpu_index
            torch.cuda.empty_cache()
        else:
            index.train(train_data)

        del train_data
        gc.collect()

        train_time = time.time() - start_time
        logger.info(f"  Training completed in {train_time:.1f}s")

        # 添加向量 - sampled_indices 已排序，可直接用于 HDF5
        batch_size = self.pool_config.gpu_add_batch
        logger.info(f"  Adding {n_samples:,} vectors (batch={batch_size:,})...")

        with h5py.File(features_path, 'r') as f:
            feats = f['features']
            for i in tqdm(range(0, n_samples, batch_size), desc="Adding vectors"):
                batch_idx = sampled_indices[i:i+batch_size]
                batch = feats[batch_idx].astype(np.float32)
                faiss.normalize_L2(batch)
                index.add(batch)
                del batch

        index.nprobe = self.pool_config.nprobe

        # 保存
        faiss.write_index(index, str(self.pool_dir / 'faiss.index'))
        logger.info(f"  Saved global index: {index.ntotal:,} vectors")

        del index
        gc.collect()

    def _build_phone_indices(
        self,
        features_path: Path,
        sampled_indices: np.ndarray,
        frame_maps: Dict
    ):
        """构建 phone 子索引"""
        frame_to_phone = frame_maps['frame_to_phone']

        # 获取采样帧的phone标签
        sampled_phones = frame_to_phone[sampled_indices]

        phone_dir = self.pool_dir / 'phone_indices'
        phone_dir.mkdir(exist_ok=True)
        phone_meta = {}

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for phone_id in tqdm(range(self.num_phones), desc="Phone indices"):
                # 找到该phone在采样集中的位置
                phone_mask = sampled_phones == phone_id
                phone_positions = np.where(phone_mask)[0]  # 在采样集中的位置
                n_frames = len(phone_positions)

                if n_frames < 100:
                    continue

                phone_name = self.idx_to_phone[phone_id]

                # 构建索引
                if n_frames < 1000:
                    sub_index = faiss.IndexFlatIP(self.feature_dim)
                else:
                    sub_nlist = min(256, int(np.sqrt(n_frames)))
                    quantizer = faiss.IndexFlatIP(self.feature_dim)
                    sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                                   sub_nlist, faiss.METRIC_INNER_PRODUCT)
                    # 训练 - phone_positions 是顺序的，sampled_indices已排序，所以结果也是排序的
                    train_size = min(30000, n_frames)
                    train_positions = phone_positions[:train_size]
                    train_original_idx = sampled_indices[train_positions]
                    # 确保排序（虽然应该已经是了）
                    train_original_idx = np.sort(train_original_idx)
                    train_data = feats[train_original_idx].astype(np.float32)
                    faiss.normalize_L2(train_data)
                    sub_index.train(train_data)
                    del train_data
                    sub_index.nprobe = min(32, sub_nlist // 4)

                # 添加 - 需要确保索引排序
                add_batch = 20000
                for i in range(0, n_frames, add_batch):
                    batch_positions = phone_positions[i:i+add_batch]
                    batch_original_idx = sampled_indices[batch_positions]
                    # HDF5 需要排序的索引
                    sort_idx = np.argsort(batch_original_idx)
                    sorted_batch_idx = batch_original_idx[sort_idx]
                    batch_features = feats[sorted_batch_idx].astype(np.float32)
                    # 恢复原始顺序
                    batch_features = batch_features[np.argsort(sort_idx)]
                    faiss.normalize_L2(batch_features)
                    sub_index.add(batch_features)
                    del batch_features

                # 保存
                faiss.write_index(sub_index, str(phone_dir / f'{phone_name}.index'))
                # 保存在采样集中的位置（用于检索后映射回原始索引）
                np.save(phone_dir / f'{phone_name}_positions.npy', phone_positions)
                phone_meta[phone_name] = {'phone_id': phone_id, 'count': n_frames}

                del sub_index
                gc.collect()

        with open(phone_dir / 'metadata.json', 'w') as f:
            json.dump(phone_meta, f, indent=2)

        logger.info(f"  Built {len(phone_meta)} phone indices")

    def _build_gender_indices(
        self,
        features_path: Path,
        sampled_indices: np.ndarray,
        frame_maps: Dict
    ):
        """构建 gender 子索引"""
        frame_to_gender = frame_maps['frame_to_gender']
        sampled_genders = frame_to_gender[sampled_indices]

        gender_dir = self.pool_dir / 'gender_indices'
        gender_dir.mkdir(exist_ok=True)

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                gender_mask = sampled_genders == gender_id
                gender_positions = np.where(gender_mask)[0]
                n_frames = len(gender_positions)

                if n_frames < 100:
                    continue

                logger.info(f"  Building {gender_name}: {n_frames:,} frames")

                nlist = min(512, int(np.sqrt(n_frames)))
                quantizer = faiss.IndexFlatIP(self.feature_dim)
                sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                               nlist, faiss.METRIC_INNER_PRODUCT)

                # 训练 - 需要排序索引
                train_size = min(50000, n_frames)
                train_positions = gender_positions[:train_size]
                train_original_idx = sampled_indices[train_positions]
                train_original_idx = np.sort(train_original_idx)  # 确保排序
                train_data = feats[train_original_idx].astype(np.float32)
                faiss.normalize_L2(train_data)
                sub_index.train(train_data)
                del train_data
                gc.collect()

                # 添加 - 需要排序索引
                add_batch = 50000
                for i in tqdm(range(0, n_frames, add_batch), desc=f"    {gender_name}", leave=False):
                    batch_positions = gender_positions[i:i+add_batch]
                    batch_original_idx = sampled_indices[batch_positions]
                    # HDF5 需要排序的索引
                    sort_idx = np.argsort(batch_original_idx)
                    sorted_batch_idx = batch_original_idx[sort_idx]
                    batch_data = feats[sorted_batch_idx].astype(np.float32)
                    # 恢复原始顺序
                    batch_data = batch_data[np.argsort(sort_idx)]
                    faiss.normalize_L2(batch_data)
                    sub_index.add(batch_data)
                    del batch_data

                sub_index.nprobe = min(32, nlist // 4)

                # 保存
                faiss.write_index(sub_index, str(gender_dir / f'{gender_name}.index'))
                np.save(gender_dir / f'{gender_name}_positions.npy', gender_positions)

                del sub_index
                gc.collect()

        logger.info(f"  Built gender indices")

    def _save_all(
        self,
        sampled_indices: np.ndarray,
        sampled_symbols: np.ndarray,
        frame_maps: Dict,
        metadata: Dict
    ):
        """保存所有数据"""
        # 1. 采样索引映射 (关键！用于检索后映射回原始特征)
        np.save(self.pool_dir / 'sampled_indices.npy', sampled_indices)

        # 2. 采样帧的属性
        n_samples = len(sampled_indices)
        sampled_phones = frame_maps['frame_to_phone'][sampled_indices]
        sampled_genders = frame_maps['frame_to_gender'][sampled_indices]
        sampled_utts = frame_maps['frame_to_utt'][sampled_indices]

        np.save(self.pool_dir / 'phones.npy', sampled_phones)
        np.save(self.pool_dir / 'genders.npy', sampled_genders)
        np.save(self.pool_dir / 'symbols.npy', sampled_symbols)
        np.save(self.pool_dir / 'frame_to_utt.npy', sampled_utts)
        np.save(self.pool_dir / 'frame_to_phone.npy', sampled_phones)
        np.save(self.pool_dir / 'frame_to_gender.npy', sampled_genders)
        np.save(self.pool_dir / 'frame_to_symbol.npy', sampled_symbols)

        # 3. Utterance 信息
        with open(self.pool_dir / 'utterances.json', 'w') as f:
            json.dump(frame_maps['utterances'], f)

        utt_ids = [utt['utt_id'] for utt in frame_maps['utterances']]
        np.save(self.pool_dir / 'utt_ids.npy', np.array(utt_ids, dtype=object))

        # 4. Phone 映射
        with open(self.pool_dir / 'symbol_index.pkl', 'wb') as f:
            pickle.dump(self.phone_to_idx, f)

        # 5. 元数据
        pool_meta = {
            'format': 'sampled',
            'original_frames': frame_maps['total_frames'],
            'sampled_frames': n_samples,
            'sample_ratio': n_samples / frame_maps['total_frames'],
            'num_utterances': len(frame_maps['utterances']),
            'num_speakers': frame_maps['n_speakers'],
            'feature_dim': self.feature_dim,
            'phone_to_idx': self.phone_to_idx,
            'n_phone_clusters': self.pool_config.n_phone_clusters,
        }
        with open(self.pool_dir / 'metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        with open(self.pool_dir / 'pool_metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        # 6. 特征软链接
        features_src = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        features_dst = self.pool_dir / 'features.h5'
        if not features_dst.exists() and features_src.exists():
            try:
                features_dst.symlink_to(features_src)
            except OSError:
                pass

        logger.info(f"\n  ✓ Saved to: {self.pool_dir}")
        logger.info(f"  Original: {frame_maps['total_frames']:,} frames")
        logger.info(f"  Sampled: {n_samples:,} frames ({n_samples/frame_maps['total_frames']:.1%})")


def run_pool_building_sampled(config: Dict) -> Dict:
    """Step 6 入口 (Sampled Version)"""
    builder = SampledTargetPoolBuilder(config)
    return builder.build()
