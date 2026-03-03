# pipelines/offline/pool_building_low_mem.py
"""
Step 6: Target Pool Construction (Low Memory Version)
专门为 120GB 内存环境优化
"""

import json
import numpy as np
import torch
import h5py
import faiss
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import logging
import pickle
import gc

logger = logging.getLogger(__name__)


@dataclass
class LowMemPoolConfig:
    """低内存 Target Pool 配置"""
    # FAISS 参数 - 减少 nlist 降低训练内存
    nlist: int = 1024  # 从4096降低到1024
    nprobe: int = 64
    train_size: int = 200000  # 从50万降低到20万

    # 批处理参数 - 关键优化
    io_batch_size: int = 50000  # 从20万降低到5万
    gpu_add_batch: int = 100000  # 从50万降低到10万

    # 子索引限制 - 防止单个索引消耗过多内存
    max_samples_per_phone: int = 500000  # 每个phone最多50万样本
    max_samples_per_gender: int = 5000000  # 每个gender最多500万样本

    # 功能开关
    use_gpu: bool = True
    gpu_id: int = 0
    build_phone_indices: bool = True
    build_gender_indices: bool = True
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8

    # 内存优化
    save_immediately: bool = True
    skip_large_phones: bool = False  # 跳过超大phone (>500万帧)


class LowMemTargetPoolBuilder:
    """
    低内存 Target Pool 构建器

    核心优化策略:
    1. 不使用 np.where 一次性获取所有索引，改用分块扫描
    2. 使用 MiniBatchKMeans 替代 KMeans
    3. 所有子索引边构建边保存
    4. 对大型 phone 使用采样而非完整构建
    """

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: Dict, pool_config: Optional[LowMemPoolConfig] = None):
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
            self.gpu_res.setTempMemory(1024 * 1024 * 1024)  # 1GB GPU临时内存

    def _load_pool_config(self) -> LowMemPoolConfig:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        knn_cfg = self.config.get('knn_vc', {})

        return LowMemPoolConfig(
            nlist=cfg.get('nlist', 1024),
            nprobe=cfg.get('nprobe', 64),
            train_size=cfg.get('train_size', 200000),
            io_batch_size=cfg.get('io_batch_size', 50000),
            gpu_add_batch=cfg.get('gpu_add_batch', 100000),
            max_samples_per_phone=cfg.get('max_samples_per_phone', 500000),
            max_samples_per_gender=cfg.get('max_samples_per_gender', 5000000),
            use_gpu=cfg.get('use_gpu', True),
            gpu_id=cfg.get('gpu_id', 0),
            build_phone_indices=cfg.get('build_phone_indices', True),
            build_gender_indices=cfg.get('build_gender_indices', True),
            build_phone_clusters=cfg.get('build_phone_clusters', True),
            n_phone_clusters=knn_cfg.get('num_clusters', 8),
        )

    def build(self) -> Dict:
        """构建完整的 Target Pool"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Target Pool Construction (Low Memory Version)")
        logger.info("=" * 60)
        logger.info(f"  Max samples per phone: {self.pool_config.max_samples_per_phone:,}")
        logger.info(f"  IO batch size: {self.pool_config.io_batch_size:,}")

        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        total_frames = metadata['total_frames']
        logger.info(f"  Total frames: {total_frames:,}")

        # 1. 构建轻量级索引映射 (只保存必要的数组)
        logger.info("\n[1/6] 构建索引映射 (流式处理)...")
        index_map = self._build_index_map_streaming(features_path, metadata)

        # 1.5. 分配 SAMM 符号
        logger.info("\n[1.5/6] 分配 SAMM 符号...")
        self._assign_symbols_streaming(features_path, index_map)

        # 2. 构建全局 FAISS 索引
        logger.info("\n[2/6] 构建全局 FAISS 索引...")
        self._build_and_save_global_index(features_path, index_map)
        gc.collect()

        # 3. 构建 Phone 子索引 (流式处理)
        if self.pool_config.build_phone_indices:
            logger.info("\n[3/6] 构建 Phone 子索引 (流式处理)...")
            self._build_phone_indices_streaming(features_path, index_map)
            gc.collect()

        # 4. 构建 Gender 子索引 (流式处理)
        if self.pool_config.build_gender_indices:
            logger.info("\n[4/6] 构建 Gender 子索引 (流式处理)...")
            self._build_gender_indices_streaming(features_path, index_map)
            gc.collect()

        # 5. 构建 Phone Clusters (采样方式)
        if self.pool_config.build_phone_clusters:
            logger.info("\n[5/6] 构建 Phone Clusters (采样方式)...")
            self._build_phone_clusters_sampled(features_path, index_map)
            gc.collect()

        # 6. 保存元数据
        logger.info("\n[6/6] 保存元数据...")
        self._save_metadata(index_map, metadata)

        return {
            'pool_dir': str(self.pool_dir),
            'total_frames': total_frames,
        }

    def _build_index_map_streaming(self, features_path: Path, metadata: Dict) -> Dict:
        """
        流式构建索引映射 - 避免大量中间数组
        只创建必要的帧级映射数组
        """
        utterances = metadata['utterances']
        total_frames = metadata['total_frames']

        # 使用 memmap 减少内存占用
        frame_to_utt = np.zeros(total_frames, dtype=np.int32)
        frame_to_phone = np.full(total_frames, self.phone_to_idx['SIL'], dtype=np.int32)
        frame_to_gender = np.zeros(total_frames, dtype=np.int8)
        frame_to_symbol = np.zeros(total_frames, dtype=np.int32)

        # 加载 phone 预测
        phone_predictions = self._load_phone_predictions()

        for utt_idx, utt in enumerate(tqdm(utterances, desc="Building index map")):
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            n_frames = e - s

            frame_to_utt[s:e] = utt_idx

            gender = utt.get('gender', 'M')
            frame_to_gender[s:e] = 0 if gender == 'M' else 1

            utt_id = utt['utt_id']
            if phone_predictions is not None and utt_id in phone_predictions:
                phones = phone_predictions[utt_id]
                if len(phones) >= n_frames:
                    frame_to_phone[s:e] = phones[:n_frames]
                else:
                    frame_to_phone[s:s+len(phones)] = phones
                    frame_to_phone[s+len(phones):e] = self.phone_to_idx['SIL']

        # 统计每个 phone 的帧数 (用于后续决策)
        phone_counts = np.bincount(frame_to_phone, minlength=self.num_phones)
        gender_counts = np.bincount(frame_to_gender, minlength=2)

        logger.info(f"  Total frames: {total_frames:,}")
        logger.info(f"  Phone distribution: max={phone_counts.max():,}, min={phone_counts[phone_counts>0].min():,}")
        logger.info(f"  Gender: M={gender_counts[0]:,}, F={gender_counts[1]:,}")

        return {
            'total_frames': total_frames,
            'utterances': utterances,
            'frame_to_utt': frame_to_utt,
            'frame_to_phone': frame_to_phone,
            'frame_to_gender': frame_to_gender,
            'frame_to_symbol': frame_to_symbol,
            'phone_counts': phone_counts,
            'gender_counts': gender_counts,
        }

    def _load_phone_predictions(self) -> Optional[Dict]:
        """加载 phone 预测"""
        path = self.cache_dir / 'features' / 'cleaned' / 'phone_predictions.h5'
        if path.exists():
            predictions = {}
            with h5py.File(path, 'r') as f:
                for utt_id in f.keys():
                    phones = f[utt_id][:]
                    if len(phones) > 0 and np.any(phones < self.num_phones):
                        predictions[utt_id] = phones
            if predictions:
                logger.info(f"  Loaded phone predictions: {len(predictions)} utts")
                return predictions
        return None

    def _assign_symbols_streaming(self, features_path: Path, index_map: Dict):
        """流式分配 SAMM 符号"""
        codebook_path = self.checkpoint_dir / 'codebook.pt'
        if not codebook_path.exists():
            logger.warning("  No codebook found, using placeholder symbols")
            return

        try:
            codebook_data = torch.load(codebook_path, map_location='cpu')
            if isinstance(codebook_data, dict) and 'codebook' in codebook_data:
                codebook = codebook_data['codebook']
            else:
                codebook = codebook_data
            codebook = torch.from_numpy(codebook).float() if isinstance(codebook, np.ndarray) else codebook

            device = torch.device(f'cuda:{self.pool_config.gpu_id}' if torch.cuda.is_available() else 'cpu')
            codebook = codebook.to(device)

            batch_size = 50000  # 5万一批，更省内存
            total_frames = index_map['total_frames']
            frame_to_symbol = index_map['frame_to_symbol']

            with h5py.File(features_path, 'r') as f:
                feats = f['features']
                for start in tqdm(range(0, total_frames, batch_size), desc="Assigning symbols"):
                    end = min(start + batch_size, total_frames)
                    batch_feats = torch.from_numpy(feats[start:end]).float().to(device)
                    dist = torch.cdist(batch_feats, codebook)
                    symbols = dist.argmin(dim=-1).cpu().numpy()
                    frame_to_symbol[start:end] = symbols
                    del batch_feats, dist
                torch.cuda.empty_cache()

            logger.info(f"  Assigned symbols to {total_frames:,} frames")
        except Exception as e:
            logger.warning(f"  Failed to assign symbols: {e}")

    def _build_and_save_global_index(self, features_path: Path, index_map: Dict):
        """构建并立即保存全局索引"""
        import time

        total_frames = index_map['total_frames']
        nlist = min(self.pool_config.nlist, int(np.sqrt(total_frames)))

        logger.info(f"  Building IVF: nlist={nlist}, total={total_frames:,}")

        quantizer = faiss.IndexFlatIP(self.feature_dim)
        index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)

        # 训练 - 使用较小的训练集
        train_size = min(self.pool_config.train_size, total_frames)
        logger.info(f"  Loading training data ({train_size:,} vectors)...")

        with h5py.File(features_path, 'r') as f:
            train_data = f['features'][:train_size].astype(np.float32)
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

        # 添加向量 - 使用较小的批次
        batch_size = self.pool_config.gpu_add_batch
        logger.info(f"  Adding {total_frames:,} vectors (batch={batch_size:,})...")

        with h5py.File(features_path, 'r') as f:
            feats = f['features']
            for start in tqdm(range(0, total_frames, batch_size), desc="Adding vectors"):
                end = min(start + batch_size, total_frames)
                batch = feats[start:end].astype(np.float32)
                faiss.normalize_L2(batch)
                index.add(batch)
                del batch

        index.nprobe = self.pool_config.nprobe

        # 立即保存
        faiss.write_index(index, str(self.pool_dir / 'faiss.index'))
        faiss_legacy_path = self.pool_dir / 'faiss_trained.index'
        if not faiss_legacy_path.exists():
            try:
                faiss_legacy_path.symlink_to('faiss.index')
            except OSError:
                pass

        logger.info(f"  Saved global index: {index.ntotal:,} vectors")
        del index
        gc.collect()

    def _build_phone_indices_streaming(self, features_path: Path, index_map: Dict):
        """
        流式构建 phone 子索引

        关键优化:
        1. 对大型 phone 使用采样
        2. 分块扫描获取索引，避免一次性 np.where
        3. 边构建边保存
        """
        frame_to_phone = index_map['frame_to_phone']
        phone_counts = index_map['phone_counts']
        total_frames = index_map['total_frames']

        phone_dir = self.pool_dir / 'phone_indices'
        phone_dir.mkdir(exist_ok=True)
        phone_meta = {}

        scan_batch = 1000000  # 100万帧一批扫描

        for phone_id in tqdm(range(self.num_phones), desc="Phone indices"):
            n_frames = phone_counts[phone_id]

            if n_frames < 100:  # 最小样本数
                continue

            phone_name = self.idx_to_phone[phone_id]

            # 决定是否需要采样
            max_samples = self.pool_config.max_samples_per_phone
            need_sampling = n_frames > max_samples
            target_samples = min(n_frames, max_samples)

            if need_sampling:
                logger.info(f"  {phone_name}: {n_frames:,} frames -> sampling {target_samples:,}")
                # 计算采样率
                sample_rate = target_samples / n_frames
            else:
                sample_rate = 1.0

            # 分块扫描收集索引
            phone_frames = []
            for scan_start in range(0, total_frames, scan_batch):
                scan_end = min(scan_start + scan_batch, total_frames)
                chunk_mask = frame_to_phone[scan_start:scan_end] == phone_id
                chunk_indices = np.where(chunk_mask)[0] + scan_start

                if need_sampling and len(chunk_indices) > 0:
                    # 按采样率随机采样
                    n_keep = max(1, int(len(chunk_indices) * sample_rate))
                    if n_keep < len(chunk_indices):
                        chunk_indices = np.random.choice(chunk_indices, n_keep, replace=False)

                phone_frames.append(chunk_indices)

            phone_frames = np.concatenate(phone_frames)
            actual_count = len(phone_frames)

            if actual_count < 100:
                continue

            # 构建索引
            with h5py.File(features_path, 'r') as f:
                feats = f['features']

                if actual_count < 1000:
                    sub_index = faiss.IndexFlatIP(self.feature_dim)
                else:
                    sub_nlist = min(256, int(np.sqrt(actual_count)))
                    quantizer = faiss.IndexFlatIP(self.feature_dim)
                    sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                                   sub_nlist, faiss.METRIC_INNER_PRODUCT)

                    # 训练: 最多使用 30000 样本
                    train_size = min(30000, actual_count)
                    train_indices = phone_frames[np.random.choice(len(phone_frames), train_size, replace=False)] if train_size < actual_count else phone_frames
                    train_data = feats[train_indices].astype(np.float32)
                    faiss.normalize_L2(train_data)
                    sub_index.train(train_data)
                    del train_data
                    sub_index.nprobe = min(32, sub_nlist // 4)

                # 分批添加
                add_batch = 20000
                for i in range(0, actual_count, add_batch):
                    batch_indices = phone_frames[i:i+add_batch]
                    batch_features = feats[batch_indices].astype(np.float32)
                    faiss.normalize_L2(batch_features)
                    sub_index.add(batch_features)
                    del batch_features

            # 保存并释放
            faiss.write_index(sub_index, str(phone_dir / f'{phone_name}.index'))
            np.save(phone_dir / f'{phone_name}_frames.npy', phone_frames)
            phone_meta[phone_name] = {'phone_id': phone_id, 'count': actual_count, 'original_count': int(n_frames)}

            del sub_index, phone_frames
            gc.collect()

        with open(phone_dir / 'metadata.json', 'w') as f:
            json.dump(phone_meta, f, indent=2)

        logger.info(f"  Built {len(phone_meta)} phone indices")

    def _build_gender_indices_streaming(self, features_path: Path, index_map: Dict):
        """流式构建 gender 子索引"""
        frame_to_gender = index_map['frame_to_gender']
        gender_counts = index_map['gender_counts']
        total_frames = index_map['total_frames']

        gender_dir = self.pool_dir / 'gender_indices'
        gender_dir.mkdir(exist_ok=True)

        scan_batch = 2000000  # 200万帧一批

        for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
            n_frames = gender_counts[gender_id]

            if n_frames < 100:
                continue

            max_samples = self.pool_config.max_samples_per_gender
            need_sampling = n_frames > max_samples
            target_samples = min(n_frames, max_samples)

            if need_sampling:
                logger.info(f"  {gender_name}: {n_frames:,} frames -> sampling {target_samples:,}")
                sample_rate = target_samples / n_frames
            else:
                sample_rate = 1.0
                logger.info(f"  {gender_name}: {n_frames:,} frames")

            # 分块收集索引
            gender_frames = []
            for scan_start in range(0, total_frames, scan_batch):
                scan_end = min(scan_start + scan_batch, total_frames)
                chunk_mask = frame_to_gender[scan_start:scan_end] == gender_id
                chunk_indices = np.where(chunk_mask)[0] + scan_start

                if need_sampling and len(chunk_indices) > 0:
                    n_keep = max(1, int(len(chunk_indices) * sample_rate))
                    if n_keep < len(chunk_indices):
                        chunk_indices = np.random.choice(chunk_indices, n_keep, replace=False)

                gender_frames.append(chunk_indices)

            gender_frames = np.concatenate(gender_frames)
            actual_count = len(gender_frames)

            logger.info(f"    Actual samples: {actual_count:,}")

            # 构建 IVF 索引
            nlist = min(512, int(np.sqrt(actual_count)))  # 降低 nlist
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                           nlist, faiss.METRIC_INNER_PRODUCT)

            with h5py.File(features_path, 'r') as f:
                feats = f['features']

                # 训练
                train_size = min(50000, actual_count)
                train_indices = gender_frames[np.random.choice(len(gender_frames), train_size, replace=False)] if train_size < actual_count else gender_frames
                train_data = feats[train_indices].astype(np.float32)
                faiss.normalize_L2(train_data)
                sub_index.train(train_data)
                del train_data
                gc.collect()

                # 分批添加
                add_batch = 50000
                for i in tqdm(range(0, actual_count, add_batch), desc=f"    {gender_name}", leave=False):
                    batch_indices = gender_frames[i:i+add_batch]
                    batch_data = feats[batch_indices].astype(np.float32)
                    faiss.normalize_L2(batch_data)
                    sub_index.add(batch_data)
                    del batch_data

            sub_index.nprobe = min(32, nlist // 4)

            # 保存并释放
            faiss.write_index(sub_index, str(gender_dir / f'{gender_name}.index'))
            np.save(gender_dir / f'{gender_name}_frames.npy', gender_frames)

            del sub_index, gender_frames
            gc.collect()

        logger.info(f"  Built gender indices")

    def _build_phone_clusters_sampled(self, features_path: Path, index_map: Dict):
        """采样方式构建 phone clusters (使用 MiniBatchKMeans)"""
        frame_to_phone = index_map['frame_to_phone']
        frame_to_gender = index_map['frame_to_gender']
        phone_counts = index_map['phone_counts']
        total_frames = index_map['total_frames']
        n_clusters = self.pool_config.n_phone_clusters

        phone_clusters = {}
        scan_batch = 1000000
        max_samples = 30000  # 每个 phone-gender 组合最多 3万样本用于聚类

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for phone_id in tqdm(range(self.num_phones), desc="Phone clusters"):
                if phone_counts[phone_id] < n_clusters:
                    continue

                for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                    # 分块采样
                    sampled_indices = []
                    target_per_chunk = max_samples // (total_frames // scan_batch + 1)

                    for scan_start in range(0, total_frames, scan_batch):
                        scan_end = min(scan_start + scan_batch, total_frames)
                        chunk_phone = frame_to_phone[scan_start:scan_end]
                        chunk_gender = frame_to_gender[scan_start:scan_end]
                        mask = (chunk_phone == phone_id) & (chunk_gender == gender_id)
                        chunk_indices = np.where(mask)[0] + scan_start

                        if len(chunk_indices) > target_per_chunk:
                            chunk_indices = np.random.choice(chunk_indices, target_per_chunk, replace=False)
                        sampled_indices.append(chunk_indices)

                    indices = np.concatenate(sampled_indices)
                    if len(indices) < n_clusters:
                        continue

                    if len(indices) > max_samples:
                        indices = np.random.choice(indices, max_samples, replace=False)

                    phone_features = feats[indices].astype(np.float32)

                    # 使用 MiniBatchKMeans 更省内存
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto',
                                             random_state=42, batch_size=1000)
                    kmeans.fit(phone_features)
                    del phone_features

                    key = f"{phone_id}_{gender_name}"
                    phone_clusters[key] = torch.from_numpy(kmeans.cluster_centers_).float()

                # 不区分性别的版本
                sampled_indices = []
                for scan_start in range(0, total_frames, scan_batch):
                    scan_end = min(scan_start + scan_batch, total_frames)
                    chunk_mask = frame_to_phone[scan_start:scan_end] == phone_id
                    chunk_indices = np.where(chunk_mask)[0] + scan_start
                    target_per_chunk = max_samples // (total_frames // scan_batch + 1)
                    if len(chunk_indices) > target_per_chunk:
                        chunk_indices = np.random.choice(chunk_indices, target_per_chunk, replace=False)
                    sampled_indices.append(chunk_indices)

                indices = np.concatenate(sampled_indices)
                if len(indices) >= n_clusters:
                    if len(indices) > max_samples:
                        indices = np.random.choice(indices, max_samples, replace=False)
                    phone_features = feats[indices].astype(np.float32)
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto',
                                             random_state=42, batch_size=1000)
                    kmeans.fit(phone_features)
                    phone_clusters[str(phone_id)] = torch.from_numpy(kmeans.cluster_centers_).float()

        torch.save(phone_clusters, self.pool_dir / 'phone_clusters.pt')
        logger.info(f"  Built {len(phone_clusters)} phone cluster sets")

    def _save_metadata(self, index_map: Dict, metadata: Dict):
        """保存所有元数据和映射数组"""
        # 帧映射
        np.save(self.pool_dir / 'frame_to_utt.npy', index_map['frame_to_utt'])
        np.save(self.pool_dir / 'frame_to_phone.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'frame_to_gender.npy', index_map['frame_to_gender'])
        np.save(self.pool_dir / 'frame_to_symbol.npy', index_map['frame_to_symbol'])

        # 兼容命名
        np.save(self.pool_dir / 'phones.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'genders.npy', index_map['frame_to_gender'])
        np.save(self.pool_dir / 'symbols.npy', index_map['frame_to_symbol'])

        with open(self.pool_dir / 'symbol_index.pkl', 'wb') as f:
            pickle.dump(self.phone_to_idx, f)

        with open(self.pool_dir / 'utterances.json', 'w') as f:
            json.dump(index_map['utterances'], f)

        utt_ids = [utt['utt_id'] for utt in index_map['utterances']]
        np.save(self.pool_dir / 'utt_ids.npy', np.array(utt_ids, dtype=object))

        pool_meta = {
            'total_frames': index_map['total_frames'],
            'num_utterances': len(index_map['utterances']),
            'feature_dim': self.feature_dim,
            'phone_to_idx': self.phone_to_idx,
            'n_phone_clusters': self.pool_config.n_phone_clusters,
        }
        with open(self.pool_dir / 'metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)

        # 创建特征软链接
        features_src = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        features_dst = self.pool_dir / 'features.h5'
        if not features_dst.exists() and features_src.exists():
            try:
                features_dst.symlink_to(features_src)
            except OSError:
                pass

        logger.info(f"\n  Saved to: {self.pool_dir}")


def run_pool_building_low_mem(config: Dict) -> Dict:
    """Step 6 入口 (低内存版本)"""
    builder = LowMemTargetPoolBuilder(config)
    return builder.build()
