# pipelines/offline/pool_building_dual_gpu.py
"""
Step 6: Target Pool Construction (Dual GPU Version)
针对双 4090 + 240GB 内存优化:
1. 双 GPU 真正并行添加向量
2. GPU 加速 L2 归一化
3. 多线程预读取数据
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
from sklearn.cluster import KMeans
import logging
import pickle
import gc
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class DualGPUPoolConfig:
    """双 GPU Target Pool 配置"""
    nlist: int = 4096
    nprobe: int = 128
    train_size: int = 500000
    min_samples_per_cluster: int = 100

    io_batch_size: int = 500000
    gpu_add_batch: int = 500000  # 降低到50万，更平衡

    use_gpu: bool = True
    gpu_ids: List[int] = None

    build_phone_indices: bool = True
    build_gender_indices: bool = True
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8

    # 内存优化选项
    phone_save_immediately: bool = True  # 立即保存 phone 索引到磁盘
    low_memory_mode: bool = True         # 低内存模式

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0, 1]


class DualGPUTargetPoolBuilder:
    """
    双 GPU Target Pool 构建器

    真正的优化策略:
    1. 使用 GPU 索引进行添加（GPU 上做聚类分配）
    2. 多线程预读取下一批数据
    3. GPU 加速 L2 归一化
    """

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: Dict, pool_config: Optional[DualGPUPoolConfig] = None):
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

        # 双 GPU 资源
        self.gpu_resources = {}
        self.n_gpus = faiss.get_num_gpus()

        if self.pool_config.use_gpu and self.n_gpus >= 1:
            for gpu_id in self.pool_config.gpu_ids:
                if gpu_id < self.n_gpus:
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB
                    self.gpu_resources[gpu_id] = res
            logger.info(f"  Initialized {len(self.gpu_resources)} GPU resources: {list(self.gpu_resources.keys())}")

    def _load_pool_config(self) -> DualGPUPoolConfig:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        knn_cfg = self.config.get('knn_vc', {})
        gpu_ids = cfg.get('gpu_ids', self.config.get('gpu_ids', [0, 1]))

        return DualGPUPoolConfig(
            nlist=cfg.get('nlist', 4096),
            nprobe=cfg.get('nprobe', 128),
            train_size=cfg.get('train_size', 500000),
            min_samples_per_cluster=cfg.get('min_samples_per_cluster', 100),
            io_batch_size=cfg.get('io_batch_size', 500000),
            gpu_add_batch=cfg.get('gpu_add_batch', 500000),
            use_gpu=cfg.get('use_gpu', True),
            gpu_ids=gpu_ids,
            build_phone_indices=cfg.get('build_phone_indices', True),
            build_gender_indices=cfg.get('build_gender_indices', True),
            build_phone_clusters=cfg.get('build_phone_clusters', True),
            n_phone_clusters=knn_cfg.get('num_clusters', 8),
            phone_save_immediately=cfg.get('phone_save_immediately', True),
            low_memory_mode=cfg.get('low_memory_mode', True),
        )

    def build(self) -> Dict:
        """构建完整的 Target Pool"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Target Pool Construction (Dual GPU Version)")
        logger.info("=" * 60)
        logger.info(f"  GPUs: {self.pool_config.gpu_ids}")
        logger.info(f"  GPU Add Batch: {self.pool_config.gpu_add_batch:,}")

        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 1. 构建索引映射
        logger.info("\n[1/6] 构建索引映射...")
        index_map = self._build_index_map(features_path, metadata)

        # 1.5. 分配 SAMM 符号
        logger.info("\n[1.5/6] 分配 SAMM 符号...")
        self._assign_symbols(features_path, index_map)

        # 2. 构建全局 FAISS 索引 (双 GPU 加速)
        logger.info("\n[2/6] 构建全局 FAISS 索引 (GPU 加速)...")
        global_index = self._build_global_index_gpu_accelerated(features_path, index_map)
        gc.collect()
        torch.cuda.empty_cache()

        # 低内存模式: 先保存全局索引，释放内存后再构建子索引
        if self.pool_config.low_memory_mode:
            logger.info("  [低内存模式] 先保存全局索引到磁盘...")
            faiss.write_index(global_index, str(self.pool_dir / 'faiss.index'))
            faiss_legacy_path = self.pool_dir / 'faiss_trained.index'
            if not faiss_legacy_path.exists():
                try:
                    faiss_legacy_path.symlink_to('faiss.index')
                except OSError:
                    pass
            del global_index
            global_index = None  # 标记已保存
            gc.collect()
            logger.info("  [低内存模式] 全局索引已保存并释放内存")

        # 3. 构建 Phone 子索引
        phone_indices = {}
        if self.pool_config.build_phone_indices:
            logger.info("\n[3/6] 构建 Phone 子索引...")
            phone_indices = self._build_phone_indices(features_path, index_map)
            gc.collect()

        # 4. 构建 Gender 子索引
        gender_indices = {}
        if self.pool_config.build_gender_indices:
            logger.info("\n[4/6] 构建 Gender 子索引...")
            gender_indices = self._build_gender_indices(features_path, index_map)
            gc.collect()

        # 5. 构建 Phone Clusters
        phone_clusters = {}
        if self.pool_config.build_phone_clusters:
            logger.info("\n[5/6] 构建 Phone Clusters...")
            phone_clusters = self._build_phone_clusters(features_path, index_map)
            gc.collect()

        # 6. 保存
        logger.info("\n[6/6] 保存 Target Pool...")
        self._save_pool(global_index, phone_indices, gender_indices,
                       phone_clusters, index_map, metadata)

        return {
            'pool_dir': str(self.pool_dir),
            'total_frames': index_map['total_frames'],
        }

    def _build_index_map(self, features_path: Path, metadata: Dict) -> Dict:
        """构建帧级索引映射"""
        utterances = metadata['utterances']
        total_frames = metadata['total_frames']

        frame_to_utt = np.zeros(total_frames, dtype=np.int32)
        frame_to_phone = np.full(total_frames, self.phone_to_idx['SIL'], dtype=np.int32)
        frame_to_gender = np.zeros(total_frames, dtype=np.int8)
        frame_to_symbol = np.zeros(total_frames, dtype=np.int32)

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
                phones = phones[:n_frames] if len(phones) >= n_frames else \
                         np.pad(phones, (0, n_frames - len(phones)),
                               constant_values=self.phone_to_idx['SIL'])
                frame_to_phone[s:e] = phones

        logger.info(f"  Total frames: {total_frames:,}")
        return {
            'total_frames': total_frames,
            'utterances': utterances,
            'frame_to_utt': frame_to_utt,
            'frame_to_phone': frame_to_phone,
            'frame_to_gender': frame_to_gender,
            'frame_to_symbol': frame_to_symbol,
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
            if predictions:
                logger.info(f"  Loaded phone predictions: {len(predictions)} utts")
                return predictions
        return None

    def _assign_symbols(self, features_path: Path, index_map: Dict):
        """分配 SAMM 符号"""
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

            # 使用 GPU 加速
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            codebook = codebook.to(device)

            batch_size = 100000  # 10万一批
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

            logger.info(f"  Assigned symbols to {total_frames:,} frames")
        except Exception as e:
            logger.warning(f"  Failed to assign symbols: {e}")

    def _build_global_index_gpu_accelerated(self, features_path: Path, index_map: Dict) -> faiss.Index:
        """
        构建全局 FAISS IVF 索引 - GPU 加速聚类分配版本

        瓶颈分析：
        - IVF add 需要计算每个向量到所有聚类中心的距离
        - 5500万 × 4096 × 1024 = 巨大计算量

        优化策略：
        1. GPU 训练 (快)
        2. GPU 批量计算聚类分配 (利用双 GPU 并行)
        3. CPU 直接插入倒排列表 (跳过距离计算)
        """
        import time

        total_frames = index_map['total_frames']

        # 减少 nlist 以加速 (4096 -> 2048)
        nlist = min(2048, int(np.sqrt(total_frames)))
        nlist = max(nlist, 1)

        logger.info(f"  Building IVF: nlist={nlist}, total={total_frames:,}")

        # 创建 CPU 索引
        quantizer = faiss.IndexFlatIP(self.feature_dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist,
                                       faiss.METRIC_INNER_PRODUCT)

        # ================================================================
        # 阶段1: GPU 加速训练
        # ================================================================
        train_size = min(self.pool_config.train_size, total_frames)
        logger.info(f"  Loading training data ({train_size:,} vectors)...")

        with h5py.File(features_path, 'r') as f:
            train_data = f['features'][:train_size].astype(np.float32)
        faiss.normalize_L2(train_data)

        logger.info(f"  Training IVF index on GPU 0...")
        start_time = time.time()

        if len(self.gpu_resources) > 0:
            gpu_id = self.pool_config.gpu_ids[0]
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources[gpu_id], gpu_id, cpu_index)
            gpu_index.train(train_data)
            cpu_index = faiss.index_gpu_to_cpu(gpu_index)
            del gpu_index
            torch.cuda.empty_cache()
        else:
            cpu_index.train(train_data)

        train_time = time.time() - start_time
        logger.info(f"  ✓ Training completed in {train_time:.1f}s")

        del train_data
        gc.collect()

        # ================================================================
        # 阶段2: GPU 加速聚类分配 + CPU 快速插入
        # ================================================================
        logger.info(f"  Adding {total_frames:,} vectors with GPU-accelerated assignment...")

        # 获取聚类中心并创建 GPU quantizer
        # 使用 reconstruct_n 兼容新版 FAISS (1.7.3+)
        centroids = cpu_index.quantizer.reconstruct_n(0, nlist)

        # 创建双 GPU quantizer 用于快速聚类分配
        if len(self.gpu_resources) >= 2:
            logger.info("  Using dual GPU for cluster assignment...")
            gpu0_quantizer = faiss.IndexFlatIP(self.feature_dim)
            gpu0_quantizer.add(centroids)
            gpu0_quantizer = faiss.index_cpu_to_gpu(self.gpu_resources[0], 0, gpu0_quantizer)

            gpu1_quantizer = faiss.IndexFlatIP(self.feature_dim)
            gpu1_quantizer.add(centroids)
            gpu1_quantizer = faiss.index_cpu_to_gpu(self.gpu_resources[1], 1, gpu1_quantizer)

            gpu_quantizers = [gpu0_quantizer, gpu1_quantizer]
        elif len(self.gpu_resources) == 1:
            logger.info("  Using single GPU for cluster assignment...")
            gpu_quantizer = faiss.IndexFlatIP(self.feature_dim)
            gpu_quantizer.add(centroids)
            gpu_quantizer = faiss.index_cpu_to_gpu(self.gpu_resources[0], 0, gpu_quantizer)
            gpu_quantizers = [gpu_quantizer]
        else:
            gpu_quantizers = None

        # 设置索引为可直接添加模式
        cpu_index.set_direct_map_type(faiss.DirectMap.NoMap)

        batch_size = 500000  # 50万一批
        n_batches = (total_frames + batch_size - 1) // batch_size
        logger.info(f"  Batch size: {batch_size:,}, Total batches: {n_batches}")

        start_time = time.time()

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for batch_idx, start in enumerate(tqdm(range(0, total_frames, batch_size), desc="Adding vectors")):
                end = min(start + batch_size, total_frames)

                # 读取并归一化
                batch = feats[start:end].astype(np.float32)
                faiss.normalize_L2(batch)

                if gpu_quantizers:
                    # GPU 加速聚类分配
                    # 交替使用双 GPU
                    gpu_idx = batch_idx % len(gpu_quantizers)
                    _, assignments = gpu_quantizers[gpu_idx].search(batch, 1)
                    assignments = assignments.flatten().astype(np.int64)

                    # 使用预计算的分配直接添加
                    cpu_index.add_core(
                        len(batch),
                        faiss.swig_ptr(batch),
                        None,  # ids
                        faiss.swig_ptr(assignments)
                    )
                else:
                    # CPU 回退
                    cpu_index.add(batch)

        # 清理 GPU quantizers
        if gpu_quantizers:
            del gpu_quantizers
            torch.cuda.empty_cache()

        add_time = time.time() - start_time
        throughput = cpu_index.ntotal / add_time
        logger.info(f"  ✓ Added {cpu_index.ntotal:,} vectors in {add_time:.1f}s")
        logger.info(f"  Throughput: {throughput:,.0f} vectors/sec")

        cpu_index.nprobe = self.pool_config.nprobe
        return cpu_index

    def _build_phone_indices(self, features_path: Path, index_map: Dict) -> Dict:
        """构建 phone 子索引 (低内存模式: 边构建边保存)"""
        phone_indices = {}
        frame_to_phone = index_map['frame_to_phone']

        # 低内存模式: 直接保存到磁盘，不在内存中累积
        save_immediately = self.pool_config.phone_save_immediately or self.pool_config.low_memory_mode

        if save_immediately:
            phone_dir = self.pool_dir / 'phone_indices'
            phone_dir.mkdir(exist_ok=True)
            phone_meta = {}

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for phone_id in tqdm(range(self.num_phones), desc="Phone indices"):
                phone_frames = np.where(frame_to_phone == phone_id)[0]
                n_frames = len(phone_frames)

                if n_frames < self.pool_config.min_samples_per_cluster:
                    continue

                phone_name = self.idx_to_phone.get(phone_id, f"phone_{phone_id}")

                # 创建索引
                if n_frames < 1000:
                    sub_index = faiss.IndexFlatIP(self.feature_dim)
                else:
                    sub_nlist = min(256, int(np.sqrt(n_frames)))
                    quantizer = faiss.IndexFlatIP(self.feature_dim)
                    sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                                   sub_nlist, faiss.METRIC_INNER_PRODUCT)
                    # 训练: 采样最多 50000 帧
                    train_size = min(50000, n_frames)
                    train_step = max(1, n_frames // train_size)
                    train_indices = phone_frames[::train_step][:train_size]
                    train_data = feats[train_indices].astype(np.float32)
                    faiss.normalize_L2(train_data)
                    sub_index.train(train_data)
                    del train_data
                    sub_index.nprobe = min(32, sub_nlist // 4)

                # 分批添加到索引，每批最多 30000 帧
                batch_size = 30000
                for i in range(0, n_frames, batch_size):
                    batch_indices = phone_frames[i:i+batch_size]
                    batch_features = feats[batch_indices].astype(np.float32)
                    faiss.normalize_L2(batch_features)
                    sub_index.add(batch_features)
                    del batch_features

                if save_immediately:
                    # 立即保存到磁盘并释放内存
                    faiss.write_index(sub_index, str(phone_dir / f'{phone_name}.index'))
                    np.save(phone_dir / f'{phone_name}_frames.npy', phone_frames)
                    phone_meta[phone_name] = {'phone_id': phone_id, 'count': n_frames}
                    del sub_index
                else:
                    phone_indices[phone_id] = {
                        'index': sub_index,
                        'frame_indices': phone_frames,
                        'count': n_frames
                    }

                del phone_frames
                gc.collect()

        if save_immediately:
            with open(phone_dir / 'metadata.json', 'w') as f:
                json.dump(phone_meta, f, indent=2)
            logger.info(f"  Built and saved {len(phone_meta)} phone indices (low memory mode)")
            return {}  # 返回空字典，已保存到磁盘
        else:
            logger.info(f"  Built {len(phone_indices)} phone indices")
            return phone_indices

    def _build_gender_indices(self, features_path: Path, index_map: Dict) -> Dict:
        """构建 gender 子索引 (低内存模式: 边构建边保存)"""
        gender_indices = {}
        frame_to_gender = index_map['frame_to_gender']

        save_immediately = self.pool_config.phone_save_immediately or self.pool_config.low_memory_mode

        if save_immediately:
            gender_dir = self.pool_dir / 'gender_indices'
            gender_dir.mkdir(exist_ok=True)

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                gender_frames = np.where(frame_to_gender == gender_id)[0]
                n_frames = len(gender_frames)

                if n_frames < self.pool_config.min_samples_per_cluster:
                    continue

                logger.info(f"  Building {gender_name}: {n_frames:,} frames")

                nlist = min(1024, int(np.sqrt(n_frames)))
                quantizer = faiss.IndexFlatIP(self.feature_dim)
                sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                               nlist, faiss.METRIC_INNER_PRODUCT)

                train_size = min(100000, n_frames)
                train_step = max(1, n_frames // train_size)
                train_sample_indices = gender_frames[::train_step][:train_size]
                train_data = feats[train_sample_indices].astype(np.float32)
                faiss.normalize_L2(train_data)
                sub_index.train(train_data)
                del train_data
                gc.collect()

                # 分批添加，每批最多 50000
                batch_size = 50000
                for i in tqdm(range(0, n_frames, batch_size), desc=f"    {gender_name}", leave=False):
                    batch_indices = gender_frames[i:i+batch_size]
                    batch_data = feats[batch_indices].astype(np.float32)
                    faiss.normalize_L2(batch_data)
                    sub_index.add(batch_data)
                    del batch_data

                sub_index.nprobe = min(64, nlist // 4)

                if save_immediately:
                    # 立即保存并释放内存
                    faiss.write_index(sub_index, str(gender_dir / f'{gender_name}.index'))
                    np.save(gender_dir / f'{gender_name}_frames.npy', gender_frames)
                    del sub_index
                else:
                    gender_indices[gender_name] = {
                        'index': sub_index,
                        'frame_indices': gender_frames,
                        'count': n_frames
                    }

                del gender_frames
                gc.collect()

        if save_immediately:
            logger.info(f"  Built and saved gender indices (low memory mode)")
            return {}
        else:
            return gender_indices

    def _build_phone_clusters(self, features_path: Path, index_map: Dict) -> Dict:
        """构建 phone clusters"""
        phone_clusters = {}
        frame_to_phone = index_map['frame_to_phone']
        frame_to_gender = index_map['frame_to_gender']
        n_clusters = self.pool_config.n_phone_clusters

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for phone_id in tqdm(range(self.num_phones), desc="Phone clusters"):
                for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                    mask = (frame_to_phone == phone_id) & (frame_to_gender == gender_id)
                    indices = np.where(mask)[0]

                    if len(indices) < n_clusters:
                        continue

                    if len(indices) > 50000:
                        indices = np.random.choice(indices, 50000, replace=False)

                    phone_features = feats[indices].astype(np.float32)
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                    kmeans.fit(phone_features)
                    del phone_features

                    key = f"{phone_id}_{gender_name}"
                    phone_clusters[key] = torch.from_numpy(kmeans.cluster_centers_).float()
                    del kmeans

                phone_indices = np.where(frame_to_phone == phone_id)[0]
                if len(phone_indices) >= n_clusters:
                    if len(phone_indices) > 100000:
                        phone_indices = np.random.choice(phone_indices, 100000, replace=False)

                    phone_features = feats[phone_indices].astype(np.float32)
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                    kmeans.fit(phone_features)
                    phone_clusters[str(phone_id)] = torch.from_numpy(kmeans.cluster_centers_).float()

        logger.info(f"  Built {len(phone_clusters)} phone cluster sets")
        return phone_clusters

    def _save_pool(self, global_index, phone_indices, gender_indices,
                   phone_clusters, index_map, metadata):
        """保存所有内容"""
        # 低内存模式下，global_index 可能已经保存并释放 (为 None)
        if global_index is not None:
            faiss.write_index(global_index, str(self.pool_dir / 'faiss.index'))
            faiss_legacy_path = self.pool_dir / 'faiss_trained.index'
            if not faiss_legacy_path.exists():
                try:
                    faiss_legacy_path.symlink_to('faiss.index')
                except OSError:
                    faiss.write_index(global_index, str(faiss_legacy_path))
        else:
            logger.info("  [低内存模式] 全局索引已在之前保存，跳过")

        # 低内存模式下，phone_indices 和 gender_indices 已经在构建时保存
        if phone_indices:
            phone_dir = self.pool_dir / 'phone_indices'
            phone_dir.mkdir(exist_ok=True)
            phone_meta = {}
            for phone_id, data in phone_indices.items():
                phone_name = self.idx_to_phone[phone_id]
                faiss.write_index(data['index'], str(phone_dir / f'{phone_name}.index'))
                np.save(phone_dir / f'{phone_name}_frames.npy', data['frame_indices'])
                phone_meta[phone_name] = {'phone_id': phone_id, 'count': data['count']}
            with open(phone_dir / 'metadata.json', 'w') as f:
                json.dump(phone_meta, f, indent=2)

        if gender_indices:
            gender_dir = self.pool_dir / 'gender_indices'
            gender_dir.mkdir(exist_ok=True)
            for gender_name, data in gender_indices.items():
                faiss.write_index(data['index'], str(gender_dir / f'{gender_name}.index'))
                np.save(gender_dir / f'{gender_name}_frames.npy', data['frame_indices'])

        if phone_clusters:
            torch.save(phone_clusters, self.pool_dir / 'phone_clusters.pt')

        np.save(self.pool_dir / 'frame_to_utt.npy', index_map['frame_to_utt'])
        np.save(self.pool_dir / 'frame_to_phone.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'frame_to_gender.npy', index_map['frame_to_gender'])
        np.save(self.pool_dir / 'frame_to_symbol.npy', index_map['frame_to_symbol'])

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

        features_src = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        features_dst = self.pool_dir / 'features.h5'
        if not features_dst.exists() and features_src.exists():
            try:
                features_dst.symlink_to(features_src)
            except OSError:
                pass

        logger.info(f"\n  ✓ Saved to: {self.pool_dir}")


def run_pool_building_dual_gpu(config: Dict) -> Dict:
    """Step 6 入口 (双 GPU 版本)"""
    builder = DualGPUTargetPoolBuilder(config)
    return builder.build()
