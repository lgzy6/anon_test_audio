# pipelines/offline/pool_building.py
"""
Step 6: Target Pool Construction
支持两种格式:
1. FAISS 索引 (全局/Phone/Gender 检索)
2. pknnvc 格式 (Phone-clustered 特征)
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

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Target Pool 配置"""
    # FAISS 参数
    nlist: int = 4096
    nprobe: int = 64
    train_size: int = 500000
    min_samples_per_cluster: int = 100
    
    # IO 参数
    io_batch_size: int = 100000
    gpu_add_batch: int = 500000
    
    # 功能开关
    use_gpu: bool = True
    gpu_id: int = 0
    build_phone_indices: bool = True
    build_gender_indices: bool = True
    
    # pknnvc 兼容
    build_phone_clusters: bool = True
    n_phone_clusters: int = 8


class TargetPoolBuilder:
    """
    Target Pool 构建器
    
    输出:
    1. FAISS 索引文件 (用于全局/约束检索)
    2. pknnvc 格式的 phone-clustered 特征 (用于 kNN-VC)
    """
    
    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]
    
    def __init__(self, config: Dict, pool_config: Optional[PoolConfig] = None):
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
    
    def _load_pool_config(self) -> PoolConfig:
        cfg = self.config.get('offline', {}).get('pool_building', {})
        knn_cfg = self.config.get('knn_vc', {})
        
        return PoolConfig(
            nlist=cfg.get('nlist', 4096),
            nprobe=cfg.get('nprobe', 64),
            train_size=cfg.get('train_size', 500000),
            min_samples_per_cluster=cfg.get('min_samples_per_cluster', 100),
            io_batch_size=cfg.get('io_batch_size', 100000),
            gpu_add_batch=cfg.get('gpu_add_batch', 500000),
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
        logger.info("Step 6: Target Pool Construction")
        logger.info("=" * 60)
        
        # 验证输入
        features_path = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        metadata_path = self.cache_dir / 'features' / 'cleaned' / 'metadata.json'
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 1. 构建索引映射
        logger.info("\n[1/6] 构建索引映射...")
        index_map = self._build_index_map(features_path, metadata)

        # 1.5. 分配 SAMM 符号 (如果有 codebook)
        logger.info("\n[1.5/6] 分配 SAMM 符号...")
        self._assign_symbols(features_path, index_map)

        # 2. 构建全局 FAISS 索引
        logger.info("\n[2/6] 构建全局 FAISS 索引...")
        global_index = self._build_global_index(features_path, index_map)
        
        # 3. 构建 Phone 子索引
        phone_indices = {}
        if self.pool_config.build_phone_indices:
            logger.info("\n[3/6] 构建 Phone 子索引...")
            phone_indices = self._build_phone_indices(features_path, index_map)
        
        # 4. 构建 Gender 子索引
        gender_indices = {}
        if self.pool_config.build_gender_indices:
            logger.info("\n[4/6] 构建 Gender 子索引...")
            gender_indices = self._build_gender_indices(features_path, index_map)
        
        # 5. 构建 Phone Clusters (pknnvc 格式)
        phone_clusters = {}
        if self.pool_config.build_phone_clusters:
            logger.info("\n[5/6] 构建 Phone Clusters (pknnvc 格式)...")
            phone_clusters = self._build_phone_clusters(features_path, index_map)
        
        # 6. 保存
        logger.info("\n[6/6] 保存 Target Pool...")
        self._save_pool(global_index, phone_indices, gender_indices, 
                       phone_clusters, index_map, metadata)
        
        return {
            'pool_dir': str(self.pool_dir),
            'total_frames': index_map['total_frames'],
            'num_phones': len(phone_indices),
            'num_genders': len(gender_indices),
            'has_phone_clusters': len(phone_clusters) > 0,
        }
    
    def _build_index_map(self, features_path: Path, metadata: Dict) -> Dict:
        """构建帧级索引映射"""
        utterances = metadata['utterances']
        total_frames = metadata['total_frames']

        frame_to_utt = np.zeros(total_frames, dtype=np.int32)
        frame_to_phone = np.full(total_frames, self.phone_to_idx['SIL'], dtype=np.int32)
        frame_to_gender = np.zeros(total_frames, dtype=np.int8)
        frame_to_symbol = np.zeros(total_frames, dtype=np.int32)  # 添加符号映射

        # 加载 phone 预测
        phone_predictions = self._load_phone_predictions()
        
        for utt_idx, utt in enumerate(tqdm(utterances, desc="Building index map")):
            s, e = utt['h5_start_idx'], utt['h5_end_idx']
            n_frames = e - s
            
            frame_to_utt[s:e] = utt_idx
            
            gender = utt.get('gender', 'M')
            if gender not in ['M', 'F']:
                gender = 'M'
            frame_to_gender[s:e] = 0 if gender == 'M' else 1
            
            utt_id = utt['utt_id']
            if phone_predictions is not None and utt_id in phone_predictions:
                phones = phone_predictions[utt_id]
                phones = phones[:n_frames] if len(phones) >= n_frames else \
                         np.pad(phones, (0, n_frames - len(phones)), 
                               constant_values=self.phone_to_idx['SIL'])
                frame_to_phone[s:e] = phones
        
        logger.info(f"  Total frames: {total_frames:,}")
        logger.info(f"  Utterances: {len(utterances):,}")

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
        paths = [
            self.cache_dir / 'features' / 'cleaned' / 'phone_predictions.h5',
            self.checkpoint_dir / 'target_pool' / 'phones.npy',
        ]

        for path in paths:
            if path.suffix == '.h5' and path.exists():
                predictions = {}
                with h5py.File(path, 'r') as f:
                    for utt_id in f.keys():
                        phones = f[utt_id][:]
                        # 验证 phone 预测
                        if not self._validate_phone_predictions(phones, utt_id):
                            logger.warning(f"  Invalid phone predictions for {utt_id}, skipping")
                            continue
                        predictions[utt_id] = phones

                if len(predictions) > 0:
                    logger.info(f"  Loaded phone predictions: {len(predictions)} utts")
                    self._log_phone_statistics(predictions)
                    return predictions
                else:
                    logger.warning("  All phone predictions failed validation")

        logger.warning("  No phone predictions found")
        return None

    def _validate_phone_predictions(self, phones: np.ndarray, utt_id: str) -> bool:
        """验证 phone 预测的有效性"""
        if len(phones) == 0:
            logger.warning(f"  Empty phone predictions for {utt_id}")
            return False

        # 检查是否所有值都在有效范围内
        if np.any(phones < 0) or np.any(phones >= self.num_phones):
            logger.warning(f"  Phone IDs out of range for {utt_id}: min={phones.min()}, max={phones.max()}")
            return False

        # 检查是否全是同一个音素（可能是错误的预测）
        if len(np.unique(phones)) == 1 and len(phones) > 10:
            logger.warning(f"  All frames have same phone for {utt_id}: {phones[0]}")
            return False

        return True

    def _log_phone_statistics(self, predictions: Dict):
        """记录 phone 预测的统计信息"""
        all_phones = []
        for phones in predictions.values():
            all_phones.extend(phones.tolist())

        unique_phones = np.unique(all_phones)
        logger.info(f"  Phone statistics: {len(unique_phones)} unique phones used")

        # 统计最常见的音素
        phone_counts = np.bincount(all_phones, minlength=self.num_phones)
        top_5_indices = np.argsort(phone_counts)[-5:][::-1]
        logger.info(f"  Top 5 phones: {[self.idx_to_phone[i] for i in top_5_indices]}")

    def _assign_symbols(self, features_path: Path, index_map: Dict):
        """分配 SAMM 符号到每一帧"""
        codebook_path = self.checkpoint_dir / 'codebook.pt'

        if not codebook_path.exists():
            logger.warning("  No codebook found, using placeholder symbols (all zeros)")
            # 使用占位符：所有帧符号为0
            return

        try:
            # 加载 codebook
            logger.info(f"  Loading codebook from {codebook_path}")
            codebook_data = torch.load(codebook_path, map_location='cpu')

            if isinstance(codebook_data, dict) and 'codebook' in codebook_data:
                codebook = codebook_data['codebook']  # [K, D]
            else:
                codebook = codebook_data

            codebook = torch.from_numpy(codebook).float() if isinstance(codebook, np.ndarray) else codebook
            K, D = codebook.shape
            logger.info(f"  Codebook: {K} codes, dim={D}")

            # 批量分配符号
            batch_size = self.pool_config.io_batch_size
            total_frames = index_map['total_frames']
            frame_to_symbol = index_map['frame_to_symbol']

            with h5py.File(features_path, 'r') as f:
                feats = f['features']

                for start in tqdm(range(0, total_frames, batch_size), desc="Assigning symbols"):
                    end = min(start + batch_size, total_frames)
                    batch_feats = torch.from_numpy(feats[start:end]).float()  # [B, D]

                    # 计算到 codebook 的距离
                    dist = torch.cdist(batch_feats, codebook)  # [B, K]
                    symbols = dist.argmin(dim=-1).numpy()  # [B]

                    frame_to_symbol[start:end] = symbols

            logger.info(f"  Assigned symbols to {total_frames:,} frames")

        except Exception as e:
            logger.warning(f"  Failed to assign symbols: {e}")
            logger.warning("  Using placeholder symbols (all zeros)")

    def _build_global_index(self, features_path: Path, index_map: Dict) -> faiss.Index:
        """构建全局 FAISS IVF 索引"""
        import time

        total_frames = index_map['total_frames']
        nlist = min(self.pool_config.nlist, int(np.sqrt(total_frames)))
        nlist = max(nlist, 1)

        logger.info(f"  Building IVF: nlist={nlist}, total={total_frames:,}")

        quantizer = faiss.IndexFlatIP(self.feature_dim)
        index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)

        # 训练 - 使用连续块采样以加速 HDF5 读取
        train_size = min(self.pool_config.train_size, total_frames)
        logger.info(f"  Sampling {train_size:,} training vectors...")

        # 从数据集开头连续读取，这是 HDF5 最快的方式
        logger.info(f"  Loading training data (fast sequential read)...")
        with h5py.File(features_path, 'r') as f:
            train_data = f['features'][:train_size].astype(np.float32)
        logger.info(f"  ✓ Loaded {len(train_data):,} vectors")
        faiss.normalize_L2(train_data)

        logger.info(f"  Training IVF index (k-means with {nlist} clusters)...")
        # 预估训练时间：基于经验公式
        estimated_time = (train_size * nlist) / 1e7  # 粗略估计（秒）
        if self.gpu_res:
            estimated_time /= 10  # GPU 加速约10倍
            logger.info(f"  Using GPU {self.pool_config.gpu_id} - Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f}min)")
        else:
            logger.info(f"  Using CPU - Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f}min)")
        logger.info(f"  Please wait...")
        start_time = time.time()

        if self.gpu_res:
            logger.info(f"  Using GPU {self.pool_config.gpu_id} for training")
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.pool_config.gpu_id, index)
            gpu_index.train(train_data)
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            logger.info(f"  Using CPU for training (this will be slow)")
            index.train(train_data)

        train_time = time.time() - start_time
        logger.info(f"  ✓ Training completed in {train_time:.1f}s ({train_time/60:.1f}min)")

        # 添加向量
        logger.info(f"  Adding {total_frames:,} vectors to index...")
        # 使用配置的批大小（240GB内存可以支持更大的批次）
        batch_size = self.pool_config.gpu_add_batch
        logger.info(f"  Batch size: {batch_size:,}")

        with h5py.File(features_path, 'r') as f:
            feats = f['features']
            for start in tqdm(range(0, total_frames, batch_size), desc="Adding vectors"):
                end = min(start + batch_size, total_frames)
                batch = feats[start:end].astype(np.float32)
                faiss.normalize_L2(batch)
                index.add(batch)

        logger.info(f"  ✓ Index built successfully: {index.ntotal:,} vectors")
        index.nprobe = self.pool_config.nprobe
        return index

    def _load_features_in_batches(self, h5_dataset, indices: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        """批量加载特征以优化内存使用"""
        n_samples = len(indices)
        features_list = []

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_features = h5_dataset[batch_indices].astype(np.float32)
            features_list.append(batch_features)

        return np.vstack(features_list)

    def _build_phone_indices(self, features_path: Path, index_map: Dict) -> Dict:
        """构建 phone 子索引"""
        phone_indices = {}
        frame_to_phone = index_map['frame_to_phone']

        with h5py.File(features_path, 'r') as f:
            feats = f['features']

            for phone_id in tqdm(range(self.num_phones), desc="Phone indices"):
                phone_frames = np.where(frame_to_phone == phone_id)[0]
                n_frames = len(phone_frames)

                if n_frames < self.pool_config.min_samples_per_cluster:
                    continue

                phone_name = self.idx_to_phone.get(phone_id, f"phone_{phone_id}")
                logger.info(f"    Building {phone_name}: {n_frames:,} frames")

                # 优化：批量加载大型 phone 特征
                if n_frames > 50000:
                    phone_features = self._load_features_in_batches(feats, phone_frames)
                else:
                    phone_features = feats[phone_frames].astype(np.float32)

                faiss.normalize_L2(phone_features)

                if n_frames < 1000:
                    sub_index = faiss.IndexFlatIP(self.feature_dim)
                else:
                    sub_nlist = min(256, int(np.sqrt(n_frames)))
                    quantizer = faiss.IndexFlatIP(self.feature_dim)
                    sub_index = faiss.IndexIVFFlat(quantizer, self.feature_dim,
                                                   sub_nlist, faiss.METRIC_INNER_PRODUCT)
                    logger.info(f"      Training {phone_name} index...")
                    sub_index.train(phone_features)
                    sub_index.nprobe = min(32, sub_nlist // 4)

                sub_index.add(phone_features)

                phone_indices[phone_id] = {
                    'index': sub_index,
                    'frame_indices': phone_frames,
                    'count': n_frames
                }

        logger.info(f"  Built {len(phone_indices)} phone indices")
        return phone_indices
    
    def _build_gender_indices(self, features_path: Path, index_map: Dict) -> Dict:
        """构建 gender 子索引"""
        gender_indices = {}
        frame_to_gender = index_map['frame_to_gender']
        
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
                logger.info(f"    Training {gender_name} index with {train_size:,} samples...")

                # 使用均匀采样加速训练数据加载
                train_step = max(1, n_frames // train_size)
                train_sample_indices = gender_frames[::train_step][:train_size]
                train_data = feats[train_sample_indices].astype(np.float32)
                faiss.normalize_L2(train_data)
                sub_index.train(train_data)
                logger.info(f"    ✓ {gender_name} training completed")

                # 批量添加向量
                logger.info(f"    Adding {n_frames:,} vectors...")
                batch_size = self.pool_config.io_batch_size
                for i in tqdm(range(0, n_frames, batch_size), desc=f"    {gender_name}", leave=False):
                    batch_indices = gender_frames[i:i+batch_size]
                    batch_data = feats[batch_indices].astype(np.float32)
                    faiss.normalize_L2(batch_data)
                    sub_index.add(batch_data)
                
                sub_index.nprobe = min(64, nlist // 4)
                
                gender_indices[gender_name] = {
                    'index': sub_index,
                    'frame_indices': gender_frames,
                    'count': n_frames
                }
        
        return gender_indices
    
    def _build_phone_clusters(self, features_path: Path, index_map: Dict) -> Dict:
        """
        构建 pknnvc 格式的 phone-clustered 特征
        
        输出格式: {phone_id: [n_clusters, feature_dim]}
        可选: 按性别分开 {f"{phone_id}_{gender}": ...}
        """
        phone_clusters = {}
        frame_to_phone = index_map['frame_to_phone']
        frame_to_gender = index_map['frame_to_gender']
        n_clusters = self.pool_config.n_phone_clusters
        
        with h5py.File(features_path, 'r') as f:
            feats = f['features']
            
            for phone_id in tqdm(range(self.num_phones), desc="Phone clusters"):
                phone_name = self.idx_to_phone[phone_id]
                
                for gender_id, gender_name in [(0, 'M'), (1, 'F')]:
                    mask = (frame_to_phone == phone_id) & (frame_to_gender == gender_id)
                    indices = np.where(mask)[0]
                    
                    if len(indices) < n_clusters:
                        continue
                    
                    # 采样 (如果太多)
                    if len(indices) > 50000:
                        indices = np.random.choice(indices, 50000, replace=False)
                    
                    phone_features = feats[indices].astype(np.float32)
                    
                    # K-Means 聚类
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                    kmeans.fit(phone_features)
                    
                    key = f"{phone_id}_{gender_name}"
                    phone_clusters[key] = torch.from_numpy(kmeans.cluster_centers_).float()
                
                # 不区分性别的版本
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
        
        # 1. 全局索引
        faiss.write_index(global_index, str(self.pool_dir / 'faiss.index'))

        # 兼容旧命名 (创建软链接而非重复保存)
        faiss_legacy_path = self.pool_dir / 'faiss_trained.index'
        if not faiss_legacy_path.exists():
            try:
                faiss_legacy_path.symlink_to('faiss.index')
            except OSError:
                # 如果软链接失败，则复制文件
                faiss.write_index(global_index, str(faiss_legacy_path))
        
        # 2. Phone 索引
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
        
        # 3. Gender 索引
        if gender_indices:
            gender_dir = self.pool_dir / 'gender_indices'
            gender_dir.mkdir(exist_ok=True)
            
            for gender_name, data in gender_indices.items():
                faiss.write_index(data['index'], str(gender_dir / f'{gender_name}.index'))
                np.save(gender_dir / f'{gender_name}_frames.npy', data['frame_indices'])
        
        # 4. Phone Clusters (pknnvc 格式)
        if phone_clusters:
            torch.save(phone_clusters, self.pool_dir / 'phone_clusters.pt')
        
        # 5. 帧映射 (numpy 格式，兼容 pknnvc)
        np.save(self.pool_dir / 'frame_to_utt.npy', index_map['frame_to_utt'])
        np.save(self.pool_dir / 'frame_to_phone.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'frame_to_gender.npy', index_map['frame_to_gender'])
        np.save(self.pool_dir / 'frame_to_symbol.npy', index_map['frame_to_symbol'])

        # 兼容命名
        np.save(self.pool_dir / 'phones.npy', index_map['frame_to_phone'])
        np.save(self.pool_dir / 'genders.npy', index_map['frame_to_gender'])
        np.save(self.pool_dir / 'symbols.npy', index_map['frame_to_symbol'])  # 关键：retriever需要
        
        # 6. 符号映射 (用于 SAMM)
        with open(self.pool_dir / 'symbol_index.pkl', 'wb') as f:
            pickle.dump(self.phone_to_idx, f)

        # 7. Utterance 信息
        with open(self.pool_dir / 'utterances.json', 'w') as f:
            json.dump(index_map['utterances'], f)

        # 7.5. Utterance IDs (用于 retriever)
        utt_ids = [utt['utt_id'] for utt in index_map['utterances']]
        np.save(self.pool_dir / 'utt_ids.npy', np.array(utt_ids, dtype=object))
        
        # 8. 元数据
        pool_meta = {
            'total_frames': index_map['total_frames'],
            'num_utterances': len(index_map['utterances']),
            'feature_dim': self.feature_dim,
            'phone_to_idx': self.phone_to_idx,
            'n_phone_clusters': self.pool_config.n_phone_clusters,
            'config': asdict(self.pool_config)
        }
        with open(self.pool_dir / 'pool_metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)
        
        with open(self.pool_dir / 'metadata.json', 'w') as f:
            json.dump(pool_meta, f, indent=2)
        
        # 9. 创建特征软链接
        features_src = self.cache_dir / 'features' / 'cleaned' / 'features.h5'
        features_dst = self.pool_dir / 'features.h5'
        if not features_dst.exists() and features_src.exists():
            try:
                features_dst.symlink_to(features_src)
            except OSError:
                pass
        
        logger.info(f"\n  Saved to: {self.pool_dir}")
        logger.info(f"  Files: faiss.index, phone_clusters.pt, phones.npy, genders.npy, symbols.npy")

        # 验证数据完整性
        self._validate_pool_integrity()

    def _validate_pool_integrity(self):
        """验证 Target Pool 的数据完整性"""
        logger.info("\n[Validation] Checking pool integrity...")

        errors = []
        warnings = []

        # 1. 检查必需文件
        required_files = [
            'faiss.index',
            'phones.npy',
            'genders.npy',
            'symbols.npy',
            'metadata.json',
            'utterances.json',
            'utt_ids.npy',
        ]

        for filename in required_files:
            filepath = self.pool_dir / filename
            if not filepath.exists():
                errors.append(f"Missing required file: {filename}")
            else:
                logger.info(f"  ✓ {filename}")

        # 2. 检查可选文件
        optional_files = [
            'phone_clusters.pt',
            'features.h5',
            'symbol_index.pkl',
        ]

        for filename in optional_files:
            filepath = self.pool_dir / filename
            if filepath.exists():
                logger.info(f"  ✓ {filename} (optional)")
            else:
                warnings.append(f"Optional file not found: {filename}")

        # 3. 验证数组维度一致性
        try:
            phones = np.load(self.pool_dir / 'phones.npy')
            genders = np.load(self.pool_dir / 'genders.npy')
            symbols = np.load(self.pool_dir / 'symbols.npy')

            if not (len(phones) == len(genders) == len(symbols)):
                errors.append(f"Array length mismatch: phones={len(phones)}, genders={len(genders)}, symbols={len(symbols)}")
            else:
                logger.info(f"  ✓ Array dimensions consistent: {len(phones)} frames")

        except Exception as e:
            errors.append(f"Failed to validate array dimensions: {e}")

        # 4. 验证 FAISS 索引
        try:
            index = faiss.read_index(str(self.pool_dir / 'faiss.index'))
            logger.info(f"  ✓ FAISS index loaded: {index.ntotal} vectors")

            if hasattr(index, 'nprobe'):
                logger.info(f"    - nprobe: {index.nprobe}")
        except Exception as e:
            errors.append(f"Failed to load FAISS index: {e}")

        # 5. 验证元数据
        try:
            with open(self.pool_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            required_keys = ['total_frames', 'num_utterances', 'feature_dim']
            for key in required_keys:
                if key not in metadata:
                    warnings.append(f"Missing metadata key: {key}")

            logger.info(f"  ✓ Metadata valid")
        except Exception as e:
            errors.append(f"Failed to validate metadata: {e}")

        # 输出结果
        if errors:
            logger.error("\n[Validation] FAILED with errors:")
            for err in errors:
                logger.error(f"  ✗ {err}")
            raise RuntimeError("Pool validation failed")

        if warnings:
            logger.warning("\n[Validation] Completed with warnings:")
            for warn in warnings:
                logger.warning(f"  ⚠ {warn}")

        logger.info("\n[Validation] ✓ Pool integrity check passed!")


def run_pool_building(config: Dict) -> Dict:
    """Step 6 入口"""
    builder = TargetPoolBuilder(config)
    return builder.build()