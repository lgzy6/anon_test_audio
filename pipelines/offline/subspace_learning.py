# pipelines/offline/subspace_learning.py
"""Step 2: 说话人子空间学习 (内存优化版)"""

import torch
import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """
    Welford 在线均值/方差计算器
    避免一次性加载所有数据到内存
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
    
    def update(self, x: np.ndarray):
        """更新统计量，x 可以是 [D] 或 [N, D]"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2
    
    def finalize(self):
        """返回均值和方差"""
        if self.count < 2:
            return self.mean, np.zeros(self.dim)
        return self.mean, self.M2 / (self.count - 1)


class SubspaceLearner:
    """
    说话人子空间学习器 (内存优化版)
    
    改进:
    1. 使用 Welford 算法计算在线均值，避免 OOM
    2. 支持限制每说话人采样帧数
    3. 添加更多质量指标
    """
    
    def __init__(
        self,
        subspace_dim: int = 64,
        min_utterances_per_speaker: int = 5,
        max_frames_per_speaker: int = 50000,
        random_state: int = 42,
    ):
        self.subspace_dim = subspace_dim
        self.min_utterances = min_utterances_per_speaker
        self.max_frames = max_frames_per_speaker
        self.rng = np.random.RandomState(random_state)
        
        self.U_s = None
        self.explained_variance_ratio = None
        self.singular_values = None
        self.global_mean = None
        self.num_speakers = 0
    
    def learn(
        self,
        features_h5_path: str,
        metadata_path: str,
    ) -> np.ndarray:
        """
        学习说话人子空间
        
        Args:
            features_h5_path: HDF5 特征文件路径
            metadata_path: 元数据 JSON 路径
        
        Returns:
            U_s: [D, D_s] 说话人子空间正交基
        """
        # 加载元数据
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_dim = metadata['feature_dim']
        utterances = metadata['utterances']
        
        # 按说话人分组
        speaker_utts = defaultdict(list)
        for utt in utterances:
            speaker_utts[utt['speaker_id']].append(utt)
        
        logger.info(f"Total speakers: {len(speaker_utts)}")
        
        # 过滤语音数量不足的说话人
        valid_speakers = {
            spk: utts for spk, utts in speaker_utts.items()
            if len(utts) >= self.min_utterances
        }
        logger.info(f"Valid speakers (>= {self.min_utterances} utts): {len(valid_speakers)}")
        self.num_speakers = len(valid_speakers)
        
        # 使用 Welford 算法计算每个说话人的在线均值
        logger.info("Computing speaker centroids (memory-efficient)...")
        speaker_centroids = self._compute_centroids_welford(
            features_h5_path, valid_speakers, feature_dim
        )
        
        # 构建 speaker direction 矩阵
        speaker_ids = list(speaker_centroids.keys())
        directions = np.stack([speaker_centroids[spk] for spk in speaker_ids])
        logger.info(f"Speaker directions shape: {directions.shape}")
        
        # 中心化
        self.global_mean = directions.mean(axis=0)
        directions_centered = directions - self.global_mean
        
        # PCA (使用 SVD 获取更多信息)
        logger.info("Running PCA via SVD...")
        U, S, Vt = np.linalg.svd(directions_centered, full_matrices=False)
        
        # 取前 subspace_dim 个主成分
        self.U_s = Vt[:self.subspace_dim].T  # [D, D_s]
        self.singular_values = S[:self.subspace_dim]
        
        # 计算解释方差比
        total_var = (S ** 2).sum()
        explained_var = (self.singular_values ** 2).sum()
        self.explained_variance_ratio = explained_var / total_var if total_var > 0 else 0
        
        self._print_quality_metrics()
        
        return self.U_s
    
    def _compute_centroids_welford(
        self,
        features_h5_path: str,
        valid_speakers: Dict,
        feature_dim: int,
    ) -> Dict[str, np.ndarray]:
        """使用 Welford 算法内存高效计算说话人 centroid"""
        speaker_centroids = {}
        
        with h5py.File(features_h5_path, 'r') as h5f:
            features = h5f['features']
            
            for spk_id, utts in tqdm(valid_speakers.items(), desc="Computing centroids"):
                accumulator = WelfordAccumulator(feature_dim)
                
                # 计算该说话人总帧数
                total_frames = sum(u['h5_end_idx'] - u['h5_start_idx'] for u in utts)
                
                # 如果帧数超过限制，进行采样
                if total_frames > self.max_frames:
                    sample_ratio = self.max_frames / total_frames
                else:
                    sample_ratio = 1.0
                
                frames_processed = 0
                for utt in utts:
                    s, e = utt['h5_start_idx'], utt['h5_end_idx']
                    utt_len = e - s
                    
                    if sample_ratio < 1.0:
                        # 随机采样
                        n_sample = max(1, int(utt_len * sample_ratio))
                        indices = self.rng.choice(utt_len, n_sample, replace=False)
                        indices.sort()
                        frames = features[s:e][indices]
                    else:
                        frames = features[s:e]
                    
                    accumulator.update(frames)
                    frames_processed += len(frames)
                    
                    # 防止超过限制
                    if frames_processed >= self.max_frames:
                        break
                
                mean, _ = accumulator.finalize()
                speaker_centroids[spk_id] = mean.astype(np.float32)
        
        return speaker_centroids
    
    def _print_quality_metrics(self):
        """打印子空间质量指标"""
        logger.info("=" * 50)
        logger.info("Subspace Learning Complete:")
        logger.info(f"  - Number of speakers: {self.num_speakers}")
        logger.info(f"  - Subspace dimension: {self.subspace_dim}")
        logger.info(f"  - U_s shape: {self.U_s.shape}")
        logger.info(f"  - Explained variance ratio: {self.explained_variance_ratio:.2%}")
        
        if self.singular_values is not None:
            # 打印各主成分贡献
            total = (self.singular_values ** 2).sum()
            cumsum = 0
            for i, sv in enumerate(self.singular_values[:10]):  # 只打印前10个
                var_ratio = (sv ** 2) / total
                cumsum += var_ratio
                logger.info(f"    PC{i+1}: {var_ratio:.2%} (累计: {cumsum:.2%})")
        
        logger.info("=" * 50)
    
    def project_out_speaker(self, features: np.ndarray) -> np.ndarray:
        """
        从特征中投影去除说话人成分
        
        Args:
            features: [T, D] 或 [B, T, D]
        Returns:
            cleaned: 去说话人特征
        """
        if self.U_s is None:
            raise ValueError("Subspace not learned yet")
        
        # P_orth = I - U_s @ U_s^T
        P_orth = np.eye(self.U_s.shape[0]) - self.U_s @ self.U_s.T
        
        original_shape = features.shape
        if features.ndim == 3:
            B, T, D = features.shape
            features = features.reshape(-1, D)
        
        cleaned = features @ P_orth
        
        if len(original_shape) == 3:
            cleaned = cleaned.reshape(original_shape)
        
        return cleaned
    
    def save(self, path: str):
        """保存子空间"""
        save_dict = {
            'U_s': torch.from_numpy(self.U_s).float(),
            'subspace_dim': self.subspace_dim,
            'explained_variance_ratio': float(self.explained_variance_ratio),
            'num_speakers': self.num_speakers,
        }
        
        if self.singular_values is not None:
            save_dict['singular_values'] = torch.from_numpy(self.singular_values).float()
        if self.global_mean is not None:
            save_dict['global_mean'] = torch.from_numpy(self.global_mean).float()
        
        torch.save(save_dict, path)
        logger.info(f"Speaker subspace saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SubspaceLearner':
        """加载子空间"""
        ckpt = torch.load(path, map_location='cpu')
        learner = cls(subspace_dim=ckpt['subspace_dim'])
        learner.U_s = ckpt['U_s'].numpy()
        learner.explained_variance_ratio = ckpt['explained_variance_ratio']
        learner.num_speakers = ckpt.get('num_speakers', 0)
        if 'singular_values' in ckpt:
            learner.singular_values = ckpt['singular_values'].numpy()
        if 'global_mean' in ckpt:
            learner.global_mean = ckpt['global_mean'].numpy()
        return learner


def run_subspace_learning(config: Dict) -> SubspaceLearner:
    """运行子空间学习"""
    cache_dir = Path(config['paths']['cache_dir'])
    checkpoint_dir = Path(config['paths']['checkpoints_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 输入路径
    features_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'
    metadata_json = cache_dir / 'features' / 'wavlm' / 'metadata.json'
    
    # 输出路径
    output_path = checkpoint_dir / 'speaker_subspace.pt'
    
    # 获取配置参数
    eta_cfg = config.get('eta_wavlm', {})
    offline_cfg = config.get('offline', {})
    subspace_cfg = offline_cfg.get('subspace_learning', {})
    
    # 学习子空间
    learner = SubspaceLearner(
        subspace_dim=eta_cfg.get('speaker_subspace_dim', 64),
        min_utterances_per_speaker=subspace_cfg.get('min_utterances_per_speaker', 5),
        max_frames_per_speaker=subspace_cfg.get('max_frames_per_speaker', 50000),
    )
    
    learner.learn(
        features_h5_path=str(features_h5),
        metadata_path=str(metadata_json),
    )
    
    learner.save(str(output_path))
    
    return learner