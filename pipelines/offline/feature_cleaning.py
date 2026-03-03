# pipelines/offline/feature_cleaning.py

"""Step 3: 去说话人特征生成"""

import torch
import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import shutil


class FeatureCleaner:
    """
    特征清洗器
    
    应用 Eta-WavLM 投影去除说话人成分
    """
    
    def __init__(
        self,
        subspace_path: str,
        batch_size: int = 10000,  # 每批处理的帧数
    ):
        # 加载子空间
        ckpt = torch.load(subspace_path, map_location='cpu')
        self.U_s = ckpt['U_s'].numpy()  # [D, D_s]
        
        # 预计算投影矩阵
        # P_orth = I - U_s @ U_s^T
        self.P_orth = np.eye(self.U_s.shape[0]) - self.U_s @ self.U_s.T
        
        self.batch_size = batch_size
        self.feature_dim = self.U_s.shape[0]
    
    def clean_features(
        self,
        input_h5_path: str,
        input_meta_path: str,
        output_dir: str,
    ):
        """
        清洗所有特征
        
        Args:
            input_h5_path: 输入 HDF5 路径
            input_meta_path: 输入元数据路径
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_h5_path = output_dir / "features.h5"
        output_meta_path = output_dir / "metadata.json"
        
        # 加载元数据
        with open(input_meta_path, 'r') as f:
            metadata = json.load(f)
        
        total_frames = metadata['total_frames']
        
        print(f"Cleaning {total_frames} frames...")
        
        # 打开输入和输出 HDF5
        with h5py.File(input_h5_path, 'r') as h5_in, \
             h5py.File(output_h5_path, 'w') as h5_out:
            
            features_in = h5_in['features']
            
            # 创建输出 dataset
            features_out = h5_out. create_dataset(
                'features',
                shape=(total_frames, self.feature_dim),
                dtype='float32',
                chunks=(1000, self.feature_dim),
            )
            
            # 批量处理
            for start in tqdm(range(0, total_frames, self.batch_size), desc="Cleaning"):
                end = min(start + self.batch_size, total_frames)
                
                # 读取原始特征
                h = features_in[start:end]  # [batch, D]
                
                # 应用投影
                h_clean = h @ self.P_orth  # [batch, D]
                
                # 写入
                features_out[start:end] = h_clean
        
        # 复制元数据 (添加 cleaned 标记)
        metadata['cleaned'] = True
        metadata['subspace_dim'] = self.U_s. shape[1]
        
        with open(output_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Feature cleaning complete:")
        print(f"  - Output:  {output_h5_path}")
        print(f"  - Metadata: {output_meta_path}")


def run_feature_cleaning(config: Dict):
    """运行特征清洗"""
    
    cache_dir = Path(config['paths']['cache_dir'])
    checkpoint_dir = Path(config['paths']['checkpoints_dir'])
    
    # 输入
    input_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'
    input_meta = cache_dir / 'features' / 'wavlm' / 'metadata.json'
    subspace_path = checkpoint_dir / 'speaker_subspace.pt'
    
    # 输出
    output_dir = cache_dir / 'features' / 'cleaned'
    
    # 清洗
    cleaner = FeatureCleaner(
        subspace_path=str(subspace_path),
        batch_size=config['offline'].get('cleaning_batch_size', 10000),
    )
    
    cleaner.clean_features(
        input_h5_path=str(input_h5),
        input_meta_path=str(input_meta),
        output_dir=str(output_dir),
    )