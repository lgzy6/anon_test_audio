# data/io/hdf5.py
"""HDF5 特征存储工具"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterator
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class HDF5Writer:
    """
    HDF5 特征写入器
    
    支持:
    - 流式写入 (自动扩展)
    - 多 dataset 写入
    - 元数据存储
    """
    
    def __init__(
        self,
        path: str,
        feature_dim: int,
        dtype: str = 'float32',
        chunk_size: int = 1000,
        compression: Optional[str] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.compression = compression
        
        self.h5f = h5py.File(self.path, 'w')
        
        # 创建主 dataset
        self.features = self.h5f.create_dataset(
            'features',
            shape=(0, feature_dim),
            maxshape=(None, feature_dim),
            dtype=dtype,
            chunks=(chunk_size, feature_dim),
            compression=compression,
        )
        
        self.current_idx = 0
        self._closed = False
    
    def write(self, feats: np.ndarray) -> Tuple[int, int]:
        """
        写入特征
        
        Args:
            feats: [N, D] 特征数组
        
        Returns:
            (start_idx, end_idx): 写入的索引范围
        """
        if self._closed:
            raise RuntimeError("Writer is closed")
        
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        
        n_frames = feats.shape[0]
        start = self.current_idx
        end = start + n_frames
        
        # 扩展 dataset
        self.features.resize((end, self.feature_dim))
        self.features[start:end] = feats.astype(self.dtype)
        
        self.current_idx = end
        
        return start, end
    
    def write_batch(self, feats_list: List[np.ndarray]) -> List[Tuple[int, int]]:
        """批量写入"""
        indices = []
        for feats in feats_list:
            idx = self.write(feats)
            indices.append(idx)
        return indices
    
    def create_auxiliary_dataset(
        self,
        name: str,
        shape: Tuple,
        maxshape: Optional[Tuple] = None,
        dtype: str = 'int32',
    ) -> h5py.Dataset:
        """创建辅助 dataset (如 phone labels, speaker ids 等)"""
        return self.h5f.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=True if maxshape else None,
        )
    
    def set_attribute(self, key: str, value):
        """设置文件属性"""
        self.h5f.attrs[key] = value
    
    def flush(self):
        """刷新到磁盘"""
        self.h5f.flush()
    
    def close(self):
        """关闭文件"""
        if not self._closed:
            self.h5f.close()
            self._closed = True
            logger.info(f"HDF5Writer closed: {self.path}, {self.current_idx} frames")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class HDF5Reader:
    """
    HDF5 特征读取器
    
    支持:
    - 随机访问
    - 流式迭代
    - 索引访问
    """
    
    def __init__(self, path: str, mode: str = 'r'):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        self.h5f = h5py.File(self.path, mode)
        self.features = self.h5f['features']
        self._closed = False
    
    def read(self, start: int, end: int) -> np.ndarray:
        """读取指定范围"""
        return self.features[start:end]
    
    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        """按索引读取 (支持 fancy indexing)"""
        return self.features[indices]
    
    def read_all(self) -> np.ndarray:
        """读取全部"""
        return self.features[:]
    
    def iterate(self, batch_size: int = 10000) -> Iterator[Tuple[int, np.ndarray]]:
        """流式迭代"""
        total = len(self.features)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield start, self.features[start:end]
    
    def get_dataset(self, name: str) -> Optional[h5py.Dataset]:
        """获取其他 dataset"""
        return self.h5f.get(name)
    
    def get_attribute(self, key: str, default=None):
        """获取属性"""
        return self.h5f.attrs.get(key, default)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.features.shape
    
    @property
    def num_frames(self) -> int:
        return self.features.shape[0]
    
    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]
    
    def close(self):
        if not self._closed:
            self.h5f.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        return self.features[idx]


# =============================================================================
# 便捷函数
# =============================================================================

@contextmanager
def open_hdf5(path: str, mode: str = 'r'):
    """上下文管理器方式打开 HDF5"""
    if mode == 'w':
        raise ValueError("Use HDF5Writer for writing")
    
    reader = HDF5Reader(path, mode)
    try:
        yield reader
    finally:
        reader.close()


def concatenate_hdf5_files(
    input_paths: List[str],
    output_path: str,
    feature_dim: int,
) -> int:
    """合并多个 HDF5 文件"""
    total_frames = 0
    
    with HDF5Writer(output_path, feature_dim) as writer:
        for path in input_paths:
            with HDF5Reader(path) as reader:
                for start, batch in reader.iterate():
                    writer.write(batch)
                total_frames += reader.num_frames
    
    logger.info(f"Concatenated {len(input_paths)} files, total {total_frames} frames")
    return total_frames


def verify_hdf5(path: str) -> Dict:
    """验证 HDF5 文件完整性"""
    try:
        with HDF5Reader(path) as reader:
            info = {
                'path': str(path),
                'shape': reader.shape,
                'num_frames': reader.num_frames,
                'feature_dim': reader.feature_dim,
                'datasets': list(reader.h5f.keys()),
                'attributes': dict(reader.h5f.attrs),
                'valid': True,
            }
            
            # 检查是否有 NaN
            sample = reader.read(0, min(1000, reader.num_frames))
            info['has_nan'] = np.isnan(sample).any()
            info['has_inf'] = np.isinf(sample).any()
            
            return info
    except Exception as e:
        return {
            'path': str(path),
            'valid': False,
            'error': str(e),
        }