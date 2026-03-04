# pipelines/offline/feature_extraction.py
"""Step 1: WavLM 特征提取 (支持断点续传)"""

import torch
import h5py
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import numpy as np

from models.ssl.wrappers import WavLMSSLExtractor
from data.datasets.librispeech import LibriSpeechDataset

logger = logging.getLogger(__name__)


@dataclass
class UtteranceMetadata:
    """单个 utterance 的元数据"""
    utt_id: str
    speaker_id: str
    gender: str
    duration_sec: float
    num_frames: int
    h5_start_idx: int
    h5_end_idx: int


class FeatureExtractor:
    """
    WavLM 特征批量提取器 (支持断点续传)
    
    改进:
    1. 支持断点续传，中断后可恢复
    2. 定期保存 checkpoint
    3. 更好的错误处理
    """
    
    def __init__(
        self,
        wavlm_ckpt: str,
        layer: int = 15,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.device = device
        self.batch_size = batch_size
        self.layer = layer
        
        # 加载 WavLM
        logger.info(f"Loading WavLM from {wavlm_ckpt}...")
        self.extractor = WavLMSSLExtractor(
            ckpt_path=wavlm_ckpt,
            layer=layer,
            device=device,
        )
        
        self.feature_dim = 1024
        self.hop_size = 320
        self.sample_rate = 16000
    
    def extract_dataset(
        self,
        dataset: LibriSpeechDataset,
        output_dir: str,
        max_utterances: Optional[int] = None,
        save_interval: int = 1000,
        resume: bool = True,
    ) -> Dict:
        """
        提取整个数据集的特征
        
        Args:
            dataset: LibriSpeech 数据集
            output_dir: 输出目录
            max_utterances: 最大处理数量
            save_interval: 保存 checkpoint 的间隔
            resume: 是否从断点恢复
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        h5_path = output_dir / "features.h5"
        meta_path = output_dir / "metadata.json"
        checkpoint_path = output_dir / "checkpoint.json"
        
        num_utterances = len(dataset) if max_utterances is None else min(len(dataset), max_utterances)
        
        # 尝试恢复
        start_idx = 0
        metadata_list: List[UtteranceMetadata] = []
        current_frame_idx = 0
        
        if resume and checkpoint_path.exists():
            start_idx, metadata_list, current_frame_idx = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming from checkpoint: {start_idx}/{num_utterances}")
        
        # 打开/创建 HDF5
        mode = 'a' if resume and h5_path.exists() else 'w'
        
        with h5py.File(h5_path, mode) as h5f:
            # 创建或获取 dataset
            if 'features' not in h5f:
                features_ds = h5f.create_dataset(
                    'features',
                    shape=(0, self.feature_dim),
                    maxshape=(None, self.feature_dim),
                    dtype='float32',
                    chunks=(1000, self.feature_dim),
                )
            else:
                features_ds = h5f['features']
                # 验证现有数据
                if current_frame_idx > 0 and features_ds.shape[0] != current_frame_idx:
                    logger.warning(f"HDF5 size mismatch: {features_ds.shape[0]} vs {current_frame_idx}")
                    current_frame_idx = features_ds.shape[0]
            
            # 批量处理
            batch_wavs = []
            batch_infos = []
            
            pbar = tqdm(range(start_idx, num_utterances), desc="Extracting features", initial=start_idx, total=num_utterances)
            
            for idx in pbar:
                try:
                    item = dataset[idx]
                    waveform = item['waveform']
                    
                    batch_wavs.append(waveform)
                    batch_infos.append({
                        'utt_id': item['utt_id'],
                        'speaker_id': item['speaker_id'],
                        'gender': item.get('gender', 'unknown'),
                        'duration_sec': len(waveform) / self.sample_rate,
                    })
                    
                    # 批次满了，处理
                    if len(batch_wavs) >= self.batch_size or idx == num_utterances - 1:
                        batch_features, batch_lengths = self._extract_batch(batch_wavs)
                        
                        for i, (feats, info) in enumerate(zip(batch_features, batch_infos)):
                            num_frames = len(feats)
                            start_frame = current_frame_idx
                            end_frame = start_frame + num_frames
                            
                            # 扩展并写入
                            features_ds.resize((end_frame, self.feature_dim))
                            features_ds[start_frame:end_frame] = feats.cpu().numpy()
                            
                            meta = UtteranceMetadata(
                                utt_id=info['utt_id'],
                                speaker_id=info['speaker_id'],
                                gender=info['gender'],
                                duration_sec=info['duration_sec'],
                                num_frames=num_frames,
                                h5_start_idx=start_frame,
                                h5_end_idx=end_frame,
                            )
                            metadata_list.append(meta)
                            current_frame_idx = end_frame
                        
                        batch_wavs = []
                        batch_infos = []
                        
                        pbar.set_postfix({'frames': current_frame_idx, 'utts': len(metadata_list)})
                    
                    # 定期保存 checkpoint
                    if (idx + 1) % save_interval == 0:
                        self._save_checkpoint(checkpoint_path, idx + 1, metadata_list, current_frame_idx)
                        h5f.flush()
                
                except Exception as e:
                    logger.error(f"Error processing {idx}: {e}")
                    continue
        
        # 保存最终元数据
        metadata = {
            'total_frames': current_frame_idx,
            'total_utterances': len(metadata_list),
            'feature_dim': self.feature_dim,
            'layer': self.layer,
            'utterances': [asdict(m) for m in metadata_list],
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 删除 checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        logger.info(f"Feature extraction complete:")
        logger.info(f"  - Total utterances: {len(metadata_list)}")
        logger.info(f"  - Total frames: {current_frame_idx}")
        logger.info(f"  - Features saved to: {h5_path}")
        
        return metadata
    
    def _save_checkpoint(self, path: Path, idx: int, metadata_list: List, frame_idx: int):
        """保存断点"""
        with open(path, 'w') as f:
            json.dump({
                'last_idx': idx,
                'frame_idx': frame_idx,
                'metadata': [asdict(m) for m in metadata_list],
            }, f)
    
    def _load_checkpoint(self, path: Path):
        """加载断点"""
        with open(path, 'r') as f:
            ckpt = json.load(f)
        
        metadata_list = [
            UtteranceMetadata(**m) for m in ckpt['metadata']
        ]
        return ckpt['last_idx'], metadata_list, ckpt['frame_idx']
    
    @torch.inference_mode()
    def _extract_batch(self, waveforms: List[torch.Tensor]):
        """提取一个批次的特征"""
        max_len = max(len(w) for w in waveforms)
        
        padded = torch.zeros(len(waveforms), max_len)
        lengths = []
        for i, w in enumerate(waveforms):
            padded[i, :len(w)] = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
            lengths.append(len(w) // self.hop_size)
        
        padded = padded.to(self.device)
        features = self.extractor(padded)
        
        results = []
        for i, length in enumerate(lengths):
            results.append(features[i, :length])
        
        return results, lengths


def run_feature_extraction(config: Dict) -> Dict:
    """运行特征提取"""
    from data.datasets.librispeech import LibriSpeechDataset

    train_split = config['offline'].get('train_split', 'train-other-500')

    dataset = LibriSpeechDataset(
        root=config['paths']['librispeech_root'],
        split=train_split,
    )

    extractor = FeatureExtractor(
        wavlm_ckpt=config['paths']['wavlm_checkpoint'],
        layer=config['ssl']['layer'],
        device=config.get('device', 'cuda'),
        batch_size=config['offline'].get('batch_size', 8),
    )

    # 根据数据集名称创建子目录
    split_name = train_split.replace('-', '_')  # train-clean-360 -> train_clean_360
    output_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name

    feat_cfg = config['offline'].get('feature_extraction', {})

    metadata = extractor.extract_dataset(
        dataset=dataset,
        output_dir=str(output_dir),
        max_utterances=feat_cfg.get('max_utterances'),
        save_interval=feat_cfg.get('save_interval', 1000),
        resume=True,
    )

    return metadata