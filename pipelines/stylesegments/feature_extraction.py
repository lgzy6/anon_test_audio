"""WavLM 特征提取 - 支持多层提取（优化版）"""

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
    utt_id: str
    speaker_id: str
    gender: str
    duration_sec: float
    num_frames: int
    h5_start_idx: int
    h5_end_idx: int


class FeatureExtractor:
    """WavLM 多层特征提取器"""

    def __init__(self, wavlm_ckpt: str, layers: List[int] = [6, 24],
                 device: str = "cuda", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        self.layers = layers

        logger.info(f"Loading WavLM, layers: {layers}")
        self.extractors = {
            layer: WavLMSSLExtractor(ckpt_path=wavlm_ckpt, layer=layer, device=device)
            for layer in layers
        }

        self.feature_dim = 1024
        self.hop_size = 320
        self.sample_rate = 16000

    def extract_dataset(self, dataset: LibriSpeechDataset, output_dir: str,
                       max_utterances: Optional[int] = None, save_interval: int = 1000,
                       resume: bool = True, sample_ratio: float = 1.0) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_utterances = len(dataset) if max_utterances is None else min(len(dataset), max_utterances)

        if sample_ratio < 1.0:
            sampled_indices = self._sample_indices(num_utterances, sample_ratio)
            logger.info(f"Sampling {len(sampled_indices)}/{num_utterances}")
        else:
            sampled_indices = list(range(num_utterances))

        h5_files = {layer: output_dir / f"layer_{layer}" / "features.h5" for layer in self.layers}
        for layer in self.layers:
            (output_dir / f"layer_{layer}").mkdir(exist_ok=True)

        meta_path = output_dir / "metadata.json"
        checkpoint_path = output_dir / "checkpoint.json"

        start_idx, metadata_list, current_frame_idx = 0, [], 0
        if resume and checkpoint_path.exists():
            start_idx, metadata_list, current_frame_idx = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming from {start_idx}/{num_utterances}")

        mode = 'a' if resume else 'w'
        h5_handles = {layer: h5py.File(h5_files[layer], mode) for layer in self.layers}
        features_datasets = {}

        for layer in self.layers:
            if 'features' not in h5_handles[layer]:
                features_datasets[layer] = h5_handles[layer].create_dataset(
                    'features', shape=(0, self.feature_dim), maxshape=(None, self.feature_dim),
                    dtype='float32', chunks=(1000, self.feature_dim))
            else:
                features_datasets[layer] = h5_handles[layer]['features']

        try:
            batch_wavs, batch_infos = [], []
            pbar = tqdm(sampled_indices[start_idx:], desc="Extracting", initial=start_idx, total=len(sampled_indices))

            for idx in pbar:
                try:
                    item = dataset[idx]
                    batch_wavs.append(item['waveform'])
                    batch_infos.append({
                        'utt_id': item['utt_id'], 'speaker_id': item['speaker_id'],
                        'gender': item.get('gender', 'unknown'),
                        'duration_sec': len(item['waveform']) / self.sample_rate
                    })

                    if len(batch_wavs) >= self.batch_size or idx == sampled_indices[-1]:
                        batch_features = self._extract_batch_multilayer(batch_wavs)

                        for i, info in enumerate(batch_infos):
                            num_frames = len(batch_features[self.layers[0]][i])
                            start_frame, end_frame = current_frame_idx, current_frame_idx + num_frames

                            for layer in self.layers:
                                features_datasets[layer].resize((end_frame, self.feature_dim))
                                features_datasets[layer][start_frame:end_frame] = batch_features[layer][i].cpu().numpy()

                            metadata_list.append(UtteranceMetadata(
                                utt_id=info['utt_id'], speaker_id=info['speaker_id'], gender=info['gender'],
                                duration_sec=info['duration_sec'], num_frames=num_frames,
                                h5_start_idx=start_frame, h5_end_idx=end_frame))
                            current_frame_idx = end_frame

                        batch_wavs, batch_infos = [], []
                        pbar.set_postfix({'frames': current_frame_idx, 'utts': len(metadata_list)})

                    if (idx + 1) % save_interval == 0:
                        self._save_checkpoint(checkpoint_path, idx + 1, metadata_list, current_frame_idx)
                        for h5 in h5_handles.values():
                            h5.flush()
                except Exception as e:
                    logger.error(f"Error {idx}: {e}")
        finally:
            for h5 in h5_handles.values():
                h5.close()

        metadata = {
            'total_frames': current_frame_idx, 'total_utterances': len(metadata_list),
            'feature_dim': self.feature_dim, 'layers': self.layers,
            'utterances': [asdict(m) for m in metadata_list]
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(f"Complete: {len(metadata_list)} utts, {current_frame_idx} frames")
        return metadata

    @torch.inference_mode()
    def _extract_batch_multilayer(self, waveforms: List[torch.Tensor]):
        max_len = max(len(w) for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len)
        lengths = []
        for i, w in enumerate(waveforms):
            padded[i, :len(w)] = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
            lengths.append(len(w) // self.hop_size)
        padded = padded.to(self.device)

        results = {}
        for layer in self.layers:
            features = self.extractors[layer](padded)
            results[layer] = [features[i, :lengths[i]] for i in range(len(lengths))]
        return results

    def _sample_indices(self, num_utterances: int, sample_ratio: float):
        import random
        random.seed(42)
        target_count = int(num_utterances * sample_ratio)
        step = num_utterances / target_count
        return [int(i * step) for i in range(target_count)]

    def _save_checkpoint(self, path: Path, idx: int, metadata_list: List, frame_idx: int):
        with open(path, 'w') as f:
            json.dump({'last_idx': idx, 'frame_idx': frame_idx,
                      'metadata': [asdict(m) for m in metadata_list]}, f)

    def _load_checkpoint(self, path: Path):
        with open(path, 'r') as f:
            ckpt = json.load(f)
        return ckpt['last_idx'], [UtteranceMetadata(**m) for m in ckpt['metadata']], ckpt['frame_idx']


def run_feature_extraction(config: Dict) -> Dict:
    train_split = config['offline'].get('train_split', 'train-other-500')
    dataset = LibriSpeechDataset(root=config['paths']['librispeech_root'], split=train_split)

    extractor = FeatureExtractor(
        wavlm_ckpt=config['paths']['wavlm_checkpoint'],
        layers=config['ssl'].get('layers', [6, 24]),
        device=config.get('device', 'cuda'),
        batch_size=config['offline'].get('batch_size', 8))

    split_name = train_split.replace('-', '_')
    output_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name
    feat_cfg = config['offline'].get('feature_extraction', {})

    return extractor.extract_dataset(
        dataset=dataset, output_dir=str(output_dir),
        max_utterances=feat_cfg.get('max_utterances'),
        save_interval=feat_cfg.get('save_interval', 1000),
        resume=True, sample_ratio=feat_cfg.get('sample_ratio', 1.0))

    def extract_dataset(self, dataset: LibriSpeechDataset, output_dir: str,
                       max_utterances: Optional[int] = None, save_interval: int = 1000,
                       resume: bool = True, sample_ratio: float = 1.0) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_utterances = len(dataset) if max_utterances is None else min(len(dataset), max_utterances)

        if sample_ratio < 1.0:
            sampled_indices = self._sample_indices(num_utterances, sample_ratio)
            logger.info(f"Sampling {len(sampled_indices)}/{num_utterances} utterances")
        else:
            sampled_indices = list(range(num_utterances))

        # 为每层创建独立的 HDF5 文件
        h5_files = {}
        for layer in self.layers:
            layer_dir = output_dir / f"layer_{layer}"
            layer_dir.mkdir(exist_ok=True)
            h5_files[layer] = layer_dir / "features.h5"

        meta_path = output_dir / "metadata.json"
        checkpoint_path = output_dir / "checkpoint.json"

        start_idx = 0
        metadata_list: List[UtteranceMetadata] = []
        current_frame_idx = 0

        if resume and checkpoint_path.exists():
            start_idx, metadata_list, current_frame_idx = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming from {start_idx}/{num_utterances}")

        # 打开所有层的 HDF5 文件
        mode = 'a' if resume else 'w'
        h5_handles = {}
        features_datasets = {}

        for layer in self.layers:
            h5_handles[layer] = h5py.File(h5_files[layer], mode)
            if 'features' not in h5_handles[layer]:
                features_datasets[layer] = h5_handles[layer].create_dataset(
                    'features', shape=(0, self.feature_dim), maxshape=(None, self.feature_dim),
                    dtype='float32', chunks=(1000, self.feature_dim)
                )
            else:
                features_datasets[layer] = h5_handles[layer]['features']

        try:
            batch_wavs = []
            batch_infos = []

            pbar = tqdm(sampled_indices[start_idx:], desc="Extracting",
                       initial=start_idx, total=len(sampled_indices))

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

                    if len(batch_wavs) >= self.batch_size or idx == sampled_indices[-1]:
                        # 提取所有层的特征
                        batch_features_all_layers = self._extract_batch_multilayer(batch_wavs)

                        for i, info in enumerate(batch_infos):
                            num_frames = len(batch_features_all_layers[self.layers[0]][i])
                            start_frame = current_frame_idx
                            end_frame = start_frame + num_frames

                            # 写入每层的特征
                            for layer in self.layers:
                                feats = batch_features_all_layers[layer][i]
                                features_datasets[layer].resize((end_frame, self.feature_dim))
                                features_datasets[layer][start_frame:end_frame] = feats.cpu().numpy()

                            meta = UtteranceMetadata(
                                utt_id=info['utt_id'], speaker_id=info['speaker_id'],
                                gender=info['gender'], duration_sec=info['duration_sec'],
                                num_frames=num_frames, h5_start_idx=start_frame, h5_end_idx=end_frame
                            )
                            metadata_list.append(meta)
                            current_frame_idx = end_frame

                        batch_wavs = []
                        batch_infos = []
                        pbar.set_postfix({'frames': current_frame_idx, 'utts': len(metadata_list)})

                    if (idx + 1) % save_interval == 0:
                        self._save_checkpoint(checkpoint_path, idx + 1, metadata_list, current_frame_idx)
                        for h5 in h5_handles.values():
                            h5.flush()

                except Exception as e:
                    logger.error(f"Error processing {idx}: {e}")
                    continue

        finally:
            for h5 in h5_handles.values():
                h5.close()

        metadata = {
            'total_frames': current_frame_idx,
            'total_utterances': len(metadata_list),
            'feature_dim': self.feature_dim,
            'layers': self.layers,
            'utterances': [asdict(m) for m in metadata_list],
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(f"Extraction complete: {len(metadata_list)} utterances, {current_frame_idx} frames")
        return metadata

    @torch.inference_mode()
    def _extract_batch_multilayer(self, waveforms: List[torch.Tensor]):
        max_len = max(len(w) for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len)
        lengths = []

        for i, w in enumerate(waveforms):
            padded[i, :len(w)] = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
            lengths.append(len(w) // self.hop_size)

        padded = padded.to(self.device)

        results = {}
        for layer in self.layers:
            features = self.extractors[layer](padded)
            results[layer] = [features[i, :lengths[i]] for i in range(len(lengths))]

        return results

    def _sample_indices(self, num_utterances: int, sample_ratio: float):
        import random
        random.seed(42)
        target_count = int(num_utterances * sample_ratio)
        step = num_utterances / target_count
        return [int(i * step) for i in range(target_count)]

    def _save_checkpoint(self, path: Path, idx: int, metadata_list: List, frame_idx: int):
        with open(path, 'w') as f:
            json.dump({'last_idx': idx, 'frame_idx': frame_idx,
                      'metadata': [asdict(m) for m in metadata_list]}, f)

    def _load_checkpoint(self, path: Path):
        with open(path, 'r') as f:
            ckpt = json.load(f)
        metadata_list = [UtteranceMetadata(**m) for m in ckpt['metadata']]
        return ckpt['last_idx'], metadata_list, ckpt['frame_idx']


def run_feature_extraction(config: Dict) -> Dict:
    """运行特征提取"""
    train_split = config['offline'].get('train_split', 'train-other-500')

    dataset = LibriSpeechDataset(
        root=config['paths']['librispeech_root'],
        split=train_split,
    )

    layers = config['ssl'].get('layers', [6, 24])

    extractor = FeatureExtractor(
        wavlm_ckpt=config['paths']['wavlm_checkpoint'],
        layers=layers,
        device=config.get('device', 'cuda'),
        batch_size=config['offline'].get('batch_size', 8),
    )

    split_name = train_split.replace('-', '_')
    output_dir = Path(config['paths']['cache_dir']) / 'features' / 'wavlm' / split_name

    feat_cfg = config['offline'].get('feature_extraction', {})

    return extractor.extract_dataset(
        dataset=dataset, output_dir=str(output_dir),
        max_utterances=feat_cfg.get('max_utterances'),
        save_interval=feat_cfg.get('save_interval', 1000),
        resume=True, sample_ratio=feat_cfg.get('sample_ratio', 1.0)
    )
