#!/usr/bin/env python3
"""
IEMOCAP 数据集 WavLM 特征提取脚本
"""

import sys
import torch
import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets.iemocap import IEMOCAPDataset
from models.ssl.wrappers import WavLMSSLExtractor


@dataclass
class UtteranceMetadata:
    """单个 utterance 的元数据"""
    utt_id: str
    speaker_id: str
    gender: str
    session_id: int
    duration_sec: float
    num_frames: int
    h5_start_idx: int
    h5_end_idx: int


class IEMOCAPFeatureExtractor:
    """IEMOCAP WavLM 特征提取器"""

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
        self.feature_dim = 1024
        self.hop_size = 320
        self.sample_rate = 16000

        print(f"Loading WavLM from {wavlm_ckpt}...")
        self.extractor = WavLMSSLExtractor(
            ckpt_path=wavlm_ckpt,
            layer=layer,
            device=device,
        )

    @torch.inference_mode()
    def _extract_batch(self, waveforms):
        """提取一个批次的特征"""
        max_len = max(len(w) for w in waveforms)

        padded = torch.zeros(len(waveforms), max_len)
        lengths = []
        for i, w in enumerate(waveforms):
            if isinstance(w, torch.Tensor):
                padded[i, :len(w)] = w
            else:
                padded[i, :len(w)] = torch.from_numpy(w)
            lengths.append(len(w) // self.hop_size)

        padded = padded.to(self.device)
        features = self.extractor(padded)

        results = []
        for i, length in enumerate(lengths):
            results.append(features[i, :length])

        return results, lengths

    def extract_dataset(self, dataset: IEMOCAPDataset, output_dir: str):
        """提取整个数据集的特征"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = output_dir / "features.h5"
        meta_path = output_dir / "metadata.json"

        num_utterances = len(dataset)
        metadata_list = []
        current_frame_idx = 0

        with h5py.File(h5_path, 'w') as h5f:
            features_ds = h5f.create_dataset(
                'features',
                shape=(0, self.feature_dim),
                maxshape=(None, self.feature_dim),
                dtype='float32',
                chunks=(1000, self.feature_dim),
            )

            batch_wavs = []
            batch_infos = []

            pbar = tqdm(range(num_utterances), desc="Extracting features")

            for idx in pbar:
                item = dataset[idx]
                waveform = item['waveform']

                batch_wavs.append(waveform)
                batch_infos.append({
                    'utt_id': item['utt_id'],
                    'speaker_id': item['speaker_id'],
                    'gender': item['gender'],
                    'session_id': item['session_id'],
                    'duration_sec': len(waveform) / self.sample_rate,
                })

                # 批次满了，处理
                if len(batch_wavs) >= self.batch_size or idx == num_utterances - 1:
                    batch_features, _ = self._extract_batch(batch_wavs)

                    for feats, info in zip(batch_features, batch_infos):
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
                            session_id=info['session_id'],
                            duration_sec=info['duration_sec'],
                            num_frames=num_frames,
                            h5_start_idx=start_frame,
                            h5_end_idx=end_frame,
                        )
                        metadata_list.append(meta)
                        current_frame_idx = end_frame

                    batch_wavs = []
                    batch_infos = []
                    pbar.set_postfix({'frames': current_frame_idx})

        # 保存元数据
        metadata = {
            'total_frames': current_frame_idx,
            'total_utterances': len(metadata_list),
            'feature_dim': self.feature_dim,
            'layer': self.layer,
            'utterances': [asdict(m) for m in metadata_list],
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nFeature extraction complete:")
        print(f"  Total utterances: {len(metadata_list)}")
        print(f"  Total frames: {current_frame_idx}")
        print(f"  Saved to: {output_dir}")

        return metadata


def main():
    """主函数"""
    config = {
        'iemocap_root': '/root/autodl-tmp/datasets/IEMOCAP_full_release/IEMOCAP_full_release',
        'wavlm_path': 'checkpoints/WavLM-Large.pt',
        'output_dir': 'cache/features/iemocap',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'layer': 15,
        'batch_size': 8,
    }

    print("=" * 60)
    print("IEMOCAP WavLM Feature Extraction")
    print("=" * 60)

    # 加载数据集
    print("\n[1/2] Loading IEMOCAP dataset...")
    dataset = IEMOCAPDataset(
        root=config['iemocap_root'],
        sessions=[1, 2, 3, 4, 5],
        sample_rate=16000,
    )

    # 提取特征
    print("\n[2/2] Extracting features...")
    extractor = IEMOCAPFeatureExtractor(
        wavlm_ckpt=config['wavlm_path'],
        layer=config['layer'],
        device=config['device'],
        batch_size=config['batch_size'],
    )

    extractor.extract_dataset(dataset, config['output_dir'])

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
