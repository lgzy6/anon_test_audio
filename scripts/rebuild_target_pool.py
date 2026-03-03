#!/usr/bin/env python3
"""
重建 Target Pool - 修复 phones/symbols 缺失问题
"""

import sys
import json
import h5py
import torch
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.samm.codebook import SAMMCodebook


class TargetPoolRebuilder:
    """重建 Target Pool"""

    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phone_to_idx = {p: i for i, p in enumerate(self.PHONE_LIST)}

    def build(self):
        """执行重建"""
        print("=" * 70)
        print("重建 Target Pool")
        print("=" * 70)

        cache_dir = Path(self.config['cache_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载数据
        print("\n[1/5] 加载数据...")
        features_path = cache_dir / "features.h5"
        metadata_path = cache_dir / "metadata.json"
        phones_path = cache_dir / "phone_predictions.h5"

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 2. 加载 Codebook (用于生成 symbols)
        print("\n[2/5] 加载 Codebook...")
        codebook = SAMMCodebook.load(
            self.config['codebook_path'],
            device=self.device
        )
        print(f"  Codebook size: {codebook.K}")

        # 3. 构建说话人索引
        print("\n[3/5] 构建说话人索引...")
        speaker_index = self._build_speaker_index(metadata)
        print(f"  说话人数量: {len(speaker_index)}")

        # 4. 构建 Prototypes
        print("\n[4/5] 构建 Prototypes...")
        pool_data = self._build_prototypes(
            features_path, phones_path, metadata,
            speaker_index, codebook
        )

        # 5. 保存
        print("\n[5/5] 保存 Pool...")
        self._save_pool(output_dir, pool_data, metadata)

        print("\n" + "=" * 70)
        print("完成!")
        print("=" * 70)

    def _build_speaker_index(self, metadata):
        """构建说话人索引"""
        speaker_index = defaultdict(lambda: {
            'utterances': [], 'gender': 'M',
            'frame_ranges': [], 'total_frames': 0
        })

        for utt in metadata['utterances']:
            spk_id = utt.get('speaker_id', utt['utt_id'].split('-')[0])
            speaker_index[spk_id]['utterances'].append(utt)
            speaker_index[spk_id]['gender'] = utt.get('gender', 'M')
            speaker_index[spk_id]['frame_ranges'].append(
                (utt['h5_start_idx'], utt['h5_end_idx'])
            )
            speaker_index[spk_id]['total_frames'] += (
                utt['h5_end_idx'] - utt['h5_start_idx']
            )

        # 过滤
        min_frames = self.config.get('min_frames_per_speaker', 100)
        return {
            spk: data for spk, data in speaker_index.items()
            if data['total_frames'] >= min_frames
        }

    def _build_prototypes(self, features_path, phones_path,
                          metadata, speaker_index, codebook):
        """构建带有正确 phones/symbols 的 prototypes"""
        n_proto = self.config.get('n_prototypes_per_speaker', 50)

        all_prototypes = []
        all_phones = []
        all_symbols = []
        all_genders = []
        all_speaker_ids = []

        with h5py.File(features_path, 'r') as f_feat, \
             h5py.File(phones_path, 'r') as f_phone:

            feats = f_feat['features']

            for spk_id, spk_data in tqdm(
                speaker_index.items(), desc="Building prototypes"
            ):
                spk_frames = []
                spk_phones = []

                for utt, (start, end) in zip(
                    spk_data['utterances'], spk_data['frame_ranges']
                ):
                    utt_id = utt['utt_id']
                    utt_frames = feats[start:end][:]
                    spk_frames.append(utt_frames)

                    # 获取 phone 预测
                    if utt_id in f_phone:
                        phones = f_phone[utt_id][:]
                        # 对齐长度
                        if len(phones) < len(utt_frames):
                            phones = np.pad(
                                phones, (0, len(utt_frames) - len(phones)),
                                constant_values=self.phone_to_idx['SIL']
                            )
                        else:
                            phones = phones[:len(utt_frames)]
                        spk_phones.append(phones)
                    else:
                        spk_phones.append(
                            np.full(len(utt_frames), self.phone_to_idx['SIL'])
                        )

                spk_frames = np.vstack(spk_frames).astype(np.float32)
                spk_phones = np.concatenate(spk_phones)

                # 生成 symbols
                with torch.no_grad():
                    spk_symbols = codebook.encode(
                        torch.from_numpy(spk_frames).to(self.device)
                    ).cpu().numpy()

                # 提取 prototypes (按音素分层)
                protos, proto_phones, proto_symbols = self._extract_prototypes(
                    spk_frames, spk_phones, spk_symbols, n_proto
                )

                all_prototypes.append(protos)
                all_phones.extend(proto_phones)
                all_symbols.extend(proto_symbols)
                all_genders.extend(
                    [0 if spk_data['gender'].upper() == 'M' else 1] * len(protos)
                )
                all_speaker_ids.extend([spk_id] * len(protos))

        return {
            'features': np.vstack(all_prototypes).astype(np.float32),
            'phones': np.array(all_phones, dtype=np.int64),
            'symbols': np.array(all_symbols, dtype=np.int64),
            'genders': np.array(all_genders, dtype=np.int64),
            'speaker_ids': all_speaker_ids,
        }

    def _extract_prototypes(self, frames, phones, symbols, n_proto):
        """按音素分层提取 prototypes"""
        unique_phones = np.unique(phones[phones >= 0])

        if len(unique_phones) == 0 or len(frames) <= n_proto:
            # Fallback: 直接返回
            indices = np.random.choice(
                len(frames), min(n_proto, len(frames)), replace=False
            )
            return frames[indices], phones[indices].tolist(), symbols[indices].tolist()

        # 按音素分配 prototype 数量
        phone_counts = {p: (phones == p).sum() for p in unique_phones}
        total = sum(phone_counts.values())

        proto_indices = []
        for phone in unique_phones:
            mask = phones == phone
            phone_frames = np.where(mask)[0]

            # 按比例分配
            n_alloc = max(1, int(n_proto * phone_counts[phone] / total))
            n_alloc = min(n_alloc, len(phone_frames))

            if n_alloc > 0:
                if len(phone_frames) <= n_alloc:
                    selected = phone_frames
                else:
                    # K-Means 选择代表性帧
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_alloc, n_init=3, random_state=42
                    )
                    kmeans.fit(frames[phone_frames])
                    # 找最近的实际帧
                    centers = kmeans.cluster_centers_
                    for c in centers:
                        dists = ((frames[phone_frames] - c) ** 2).sum(axis=1)
                        nearest = phone_frames[dists.argmin()]
                        if nearest not in proto_indices:
                            proto_indices.append(nearest)

        # 补足数量
        if len(proto_indices) < n_proto:
            remaining = set(range(len(frames))) - set(proto_indices)
            extra = np.random.choice(
                list(remaining),
                min(n_proto - len(proto_indices), len(remaining)),
                replace=False
            )
            proto_indices.extend(extra)

        proto_indices = np.array(proto_indices[:n_proto])

        return (
            frames[proto_indices],
            phones[proto_indices].tolist(),
            symbols[proto_indices].tolist()
        )

    def _save_pool(self, output_dir, pool_data, metadata):
        """保存 Pool"""
        # 保存特征
        np.save(output_dir / "features.npy", pool_data['features'])
        np.save(output_dir / "phones.npy", pool_data['phones'])
        np.save(output_dir / "symbols.npy", pool_data['symbols'])
        np.save(output_dir / "genders.npy", pool_data['genders'])

        # 构建 FAISS 索引
        print("  构建 FAISS 索引...")
        features = pool_data['features'].copy()
        faiss.normalize_L2(features)

        dim = features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(features)
        faiss.write_index(index, str(output_dir / "faiss.index"))

        # 保存元数据
        pool_metadata = {
            'n_prototypes': len(pool_data['features']),
            'feature_dim': pool_data['features'].shape[1],
            'n_speakers': len(set(pool_data['speaker_ids'])),
            'phones_valid': True,
            'symbols_valid': True,
            'phone_to_idx': self.phone_to_idx,
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(pool_metadata, f, indent=2)

        print(f"  ✓ 保存完成: {output_dir}")
        print(f"    - Prototypes: {len(pool_data['features']):,}")
        print(f"    - Phones 唯一值: {len(np.unique(pool_data['phones']))}")
        print(f"    - Symbols 唯一值: {len(np.unique(pool_data['symbols']))}")


def main():
    config = {
        'cache_dir': 'data/samm_anon/cache/features/cleaned',
        'output_dir': 'data/samm_anon/checkpoints/target_pool_fixed',
        'codebook_path': 'data/samm_anon/checkpoints/codebook.pt',
        'n_prototypes_per_speaker': 50,
        'min_frames_per_speaker': 100,
    }

    rebuilder = TargetPoolRebuilder(config)
    rebuilder.build()


if __name__ == "__main__":
    main()
