"""
帧级别 HDF5 数据集，用于 Purifier 训练

从预提取的 HDF5 文件中按帧加载:
  - L24 特征 (1024-d) 作为输入
  - Phone ID 作为内容标签
  - Speaker ID 作为对抗标签（从 metadata.json 映射）
"""

import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PurifierFrameDataset(Dataset):
    """
    帧级别数据集：将所有 utterance 的帧打平，按索引随机访问。

    为避免一次性加载全部特征到内存，采用 HDF5 随机读取。
    如果数据量可控（< 30GB），也可用 load_into_memory=True 预加载到 RAM。
    """

    def __init__(self, feature_dir: str, load_into_memory: bool = False):
        self.feature_dir = feature_dir

        # 读取元数据
        with open(f"{feature_dir}/metadata.json", "r") as f:
            meta = json.load(f)

        self.total_frames = meta["total_frames"]
        self.utterances = meta["utterances"]

        # 构建 speaker_id -> 连续整数 映射
        unique_speakers = sorted(set(u["speaker_id"] for u in self.utterances))
        self.spk2idx = {spk: i for i, spk in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)

        # 构建帧级别的 speaker 标签数组
        # 每一帧都有对应的 speaker index
        self.frame_spk_labels = np.empty(self.total_frames, dtype=np.int64)
        for utt in self.utterances:
            s, e = utt["h5_start_idx"], utt["h5_end_idx"]
            self.frame_spk_labels[s:e] = self.spk2idx[utt["speaker_id"]]

        # 打开 HDF5
        self._h5_feat = None
        self._h5_phone = None
        self.feat_path = f"{feature_dir}/layer_24.h5"
        self.phone_path = f"{feature_dir}/phones.h5"

        self.in_memory = load_into_memory
        if load_into_memory:
            print("预加载特征到内存...")
            with h5py.File(self.feat_path, "r") as f:
                self.feat_data = f["features"][:]
            with h5py.File(self.phone_path, "r") as f:
                self.phone_data = f["phones"][:]
            print(f"  特征: {self.feat_data.shape}, 音素: {self.phone_data.shape}")

    def _open_h5(self):
        """懒加载 HDF5（兼容多 worker DataLoader）"""
        if self._h5_feat is None:
            self._h5_feat = h5py.File(self.feat_path, "r")
            self._h5_phone = h5py.File(self.phone_path, "r")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.in_memory:
            feat = self.feat_data[idx]
            phone = self.phone_data[idx]
        else:
            self._open_h5()
            feat = self._h5_feat["features"][idx]
            phone = self._h5_phone["phones"][idx]

        return (
            torch.from_numpy(feat.astype(np.float32)),
            torch.tensor(phone, dtype=torch.long),
            torch.tensor(self.frame_spk_labels[idx], dtype=torch.long),
        )

    def close(self):
        if self._h5_feat is not None:
            self._h5_feat.close()
            self._h5_phone.close()


def build_dataloader(
    feature_dir: str,
    batch_size: int = 4096,
    num_workers: int = 4,
    load_into_memory: bool = False,
):
    """构建训练用 DataLoader"""
    ds = PurifierFrameDataset(feature_dir, load_into_memory=load_into_memory)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if not load_into_memory else 0,
        pin_memory=True,
        drop_last=True,
    )

    print(f"数据集: {ds.total_frames} 帧, {ds.num_speakers} 说话人")
    return loader, ds