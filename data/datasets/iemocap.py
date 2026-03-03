# data/datasets/iemocap.py

"""IEMOCAP 数据集加载"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP 数据集

    目录结构:
    IEMOCAP_full_release/
    ├── Session1/
    │   ├── sentences/
    │   │   └── wav/
    │   │       ├── Ses01F_impro01/
    │   │       │   ├── Ses01F_impro01_F000.wav
    │   │       │   └── ...
    │   │       └── ...
    │   └── dialog/
    │       └── EmoEvaluation/
    │           └── *.txt (情感标注)
    ├── Session2/
    └── ...

    说话人命名规则:
    - Ses01F: Session1 Female
    - Ses01M: Session1 Male
    - 文件名中 _F000 表示 Female 说话人的第0句
    - 文件名中 _M000 表示 Male 说话人的第0句
    """

    def __init__(
        self,
        root: str,
        sessions: Optional[List[int]] = None,
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.sessions = sessions or [1, 2, 3, 4, 5]

        if not self.root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        # 扫描所有音频文件
        self.utterances = self._scan_utterances()

        # 统计信息
        speakers = set(u['speaker_id'] for u in self.utterances)
        print(f"IEMOCAPDataset: {len(self.utterances)} utterances, "
              f"{len(speakers)} speakers")

    def _scan_utterances(self) -> List[Dict]:
        """扫描所有 utterance"""
        utterances = []

        for session_id in self.sessions:
            session_dir = self.root / f"Session{session_id}"
            wav_dir = session_dir / "sentences" / "wav"

            if not wav_dir.exists():
                print(f"警告: {wav_dir} 不存在，跳过")
                continue

            # 遍历每个对话文件夹
            for dialog_dir in sorted(wav_dir.iterdir()):
                if not dialog_dir.is_dir():
                    continue

                # 遍历每个 wav 文件 (过滤掉 macOS 隐藏文件)
                for wav_file in sorted(dialog_dir.glob("*.wav")):
                    # 跳过 macOS 隐藏文件
                    if wav_file.name.startswith('._'):
                        continue
                    utt_id = wav_file.stem
                    speaker_id, gender = self._parse_speaker_info(utt_id)

                    utterances.append({
                        'utt_id': utt_id,
                        'speaker_id': speaker_id,
                        'gender': gender,
                        'session_id': session_id,
                        'audio_path': str(wav_file),
                    })

        return utterances

    def _parse_speaker_info(self, utt_id: str) -> tuple:
        """
        从 utterance ID 解析说话人信息

        例如: Ses01F_impro01_F000
        - Ses01F: Session1 的对话 (F表示这是Female主导的对话)
        - _F000: 这句话是 Female 说的

        说话人ID格式: Ses{session}{dialog_gender}_{utterance_gender}
        """
        parts = utt_id.split('_')

        # 获取 session 信息 (如 Ses01F)
        session_part = parts[0]  # Ses01F
        session_num = session_part[3:5]  # 01

        # 获取说话人性别 (从最后一个部分，如 F000 或 M000)
        last_part = parts[-1]
        if last_part.startswith('F'):
            gender = 'f'
            spk_gender = 'F'
        elif last_part.startswith('M'):
            gender = 'm'
            spk_gender = 'M'
        else:
            gender = 'unknown'
            spk_gender = 'U'

        # 构建唯一的说话人ID
        # 每个 session 有一男一女，共10个说话人
        speaker_id = f"Ses{session_num}{spk_gender}"

        return speaker_id, gender

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict:
        utt_info = self.utterances[idx]

        # 加载音频
        waveform, sr = torchaudio.load(utt_info['audio_path'])

        # 重采样到目标采样率
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [T]

        return {
            'utt_id': utt_info['utt_id'],
            'speaker_id': utt_info['speaker_id'],
            'gender': utt_info['gender'],
            'session_id': utt_info['session_id'],
            'waveform': waveform,
            'audio_path': utt_info['audio_path'],
        }

    def get_speaker_utterances(self, speaker_id: str) -> List[int]:
        """获取某说话人的所有 utterance 索引"""
        return [
            i for i, utt in enumerate(self.utterances)
            if utt['speaker_id'] == speaker_id
        ]

    def get_all_speakers(self) -> List[str]:
        """获取所有说话人 ID"""
        return list(set(utt['speaker_id'] for utt in self.utterances))
