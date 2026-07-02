"""Emotional Speech Dataset (ESD) 加载器
说话人 0001-0010: 中文, 0011-0020: 英文
"""

import torchaudio
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset

EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# ESD 论文中的性别标注 (1-10中文, 11-20英文)
SPEAKER_GENDER = {
    "0001": "f", "0002": "m", "0003": "m", "0004": "f", "0005": "f",
    "0006": "m", "0007": "f", "0008": "m", "0009": "f", "0010": "m",
    "0011": "m", "0012": "f", "0013": "m", "0014": "f", "0015": "m",
    "0016": "f", "0017": "m", "0018": "f", "0019": "m", "0020": "f",
}


class ESDDataset(Dataset):
    def __init__(self, root: str, language: str = "both", sample_rate: int = 16000):
        """
        Args:
            language: "zh" | "en" | "both"
        """
        assert language in ("zh", "en", "both")
        self.root = Path(root)
        self.sample_rate = sample_rate

        if language == "zh":
            spk_range = range(1, 11)
        elif language == "en":
            spk_range = range(11, 21)
        else:
            spk_range = range(1, 21)

        self.utterances = self._scan(spk_range)
        print(f"ESDDataset ({language}): {len(self.utterances)} utterances, "
              f"{len(set(u['speaker_id'] for u in self.utterances))} speakers")

    def _scan(self, spk_range) -> List[Dict]:
        utterances = []
        for spk_num in spk_range:
            spk_id = f"{spk_num:04d}"
            spk_dir = self.root / spk_id
            if not spk_dir.exists():
                continue
            lang = "zh" if spk_num <= 10 else "en"
            for emotion in EMOTIONS:
                emo_dir = spk_dir / emotion
                if not emo_dir.exists():
                    continue
                for wav_file in sorted(emo_dir.glob("*.wav")):
                    utterances.append({
                        "utt_id": wav_file.stem,
                        "speaker_id": spk_id,
                        "gender": SPEAKER_GENDER.get(spk_id, "unknown"),
                        "emotion": emotion.lower(),
                        "language": lang,
                        "audio_path": str(wav_file),
                    })
        return utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict:
        info = self.utterances[idx]
        waveform, sr = torchaudio.load(info["audio_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return {**info, "waveform": waveform}
