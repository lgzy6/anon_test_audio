"""RAVDESS dataset loader
Filename: 03-01-06-01-02-01-12.wav
Fields:   modality-vocal_channel-emotion-intensity-statement-repetition-actor
Actor 01-24: odd=male, even=female
"""
import torchaudio
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


class RAVDESSDataset(Dataset):
    def __init__(self, root: str, sample_rate: int = 16000):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.utterances = self._scan()
        print(f"RAVDESSDataset: {len(self.utterances)} utterances, "
              f"{len(set(u['speaker_id'] for u in self.utterances))} speakers")

    def _scan(self) -> List[Dict]:
        utterances = []
        for actor_dir in sorted(self.root.glob("Actor_*")):
            for wav_file in sorted(actor_dir.glob("*.wav")):
                parts = wav_file.stem.split("-")
                if len(parts) != 7:
                    continue
                actor_id = int(parts[6])
                utterances.append({
                    "utt_id": f"ravdess_{wav_file.stem}",
                    "speaker_id": f"ravdess_actor{actor_id:02d}",
                    "gender": "m" if actor_id % 2 == 1 else "f",
                    "emotion": EMOTION_MAP.get(parts[2], "unknown"),
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
        return {**info, "waveform": waveform.squeeze(0)}
