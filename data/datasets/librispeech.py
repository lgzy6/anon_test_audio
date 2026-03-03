# data/datasets/librispeech. py

"""LibriSpeech / LibriTTS 数据集加载"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech / LibriTTS 数据集
    
    支持: 
    - LibriSpeech:  . flac 格式
    - LibriTTS: .wav 格式
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train-other-500",
        sample_rate: int = 16000,
        audio_ext: Optional[str] = None,  # 自动检测或指定 ". flac" / ".wav"
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.data_dir = self.root / split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 自动检测音频格式
        if audio_ext is None:
            self.audio_ext = self._detect_audio_format()
        else:
            self.audio_ext = audio_ext
        
        print(f"检测到音频格式: {self.audio_ext}")
        
        # 扫描所有音频文件
        self.utterances = self._scan_utterances()
        
        # 加载说话人信息
        self.speakers_info = self._load_speakers_info()
        
        print(f"LibriSpeechDataset:  {len(self.utterances)} utterances, "
              f"{len(set(u['speaker_id'] for u in self.utterances))} speakers")
    
    def _detect_audio_format(self) -> str:
        """自动检测音频格式"""
        # 检查是否有 .flac 文件
        flac_files = list(self. data_dir.rglob("*.flac"))
        if flac_files:
            return ".flac"
        
        # 检查是否有 . wav 文件
        wav_files = list(self.data_dir.rglob("*.wav"))
        if wav_files: 
            return ".wav"
        
        raise ValueError(f"在 {self.data_dir} 中未找到 .flac 或 .wav 文件")
    
    def _scan_utterances(self) -> List[Dict]:
        """扫描所有 utterance"""
        utterances = []
        
        # 目录结构: {split}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utt_id}.ext
        for speaker_dir in sorted(self.data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                
                # 使用检测到的音频扩展名
                pattern = f"*{self.audio_ext}"
                for audio_file in sorted(chapter_dir.glob(pattern)):
                    utt_id = audio_file. stem
                    
                    utterances.append({
                        'utt_id': utt_id,
                        'speaker_id': speaker_id,
                        'audio_path': str(audio_file),
                    })
        
        return utterances
    
    def _load_speakers_info(self) -> Dict:
        """加载说话人信息 (性别等)"""
        speakers_info = {}
        
        # 尝试多种可能的文件名
        possible_files = [
            self.root / "SPEAKERS.TXT",      # LibriSpeech
            self. root / "SPEAKERS.txt",      # LibriTTS
            self.root / "speakers.tsv",      # LibriTTS 另一种格式
        ]
        
        for speakers_file in possible_files: 
            if speakers_file.exists():
                print(f"加载说话人信息: {speakers_file}")
                
                if speakers_file.suffix. lower() == '.tsv':
                    # TSV 格式
                    with open(speakers_file, 'r') as f:
                        header = f.readline()  # 跳过表头
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                spk_id = parts[0]. strip()
                                gender = parts[1].strip().lower()
                                speakers_info[spk_id] = {'gender': gender}
                else: 
                    # TXT 格式 (pipe 分隔)
                    with open(speakers_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line. startswith(';') or not line:
                                continue
                            parts = line.split('|')
                            if len(parts) >= 2:
                                spk_id = parts[0].strip()
                                gender = parts[1].strip().lower()
                                speakers_info[spk_id] = {'gender':  gender}
                break
        
        return speakers_info
    
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
        
        # 获取性别
        speaker_id = utt_info['speaker_id']
        gender = self.speakers_info. get(speaker_id, {}).get('gender', 'unknown')
        
        return {
            'utt_id': utt_info['utt_id'],
            'speaker_id': speaker_id,
            'gender': gender,
            'waveform': waveform,
            'audio_path': utt_info['audio_path'],
        }
    
    def get_speaker_utterances(self, speaker_id:  str) -> List[int]:
        """获取某说话人的所有 utterance 索引"""
        return [
            i for i, utt in enumerate(self.utterances)
            if utt['speaker_id'] == speaker_id
        ]
    
    def get_all_speakers(self) -> List[str]:
        """获取所有说话人 ID"""
        return list(set(utt['speaker_id'] for utt in self.utterances))