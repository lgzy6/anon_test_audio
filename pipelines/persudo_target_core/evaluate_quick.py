#!/usr/bin/env python3
"""WER 评估：original vs anon_cross_f / anon_same_m（使用 Whisper + torchaudio，无需 ffmpeg）"""

import torch
import jiwer
import whisper
import numpy as np
import torchaudio
from pathlib import Path

OUTPUT_DIR = Path('/root/autodl-tmp/anon_test/outputs/test_anonymization_v1')


def load_audio_whisper(path):
    """用 torchaudio 加载，转为 whisper 需要的 float32 numpy [T]"""
    wav, sr = torchaudio.load(str(path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(dim=0).numpy().astype(np.float32)


def transcribe(model, path):
    audio = load_audio_whisper(path)
    result = model.transcribe(audio)
    return result['text'].strip().lower()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("加载 Whisper 模型...")
    model = whisper.load_model("base", device=device)

    ref       = transcribe(model, OUTPUT_DIR / 'original.wav')
    hyp_cross = transcribe(model, OUTPUT_DIR / 'anon_cross_f.wav')
    hyp_same  = transcribe(model, OUTPUT_DIR / 'anon_same_m.wav')

    wer_cross = jiwer.wer(ref, hyp_cross)
    wer_same  = jiwer.wer(ref, hyp_same)

    print("\n" + "=" * 55)
    print(f"源转录      : {ref}")
    print(f"cross 转录  : {hyp_cross}")
    print(f"same  转录  : {hyp_same}")
    print("-" * 55)
    print(f"WER (cross) : {wer_cross:.4f}")
    print(f"WER (same)  : {wer_same:.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()
