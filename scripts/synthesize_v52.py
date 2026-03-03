#!/usr/bin/env python3
"""
使用 v5.2 Target Pool 进行语音合成听感测试
"""

import sys
import torch
import torchaudio
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ssl.wrappers import WavLMSSLExtractor
from models.vocoder.hifigan import HiFiGAN


class V52Synthesizer:
    """使用 v5.2 Pool 进行语音合成"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self._load_components()

    def _load_components(self):
        """加载各组件"""
        print("Loading components...")

        # SSL 特征提取器
        print("  Loading WavLM...")
        self.ssl = WavLMSSLExtractor(
            ckpt_path=self.config['wavlm_path'],
            layer=15,
            device=self.device
        )

        # 加载 Target Pool
        print("  Loading Target Pool...")
        pool_dir = Path(self.config['pool_dir'])
        self.pool_features = np.load(pool_dir / "features.npy")
        self.pool_patterns = np.load(pool_dir / "patterns.npy")
        self.pool_genders = np.load(pool_dir / "genders.npy")
        self.faiss_index = faiss.read_index(str(pool_dir / "faiss.index"))

        # Vocoder
        print("  Loading HiFiGAN...")
        self.vocoder = HiFiGAN.load(
            self.config['vocoder_path'],
            device=self.device
        )
        print("  Done!")

    @torch.no_grad()
    def synthesize(self, audio_path: str, output_path: str, k: int = 4):
        """合成匿名化语音"""
        # 1. 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(self.device)

        # 2. 提取 SSL 特征
        print("Extracting SSL features...")
        h = self.ssl(waveform).squeeze(0).cpu().numpy()
        print(f"  Features shape: {h.shape}")

        # 3. kNN 检索
        print("Running kNN retrieval...")
        h_norm = h.copy()
        faiss.normalize_L2(h_norm)
        _, indices = self.faiss_index.search(h_norm, k)
        h_anon = self.pool_features[indices[:, 0]]

        # 4. Vocoder 合成
        print("Synthesizing with HiFiGAN...")
        h_anon_t = torch.from_numpy(h_anon).float().to(self.device)
        wav_anon = self.vocoder(h_anon_t)

        # 5. 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav_anon.cpu().unsqueeze(0), 16000)
        print(f"Saved to: {output_path}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"outputs/v52_synth/{ts}/anonymized.wav"

    config = {
        'wavlm_path': 'checkpoints/WavLM-Large.pt',
        'pool_dir': 'data/samm_anon/target_pool_v52',
        'vocoder_path': 'checkpoints/hifigan.pt',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    synth = V52Synthesizer(config)
    synth.synthesize(args.audio, args.output)


if __name__ == "__main__":
    main()
