# models/ssl/wrappers.py

import torch
import torch.nn as nn
from .wavlm_model import WavLM, WavLMConfig


class WavLMSSLExtractor(nn.Module):
    """
    Minimal, frozen WavLM SSL feature extractor.
    waveform -> frame-level SSL features
    """

    def __init__(
        self,
        ckpt_path: str,
        layer: int,
        device: str = "cuda",
    ):
        super().__init__()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = WavLMConfig(ckpt["cfg"])

        model = WavLM(cfg)
        model.load_state_dict(ckpt["model"])

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model.to(device)
        self.layer = layer
        self.device = device

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, T] float32 waveform
        Returns:
            feats: [B, T', D]
        """
        wav = wav.to(self.device)

        feats, _ = self.model.extract_features(
            wav,
            output_layer=self.layer,
        )
        return feats
