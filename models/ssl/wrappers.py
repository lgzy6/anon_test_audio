# models/ssl/wrappers.py

import torch
import torch.nn as nn
from .wavlm_model import WavLM, WavLMConfig


class WavLMSSLExtractor(nn.Module):
    
    def __init__(self, ckpt_path, layer, device="cuda"):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = WavLMConfig(ckpt["cfg"])
        model = WavLM(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(device)
        self.layer  = layer   # 保留，兼容原有单层调用
        self.device = device

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """原有接口不变，单层输出"""
        wav = wav.to(self.device)
        feats, _ = self.model.extract_features(wav, output_layer=self.layer)
        return feats

    @torch.inference_mode()
    def forward_multi_layer(
        self,
        wav: torch.Tensor,
        layers: list = [6, 12, 24]
    ) -> dict:
        """
        单次前向，返回多层特征

        Returns:
            {6: [B, T', 1024], 12: [B, T', 1024], 24: [B, T', 1024]}
        """
        wav = wav.to(self.device)
        max_layer = max(layers)

        # ret_layer_results=True 返回 (feature, layer_results)
        (feats, layer_results), _ = self.model.extract_features(
            wav,
            output_layer=max_layer,
            ret_layer_results=True
        )

        # layer_results 是 list[(x, z)]，索引对应层号
        # x: [T, B, D] -> 需要转置为 [B, T, D]
        return {
            layer_idx: layer_results[layer_idx][0].transpose(0, 1)
            for layer_idx in layers
        }