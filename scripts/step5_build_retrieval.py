#!/usr/bin/env python3
"""
步骤 5：根据步骤 4 的判定，构建检索模块

两条路线：
  路线 A (CONTINUOUS): α 加权 cosine similarity
  路线 B (VQ): SimpleVQOnEta + 离散符号匹配

可嵌入到 run_full_pipeline_test.py 的 knn_retrieve_from_pool() 中

使用方法：
    cd /root/autodl-tmp/anon_test
    # 作为模块导入
    from scripts.step5_build_retrieval import EtaStyleRetriever
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / 'checkpoints'


class EtaStyleRetriever(nn.Module):
    """
    η_style 空间的检索模块 (nn.Module 版本)

    嵌入到 run_full_pipeline_test.py 的 knn_retrieve_from_pool() 中
    使用 α 加权 cosine similarity 进行 content + style 联合检索
    """

    def __init__(self, eta_projection_path=None, P_orth_path=None,
                 alpha=0.3, device='cuda'):
        """
        Args:
            eta_projection_path: checkpoints/eta_projection.pt 路径
            P_orth_path: outputs/eta_validation/P_orth_on_eta.npy 路径
            alpha: style_sim 权重 (0 = 纯 content, 1 = 纯 style)
            device: 计算设备
        """
        super().__init__()
        self.alpha = alpha

        if eta_projection_path is None:
            eta_projection_path = str(CKPT_DIR / 'eta_projection.pt')

        # 加载 Eta-WavLM
        ckpt = torch.load(eta_projection_path, map_location='cpu')
        self.register_buffer('A_star', ckpt['A_star'])   # (P, Q)
        self.register_buffer('b_star', ckpt['b_star'])   # (Q,)
        self.pca_dim = ckpt['pca_components']

        # 加载 PCA 模型
        pca_path = CKPT_DIR / 'speaker_pca_model.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
        else:
            self.pca_model = None

        # 加载 content 投影矩阵（可选）
        if P_orth_path is None:
            P_orth_path = str(BASE_DIR / 'outputs' / 'eta_validation' / 'P_orth_on_eta.npy')
        if Path(P_orth_path).exists():
            P_orth_np = np.load(P_orth_path)
            self.register_buffer('P_orth', torch.from_numpy(P_orth_np).float())
        else:
            self.P_orth = None

        # Speaker encoder (延迟加载)
        self._speaker_encoder = None

        self.to(device)

        print(f"EtaStyleRetriever initialized:")
        print(f"  A_star: {self.A_star.shape}")
        print(f"  P_orth: {'loaded' if self.P_orth is not None else 'not available'}")
        print(f"  PCA model: {'loaded' if self.pca_model is not None else 'not available'}")
        print(f"  alpha: {self.alpha}")

    def _get_speaker_encoder(self):
        """延迟加载 ECAPA-TDNN (只在需要实时提取时加载)"""
        if self._speaker_encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier

            device = str(self.A_star.device)
            self._speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                savedir=str(BASE_DIR / "cache" / "models" / "speaker")
            )
        return self._speaker_encoder

    def extract_speaker_embed(self, waveform):
        """
        从音频波形实时提取 PCA speaker embedding

        Args:
            waveform: (1, L) 或 (L,) 16kHz 音频

        Returns:
            embed_pca: (P,) PCA speaker embedding tensor
        """
        if self.pca_model is None:
            raise RuntimeError("PCA model not loaded, cannot extract speaker embedding")

        device = self.A_star.device
        encoder = self._get_speaker_encoder()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)

        with torch.no_grad():
            raw_emb = encoder.encode_batch(waveform)
            if raw_emb.dim() == 3:
                raw_emb = raw_emb.squeeze(1)
            raw_np = raw_emb[0].cpu().numpy().reshape(1, -1)

        embed_pca = self.pca_model.transform(raw_np)[0]
        return torch.from_numpy(embed_pca).float().to(device)

    def extract_speaker_embed_from_file(self, audio_path):
        """
        从音频文件提取 PCA speaker embedding

        Args:
            audio_path: 音频文件路径

        Returns:
            embed_pca: (P,) PCA speaker embedding tensor
        """
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return self.extract_speaker_embed(waveform)

    def compute_eta(self, feats, speaker_embed_pca):
        """
        计算 η = s - A^T d - b

        Args:
            feats: (T, Q) WavLM 帧级特征
            speaker_embed_pca: (P,) PCA speaker embedding

        Returns:
            η: (T, Q)
        """
        speaker_component = speaker_embed_pca @ self.A_star + self.b_star
        return feats - speaker_component.unsqueeze(0)

    def compute_eta_style(self, feats, speaker_embed_pca):
        """
        计算 η_style = P_orth @ η

        Args:
            feats: (T, Q)
            speaker_embed_pca: (P,)

        Returns:
            η_style: (T, Q)
        """
        eta = self.compute_eta(feats, speaker_embed_pca)
        if self.P_orth is not None:
            return eta @ self.P_orth
        return eta

    @torch.inference_mode()
    def forward(self, feats, speaker_embed_pca):
        """nn.Module forward: 返回 η_style"""
        return self.compute_eta_style(feats, speaker_embed_pca)

    def retrieve_with_style(self, src_feats, tgt_feats,
                            src_speaker_embed, tgt_speaker_embed):
        """
        带 style 感知的检索

        Args:
            src_feats: (T, Q) 源帧特征
            tgt_feats: (K, Q) 目标候选特征
            src_speaker_embed: (P,) 源说话人 PCA embedding
            tgt_speaker_embed: (P,) 目标说话人 PCA embedding

        Returns:
            selected_indices: (T,) 每帧选中的目标索引
            final_sim: (T, K) 综合相似度矩阵
        """
        # content 通道
        src_norm = F.normalize(src_feats, dim=-1)
        tgt_norm = F.normalize(tgt_feats, dim=-1)
        content_sim = torch.mm(src_norm, tgt_norm.T)  # (T, K)

        if self.alpha > 0:
            # style 通道
            eta_src = self.compute_eta_style(src_feats, src_speaker_embed)
            eta_tgt = self.compute_eta_style(tgt_feats, tgt_speaker_embed)

            eta_src_norm = F.normalize(eta_src, dim=-1)
            eta_tgt_norm = F.normalize(eta_tgt, dim=-1)
            style_sim = torch.mm(eta_src_norm, eta_tgt_norm.T)  # (T, K)

            final_sim = (1 - self.alpha) * content_sim + self.alpha * style_sim
        else:
            final_sim = content_sim

        return final_sim.argmax(dim=-1), final_sim

    @torch.inference_mode()
    def enhanced_knn_retrieve(self, h_ssl, phones, pool_features, pool_genders,
                              src_speaker_embed, tgt_speaker_embed,
                              target_gender=0, pool_phones=None,
                              batch_size=512):
        """
        增强版 kNN 检索：替换 run_full_pipeline_test.py 中的 knn_retrieve_from_pool

        Args:
            h_ssl: (T, Q) 源 WavLM 特征
            phones: (T,) 音素 ID
            pool_features: (N, Q) 目标池特征
            pool_genders: (N,) 性别标签
            src_speaker_embed: (P,) 源说话人 PCA embedding
            tgt_speaker_embed: (P,) 目标说话人 PCA embedding
            target_gender: 目标性别 (0=M, 1=F)
            pool_phones: (N,) 目标池音素标签 (可选，用于 phone 约束)
            batch_size: 批处理大小

        Returns:
            h_anon: (T, Q) 匿名化特征
        """
        device = self.A_star.device

        # 性别约束
        gender_mask = (pool_genders == target_gender)
        if gender_mask.sum() == 0:
            gender_mask = torch.ones(len(pool_features), dtype=torch.bool, device=device)

        T = h_ssl.shape[0]
        h_anon = torch.zeros_like(h_ssl)

        # 如果有 phone 标签，按 phone 分组检索
        if pool_phones is not None:
            unique_phones = phones.unique()
            for phone_id in unique_phones:
                src_mask = (phones == phone_id)
                if src_mask.sum() == 0:
                    continue

                # phone + gender 联合约束
                tgt_mask = (pool_phones == phone_id) & gender_mask
                if tgt_mask.sum() < 1:
                    # fallback: 仅 gender 约束
                    tgt_mask = gender_mask

                candidates = pool_features[tgt_mask]
                h_src = h_ssl[src_mask]

                # 检索
                selected = self._retrieve_batch(
                    h_src, candidates, src_speaker_embed, tgt_speaker_embed,
                    batch_size
                )
                h_anon[src_mask] = selected
        else:
            # 无 phone 约束：全局检索
            candidates = pool_features[gender_mask]
            h_anon = self._retrieve_batch(
                h_ssl, candidates, src_speaker_embed, tgt_speaker_embed,
                batch_size
            )

        return h_anon

    def _retrieve_batch(self, h_src, candidates, src_embed, tgt_embed,
                        batch_size=512):
        """分批检索"""
        # 预计算目标侧
        candidates_norm = F.normalize(candidates, dim=-1)

        if self.alpha > 0:
            eta_tgt = self.compute_eta_style(candidates, tgt_embed)
            eta_tgt_norm = F.normalize(eta_tgt, dim=-1)
        else:
            eta_tgt_norm = None

        T = h_src.shape[0]
        result = torch.zeros_like(h_src)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            h_batch = h_src[start:end]

            # Content similarity
            h_norm = F.normalize(h_batch, dim=-1)
            content_sim = torch.mm(h_norm, candidates_norm.T)

            if self.alpha > 0 and eta_tgt_norm is not None:
                # Style similarity
                eta_src = self.compute_eta_style(h_batch, src_embed)
                eta_src_norm = F.normalize(eta_src, dim=-1)
                style_sim = torch.mm(eta_src_norm, eta_tgt_norm.T)

                final_sim = (1 - self.alpha) * content_sim + self.alpha * style_sim
            else:
                final_sim = content_sim

            nearest_idx = final_sim.argmax(dim=-1)
            result[start:end] = candidates[nearest_idx]

        return result


def patch_knn_retrieve(retriever, src_audio_path=None):
    """
    返回一个可直接替换 knn_retrieve_from_pool 的函数

    使用方法:
        retriever = EtaStyleRetriever(alpha=0.3)
        # 带源音频路径 → 实时提取 speaker embedding
        knn_fn = patch_knn_retrieve(retriever, src_audio_path='/path/to/audio.flac')
        # 或不带 → fallback 到零向量 (效果降低)
        knn_fn = patch_knn_retrieve(retriever)

    Args:
        retriever: EtaStyleRetriever 实例
        src_audio_path: 源音频路径 (用于提取 speaker embedding)
    """
    # 预提取源说话人 embedding (如果有音频)
    _src_embed = None
    if src_audio_path is not None and retriever.pca_model is not None:
        try:
            _src_embed = retriever.extract_speaker_embed_from_file(src_audio_path)
            print(f"  Source speaker embedding extracted from {src_audio_path}")
        except Exception as e:
            print(f"  WARNING: Failed to extract speaker embedding: {e}")

    def knn_retrieve_from_pool(h_ssl, phones, source_gender_arg, device):
        pool_dir = BASE_DIR / 'data' / 'samm_anon' / 'target_pool_v52'

        pool_features = torch.from_numpy(
            np.load(str(pool_dir / 'features.npy'))
        ).float().to(device)
        pool_genders = torch.from_numpy(
            np.load(str(pool_dir / 'genders.npy'))
        ).long().to(device)

        # phone 标签 (可选)
        pool_phones_path = pool_dir / 'phones.npy'
        pool_phones = None
        if pool_phones_path.exists():
            pool_phones = torch.from_numpy(
                np.load(str(pool_phones_path))
            ).long().to(device)

        target_gender = 0 if source_gender_arg == 'M' else 1

        # 使用预提取的 embedding 或 fallback
        src_embed = _src_embed if _src_embed is not None else \
            torch.zeros(retriever.pca_dim, device=device)
        tgt_embed = torch.zeros(retriever.pca_dim, device=device)

        h_anon = retriever.enhanced_knn_retrieve(
            h_ssl, phones, pool_features, pool_genders,
            src_embed, tgt_embed, target_gender,
            pool_phones=pool_phones
        )

        del pool_features, pool_genders
        if pool_phones is not None:
            del pool_phones
        torch.cuda.empty_cache()

        return h_anon

    return knn_retrieve_from_pool


if __name__ == '__main__':
    print("EtaStyleRetriever module")
    print("Usage: from scripts.step5_build_retrieval import EtaStyleRetriever")

    # 检查所需文件
    eta_path = CKPT_DIR / 'eta_projection.pt'
    pca_path = CKPT_DIR / 'speaker_pca_model.pkl'
    p_orth_path = BASE_DIR / 'outputs' / 'eta_validation' / 'P_orth_on_eta.npy'

    print(f"\nRequired files:")
    print(f"  eta_projection.pt:  {'OK' if eta_path.exists() else 'MISSING (run step2)'}")
    print(f"  speaker_pca_model:  {'OK' if pca_path.exists() else 'MISSING (run step1)'}")
    print(f"  P_orth_on_eta.npy:  {'OK' if p_orth_path.exists() else 'MISSING (run step3)'}")

    if eta_path.exists():
        print("\nInitializing retriever...")
        retriever = EtaStyleRetriever(alpha=0.3)
        print("Ready for use.")
