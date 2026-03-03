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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / 'checkpoints'


class EtaStyleRetriever:
    """
    η_style 空间的检索模块

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
        self.device = device
        self.alpha = alpha

        if eta_projection_path is None:
            eta_projection_path = str(CKPT_DIR / 'eta_projection.pt')

        # 加载 Eta-WavLM
        ckpt = torch.load(eta_projection_path, map_location=device)
        self.A_star = ckpt['A_star'].to(device)  # (P, Q)
        self.b_star = ckpt['b_star'].to(device)  # (Q,)
        self.pca_dim = ckpt['pca_components']

        # 加载 PCA 模型
        pca_path = CKPT_DIR / 'speaker_pca_model.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
        else:
            self.pca_model = None

        # 加载 content 投影矩阵（可选）
        self.P_orth = None
        if P_orth_path is None:
            P_orth_path = str(BASE_DIR / 'outputs' / 'eta_validation' / 'P_orth_on_eta.npy')
        if Path(P_orth_path).exists():
            P_orth_np = np.load(P_orth_path)
            self.P_orth = torch.from_numpy(P_orth_np).float().to(device)

        print(f"EtaStyleRetriever initialized:")
        print(f"  A_star: {self.A_star.shape}")
        print(f"  P_orth: {'loaded' if self.P_orth is not None else 'not available'}")
        print(f"  alpha: {self.alpha}")

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
        # content 通道（原始 cosine similarity）
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

    def enhanced_knn_retrieve(self, h_ssl, phones, pool_features, pool_genders,
                              src_speaker_embed, tgt_speaker_embed,
                              target_gender=0, batch_size=512):
        """
        增强版 kNN 检索：替换 run_full_pipeline_test.py 中的 knn_retrieve_from_pool

        Args:
            h_ssl: (T, Q) 源 WavLM 特征
            phones: (T,) 音素 ID
            pool_features: (N, Q) 目标池特征
            pool_genders: (N,) 性别标签
            src_speaker_embed: (P,) 源说话人 PCA embedding
            tgt_speaker_embed: (P,) 目标说话人 PCA embedding（可以是池的平均）
            target_gender: 目标性别 (0=M, 1=F)
            batch_size: 批处理大小

        Returns:
            h_anon: (T, Q) 匿名化特征
        """
        device = self.device

        # 性别约束
        gender_mask = (pool_genders == target_gender)
        if gender_mask.sum() == 0:
            gender_mask = torch.ones(len(pool_features), dtype=torch.bool, device=device)

        candidates = pool_features[gender_mask]

        # 预计算目标 η_style
        eta_tgt = self.compute_eta_style(candidates, tgt_speaker_embed)
        eta_tgt_norm = F.normalize(eta_tgt, dim=-1)
        candidates_norm = F.normalize(candidates, dim=-1)

        T = h_ssl.shape[0]
        h_anon = torch.zeros_like(h_ssl)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            h_batch = h_ssl[start:end]

            # Content similarity
            h_norm = F.normalize(h_batch, dim=-1)
            content_sim = torch.mm(h_norm, candidates_norm.T)

            if self.alpha > 0:
                # Style similarity
                eta_src = self.compute_eta_style(h_batch, src_speaker_embed)
                eta_src_norm = F.normalize(eta_src, dim=-1)
                style_sim = torch.mm(eta_src_norm, eta_tgt_norm.T)

                final_sim = (1 - self.alpha) * content_sim + self.alpha * style_sim
            else:
                final_sim = content_sim

            nearest_idx = final_sim.argmax(dim=-1)
            h_anon[start:end] = candidates[nearest_idx]

        return h_anon


def patch_knn_retrieve(retriever, source_gender='M'):
    """
    返回一个可直接替换 knn_retrieve_from_pool 的函数

    使用方法:
        retriever = EtaStyleRetriever(alpha=0.3)
        knn_retrieve_from_pool = patch_knn_retrieve(retriever, 'M')
    """
    def knn_retrieve_from_pool(h_ssl, phones, source_gender_arg, device):
        pool_dir = BASE_DIR / 'data' / 'samm_anon' / 'target_pool_v52'

        pool_features = torch.from_numpy(
            np.load(str(pool_dir / 'features.npy'))
        ).float().to(device)
        pool_genders = torch.from_numpy(
            np.load(str(pool_dir / 'genders.npy'))
        ).long().to(device)

        target_gender = 0 if source_gender_arg == 'M' else 1

        # 使用零向量作为默认说话人 embedding
        src_embed = torch.zeros(retriever.pca_dim, device=device)
        tgt_embed = torch.zeros(retriever.pca_dim, device=device)

        h_anon = retriever.enhanced_knn_retrieve(
            h_ssl, phones, pool_features, pool_genders,
            src_embed, tgt_embed, target_gender
        )

        del pool_features, pool_genders
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
