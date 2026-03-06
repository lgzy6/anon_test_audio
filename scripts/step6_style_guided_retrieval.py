#!/usr/bin/env python3
"""
步骤 6：风格引导的匿名化检索

三种 System 配置:
  System A (baseline): η + 随机目标 (无风格引导)
  System B (proposed): η + 风格引导目标选择
  System C (transfer): η + 指定风格模板

使用方法：
    python scripts/step6_style_guided_retrieval.py --system B --audio input.flac
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent


class StyleGuidedRetriever:
    """
    风格引导的匿名化检索器

    论文核心模块：在 phone-constrained kNN 的基础上，
    通过段级风格相似度选择"说话方式匹配"的目标话语
    """

    def __init__(self, eta_projection_path, style_extractor_path,
                 pool_styles_path, pool_dir, device='cuda'):
        self.device = device

        # 加载 Eta-WavLM
        eta_ckpt = torch.load(eta_projection_path, map_location=device)
        self.A_star = eta_ckpt['A_star'].to(device)
        self.b_star = eta_ckpt['b_star'].to(device)
        self.pca_dim = eta_ckpt['pca_components']

        # 加载 PCA
        pca_path = Path(eta_projection_path).parent / 'speaker_pca_model.pkl'
        with open(pca_path, 'rb') as f:
            self.pca_model = pickle.load(f)

        # 加载段级风格提取器
        from scripts.step4_build_style_extractor import SegmentStyleExtractor
        self.style_extractor = SegmentStyleExtractor.load(style_extractor_path)

        # 加载目标池
        pool_dir = Path(pool_dir)
        self.pool_features = torch.from_numpy(
            np.load(str(pool_dir / 'features.npy'))
        ).float().to(device)
        self.pool_genders = torch.from_numpy(
            np.load(str(pool_dir / 'genders.npy'))
        ).long().to(device)

        # 加载预计算的目标池风格
        pool_data = np.load(str(pool_styles_path))
        self.pool_utt_styles = pool_data['styles']         # (N_utt, style_dim)
        self.pool_utt_boundaries = pool_data['utt_boundaries']  # (N_utt, 2)
        self.pool_utt_speakers = pool_data['speaker_ids']
        self.pool_phones = torch.from_numpy(pool_data['phones']).long().to(device)

        print(f"StyleGuidedRetriever initialized:")
        print(f"  Pool: {len(self.pool_features)} frames, {len(self.pool_utt_styles)} utterances")

    def compute_eta(self, feats, speaker_embed_pca):
        """η = s - A*d - b"""
        d = torch.from_numpy(speaker_embed_pca).float().to(self.device)
        speaker_component = d @ self.A_star + self.b_star
        return feats - speaker_component.unsqueeze(0)

    def _select_target_utterance(self, source_style, target_gender, mode='preserve'):
        """
        根据风格选择目标话语

        mode:
          'preserve' → 找风格最相似的
          'random'   → 随机选择 (baseline)
        """
        # 性别过滤
        valid_utts = []
        for i, (s, e) in enumerate(self.pool_utt_boundaries):
            utt_genders = self.pool_genders[s:e]
            if (utt_genders == target_gender).sum() > len(utt_genders) * 0.5:
                valid_utts.append(i)

        if not valid_utts:
            valid_utts = list(range(len(self.pool_utt_styles)))

        if mode == 'random':
            return valid_utts[np.random.randint(len(valid_utts))]

        # 风格相似度
        valid_styles = self.pool_utt_styles[valid_utts]
        source_norm = source_style / (np.linalg.norm(source_style) + 1e-8)
        valid_norms = valid_styles / (np.linalg.norm(valid_styles, axis=1, keepdims=True) + 1e-8)
        sims = valid_norms @ source_norm
        best_local_idx = sims.argmax()
        return valid_utts[best_local_idx]

    def anonymize(self, source_features, source_phones,
                  source_speaker_embed_pca, target_gender=0,
                  mode='preserve', top_k=4):
        """
        端到端匿名化

        Args:
            source_features: (T, 1024) WavLM 帧特征
            source_phones: (T,) phone 序列
            source_speaker_embed_pca: (P,) 源说话人 PCA embedding
            target_gender: 0=M, 1=F
            mode: 'preserve' / 'random' / 'transfer'
            top_k: kNN 的 k 值

        Returns:
            h_anon: (T, 1024) 匿名化特征
            info: dict 包含匹配信息
        """
        device = self.device

        # 1. 计算源 η
        src_feats = torch.from_numpy(source_features).float().to(device)
        eta_src = self.compute_eta(src_feats, source_speaker_embed_pca)

        # 2. 提取源话语的段级风格
        eta_src_np = eta_src.cpu().numpy()
        source_style = self.style_extractor.extract_utterance_style(
            eta_src_np, source_phones
        )

        # 3. 选择目标话语
        target_utt_idx = self._select_target_utterance(
            source_style, target_gender, mode
        )
        t_start, t_end = self.pool_utt_boundaries[target_utt_idx]

        # 4. Phone-constrained kNN 在目标话语内检索
        target_feats = self.pool_features[t_start:t_end]
        target_phones = self.pool_phones[t_start:t_end]

        T = len(source_features)
        h_anon = torch.zeros(T, source_features.shape[1], device=device)

        for phone_id in np.unique(source_phones):
            if phone_id == 0:
                h_anon[source_phones == 0] = src_feats[source_phones == 0]
                continue

            src_mask = torch.from_numpy(source_phones == phone_id).to(device)
            tgt_mask = (target_phones == phone_id)

            if tgt_mask.sum() == 0:
                # Fallback: 全局池
                global_mask = (self.pool_phones == phone_id)
                if global_mask.sum() == 0:
                    h_anon[src_mask] = src_feats[src_mask]
                    continue
                tgt_feats = self.pool_features[global_mask]
            else:
                tgt_feats = target_feats[tgt_mask]

            # Cosine kNN
            src_batch = F.normalize(src_feats[src_mask], dim=-1)
            tgt_batch = F.normalize(tgt_feats, dim=-1)
            sims = torch.mm(src_batch, tgt_batch.T)

            if top_k == 1:
                indices = sims.argmax(dim=-1)
            else:
                k = min(top_k, tgt_batch.shape[0])
                _, top_indices = sims.topk(k, dim=-1)
                # 加权平均 top-k
                top_sims = torch.gather(sims, 1, top_indices)
                top_weights = F.softmax(top_sims * 10, dim=-1)
                selected = tgt_feats[top_indices]  # (n_src, k, 1024)
                h_anon[src_mask] = (selected * top_weights.unsqueeze(-1)).sum(dim=1)
                continue

            h_anon[src_mask] = tgt_feats[indices]

        # 风格保留度量
        target_style = self.pool_utt_styles[target_utt_idx]
        style_sim = np.dot(source_style, target_style) / (
            np.linalg.norm(source_style) * np.linalg.norm(target_style) + 1e-8
        )

        return h_anon.cpu().numpy(), {
            'target_utt_idx': target_utt_idx,
            'target_speaker': self.pool_utt_speakers[target_utt_idx],
            'style_similarity': float(style_sim),
            'mode': mode,
        }


def main():
    """测试 Step 6 风格引导检索"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-utt-idx', type=int, default=0, help='测试话语索引')
    parser.add_argument('--mode', type=str, default='preserve', choices=['preserve', 'random'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载配置
    config_path = BASE_DIR / 'configs' / 'base.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ckpt_dir = Path(config['paths']['checkpoints_dir'])

    # 检查必要文件
    required_files = [
        ckpt_dir / 'eta_projection.pt',
        ckpt_dir / 'style_extractor.pkl',
        ckpt_dir / 'utterance_styles.npz'
    ]

    for f in required_files:
        if not f.exists():
            print(f"错误: 缺少文件 {f}")
            print("请先运行 Step 1-5")
            return

    print(f"[Step 6 测试] 使用话语 {args.test_utt_idx}, 模式: {args.mode}")
    print("注意: 完整的 Step 6 需要在实际匿名化流程中调用")
    print("当前仅验证模块加载和风格检索功能\n")

    # 加载 utterance_styles
    styles_data = np.load(ckpt_dir / 'utterance_styles.npz')
    print(f"已加载风格数据:")
    print(f"  话语数: {len(styles_data['styles'])}")
    print(f"  说话人数: {len(np.unique(styles_data['speaker_ids']))}")
    print(f"  风格维度: {styles_data['styles'].shape[1]}")

    print("\n✓ Step 6 模块就绪")
    print("提示: 完整测试需要准备测试音频并运行完整的匿名化流程")


if __name__ == '__main__':
    main()