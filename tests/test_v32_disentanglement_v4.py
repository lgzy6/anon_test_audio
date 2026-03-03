#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 Test A: 解耦验证 (V4 - 说话人解耦)

新增优化方案:
- 方案 A: 说话人对抗训练 (同时对抗音素+说话人)
- 方案 B: 说话人归一化 (去除说话人均值偏移)

目标: H_style 应同时去除内容信息和说话人身份信息
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor


class GPULinearProbe:
    """GPU 加速的 Linear Probe"""

    def __init__(self, input_dim: int, n_classes: int, device='cuda'):
        self.device = device
        self.model = nn.Linear(input_dim, n_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 50, batch_size: int = 2048, lr: float = 1e-3):
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        n_samples = len(X_t)

        self.model.train()
        for epoch in range(epochs):
            indices = torch.randperm(n_samples, device=self.device)
            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X_t[batch_idx], y_t[batch_idx]

                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).argmax(dim=1)
        return (preds.cpu().numpy() == y).mean()


# ============================================================
# 方案 B: 说话人归一化
# ============================================================

class SpeakerNormalizedProjector:
    """
    说话人归一化投影器

    步骤:
    1. 计算每个说话人的特征均值
    2. 从每帧减去对应说话人均值
    3. 再应用内容子空间投影
    """

    def __init__(self, n_phones: int = 41, feature_dim: int = 1024,
                 n_components: int = 100):
        self.n_phones = n_phones
        self.feature_dim = feature_dim
        self.n_components = n_components

        self.speaker_means = {}
        self.global_mean = None
        self.U_c = None
        self.P_orth = None

    def fit(self, features: np.ndarray, phones: np.ndarray,
            speaker_ids: np.ndarray):
        """学习说话人归一化 + 内容子空间"""
        print("[SpeakerNorm] Learning speaker normalization...")

        # Step 1: 计算每个说话人的均值
        unique_speakers = np.unique(speaker_ids)
        print(f"  Found {len(unique_speakers)} speakers")

        for spk in unique_speakers:
            mask = (speaker_ids == spk)
            self.speaker_means[spk] = features[mask].mean(axis=0)

        self.global_mean = features.mean(axis=0)

        # Step 2: 说话人归一化
        features_norm = self._normalize(features, speaker_ids)

        # Step 3: 学习内容子空间 (在归一化后的特征上)
        self._fit_content_subspace(features_norm, phones)

        return self

    def _normalize(self, features: np.ndarray, speaker_ids: np.ndarray) -> np.ndarray:
        """对特征进行说话人归一化"""
        features_norm = features.copy()
        for spk, mean in self.speaker_means.items():
            mask = (speaker_ids == spk)
            features_norm[mask] -= mean
        return features_norm

    def _fit_content_subspace(self, features: np.ndarray, phones: np.ndarray):
        """学习内容子空间"""
        # 平衡采样
        balanced_feats, balanced_phones = [], []
        for phone_id in range(self.n_phones):
            if phone_id == 0:  # 跳过 silence
                continue
            mask = (phones == phone_id)
            if mask.sum() == 0:
                continue
            indices = np.where(mask)[0]
            if len(indices) > 2000:
                indices = np.random.choice(indices, 2000, replace=False)
            balanced_feats.append(features[indices])
            balanced_phones.append(phones[indices])

        if not balanced_feats:
            return

        features_bal = np.vstack(balanced_feats)
        phones_bal = np.concatenate(balanced_phones)

        # 计算类中心
        active_phones = np.unique(phones_bal)
        centroids = []
        for phone_id in active_phones:
            mask = (phones_bal == phone_id)
            centroids.append(features_bal[mask].mean(axis=0))
        centroids = np.array(centroids)

        # PCA on centroids
        centroids_centered = centroids - centroids.mean(axis=0)
        U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)

        n_comp = min(self.n_components, len(S))
        self.U_c = Vt[:n_comp].T.astype(np.float32)
        self.P_orth = np.eye(self.feature_dim, dtype=np.float32) - self.U_c @ self.U_c.T

        print(f"  Content subspace dim: {n_comp}")

    def project_to_style(self, features: np.ndarray,
                         speaker_ids: np.ndarray) -> np.ndarray:
        """投影到风格子空间 (先归一化再投影)"""
        features_norm = self._normalize(features, speaker_ids)
        if self.P_orth is not None:
            return (features_norm @ self.P_orth).astype(np.float32)
        return features_norm.astype(np.float32)


# ============================================================
# 方案 A: 说话人对抗训练
# ============================================================

class DualAdversarialNetwork(nn.Module):
    """
    双对抗网络: 同时对抗音素和说话人分类器
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 512,
                 n_phones: int = 41, n_speakers: int = 100):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Phone Classifier
        self.phone_cls = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_phones),
        )

        # Speaker Classifier
        self.speaker_cls = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_speakers),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        phone_logits = self.phone_cls(z)
        speaker_logits = self.speaker_cls(z)
        return z, phone_logits, speaker_logits


class DualAdversarialTrainer:
    """双对抗训练器"""

    def __init__(self, model: DualAdversarialNetwork,
                 n_phones: int = 41, n_speakers: int = 100):
        self.model = model
        self.n_phones = n_phones
        self.n_speakers = n_speakers
        self.device = next(model.parameters()).device

    def train(self, features: np.ndarray, phones: np.ndarray,
              speaker_ids: np.ndarray, epochs: int = 100,
              batch_size: int = 2048, adv_phone: float = 1.0,
              adv_speaker: float = 1.0):
        """双对抗训练"""
        print(f"[DualAdv] Training with adv_phone={adv_phone}, adv_speaker={adv_speaker}")

        X = torch.from_numpy(features).float().to(self.device)
        y_phone = torch.from_numpy(phones).long().to(self.device)
        y_spk = torch.from_numpy(speaker_ids).long().to(self.device)
        n_samples = len(X)

        opt_enc = torch.optim.Adam(self.model.encoder.parameters(), lr=1e-3)
        opt_cls = torch.optim.Adam(
            list(self.model.phone_cls.parameters()) +
            list(self.model.speaker_cls.parameters()), lr=1e-3
        )
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(n_samples, device=self.device)
            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_b = X[batch_idx]
                y_ph_b = y_phone[batch_idx]
                y_spk_b = y_spk[batch_idx]

                # Step 1: 训练分类器
                z, ph_logits, spk_logits = self.model(X_b)
                loss_cls = ce_loss(ph_logits, y_ph_b) + ce_loss(spk_logits, y_spk_b)
                opt_cls.zero_grad()
                loss_cls.backward()
                opt_cls.step()

                # Step 2: 训练编码器 (对抗)
                z, ph_logits, spk_logits = self.model(X_b)
                uniform_ph = torch.ones_like(ph_logits) / self.n_phones
                uniform_spk = torch.ones_like(spk_logits) / self.n_speakers

                loss_adv_ph = F.kl_div(
                    F.log_softmax(ph_logits, dim=1), uniform_ph, reduction='batchmean')
                loss_adv_spk = F.kl_div(
                    F.log_softmax(spk_logits, dim=1), uniform_spk, reduction='batchmean')

                loss_enc = adv_phone * loss_adv_ph + adv_speaker * loss_adv_spk
                opt_enc.zero_grad()
                loss_enc.backward()
                opt_enc.step()

                total_loss += loss_enc.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

        return self

    @torch.no_grad()
    def transform(self, features: np.ndarray) -> np.ndarray:
        """变换特征"""
        self.model.eval()
        X = torch.from_numpy(features).float().to(self.device)
        z = self.model.encode(X)
        return z.cpu().numpy()


# ============================================================
# 数据加载
# ============================================================

def load_data_with_speakers(config_path: str, max_samples: int = 100000):
    """加载数据，包含说话人信息"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config['paths']['cache_dir'])
    features_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'
    metadata_json = cache_dir / 'features' / 'wavlm' / 'metadata.json'

    # 加载元数据
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)

    # 加载 phone predictor
    phone_ckpt = config.get('phone_predictor', {}).get(
        'checkpoint', './checkpoints/phone_decoder.pt')
    phone_predictor = PhonePredictor.load(phone_ckpt, device='cuda') \
        if Path(phone_ckpt).exists() else None

    # 构建帧级说话人映射
    print("Building frame-to-speaker mapping...")
    utterances = metadata['utterances']
    speaker_map = {}
    for spk in set(u['speaker_id'] for u in utterances):
        speaker_map[spk] = len(speaker_map)

    frame_speakers = []
    for utt in utterances:
        spk_id = speaker_map[utt['speaker_id']]
        n_frames = utt['h5_end_idx'] - utt['h5_start_idx']
        frame_speakers.extend([spk_id] * n_frames)

    frame_speakers = np.array(frame_speakers)

    # 加载特征
    print(f"Loading features from {features_h5}...")
    with h5py.File(features_h5, 'r') as f:
        total = f['features'].shape[0]
        if total > max_samples:
            idx = np.sort(np.random.choice(total, max_samples, replace=False))
            features = f['features'][idx]
            frame_speakers = frame_speakers[idx]
        else:
            features = f['features'][:]

    print(f"Loaded {len(features)} frames")

    # 预测音素
    if phone_predictor:
        print("Predicting phones...")
        feat_t = torch.from_numpy(features).float().cuda()
        phones = []
        for i in range(0, len(feat_t), 10000):
            phones.append(phone_predictor(feat_t[i:i+10000]).cpu().numpy())
        phones = np.concatenate(phones)
    else:
        phones = np.random.randint(0, 41, len(features))

    n_speakers = len(speaker_map)
    print(f"  Speakers: {n_speakers}, Phones: {len(np.unique(phones))}")

    return features, phones, frame_speakers, n_speakers


def run_probe(features, labels, name, n_classes):
    """运行 Linear Probe"""
    print(f"\n  Probing: {name}")
    n = len(features)
    idx = np.random.permutation(n)
    tr, te = idx[:int(n*0.8)], idx[int(n*0.8):]

    probe = GPULinearProbe(features.shape[1], n_classes)
    probe.fit(features[tr], labels[tr], epochs=50)
    acc = probe.score(features[te], labels[te])
    print(f"    Test Acc: {acc:.2%}")
    return acc


def main():
    """主测试流程"""
    print("="*60)
    print("DS-SAMM v3.2 Test: Speaker Disentanglement (V4)")
    print("="*60)

    np.random.seed(42)
    torch.manual_seed(42)

    config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'
    features, phones, speakers, n_spk = load_data_with_speakers(
        str(config_path), max_samples=100000
    )

    # 剔除 silence
    mask = phones != 0
    feat_ns = features[mask]
    ph_ns = phones[mask]
    spk_ns = speakers[mask]
    print(f"\nAfter removing silence: {len(feat_ns)} frames")

    results = {}

    # Baseline
    print("\n" + "="*50)
    print("Baseline: Original Features")
    print("="*50)
    results['orig_phone'] = run_probe(feat_ns, ph_ns, "Phone", 41)
    results['orig_spk'] = run_probe(feat_ns, spk_ns, "Speaker", n_spk)

    # 方案 B: 说话人归一化
    print("\n" + "="*50)
    print("Method B: Speaker Normalization")
    print("="*50)
    proj_norm = SpeakerNormalizedProjector(n_components=100)
    proj_norm.fit(features, phones, speakers)
    feat_norm = proj_norm.project_to_style(feat_ns, spk_ns)
    results['norm_phone'] = run_probe(feat_norm, ph_ns, "Phone", 41)
    results['norm_spk'] = run_probe(feat_norm, spk_ns, "Speaker", n_spk)

    # 方案 A: 双对抗训练
    print("\n" + "="*50)
    print("Method A: Dual Adversarial Training")
    print("="*50)
    model = DualAdversarialNetwork(
        feature_dim=features.shape[1],
        n_phones=41,
        n_speakers=n_spk
    ).cuda()

    trainer = DualAdversarialTrainer(model, n_phones=41, n_speakers=n_spk)
    trainer.train(feat_ns, ph_ns, spk_ns, epochs=100,
                  adv_phone=1.0, adv_speaker=2.0)

    feat_adv = trainer.transform(feat_ns)
    results['adv_phone'] = run_probe(feat_adv, ph_ns, "Phone", 41)
    results['adv_spk'] = run_probe(feat_adv, spk_ns, "Speaker", n_spk)

    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBaseline:")
    print(f"  Phone Acc: {results['orig_phone']:.2%}")
    print(f"  Speaker Acc: {results['orig_spk']:.2%}")

    print(f"\nMethod B (Speaker Norm):")
    print(f"  Phone Acc: {results['norm_phone']:.2%}")
    print(f"  Speaker Acc: {results['norm_spk']:.2%}")

    print(f"\nMethod A (Dual Adversarial):")
    print(f"  Phone Acc: {results['adv_phone']:.2%}")
    print(f"  Speaker Acc: {results['adv_spk']:.2%}")

    return results


if __name__ == '__main__':
    main()
