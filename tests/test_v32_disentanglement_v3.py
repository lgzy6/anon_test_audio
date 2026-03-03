#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 Test A: 正交解耦有效性验证 (V3 优化版)

优化方案:
1. 增加子空间维度 (Expanded Subspace)
2. 对抗神经网络 (Adversarial Network)

成功标准:
- H_style 上的音素分类准确率应显著低于 H 原始特征
- 理想情况: < 30%
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

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
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).argmax(dim=1)
        return (preds.cpu().numpy() == y).mean()


# ============================================================
# 方案1: 增加子空间维度 (Expanded Subspace)
# ============================================================

class ExpandedSubspaceProjector:
    """
    扩展子空间投影器

    改进:
    1. 使用更大的子空间维度 (100-200)
    2. 迭代式投影，逐步去除内容
    3. 结合 PCA 和类内方差分析
    """

    def __init__(self, n_phones: int = 41, feature_dim: int = 1024,
                 silence_id: int = 0, n_components: int = 100):
        self.n_phones = n_phones
        self.feature_dim = feature_dim
        self.silence_id = silence_id
        self.n_components = n_components

        self.U_c = None
        self.P_orth = None
        self.global_mean = None

    def _balance_sample(self, features: np.ndarray, phones: np.ndarray,
                        max_per_class: int = 2000) -> tuple:
        """平衡采样"""
        balanced_feats, balanced_phones = [], []

        for phone_id in range(self.n_phones):
            if phone_id == self.silence_id:
                continue
            mask = (phones == phone_id)
            if mask.sum() == 0:
                continue
            indices = np.where(mask)[0]
            if len(indices) > max_per_class:
                indices = np.random.choice(indices, max_per_class, replace=False)
            balanced_feats.append(features[indices])
            balanced_phones.append(phones[indices])

        return np.vstack(balanced_feats), np.concatenate(balanced_phones)

    def fit(self, features: np.ndarray, phones: np.ndarray,
            max_per_class: int = 2000, n_iterations: int = 2):
        """
        学习扩展内容子空间

        使用迭代式方法:
        1. 计算类中心的 PCA
        2. 计算类内高方差方向
        3. 合并并正交化
        """
        print(f"[ExpandedSubspace] Learning content subspace...")
        print(f"  Target dimensions: {self.n_components}")

        # 平衡采样
        features_bal, phones_bal = self._balance_sample(features, phones, max_per_class)
        print(f"  Balanced samples: {len(features_bal)}")

        self.global_mean = features_bal.mean(axis=0)
        features_centered = features_bal - self.global_mean

        active_phones = np.unique(phones_bal)
        n_active = len(active_phones)
        print(f"  Active phones: {n_active}")

        # Step 1: 类中心 PCA (捕获类间差异)
        centroids = np.zeros((n_active, self.feature_dim), dtype=np.float32)
        for i, phone_id in enumerate(active_phones):
            mask = (phones_bal == phone_id)
            centroids[i] = features_centered[mask].mean(axis=0)

        U_centers, S_centers, Vt_centers = np.linalg.svd(centroids, full_matrices=False)
        n_center_dims = min(n_active - 1, self.n_components // 2)
        V_centers = Vt_centers[:n_center_dims].T

        print(f"  Center PCA dims: {n_center_dims}")

        # Step 2: 类内高方差方向 (捕获类内的内容变化)
        within_class_dirs = []
        for phone_id in active_phones:
            mask = (phones_bal == phone_id)
            X_class = features_centered[mask]
            if len(X_class) < 10:
                continue

            # 类内 PCA
            X_class_centered = X_class - X_class.mean(axis=0)
            U, S, Vt = np.linalg.svd(X_class_centered, full_matrices=False)

            # 取前几个主方向
            n_take = min(3, len(S))
            within_class_dirs.append(Vt[:n_take].T)

        if within_class_dirs:
            V_within = np.hstack(within_class_dirs)
            print(f"  Within-class dirs: {V_within.shape[1]}")
        else:
            V_within = np.zeros((self.feature_dim, 0))

        # Step 3: 合并并正交化
        V_combined = np.hstack([V_centers, V_within])

        # QR 分解正交化
        Q, R = np.linalg.qr(V_combined)

        # 取前 n_components 个方向
        n_final = min(self.n_components, Q.shape[1])
        self.U_c = Q[:, :n_final].astype(np.float32)

        print(f"  Final subspace dim: {n_final}")

        # 构建正交投影矩阵
        self.P_orth = np.eye(self.feature_dim, dtype=np.float32) - self.U_c @ self.U_c.T

        return self

    def project_to_style(self, features: np.ndarray) -> np.ndarray:
        """投影到风格子空间"""
        return (features @ self.P_orth).astype(np.float32)

    def get_content_component(self, features: np.ndarray) -> np.ndarray:
        """提取内容成分"""
        return (features @ self.U_c @ self.U_c.T).astype(np.float32)


# ============================================================
# 方案2: 对抗神经网络 (Adversarial Network)
# ============================================================

class ContentRemovalNetwork(nn.Module):
    """
    对抗式内容去除网络

    结构:
    - Encoder: 学习去除内容的变换
    - Phone Classifier: 对抗目标，尝试从变换后特征预测音素
    - Reconstruction: 保持特征可用性

    训练目标:
    - 最小化重建误差
    - 最大化音素分类器的困惑度 (对抗)
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 512,
                 n_phones: int = 41, device: str = 'cuda'):
        super().__init__()
        self.device = device

        # Encoder: 去除内容的变换
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Decoder: 重建原始特征
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Phone Classifier (对抗目标)
        self.phone_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_phones),
        )

        self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码 (去除内容)"""
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        """前向传播"""
        z = self.encode(x)  # 去除内容后的特征
        x_recon = self.decoder(z)  # 重建
        phone_logits = self.phone_classifier(z)  # 音素预测
        return z, x_recon, phone_logits


class AdversarialTrainer:
    """对抗训练器"""

    def __init__(self, model: ContentRemovalNetwork, n_phones: int = 41):
        self.model = model
        self.n_phones = n_phones
        self.device = model.device

    def train(self, features: np.ndarray, phones: np.ndarray,
              epochs: int = 100, batch_size: int = 2048,
              lr_enc: float = 1e-3, lr_cls: float = 1e-3,
              adv_weight: float = 1.0, recon_weight: float = 0.1):
        """
        对抗训练

        Args:
            adv_weight: 对抗损失权重 (越大越强调去除内容)
            recon_weight: 重建损失权重 (保持特征可用性)
        """
        print(f"[Adversarial] Training content removal network...")
        print(f"  adv_weight={adv_weight}, recon_weight={recon_weight}")

        X = torch.from_numpy(features).float().to(self.device)
        y = torch.from_numpy(phones).long().to(self.device)
        n_samples = len(X)

        # 分离优化器
        opt_enc = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()),
            lr=lr_enc
        )
        opt_cls = torch.optim.Adam(
            self.model.phone_classifier.parameters(),
            lr=lr_cls
        )

        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(n_samples, device=self.device)

            total_loss_enc, total_loss_cls = 0, 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                # Step 1: 训练分类器 (尝试预测音素)
                z, _, phone_logits = self.model(X_batch)
                loss_cls = ce_loss(phone_logits, y_batch)

                opt_cls.zero_grad()
                loss_cls.backward()
                opt_cls.step()

                # Step 2: 训练编码器 (对抗 + 重建)
                z, x_recon, phone_logits = self.model(X_batch)

                # 对抗损失: 让分类器输出接近均匀分布
                uniform = torch.ones_like(phone_logits) / self.n_phones
                loss_adv = F.kl_div(
                    F.log_softmax(phone_logits, dim=1),
                    uniform,
                    reduction='batchmean'
                )

                # 重建损失
                loss_recon = mse_loss(x_recon, X_batch)

                loss_enc = adv_weight * loss_adv + recon_weight * loss_recon

                opt_enc.zero_grad()
                loss_enc.backward()
                opt_enc.step()

                total_loss_enc += loss_enc.item()
                total_loss_cls += loss_cls.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, "
                      f"L_enc={total_loss_enc/n_batches:.4f}, "
                      f"L_cls={total_loss_cls/n_batches:.4f}")

        return self

    @torch.no_grad()
    def transform(self, features: np.ndarray) -> np.ndarray:
        """变换特征 (去除内容)"""
        self.model.eval()
        X = torch.from_numpy(features).float().to(self.device)
        z = self.model.encode(X)
        return z.cpu().numpy()


# ============================================================
# 数据加载和辅助函数
# ============================================================

def load_test_data(config_path: str, max_samples: int = 100000):
    """加载测试数据"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config['paths']['cache_dir'])
    features_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'

    if not features_h5.exists():
        raise FileNotFoundError(f"Features not found: {features_h5}")

    phone_ckpt = config.get('phone_predictor', {}).get('checkpoint',
                           './checkpoints/phone_decoder.pt')

    if Path(phone_ckpt).exists():
        phone_predictor = PhonePredictor.load(phone_ckpt, device='cuda')
    else:
        phone_predictor = None

    print(f"Loading features from {features_h5}...")
    with h5py.File(features_h5, 'r') as f:
        total_frames = f['features'].shape[0]
        if total_frames > max_samples:
            indices = np.random.choice(total_frames, max_samples, replace=False)
            indices.sort()
            features = f['features'][indices]
        else:
            features = f['features'][:]

    print(f"Loaded {len(features)} frames, shape: {features.shape}")

    if phone_predictor is not None:
        print("Predicting phones...")
        features_tensor = torch.from_numpy(features).float().cuda()
        batch_size = 10000
        phones_list = []
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i:i+batch_size]
            phones_batch = phone_predictor(batch).cpu().numpy()
            phones_list.append(phones_batch)
        phones = np.concatenate(phones_list)
    else:
        phones = np.random.randint(0, 41, size=len(features))

    return features, phones


def run_linear_probe(features: np.ndarray, phones: np.ndarray,
                     name: str, n_classes: int = 41):
    """运行 Linear Probe 测试"""
    print(f"\n{'='*50}")
    print(f"Linear Probe: {name}")
    print(f"{'='*50}")

    n_samples = len(features)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = phones[train_idx], phones[test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    probe = GPULinearProbe(features.shape[1], n_classes)
    probe.fit(X_train, y_train, epochs=50, batch_size=2048)

    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)

    print(f"\nResults for {name}:")
    print(f"  Train Acc: {train_acc:.2%}")
    print(f"  Test Acc:  {test_acc:.2%}")

    return test_acc


def main():
    """主测试流程"""
    print("="*60)
    print("DS-SAMM-Anon v3.2 Test: Disentanglement (V3 Optimized)")
    print("="*60)

    np.random.seed(42)
    torch.manual_seed(42)

    config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'

    if not config_path.exists():
        print("Config not found, using synthetic data...")
        n_samples = 30000
        feature_dim = 1024
        n_phones = 41
        phones = np.random.randint(0, n_phones, size=n_samples)
        phone_centers = np.random.randn(n_phones, feature_dim).astype(np.float32)
        features = phone_centers[phones] + np.random.randn(n_samples, feature_dim).astype(np.float32) * 0.5
    else:
        features, phones = load_test_data(str(config_path), max_samples=100000)

    # 剔除 silence
    non_sil_mask = phones != 0
    features_no_sil = features[non_sil_mask]
    phones_no_sil = phones[non_sil_mask]
    print(f"\nAfter removing silence: {len(features_no_sil)} frames")

    # Test 1: 原始特征基准
    acc_original = run_linear_probe(
        features_no_sil, phones_no_sil,
        "H_original (baseline)"
    )

    results = {'original': acc_original}

    # ============================================================
    # 方案1: 扩展子空间 (测试不同维度)
    # ============================================================
    print("\n" + "="*60)
    print("Method 1: Expanded Subspace")
    print("="*60)

    for n_comp in [50, 100, 200]:
        print(f"\n--- Testing n_components={n_comp} ---")
        projector = ExpandedSubspaceProjector(
            n_phones=41,
            feature_dim=features.shape[1],
            n_components=n_comp
        )
        projector.fit(features, phones, max_per_class=2000)

        features_style = projector.project_to_style(features_no_sil)
        acc = run_linear_probe(
            features_style, phones_no_sil,
            f"H_style (expanded, dim={n_comp})"
        )
        results[f'expanded_{n_comp}'] = acc

    # ============================================================
    # 方案2: 对抗神经网络
    # ============================================================
    print("\n" + "="*60)
    print("Method 2: Adversarial Network")
    print("="*60)

    for adv_w in [0.5, 1.0, 2.0]:
        print(f"\n--- Testing adv_weight={adv_w} ---")
        model = ContentRemovalNetwork(
            feature_dim=features.shape[1],
            hidden_dim=512,
            n_phones=41
        )
        trainer = AdversarialTrainer(model, n_phones=41)
        trainer.train(
            features_no_sil, phones_no_sil,
            epochs=100,
            adv_weight=adv_w,
            recon_weight=0.1
        )

        features_adv = trainer.transform(features_no_sil)
        acc = run_linear_probe(
            features_adv, phones_no_sil,
            f"H_style (adversarial, w={adv_w})"
        )
        results[f'adversarial_{adv_w}'] = acc

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: Disentanglement V3 Results")
    print("="*60)

    print(f"\nBaseline:")
    print(f"  H_original: {results['original']:.2%}")

    print(f"\nMethod 1 - Expanded Subspace:")
    for n_comp in [50, 100, 200]:
        key = f'expanded_{n_comp}'
        reduction = (1 - results[key] / results['original']) * 100
        print(f"  dim={n_comp}: {results[key]:.2%} (reduction: {reduction:.1f}%)")

    print(f"\nMethod 2 - Adversarial Network:")
    for adv_w in [0.5, 1.0, 2.0]:
        key = f'adversarial_{adv_w}'
        reduction = (1 - results[key] / results['original']) * 100
        print(f"  w={adv_w}: {results[key]:.2%} (reduction: {reduction:.1f}%)")

    # 找最佳方法
    best_key = min(results.keys(), key=lambda k: results[k] if k != 'original' else 1.0)
    best_acc = results[best_key]
    best_reduction = (1 - best_acc / results['original']) * 100

    print(f"\nBest method: {best_key}")
    print(f"  Accuracy: {best_acc:.2%}")
    print(f"  Content reduction: {best_reduction:.1f}%")

    if best_acc < 0.30:
        print("\n✓ SUCCESS: Content effectively removed!")
    elif best_acc < 0.50:
        print("\n◐ GOOD: Significant content reduction achieved")
    else:
        print("\n✗ NEEDS IMPROVEMENT: Try higher adv_weight or more dimensions")

    return results


if __name__ == '__main__':
    results = main()
