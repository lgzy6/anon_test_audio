#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 Test A: 正交解耦有效性验证 (修复版)

修复内容:
1. 剔除 Silence 帧 (phone_id=0)
2. 对各音素进行平衡采样，避免数据偏斜
3. 强制子空间维度为 n_phones-1
4. 使用 GPU Linear Probe 替代 sklearn
5. 可选: 使用 LDA 替代 PCA

成功标准:
- H_style 上的音素分类准确率应显著低于 H 原始特征
- 理想情况: < 30-40%
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import h5py
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor


class GPULinearProbe:
    """GPU 加速的 Linear Probe，替代 sklearn"""
    
    def __init__(self, input_dim: int, n_classes: int, device='cuda'):
        self.device = device
        self.model = nn.Linear(input_dim, n_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 50, batch_size: int = 2048, lr: float = 1e-3):
        """训练线性分类器"""
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1)
        return preds.cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        preds = self.predict(X)
        return (preds == y).mean()


class ContentSubspaceProjectorV2:
    """
    v3.2 内容子空间投影器 (修复版)
    
    修复:
    1. 剔除 Silence 帧
    2. 平衡采样各音素
    3. 强制使用 n_phones-1 维子空间
    4. 可选 LDA 模式
    """
    
    def __init__(self, n_phones: int = 41, feature_dim: int = 1024, 
                 silence_id: int = 0, use_lda: bool = False):
        self.n_phones = n_phones
        self.feature_dim = feature_dim
        self.silence_id = silence_id
        self.use_lda = use_lda
        
        self.phone_centroids = None
        self.U_c = None
        self.P_orth = None
        self.active_phones = None  # 非silence的音素列表
        
    def _balance_sample(self, features: np.ndarray, phones: np.ndarray, 
                        max_per_class: int = 2000) -> tuple:
        """平衡采样，确保每个音素贡献均等"""
        balanced_feats = []
        balanced_phones = []
        
        for phone_id in range(self.n_phones):
            if phone_id == self.silence_id:
                continue  # 跳过 silence
                
            mask = (phones == phone_id)
            count = mask.sum()
            
            if count == 0:
                continue
                
            indices = np.where(mask)[0]
            if count > max_per_class:
                indices = np.random.choice(indices, max_per_class, replace=False)
            
            balanced_feats.append(features[indices])
            balanced_phones.append(phones[indices])
        
        return np.vstack(balanced_feats), np.concatenate(balanced_phones)
    
    def fit(self, features: np.ndarray, phones: np.ndarray, 
            max_per_class: int = 2000):
        """
        学习内容子空间 (修复版)
        """
        print(f"[V2] Learning content subspace...")
        print(f"  Original: {len(features)} frames")
        
        # Step 1: 剔除 Silence 并平衡采样
        features_bal, phones_bal = self._balance_sample(
            features, phones, max_per_class
        )
        print(f"  After balance sampling: {len(features_bal)} frames")
        
        # 统计活跃音素
        self.active_phones = np.unique(phones_bal)
        n_active = len(self.active_phones)
        print(f"  Active phones (excl. silence): {n_active}")
        
        # Step 2: 计算每个音素的中心向量
        self.phone_centroids = np.zeros((self.n_phones, self.feature_dim), dtype=np.float32)
        phone_counts = np.zeros(self.n_phones)
        
        for phone_id in self.active_phones:
            mask = (phones_bal == phone_id)
            self.phone_centroids[phone_id] = features_bal[mask].mean(axis=0)
            phone_counts[phone_id] = mask.sum()
        
        active_counts = phone_counts[self.active_phones]
        print(f"  Phone distribution: min={active_counts.min():.0f}, "
              f"max={active_counts.max():.0f}, mean={active_counts.mean():.0f}")
        
        # Step 3: 构建内容子空间
        if self.use_lda:
            self._fit_lda(features_bal, phones_bal, n_active)
        else:
            self._fit_pca(n_active)
        
        return self
    
    def _fit_pca(self, n_active: int):
        """使用 PCA 构建子空间 (强制维度)"""
        centroids_active = self.phone_centroids[self.active_phones]
        centroids_centered = centroids_active - centroids_active.mean(axis=0)
        
        U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
        
        # 强制使用 n_active - 1 维 (不再依赖方差阈值)
        n_components = n_active - 1
        
        total_var = (S ** 2).sum()
        explained_var = (S[:n_components] ** 2).sum() / total_var
        
        print(f"  [PCA] Content subspace dim: {n_components} "
              f"(explains {explained_var:.1%} variance)")
        
        self.U_c = Vt[:n_components].T
        self.P_orth = np.eye(self.feature_dim) - self.U_c @ self.U_c.T
    
    def _fit_lda(self, features: np.ndarray, phones: np.ndarray, n_active: int):
        """使用 LDA 构建子空间 (找最能区分音素的方向)"""
        print("  [LDA] Computing scatter matrices...")
        
        # 全局均值
        global_mean = features.mean(axis=0)
        
        # 类间散度 S_b 和类内散度 S_w
        S_b = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        S_w = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        
        for phone_id in self.active_phones:
            mask = (phones == phone_id)
            X_k = features[mask]
            n_k = len(X_k)
            
            mean_k = X_k.mean(axis=0)
            diff = (mean_k - global_mean).reshape(-1, 1)
            S_b += n_k * (diff @ diff.T)
            
            # 类内散度
            X_centered = X_k - mean_k
            S_w += X_centered.T @ X_centered
        
        # 正则化 S_w (防止奇异)
        S_w += np.eye(self.feature_dim) * 1e-6
        
        # 求解广义特征值问题: S_b @ v = lambda * S_w @ v
        print("  [LDA] Solving generalized eigenvalue problem...")
        try:
            # 等价于求 inv(S_w) @ S_b 的特征向量
            S_w_inv = np.linalg.inv(S_w)
            M = S_w_inv @ S_b
            eigenvalues, eigenvectors = np.linalg.eig(M)
            
            # 取实部并按特征值排序
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 取前 n_active - 1 个方向
            n_components = n_active - 1
            self.U_c = eigenvectors[:, :n_components].astype(np.float32)
            
            print(f"  [LDA] Content subspace dim: {n_components}")
            
        except np.linalg.LinAlgError as e:
            print(f"  [LDA] Failed: {e}, falling back to PCA")
            self._fit_pca(n_active)
            return
        
        self.P_orth = np.eye(self.feature_dim) - self.U_c @ self.U_c.T
    
    def project_to_style(self, features: np.ndarray) -> np.ndarray:
        """投影到风格子空间 (去除内容)"""
        if self.P_orth is None:
            raise ValueError("Must call fit() first")
        return (features @ self.P_orth).astype(np.float32)
    
    def get_content_component(self, features: np.ndarray) -> np.ndarray:
        """提取内容成分"""
        return (features - self.project_to_style(features)).astype(np.float32)


def load_test_data(config_path: str, max_samples: int = 50000):
    """加载测试数据"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cache_dir = Path(config['paths']['cache_dir'])
    features_h5 = cache_dir / 'features' / 'wavlm' / 'features.h5'
    metadata_json = cache_dir / 'features' / 'wavlm' / 'metadata.json'
    
    if not features_h5.exists():
        raise FileNotFoundError(f"Features not found: {features_h5}")
    
    phone_ckpt = config.get('phone_predictor', {}).get('checkpoint',
                           './checkpoints/phone_decoder.pt')
    
    if Path(phone_ckpt).exists():
        phone_predictor = PhonePredictor.load(phone_ckpt, device='cuda')
    else:
        print(f"Warning: Phone predictor not found, using random labels")
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


def run_gpu_linear_probe(features: np.ndarray, phones: np.ndarray, 
                         name: str, n_classes: int = 41):
    """运行 GPU Linear Probe 测试"""
    print(f"\n{'='*50}")
    print(f"GPU Linear Probe Test: {name}")
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
    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy:  {test_acc:.2%}")
    print(f"  Random Baseline: {1/n_classes:.2%}")
    
    return test_acc


def analyze_phone_distribution(phones: np.ndarray, n_phones: int = 41):
    """分析音素分布"""
    print("\n" + "="*50)
    print("Phone Distribution Analysis")
    print("="*50)
    
    counts = np.bincount(phones, minlength=n_phones)
    total = len(phones)
    
    print(f"Total frames: {total}")
    print(f"\nTop 5 phones by frequency:")
    top_indices = np.argsort(counts)[::-1][:5]
    for idx in top_indices:
        print(f"  Phone {idx}: {counts[idx]:,} ({counts[idx]/total:.1%})")
    
    # 检查 silence (假设 id=0)
    silence_ratio = counts[0] / total
    print(f"\nSilence (phone 0) ratio: {silence_ratio:.1%}")
    
    if silence_ratio > 0.5:
        print("⚠️  WARNING: Silence dominates the data!")
    
    return counts


def main():
    """主测试流程"""
    print("="*60)
    print("DS-SAMM-Anon v3.2 Test A: Disentanglement Validation (V2)")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'
    
    if not config_path.exists():
        print(f"Config not found, using synthetic data...")
        n_samples = 20000
        feature_dim = 1024
        n_phones = 41
        
        phones = np.random.randint(0, n_phones, size=n_samples)
        phone_centers = np.random.randn(n_phones, feature_dim).astype(np.float32)
        features = phone_centers[phones] + np.random.randn(n_samples, feature_dim).astype(np.float32) * 0.5
    else:
        # 增加采样量以确保各音素有足够样本
        features, phones = load_test_data(str(config_path), max_samples=100000)
    
    # 分析音素分布
    phone_counts = analyze_phone_distribution(phones)
    
    # Test 1: 原始特征 (排除 silence 的测试)
    non_silence_mask = phones != 0
    features_no_sil = features[non_silence_mask]
    phones_no_sil = phones[non_silence_mask]
    
    print(f"\nAfter removing silence: {len(features_no_sil)} frames")
    
    acc_original = run_gpu_linear_probe(
        features_no_sil, phones_no_sil, 
        "H_original (WavLM, no silence)"
    )
    
    # Test 2: 学习内容子空间 (PCA 版本)
    print("\n" + "="*50)
    print("Learning Content Subspace (PCA, balanced)...")
    print("="*50)
    
    projector_pca = ContentSubspaceProjectorV2(
        n_phones=41, 
        feature_dim=features.shape[1],
        silence_id=0,
        use_lda=False
    )
    projector_pca.fit(features, phones, max_per_class=2000)
    
    features_style_pca = projector_pca.project_to_style(features_no_sil)
    acc_style_pca = run_gpu_linear_probe(
        features_style_pca, phones_no_sil,
        "H_style (PCA, content removed)"
    )
    
    # Test 3: 学习内容子空间 (LDA 版本)
    print("\n" + "="*50)
    print("Learning Content Subspace (LDA, balanced)...")
    print("="*50)
    
    projector_lda = ContentSubspaceProjectorV2(
        n_phones=41,
        feature_dim=features.shape[1],
        silence_id=0,
        use_lda=True
    )
    projector_lda.fit(features, phones, max_per_class=2000)
    
    features_style_lda = projector_lda.project_to_style(features_no_sil)
    acc_style_lda = run_gpu_linear_probe(
        features_style_lda, phones_no_sil,
        "H_style (LDA, content removed)"
    )
    
    # Test 4: 内容成分验证
    features_content = projector_lda.get_content_component(features_no_sil)
    acc_content = run_gpu_linear_probe(
        features_content, phones_no_sil,
        "H_content (content only)"
    )
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY: Disentanglement Validation (V2)")
    print("="*60)
    print(f"  H_original accuracy:    {acc_original:.2%}")
    print(f"  H_style (PCA) accuracy: {acc_style_pca:.2%}  (should be LOW)")
    print(f"  H_style (LDA) accuracy: {acc_style_lda:.2%}  (should be LOW)")
    print(f"  H_content accuracy:     {acc_content:.2%}  (should be HIGH)")
    print(f"  Random baseline:        {1/41:.2%}")
    print()
    
    # 计算降幅
    reduction_pca = (1 - acc_style_pca / acc_original) * 100
    reduction_lda = (1 - acc_style_lda / acc_original) * 100
    
    print(f"Content reduction (PCA): {reduction_pca:.1f}%")
    print(f"Content reduction (LDA): {reduction_lda:.1f}%")
    print()
    
    # 判断成功
    best_style_acc = min(acc_style_pca, acc_style_lda)
    if best_style_acc < 0.40:
        print("✓ SUCCESS: H_style has significantly less content information")
    elif best_style_acc < acc_original * 0.6:
        print("◐ PARTIAL: Some content removed, but could be better")
    else:
        print("✗ FAILED: H_style still contains too much content")
        print("  Consider: more aggressive subspace, nonlinear methods")
    
    return {
        'acc_original': acc_original,
        'acc_style_pca': acc_style_pca,
        'acc_style_lda': acc_style_lda,
        'acc_content': acc_content,
    }


if __name__ == '__main__':
    results = main()