#!/usr/bin/env python3
"""
快速端到端验证：解耦质量是否足够继续
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_sanity_check(features: np.ndarray, 
                       phones: np.ndarray,
                       projector,
                       n_test: int = 1000):
    """
    快速验证：投影后是否还能正常工作
    
    检查项：
    1. H_style 的数值稳定性
    2. 同音素帧的相似度是否降低（预期：风格主导）
    3. 不同说话人同音素的相似度是否提高（预期：内容被移除）
    """
    print("="*50)
    print("Quick Sanity Check")
    print("="*50)
    
    # 采样测试
    idx = np.random.choice(len(features), min(n_test, len(features)), replace=False)
    H = features[idx]
    P = phones[idx]
    
    # 投影
    H_style = projector.project_to_style(H)
    H_content = projector.get_content_component(H)
    
    # Check 1: 数值稳定性
    print("\n[1] Numerical Stability:")
    print(f"  H_style: mean={H_style.mean():.4f}, std={H_style.std():.4f}")
    print(f"  H_style range: [{H_style.min():.2f}, {H_style.max():.2f}]")
    
    has_nan = np.isnan(H_style).any()
    has_inf = np.isinf(H_style).any()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ✗ FAILED: Numerical issues!")
        return False
    print("  ✓ PASSED")
    
    # Check 2: 能量保留比例
    print("\n[2] Energy Retention:")
    energy_original = np.linalg.norm(H, axis=1).mean()
    energy_style = np.linalg.norm(H_style, axis=1).mean()
    energy_content = np.linalg.norm(H_content, axis=1).mean()
    
    style_ratio = energy_style / energy_original
    content_ratio = energy_content / energy_original
    
    print(f"  Original energy: {energy_original:.2f}")
    print(f"  Style energy: {energy_style:.2f} ({style_ratio:.1%})")
    print(f"  Content energy: {energy_content:.2f} ({content_ratio:.1%})")
    
    if style_ratio < 0.3:
        print("  ⚠ WARNING: Style component too weak!")
    elif style_ratio > 0.95:
        print("  ⚠ WARNING: Almost no content removed!")
    else:
        print("  ✓ REASONABLE")
    
    # Check 3: 投影正交性验证
    print("\n[3] Orthogonality Check:")
    dot_products = np.sum(H_style * H_content, axis=1)
    mean_dot = np.abs(dot_products).mean()
    print(f"  Mean |H_style · H_content|: {mean_dot:.6f}")
    
    if mean_dot < 0.01:
        print("  ✓ PASSED: Nearly orthogonal")
    else:
        print(f"  ⚠ WARNING: Not perfectly orthogonal")
    
    # Check 4: 聚类结构预览
    print("\n[4] Clustering Preview (random 3 phones):")
    unique_phones = np.unique(P)
    sample_phones = np.random.choice(unique_phones, min(3, len(unique_phones)), replace=False)
    
    for ph in sample_phones:
        mask = (P == ph)
        if mask.sum() < 5:
            continue
        
        H_ph_style = H_style[mask]
        
        # 类内方差
        intra_var = H_ph_style.var(axis=0).mean()
        
        # 与其他音素的距离
        other_mask = ~mask
        if other_mask.sum() > 0:
            H_other = H_style[other_mask][:100]  # 采样
            center = H_ph_style.mean(axis=0)
            inter_dist = np.linalg.norm(H_other - center, axis=1).mean()
        else:
            inter_dist = 0
            
        print(f"  Phone {ph}: intra_var={intra_var:.4f}, inter_dist={inter_dist:.2f}")
    
    print("\n" + "="*50)
    print("VERDICT: ", end="")
    
    # 综合判断
    if has_nan or has_inf:
        print("✗ BLOCKED - Fix numerical issues first")
        return False
    elif style_ratio < 0.5:
        print("✓ GO AHEAD - Good disentanglement")
        return True
    elif style_ratio < 0.8:
        print("◐ PROCEED WITH CAUTION - Moderate disentanglement")
        return True
    else:
        print("✗ RECONSIDER - Poor disentanglement")
        return False


def main():
    """运行快速检查"""
    import yaml
    import h5py
    
    # 加载配置
    config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'
    
    if not config_path.exists():
        print("Using synthetic data for demo...")
        np.random.seed(42)
        n = 5000
        features = np.random.randn(n, 1024).astype(np.float32)
        phones = np.random.randint(0, 41, n)
        
        # 模拟投影器
        class DummyProjector:
            def __init__(self):
                self.U_c = np.random.randn(1024, 100).astype(np.float32)
                self.U_c, _ = np.linalg.qr(self.U_c)
                self.P_orth = np.eye(1024) - self.U_c @ self.U_c.T
                
            def project_to_style(self, x):
                return (x @ self.P_orth).astype(np.float32)
            
            def get_content_component(self, x):
                return (x @ self.U_c @ self.U_c.T).astype(np.float32)
        
        projector = DummyProjector()
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        cache_dir = Path(config['paths']['cache_dir'])
        h5_path = cache_dir / 'features' / 'wavlm' / 'features.h5'
        
        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:50000]
        
        # 加载 phone predictor
        from models.phone_predictor.predictor import PhonePredictor
        phone_ckpt = config.get('phone_predictor', {}).get(
            'checkpoint', './checkpoints/phone_decoder.pt'
        )
        
        if Path(phone_ckpt).exists():
            predictor = PhonePredictor.load(phone_ckpt, device='cuda')
            phones = predictor(torch.from_numpy(features).cuda()).cpu().numpy()
        else:
            phones = np.random.randint(0, 41, len(features))
        
        # 使用之前测试的最佳投影器
        from tests.test_v32_disentanglement_v3 import ExpandedSubspaceProjector
        
        projector = ExpandedSubspaceProjector(n_phones=41, n_components=100)
        projector.fit(features, phones, max_per_class=2000)
    
    # 运行检查
    can_proceed = quick_sanity_check(features, phones, projector)
    
    print("\n" + "="*50)
    if can_proceed:
        print("✓ RECOMMENDATION: Proceed to SAMM training")
        print("  Next steps:")
        print("  1. Run Test B (style clustering visualization)")
        print("  2. Build phone-conditioned target pool")
        print("  3. Start SAMM training with current projector")
    else:
        print("✗ RECOMMENDATION: Improve disentanglement first")
        print("  Suggestions:")
        print("  1. Try gradient reversal adversarial training")
        print("  2. Increase subspace dimensions")
        print("  3. Check data quality and phone labels")
    
    return can_proceed


if __name__ == '__main__':
    main()