#!/usr/bin/env python3
"""
系统诊断脚本 - 诊断语义丢失的根本原因
"""

import sys
import torch
import yaml
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_config():
    """检查配置是否正确"""
    print("\n" + "="*70)
    print("1. 配置检查")
    print("="*70)

    config_path = Path("configs/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eta_wavlm = config.get('eta_wavlm', {})
    knn_vc = config.get('knn_vc', {})

    print(f"\n✓ Eta-WavLM 配置:")
    print(f"  - enabled: {eta_wavlm.get('enabled', 'NOT SET')}")

    print(f"\n✓ kNN-VC 配置:")
    print(f"  - use_top1: {knn_vc.get('use_top1', 'NOT SET')}")
    print(f"  - use_cosine: {knn_vc.get('use_cosine', 'NOT SET')}")
    print(f"  - k_neighbors: {knn_vc.get('k_neighbors', 4)}")

    issues = []
    if not eta_wavlm.get('enabled', False):
        issues.append("⚠️  Eta-WavLM 未启用")
    if not knn_vc.get('use_top1', False):
        issues.append("⚠️  Top-1 未启用")
    if not knn_vc.get('use_cosine', False):
        issues.append("⚠️  余弦相似度未启用")

    if issues:
        print(f"\n问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✓ 配置正确")

    return len(issues) == 0


def check_target_pool():
    """检查 Target Pool 是否使用了 cleaned 特征"""
    print("\n" + "="*70)
    print("2. Target Pool 检查")
    print("="*70)

    pool_dir = Path("checkpoints/target_pool")
    if not pool_dir.exists():
        pool_dir = Path("data/samm_anon/checkpoints/target_pool")

    if not pool_dir.exists():
        print("✗ Target Pool 不存在")
        return False

    # 检查 metadata
    metadata_path = pool_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        cleaned = metadata.get('cleaned', False)
        print(f"\n✓ Target Pool metadata:")
        print(f"  - 使用 cleaned 特征: {cleaned}")
        print(f"  - 总帧数: {metadata.get('total_frames', 'N/A')}")

        if not cleaned:
            print(f"\n⚠️  严重问题: Target Pool 使用的是原始特征，不是 cleaned 特征!")
            print(f"  这会导致特征空间不匹配，必须重建 Target Pool")
            return False
    else:
        print(f"\n⚠️  未找到 metadata.json")

    # 检查 features
    features_path = pool_dir / "features.npy"
    if features_path.exists():
        features = np.load(features_path)
        print(f"\n✓ Features:")
        print(f"  - 形状: {features.shape}")
        print(f"  - 均值: {features.mean():.4f}")
        print(f"  - 标准差: {features.std():.4f}")

    return True


def check_retriever_actual_behavior():
    """检查检索器实际运行时的行为"""
    print("\n" + "="*70)
    print("3. 检索器运行时检查")
    print("="*70)

    from models.knn_vc.retriever import ConstrainedKNNRetriever

    try:
        retriever = ConstrainedKNNRetriever(
            target_pool_path="data/samm_anon/checkpoints/target_pool",
            k=4,
            use_top1=True,
            use_cosine=True,
            device='cpu'
        )

        print(f"\n✓ 检索器初始化成功:")
        print(f"  - use_top1: {retriever.use_top1}")
        print(f"  - use_cosine: {retriever.use_cosine}")
        print(f"  - k: {retriever.k}")
        print(f"  - 目标池大小: {len(retriever.features)}")

        if hasattr(retriever, 'phones') and retriever.phones is not None:
            print(f"  - 音素标签: 存在")
        else:
            print(f"  - 音素标签: ⚠️  不存在 (音素约束无效)")

        return True

    except Exception as e:
        print(f"\n✗ 检索器初始化失败: {e}")
        return False


def check_samm_impact():
    """检查 SAMM 掩码的影响"""
    print("\n" + "="*70)
    print("4. SAMM 掩码影响检查")
    print("="*70)

    # 检查最近的测试结果
    outputs_dir = Path("outputs")
    test_dirs = sorted(outputs_dir.glob("test_v2.1_*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    if not test_dirs:
        print("✗ 未找到测试结果")
        return

    latest = test_dirs[0]
    print(f"\n分析最新测试: {latest.name}")

    # 如果有中间特征，分析它们
    intermediates_dir = latest / "intermediates"
    if intermediates_dir.exists():
        print("\n✓ 找到中间特征，分析中...")
        # TODO: 加载并分析中间特征
    else:
        print("\n⚠️  未保存中间特征，无法详细分析")


def check_phone_predictor():
    """检查音素预测器"""
    print("\n" + "="*70)
    print("5. 音素预测器检查")
    print("="*70)

    phone_ckpt = Path("checkpoints/phone_decoder.pt")
    if not phone_ckpt.exists():
        print("✗ 音素预测器不存在")
        return False

    print(f"✓ 音素预测器存在: {phone_ckpt}")
    return True


def diagnose_failure_mode():
    """诊断失败模式"""
    print("\n" + "="*70)
    print("诊断总结与建议")
    print("="*70)

    print(f"\n根据 WER 87.50% 的结果，可能的原因:")

    print(f"\n1. 【高优先级】Target Pool 特征空间不匹配")
    print(f"   - 症状: 检索到的特征与源特征不在同一空间")
    print(f"   - 检查: Target Pool metadata 中的 'cleaned' 字段")
    print(f"   - 修复: 重建 Target Pool")

    print(f"\n2. 【高优先级】SAMM 掩码破坏了语义")
    print(f"   - 症状: 掩码符号 -1 导致检索失败")
    print(f"   - 检查: 掩码比例是否过高 (>25%)")
    print(f"   - 修复: 降低掩码比例或改进 Pattern Matrix")

    print(f"\n3. 【中优先级】kNN 约束过于严格")
    print(f"   - 症状: 很多帧找不到足够的候选")
    print(f"   - 检查: 音素约束 + 性别约束 + 符号约束同时启用")
    print(f"   - 修复: 放松符号约束或扩大 k")

    print(f"\n4. 【中优先级】时长调整破坏了对齐")
    print(f"   - 症状: 时长变化过大 (原始 4.39s → 匿名 4.84s)")
    print(f"   - 检查: 时长预测器的权重")
    print(f"   - 修复: 降低 duration_predictor_weight")

    print(f"\n5. 【低优先级】WavLM Layer 15 泄漏过多身份")
    print(f"   - 症状: 即使用 Eta-WavLM 也无法完全去除")
    print(f"   - 修复: 改用 Layer 6 或 9")


def main():
    """主诊断流程"""
    print("="*70)
    print("anon_test v2.1 - 系统诊断")
    print("="*70)

    all_ok = True

    # 1. 配置检查
    if not check_config():
        all_ok = False

    # 2. Target Pool 检查
    if not check_target_pool():
        all_ok = False

    # 3. 检索器检查
    if not check_retriever_actual_behavior():
        all_ok = False

    # 4. SAMM 影响
    check_samm_impact()

    # 5. 音素预测器
    check_phone_predictor()

    # 6. 诊断总结
    diagnose_failure_mode()

    if not all_ok:
        print(f"\n" + "="*70)
        print("⚠️  发现问题，建议按以下顺序修复:")
        print("="*70)
        print(f"\n1. 检查并重建 Target Pool (如果使用了原始特征)")
        print(f"2. 验证配置文件是否正确")
        print(f"3. 运行简化测试 (禁用 SAMM 掩码)")
        print(f"\n运行修复命令:")
        print(f"  python scripts/rebuild_target_pool.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
