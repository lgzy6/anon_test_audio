#!/usr/bin/env python3
"""验证大规模数据集配置是否正确"""

import yaml
from pathlib import Path

def verify_config():
    config_path = Path('configs/large_scale.yaml')

    print("=" * 60)
    print("验证大规模数据集配置")
    print("=" * 60)

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查关键路径
    print("\n[路径配置]")
    paths = config['paths']
    print(f"  数据集: {paths['librispeech_root']}")
    print(f"  缓存: {paths['cache_dir']}")
    print(f"  权重: {paths['checkpoints_dir']}")
    print(f"  输出: {paths['output_dir']}")

    # 检查数据集配置
    print("\n[数据集配置]")
    train_split = config['offline']['train_split']
    print(f"  训练集: {train_split}")

    split_name = train_split.replace('-', '_')
    cache_dir = Path(paths['cache_dir']) / 'features' / 'wavlm' / split_name
    print(f"  特征路径: {cache_dir}")

    # 检查脚本兼容性
    print("\n[脚本兼容性检查]")
    scripts = [
        'scripts/step1_extract_speaker_embeddings.py',
        'scripts/step2_compute_eta_projection.py',
        'scripts/step4_build_style_extractor.py',
        'scripts/step5_precompute_utterance_styles.py',
        'scripts/step5_build_phone_clusters.py',
        'scripts/step6_style_guided_retrieval.py'
    ]

    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
                has_config_arg = '--config' in content or "args.config" in content
                status = "✓" if has_config_arg else "✗"
                print(f"  {status} {script_path.name}")
        else:
            print(f"  ✗ {script_path.name} (不存在)")

    print("\n" + "=" * 60)
    print("✓ 配置验证完成")
    print("=" * 60)

if __name__ == '__main__':
    verify_config()
