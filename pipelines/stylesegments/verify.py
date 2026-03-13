#!/usr/bin/env python3
"""验证stylesegments管线配置"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_imports():
    """检查模块导入"""
    print("Checking imports...")
    try:
        from pipelines.stylesegments import (
            FeatureExtractor, run_feature_extraction,
            extract_speaker_embeddings, compute_eta_projection,
            precompute_phones, SegmentStyleExtractor,
            build_style_extractor, build_phone_clusters
        )
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def check_config():
    """检查配置文件"""
    print("\nChecking config...")
    import yaml
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'base.yaml'

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'layers' in config.get('ssl', {}):
        layers = config['ssl']['layers']
        print(f"✓ Config loaded, layers: {layers}")
        return True
    else:
        print("✗ 'layers' not found in ssl config")
        return False

def check_structure():
    """检查目录结构"""
    print("\nChecking structure...")
    base = Path(__file__).parent
    required = [
        '__init__.py', 'feature_extraction.py', 'speaker_embeddings.py',
        'eta_projection.py', 'phone_precompute.py', 'style_extractor.py',
        'phone_clusters.py', 'runner.py', 'README.md'
    ]

    missing = [f for f in required if not (base / f).exists()]
    if missing:
        print(f"✗ Missing files: {missing}")
        return False

    print(f"✓ All {len(required)} files present")
    return True

if __name__ == '__main__':
    print("="*60)
    print("Style Segments Pipeline Verification")
    print("="*60)

    results = [
        check_structure(),
        check_imports(),
        check_config()
    ]

    print("\n" + "="*60)
    if all(results):
        print("✓ All checks passed!")
        print("\nUsage:")
        print("  python pipelines/stylesegments/runner.py --config configs/base.yaml")
    else:
        print("✗ Some checks failed")
        sys.exit(1)
    print("="*60)
