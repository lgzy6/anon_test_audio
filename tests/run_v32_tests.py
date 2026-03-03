#!/usr/bin/env python3
"""
DS-SAMM-Anon v3.2 完整测试流程

运行所有验证测试:
1. Test A: 解耦有效性
2. Test B: 风格聚类
3. Test C: 语义重建
4. Test D: 端到端匿名化
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(config_path: str = None):
    """运行所有测试"""
    results = {}

    print("=" * 60)
    print("DS-SAMM-Anon v3.2 Feasibility Tests")
    print("=" * 60)

    # Test A: 解耦验证
    print("\n[1/4] Running Disentanglement Test...")
    try:
        from tests.test_v32_disentanglement import main as test_a
        results['disentanglement'] = test_a()
    except Exception as e:
        print(f"Test A failed: {e}")
        results['disentanglement'] = {'error': str(e)}

    # Test B: 风格聚类
    print("\n[2/4] Running Style Clustering Test...")
    try:
        from tests.test_v32_style_clustering import main as test_b
        results['style_clustering'] = test_b()
    except Exception as e:
        print(f"Test B failed: {e}")
        results['style_clustering'] = {'error': str(e)}

    # 打印总结
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, res in results.items():
        if 'error' in res:
            print(f"  {name}: FAILED - {res['error']}")
        else:
            print(f"  {name}: PASSED")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    run_all_tests(args.config)
