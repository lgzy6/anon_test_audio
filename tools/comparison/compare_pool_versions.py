#!/usr/bin/env python3
"""
Target Pool 版本对比测试脚本
对比不同版本的 Target Pool 对隐私和效用的影响
"""

import sys
import json
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class PoolVersion:
    """Target Pool 版本信息"""
    name: str
    path: Path
    description: str
    size_mb: float
    has_phone_clusters: bool
    has_cleaned_flag: bool
    num_features: int


def analyze_pool_version(pool_path: Path) -> Optional[PoolVersion]:
    """分析 Target Pool 版本"""
    if not pool_path.exists():
        return None

    # 读取 metadata
    metadata_path = pool_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # 检查 features
    features_path = pool_path / "features.npy"
    if not features_path.exists():
        features_path = pool_path / "prototypes.npy"  # broken 版本使用的名字

    num_features = 0
    if features_path.exists():
        features = np.load(features_path, mmap_mode='r')
        num_features = len(features)

    # 检查 phone_clusters
    has_phone_clusters = (pool_path / "phone_clusters.pt").exists()

    # 检查是否使用 cleaned 特征
    has_cleaned_flag = metadata.get('cleaned', False)

    # 计算大小
    size_mb = sum(f.stat().st_size for f in pool_path.glob("*") if f.is_file()) / 1024 / 1024

    return PoolVersion(
        name=pool_path.name,
        path=pool_path,
        description=metadata.get('description', ''),
        size_mb=size_mb,
        has_phone_clusters=has_phone_clusters,
        has_cleaned_flag=has_cleaned_flag,
        num_features=num_features
    )


def find_all_pool_versions(base_dir: Path) -> List[PoolVersion]:
    """找到所有 Target Pool 版本"""
    versions = []

    # 搜索所有可能的位置
    search_paths = [
        base_dir / "data/samm_anon/checkpoints",
        base_dir / "checkpoints",
    ]

    found_pools = set()

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for pool_dir in search_path.glob("target_pool*"):
            # 跳过软链接
            if pool_dir.is_symlink():
                continue

            # 跳过已经找到的
            if pool_dir.resolve() in found_pools:
                continue

            version = analyze_pool_version(pool_dir)
            if version and version.num_features > 0:
                versions.append(version)
                found_pools.add(pool_dir.resolve())

    return sorted(versions, key=lambda v: v.name)


def run_anonymization_test(pool_version: PoolVersion, test_audio: str, config_template: str,
                           output_base: Path) -> Optional[Dict]:
    """运行单个版本的匿名化测试"""
    from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig

    print(f"\n{'='*70}")
    print(f"测试 Target Pool: {pool_version.name}")
    print(f"{'='*70}")
    print(f"路径: {pool_version.path}")
    print(f"特征数: {pool_version.num_features:,}")
    print(f"Phone Clusters: {'✓' if pool_version.has_phone_clusters else '✗'}")
    print(f"Cleaned 特征: {'✓' if pool_version.has_cleaned_flag else '✗'}")
    print(f"大小: {pool_version.size_mb:.1f} MB")

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_base / f"{pool_version.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 临时切换 target pool
    original_pool = Path("data/samm_anon/checkpoints/target_pool")
    temp_pool_link = Path("data/samm_anon/checkpoints/target_pool_temp")

    try:
        # 创建临时软链接
        if temp_pool_link.exists() or temp_pool_link.is_symlink():
            temp_pool_link.unlink()
        temp_pool_link.symlink_to(pool_version.path.resolve())

        # 加载配置
        config = AnonymizerConfig.from_yaml(config_template)
        config.target_pool_path = str(temp_pool_link)

        print(f"\n运行匿名化...")
        print(f"  配置: use_eta_wavlm={config.use_eta_wavlm}, use_top1={config.use_top1}")

        # 初始化匿名器
        anonymizer = SpeechAnonymizer(config)

        # 执行匿名化
        result = anonymizer.anonymize_file(
            input_path=test_audio,
            output_path=str(output_dir / "anonymized.wav"),
            source_gender='M',
            save_original=True
        )

        print(f"✓ 匿名化完成")

        # 保存版本信息
        version_info = {
            'pool_name': pool_version.name,
            'pool_path': str(pool_version.path),
            'num_features': pool_version.num_features,
            'has_phone_clusters': pool_version.has_phone_clusters,
            'has_cleaned_flag': pool_version.has_cleaned_flag,
            'size_mb': pool_version.size_mb,
            'config': {
                'use_eta_wavlm': config.use_eta_wavlm,
                'use_top1': config.use_top1,
                'use_cosine': config.use_cosine,
                'ssl_layer': config.ssl_layer
            }
        }

        with open(output_dir / "pool_version.json", 'w') as f:
            json.dump(version_info, f, indent=2)

        return {
            'pool_version': pool_version,
            'output_dir': output_dir,
            'success': True
        }

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pool_version': pool_version,
            'output_dir': output_dir,
            'success': False,
            'error': str(e)
        }

    finally:
        # 清理临时链接
        if temp_pool_link.exists() or temp_pool_link.is_symlink():
            temp_pool_link.unlink()


def evaluate_result(output_dir: Path) -> Optional[Dict]:
    """评估单个测试结果"""
    import subprocess

    original = output_dir / "original.wav"
    anonymized = output_dir / "anonymized.wav"

    if not original.exists() or not anonymized.exists():
        return None

    print(f"\n评估语义保留...")

    try:
        result = subprocess.run(
            ['python', 'evaluate_semantic_preservation.py',
             '--original', str(original),
             '--anonymized', str(anonymized),
             '--output', str(output_dir / 'evaluation')],
            capture_output=True,
            text=True,
            check=True
        )

        # 读取评估结果
        eval_file = output_dir / 'evaluation' / 'semantic_evaluation.json'
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            return eval_data['metrics']

    except subprocess.CalledProcessError as e:
        print(f"✗ 评估失败: {e}")
        return None


def generate_comparison_report(results: List[Dict], output_path: Path):
    """生成对比报告"""
    print("\n" + "="*70)
    print("对比报告")
    print("="*70)

    # 准备表格数据
    table_data = []

    for result in results:
        if not result['success']:
            continue

        pool_version = result['pool_version']
        metrics = result.get('metrics', {})

        row = {
            'Target Pool': pool_version.name,
            '特征数': f"{pool_version.num_features:,}",
            'Phone Clusters': '✓' if pool_version.has_phone_clusters else '✗',
            'Cleaned': '✓' if pool_version.has_cleaned_flag else '✗',
            'WER': f"{metrics.get('wer', 0):.2%}" if metrics else 'N/A',
            '相似度': f"{metrics.get('jaccard_similarity', 0):.2%}" if metrics else 'N/A',
            '大小(MB)': f"{pool_version.size_mb:.1f}"
        }
        table_data.append(row)

    # 打印表格
    if table_data:
        headers = list(table_data[0].keys())
        col_widths = {h: max(len(h), max(len(str(row[h])) for row in table_data)) for h in headers}

        # 打印表头
        print("\n" + "-"*70)
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-"*70)

        # 打印数据
        for row in table_data:
            data_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            print(data_line)

        print("-"*70)

    # 保存详细报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': table_data,
        'details': results
    }

    with open(output_path / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存: {output_path / 'comparison_report.json'}")

    # 推荐
    if table_data:
        print(f"\n推荐:")
        # 找 WER 最低的
        valid_results = [r for r in table_data if r['WER'] != 'N/A']
        if valid_results:
            best = min(valid_results, key=lambda r: float(r['WER'].rstrip('%')))
            print(f"  最佳语义保留: {best['Target Pool']} (WER: {best['WER']})")


def main():
    """主函数"""
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Target Pool 版本对比测试")
    parser.add_argument("--audio", "-a", help="测试音频路径")
    parser.add_argument("--config", "-c", default="configs/test_no_eta.yaml",
                       help="配置模板")
    parser.add_argument("--output", "-o", default="outputs/pool_comparison",
                       help="输出目录")

    args = parser.parse_args()

    base_dir = Path.cwd()
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Target Pool 版本对比测试")
    print("="*70)

    # 1. 找到所有版本
    print("\n[1/4] 搜索 Target Pool 版本...")
    versions = find_all_pool_versions(base_dir)

    if not versions:
        print("✗ 未找到任何 Target Pool")
        return

    print(f"\n找到 {len(versions)} 个版本:")
    for v in versions:
        print(f"  - {v.name}: {v.num_features:,} 特征, "
              f"Phone Clusters: {'✓' if v.has_phone_clusters else '✗'}, "
              f"Cleaned: {'✓' if v.has_cleaned_flag else '✗'}")

    # 2. 选择测试音频
    print("\n[2/4] 准备测试音频...")
    if args.audio:
        test_audio = args.audio
    else:
        # 随机选择
        audio_files = list(Path('/root/autodl-tmp/datasets/LibriSpeech/test-clean').glob('**/*.flac'))
        test_audio = str(random.choice(audio_files))

    print(f"测试音频: {test_audio}")

    # 3. 运行测试
    print("\n[3/4] 运行匿名化测试...")
    results = []

    for version in versions:
        result = run_anonymization_test(
            version, test_audio, args.config, output_base
        )
        if result and result['success']:
            # 评估
            metrics = evaluate_result(result['output_dir'])
            result['metrics'] = metrics
        results.append(result)

    # 4. 生成报告
    print("\n[4/4] 生成对比报告...")
    generate_comparison_report(results, output_base)

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
    print(f"所有结果保存在: {output_base}")


if __name__ == "__main__":
    main()
