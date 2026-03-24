#!/usr/bin/env python3
"""批量匿名化脚本 - 处理整个测试集"""

import sys
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent

from pipelines.persudo_target_person.synthesize_pseudo import (
    load_models, extract_source_features, anonymize_with_duration, synthesize_audio
)


def batch_anonymize(input_dir, output_dir, bank_path, metadata_path=None, k=4, dur_weight=0.5, device='cuda'):
    """批量匿名化"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    models = load_models(bank_path, device, dur_weight)

    # 获取音频文件列表
    audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.flac"))

    print(f"\n找到 {len(audio_files)} 个音频文件")

    results = []
    for audio_path in tqdm(audio_files, desc="批量处理"):
        try:
            # 保持目录结构
            rel_path = audio_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.wav')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 提取特征
            source = extract_source_features(str(audio_path), models, device)

            # 匿名化
            h_anon = anonymize_with_duration(
                source['wavlm'], source['phones'], models, device, k
            )

            # 合成
            synthesize_audio(h_anon, models['vocoder'], str(output_path))

            results.append({
                'input': str(audio_path),
                'output': str(output_path),
                'status': 'success'
            })

        except Exception as e:
            print(f"\n错误: {audio_path} - {e}")
            results.append({
                'input': str(audio_path),
                'status': 'failed',
                'error': str(e)
            })

    # 保存结果
    result_file = output_dir / 'batch_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n完成: {success}/{len(results)} 成功")
    print(f"结果保存至: {result_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--bank', default='checkpoints/pseudo_bank.pt')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--dur_weight', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    batch_anonymize(
        args.input, args.output, args.bank,
        k=args.k, dur_weight=args.dur_weight, device=args.device
    )
