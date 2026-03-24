#!/usr/bin/env python3
"""完整的匿名化流程：批量处理 + 评估"""

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


def batch_anonymize(input_dir, output_base='outputs/persudo_target', bank_path='checkpoints/pseudo_bank.pt',
                    k=4, dur_weight=0.5, device='cuda'):
    """批量匿名化，保持原始目录结构"""
    input_dir = Path(input_dir)
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(bank_path, device, dur_weight)
    audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.flac"))

    print(f"\n找到 {len(audio_files)} 个音频文件")

    results = []
    for audio_path in tqdm(audio_files, desc="批量处理"):
        try:
            # 保持原始目录结构
            rel_path = audio_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.wav')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            source = extract_source_features(str(audio_path), models, device)
            h_anon = anonymize_with_duration(source['wavlm'], source['phones'], models, device, k)
            synthesize_audio(h_anon, models['vocoder'], str(output_path))

            results.append({'input': str(rel_path), 'output': str(output_path), 'status': 'success'})
        except Exception as e:
            print(f"\n错误: {audio_path} - {e}")
            results.append({'input': str(audio_path.relative_to(input_dir)), 'status': 'failed', 'error': str(e)})

    result_file = output_dir / 'batch_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n完成: {success}/{len(results)} 成功")
    return output_dir, results


def prepare_eval_files(anon_dir, original_dir):
    """准备评估所需的文件映射"""
    anon_dir = Path(anon_dir)
    original_dir = Path(original_dir)

    mapping_file = anon_dir / 'eval_mapping.json'
    mapping = {}

    for anon_path in sorted(anon_dir.rglob("*.wav")):
        rel_path = anon_path.relative_to(anon_dir)
        # 原始文件可能是 .flac 格式
        orig_flac = original_dir / rel_path.with_suffix('.flac')
        orig_wav = original_dir / rel_path

        if orig_flac.exists():
            mapping[str(rel_path)] = {'original': str(orig_flac), 'anonymized': str(anon_path)}
        elif orig_wav.exists():
            mapping[str(rel_path)] = {'original': str(orig_wav), 'anonymized': str(anon_path)}

    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"生成评估映射: {mapping_file} ({len(mapping)} 对文件)")
    return mapping_file


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', default='outputs/persudo_target', help='输出基础目录')
    parser.add_argument('--bank', default='checkpoints/pseudo_bank.pt')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--dur_weight', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # 批量匿名化
    output_dir, results = batch_anonymize(
        args.input, args.output, args.bank, args.k, args.dur_weight, args.device
    )

    # 生成评估映射
    prepare_eval_files(output_dir, args.input)

    print("\n✓ 全部完成")


if __name__ == '__main__':
    main()
