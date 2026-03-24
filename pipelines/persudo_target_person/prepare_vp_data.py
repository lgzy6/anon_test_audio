#!/usr/bin/env python3
"""准备 VoicePrivacy 评估数据 - 完整版"""

import json
import shutil
from pathlib import Path

def prepare_kaldi_data(anon_dir, original_dir, output_dir, suffix='_persudo'):
    """准备 Kaldi 格式数据"""
    anon_dir = Path(anon_dir)
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取映射
    with open(anon_dir / 'eval_mapping.json') as f:
        mapping = json.load(f)

    # wav.scp
    with open(output_dir / 'wav.scp', 'w') as f:
        for rel_path, paths in sorted(mapping.items()):
            utt_id = rel_path.replace('/', '-').replace('.wav', '')
            f.write(f"{utt_id} {paths['anonymized']}\n")

    # utt2spk
    with open(output_dir / 'utt2spk', 'w') as f:
        for rel_path in sorted(mapping.keys()):
            utt_id = rel_path.replace('/', '-').replace('.wav', '')
            spk_id = rel_path.split('/')[0]
            f.write(f"{utt_id} {spk_id}\n")

    # spk2utt (从 utt2spk 生成)
    spk2utt = {}
    for rel_path in mapping.keys():
        utt_id = rel_path.replace('/', '-').replace('.wav', '')
        spk_id = rel_path.split('/')[0]
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(utt_id)

    with open(output_dir / 'spk2utt', 'w') as f:
        for spk_id in sorted(spk2utt.keys()):
            utts = ' '.join(sorted(spk2utt[spk_id]))
            f.write(f"{spk_id} {utts}\n")

    print(f"✓ 生成 Kaldi 数据: {output_dir}")
    print(f"  文件数: {len(mapping)}, 说话人数: {len(spk2utt)}")


def copy_enrolls_trials(original_data_dir, output_dir):
    """复制 enrolls 和 trials 文件"""
    original_data_dir = Path(original_data_dir)
    output_dir = Path(output_dir)

    for file in ['enrolls', 'trials_f', 'trials_m', 'trials']:
        src = original_data_dir / file
        if src.exists():
            shutil.copy(src, output_dir / file)
            print(f"✓ 复制: {file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anon_dir', default='outputs/persudo_target')
    parser.add_argument('--original_dir', default='/root/autodl-tmp/datasets/LibriSpeech/test-clean')
    parser.add_argument('--output_dir', default='outputs/voiceprivacy_data')
    parser.add_argument('--vp_original_data', help='VoicePrivacy原始数据目录，用于复制enrolls/trials')
    args = parser.parse_args()

    prepare_kaldi_data(args.anon_dir, args.original_dir, args.output_dir)

    if args.vp_original_data:
        copy_enrolls_trials(args.vp_original_data, args.output_dir)
