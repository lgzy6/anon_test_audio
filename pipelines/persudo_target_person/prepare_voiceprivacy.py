#!/usr/bin/env python3
"""准备 VoicePrivacy 评估所需的 Kaldi 格式数据"""

import json
from pathlib import Path

# 读取映射
with open('outputs/persudo_target/eval_mapping.json') as f:
    mapping = json.load(f)

# 创建输出目录
vp_data_dir = Path('outputs/voiceprivacy_eval')
vp_data_dir.mkdir(exist_ok=True)

# 生成 wav.scp (音频路径)
with open(vp_data_dir / 'wav.scp', 'w') as f:
    for rel_path, paths in mapping.items():
        utt_id = rel_path.replace('/', '-').replace('.wav', '')
        f.write(f"{utt_id} {paths['anonymized']}\n")

# 生成 utt2spk (话语到说话人映射)
with open(vp_data_dir / 'utt2spk', 'w') as f:
    for rel_path in mapping.keys():
        utt_id = rel_path.replace('/', '-').replace('.wav', '')
        spk_id = rel_path.split('/')[0]  # 第一级目录是说话人ID
        f.write(f"{utt_id} {spk_id}\n")

print(f"生成 Kaldi 格式数据: {vp_data_dir}")
print(f"文件数: {len(mapping)}")
