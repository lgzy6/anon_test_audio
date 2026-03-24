#!/usr/bin/env python3
"""准备 spkanon 评估所需的文件列表"""

import json
from pathlib import Path

anon_dir = Path('outputs/persudo_target')
output_file = anon_dir / 'anon_eval.txt'

with open(anon_dir / 'eval_mapping.json') as f:
    mapping = json.load(f)

with open(output_file, 'w') as f:
    for rel_path, paths in mapping.items():
        f.write(f"{paths['anonymized']}\n")

print(f"生成 spkanon 评估列表: {output_file}")
