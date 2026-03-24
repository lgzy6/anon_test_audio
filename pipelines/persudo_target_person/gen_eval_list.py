#!/usr/bin/env python3
"""生成 spkanon 所需的数据列表"""

import json
from pathlib import Path

# 读取映射文件
with open('outputs/persudo_target/eval_mapping.json') as f:
    mapping = json.load(f)

# 生成匿名化音频列表
anon_list = []
for rel_path, paths in mapping.items():
    anon_list.append(paths['anonymized'])

# 保存为 spkanon 格式
output_dir = Path('outputs/persudo_target')
with open(output_dir / 'anon_eval.txt', 'w') as f:
    for path in anon_list:
        f.write(f"{path}\n")

print(f"生成 {len(anon_list)} 个文件的评估列表")
