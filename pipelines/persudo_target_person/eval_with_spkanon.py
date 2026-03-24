#!/usr/bin/env python3
"""使用 spkanon 框架评估匿名化效果"""

import os
import sys
import subprocess
from pathlib import Path

SPKANON_DIR = "/root/autodl-tmp/spkanon/spane"


def run_spkanon_eval(anon_dir, config_path=None, device='cuda'):
    """运行 spkanon 评估"""

    if config_path is None:
        config_path = f"{SPKANON_DIR}/config/config.yaml"

    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        return

    cmd = [
        "python", f"{SPKANON_DIR}/run.py",
        config_path,
        "--device", device
    ]

    print(f"运行评估命令: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SPKANON_DIR)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anon_dir', required=True, help='匿名化音频目录')
    parser.add_argument('--config', help='spkanon配置文件')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_spkanon_eval(args.anon_dir, args.config, args.device)
