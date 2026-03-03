# scripts/run_offline.py
#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from pipelines.offline.runner import OfflineRunner


def main():
    parser = argparse.ArgumentParser(description='SAMM-Anon Offline Pipeline')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/server.yaml',
        help='配置文件路径 (默认: configs/server.yaml)'
    )
    
    parser.add_argument(
        '--step', '-s',
        type=int,
        nargs='+',
        default=None,
        help='指定运行的步骤 (1-6)，如:  --step 1 2 3'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='起始步骤 (默认: 1)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=6,
        help='结束步骤 (默认: 6)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='仅显示状态，不运行'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 创建 Runner
    runner = OfflineRunner(config)
    
    # 显示状态
    if args.status:
        runner.print_status()
        return
    
    # 运行
    if args.step:
        runner.run(steps=args.step)
    else:
        runner.run(start_step=args.start, end_step=args.end)


if __name__ == '__main__':
    main()

