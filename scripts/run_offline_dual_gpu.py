# scripts/run_offline_dual_gpu.py
#!/usr/bin/env python
"""
SAMM-Anon Offline Pipeline (双 4090 + 240GB 内存优化版本)

使用方法:
    # 运行全部步骤
    python scripts/run_offline_dual_gpu.py

    # 只运行 step 6 (Target Pool 构建)
    python scripts/run_offline_dual_gpu.py --step 6

    # 查看状态
    python scripts/run_offline_dual_gpu.py --status

    # 使用自定义配置
    python scripts/run_offline_dual_gpu.py --config configs/dual_4090.yaml
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description='SAMM-Anon Offline Pipeline (Dual GPU Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行全部步骤
  python scripts/run_offline_dual_gpu.py

  # 只运行 step 6
  python scripts/run_offline_dual_gpu.py --step 6

  # 从 step 4 开始运行
  python scripts/run_offline_dual_gpu.py --start 4

  # 查看当前状态
  python scripts/run_offline_dual_gpu.py --status
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/dual_4090.yaml',
        help='配置文件路径 (默认: configs/dual_4090.yaml)'
    )

    parser.add_argument(
        '--step', '-s',
        type=int,
        nargs='+',
        default=None,
        help='指定运行的步骤 (1-6)，如: --step 1 2 3'
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

    # 导入双 GPU Runner
    from pipelines.offline.runner_dual_gpu import DualGPUOfflineRunner

    # 创建 Runner
    runner = DualGPUOfflineRunner(config)

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
