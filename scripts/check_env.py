#!/usr/bin/env python
"""
环境和配置检查工具

检查:
1. Python 包依赖
2. GPU 可用性
3. FAISS GPU 支持
4. 配置文件完整性
5. 数据集路径
6. 磁盘空间
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_packages():
    """检查必要的 Python 包"""
    print("\n" + "=" * 60)
    print("1️⃣  Python 包检查")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'h5py': 'HDF5',
        'faiss': 'FAISS',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name:20s} - 已安装")
        except ImportError:
            print(f"  ✗ {name:20s} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  缺少 {len(missing)} 个包，请安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_gpu():
    """检查 GPU 可用性"""
    print("\n" + "=" * 60)
    print("2️⃣  GPU 检查")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("  ✗ CUDA 不可用")
            return False
        
        num_gpus = torch.cuda.device_count()
        print(f"  ✓ 检测到 {num_gpus} 个 GPU")
        
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {name} ({memory:.1f} GB)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GPU 检查失败: {e}")
        return False


def check_faiss_gpu():
    """检查 FAISS GPU 支持"""
    print("\n" + "=" * 60)
    print("3️⃣  FAISS GPU 检查")
    print("=" * 60)
    
    try:
        import faiss
        
        num_gpus = faiss.get_num_gpus()
        print(f"  FAISS 版本: {faiss.__version__}")
        print(f"  GPU 数量: {num_gpus}")
        
        if num_gpus == 0:
            print("  ⚠️  FAISS GPU 不可用（使用 CPU 版本）")
            print("  提示: 安装 GPU 版本以加速 (pip install faiss-gpu)")
            return False
        else:
            print(f"  ✓ FAISS GPU 可用")
            return True
            
    except Exception as e:
        print(f"  ✗ FAISS 检查失败: {e}")
        return False


def check_config(config_path: str):
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("4️⃣  配置文件检查")
    print("=" * 60)
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"  ✗ 配置文件不存在: {config_path}")
        return False
    
    print(f"  ✓ 配置文件: {config_path}")
    
    try:
        from utils.config import load_config
        config = load_config(str(config_path))
        
        # 检查必要字段
        required_fields = [
            ('paths', 'cache_dir'),
            ('paths', 'checkpoints_dir'),
            ('paths', 'librispeech_root'),
            ('paths', 'wavlm_checkpoint'),
            ('ssl', 'model'),
            ('samm', 'codebook_size'),
        ]
        
        missing = []
        for *path, key in required_fields:
            d = config
            for p in path:
                d = d.get(p, {})
            if key not in d:
                missing.append('.'.join(path + [key]))
        
        if missing:
            print(f"  ⚠️  缺少配置字段:")
            for m in missing:
                print(f"      - {m}")
            return False
        
        print("  ✓ 配置字段完整")
        
        # 检查路径
        paths_to_check = {
            'LibriSpeech': config['paths']['librispeech_root'],
            'WavLM checkpoint': config['paths']['wavlm_checkpoint'],
        }
        
        path_ok = True
        for name, path in paths_to_check.items():
            if Path(path).exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name} 不存在: {path}")
                path_ok = False
        
        return path_ok
        
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return False


def check_disk_space(config_path: str):
    """检查磁盘空间"""
    print("\n" + "=" * 60)
    print("5️⃣  磁盘空间检查")
    print("=" * 60)
    
    try:
        import shutil
        from utils.config import load_config
        
        config = load_config(config_path)
        
        paths = {
            'Cache 目录': config['paths']['cache_dir'],
            'Checkpoint 目录': config['paths']['checkpoints_dir'],
        }
        
        for name, path in paths.items():
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            total, used, free = shutil.disk_usage(path)
            
            print(f"\n  {name}: {path}")
            print(f"    总容量: {total / (1024**3):.1f} GB")
            print(f"    已使用: {used / (1024**3):.1f} GB")
            print(f"    可用:   {free / (1024**3):.1f} GB")
            
            # Pool Building 需要约 35 GB
            required_gb = 40
            if free / (1024**3) < required_gb:
                print(f"    ⚠️  可用空间不足 {required_gb} GB")
            else:
                print(f"    ✓ 空间充足")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 磁盘空间检查失败: {e}")
        return False


def print_recommendations(results: dict):
    """打印建议"""
    print("\n" + "=" * 60)
    print("💡 建议")
    print("=" * 60)
    
    if not results['packages']:
        print("  ⚠️  请先安装缺少的 Python 包")
    
    if not results['gpu']:
        print("  ⚠️  未检测到 GPU，训练速度会很慢")
    
    if not results['faiss_gpu']:
        print("  💡 建议安装 FAISS GPU 版本以加速 Step 6:")
        print("     pip uninstall faiss-cpu")
        print("     pip install faiss-gpu")
    
    if not results['config']:
        print("  ⚠️  请检查配置文件")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("  ✓ 环境配置良好，可以开始训练!")
        print("\n运行示例:")
        print("  python scripts/train_offline.py --config configs/server.yaml")
    else:
        print("\n  ⚠️  请先解决上述问题")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='环境和配置检查工具')
    parser.add_argument(
        '--config', '-c',
        default='configs/server.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(" " * 15 + "SAMM-Anon 环境检查")
    print("=" * 60)
    
    results = {
        'packages': check_packages(),
        'gpu': check_gpu(),
        'faiss_gpu': check_faiss_gpu(),
        'config': check_config(args.config),
        'disk': check_disk_space(args.config),
    }
    
    print_recommendations(results)


if __name__ == '__main__':
    main()