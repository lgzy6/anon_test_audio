#!/bin/bash
# setup_env.sh - SAMM-Anon 服务器环境创建脚本
# 使用方法:  bash setup_env.sh

set -e  # 遇到错误立即退出

# 先初始化 conda，确保后续 conda activate 可用（即使脚本被 bash 直接执行）
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" || true
fi

echo "=============================================="
echo "SAMM-Anon 环境创建脚本"
echo "Python 3.10 + PyTorch + CUDA"
echo "=============================================="

ENV_NAME="anon_test"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"  # 可选:  11.8, 12.1, cpu

# -------------------------------------------------
# 防呆：历史版本里曾误留一行单字符（例如 'c'）会导致：command not found
# 如你文件第 29 行出现了单独的 c/==== 等，请确保已删除。
# -------------------------------------------------

echo ""
echo "[1/7] 检查 Conda..."

if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

echo "Conda 版本: $(conda --version)"

# 避免 conda 缓存/损坏 repodata 导致 JSONDecodeError（网络/镜像返回非 JSON 时常见）
# 这里做一次轻量清理；失败不致命
conda clean -i -t -y &> /dev/null || true

# 给 conda 增加稳健参数，尽量减少 repodata/网络抖动导致的 JSONDecodeError 噪声
# 说明：这些参数对于老版本 conda 可能部分不生效，但不会影响主要流程
export CONDA_ALWAYS_YES=true

echo ""
echo "[2/7] 检查 CUDA..."

if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    SYSTEM_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "系统 CUDA 版本: $SYSTEM_CUDA"

    # 若驱动显示 CUDA 13.x 等，不代表你必须安装 CUDA13。
    # PyTorch 官方 wheel 目前常用 cu118/cu121，驱动足够新即可运行。
    if [[ "$SYSTEM_CUDA" =~ ^13\.[0-9]+$ ]]; then
        echo "提示: 驱动显示 CUDA $SYSTEM_CUDA，仍将使用 PyTorch cu121 wheel（更常见/兼容）。"
        CUDA_VERSION="12.1"
    fi
else
    echo "警告: 未检测到 GPU，将安装 CPU 版本"
    CUDA_VERSION="cpu"
fi

# 若用户手动指定 CUDA_VERSION 不在 cpu/11.8/12.1，则回退到 12.1。
case "$CUDA_VERSION" in
  cpu|11.8|12.1) ;;
  *)
    echo "警告: CUDA_VERSION=\"$CUDA_VERSION\" 不在支持列表，回退为 12.1"
    CUDA_VERSION="12.1"
    ;;
esac

echo ""
echo "[3/7] 创建 Conda 环境..."

# 确保 conda activate 可用
eval "$(conda shell.bash hook)" || true

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，是否删除重建?  (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境，跳过创建步骤"
        conda activate ${ENV_NAME}
    fi
fi

# 创建新环境（对 repodata 异常做一次重试）
if ! conda env list | grep -q "^${ENV_NAME} "; then
    set +e
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "警告: conda create 失败，尝试清理索引并重试一次（常见原因：镜像/网络返回异常导致 JSONDecodeError）"

        # 更彻底的缓存清理
        conda clean -a -y || true

        # 对部分环境：切换为 classic solver 可能更稳定
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y --solver=classic
        rc=$?
    fi
    set -e
    if [ $rc -ne 0 ]; then
        echo "错误: conda create 仍失败。建议检查网络/镜像源：conda config --show-sources"
        echo "建议操作: conda clean -a -y && conda update -n base -c defaults conda"
        exit $rc
    fi
fi

echo ""
echo "[4/7] 激活环境..."

# 初始化 conda (确保 activate 可用)
eval "$(conda shell.bash hook)" || true
conda activate ${ENV_NAME}

echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"

echo ""
echo "[5/7] 安装 PyTorch..."

# 升级 pip，避免老 pip 解析 wheel/依赖失败
python -m pip install -U pip setuptools wheel

if [ "$CUDA_VERSION" = "cpu" ]; then
    echo "安装 CPU 版本 PyTorch..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
elif [ "$CUDA_VERSION" = "11.8" ]; then
    echo "安装 CUDA 11.8 版本 PyTorch..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "12.1" ]; then
    echo "安装 CUDA 12.1 版本 PyTorch..."
    pip install torch==2.1.0 torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "安装默认 PyTorch..."
    pip install torch torchaudio
fi

echo ""
echo "[6/7] 安装其他依赖..."

# 核心依赖
pip install numpy==1.24.3
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0

# 音频处理
pip install librosa>=0.10.0
pip install soundfile>=0.12.0

# 数据存储
pip install h5py>=3.9.0
pip install pyyaml>=6.0.1

# 进度条
pip install tqdm>=4.66.0

# FAISS (向量检索)
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install faiss-cpu>=1.7.4
else
    # 尝试安装 GPU 版本，失败则安装 CPU 版本
    pip install faiss-gpu>=1.7.4 || pip install faiss-cpu>=1.7.4
fi

# 可选依赖 (根据需要取消注释)
pip install speechbrain>=1.0.0      # ECAPA-TDNN 说话人编码器
pip install kaldiio>=2.18.0         # Kaldi 格式支持
pip install tensorboard>=2.14.0     # 可视化
pip install matplotlib>=3.7.0       # 绘图

echo ""
echo "[7/7] 验证安装..."

python << 'EOF'
import sys
print("=" * 50)
print("环境验证")
print("=" * 50)

# Python
print(f"Python:  {sys.version}")

# PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Torchaudio
import torchaudio
print(f"Torchaudio: {torchaudio.__version__}")

# 其他依赖
import numpy as np
print(f"NumPy: {np.__version__}")

import h5py
print(f"H5py: {h5py.__version__}")

import sklearn
print(f"Scikit-learn: {sklearn.__version__}")

import yaml
print(f"PyYAML: OK")

import tqdm
print(f"tqdm: OK")

try:
    import faiss
    print(f"FAISS:  OK (GPU:  {faiss.get_num_gpus() > 0})")
except ImportError: 
    print("FAISS:  未安装")

print("=" * 50)
print("✓ 环境验证完成!")
print("=" * 50)
EOF

echo ""
echo "=============================================="
echo "环境创建完成!"
echo "=============================================="
echo ""
echo "使用方法:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "运行 Offline Pipeline:"
echo "  python scripts/run_offline.py -c configs/server.yaml --status"
echo "  python scripts/run_offline.py -c configs/server.yaml --step 1"
echo ""