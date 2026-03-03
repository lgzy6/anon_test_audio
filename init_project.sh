#!/bin/bash
# init_project.sh - 初始化项目目录结构

PROJECT_ROOT="samm_anon"

# 创建主目录
mkdir -p $PROJECT_ROOT
cd $PROJECT_ROOT

# 创建子目录
mkdir -p configs
mkdir -p data/{kaldi_io,datasets,processors}
mkdir -p models/{ssl,eta_wavlm,samm,knn_vc}
mkdir -p offline
mkdir -p online
mkdir -p evaluation
mkdir -p utils
mkdir -p scripts
mkdir -p checkpoints/target_pool
mkdir -p cache/{wavlm_features,cleaned_features,metadata}
mkdir -p outputs/{anonymized,evaluation}

# 创建 __init__.py
touch data/__init__.py
touch data/kaldi_io/__init__. py
touch data/datasets/__init__.py
touch data/processors/__init__.py
touch models/__init__.py
touch models/ssl/__init__.py
touch models/eta_wavlm/__init__.py
touch models/samm/__init__.py
touch models/knn_vc/__init__. py
touch offline/__init__.py
touch online/__init__.py
touch evaluation/__init__.py
touch utils/__init__.py

echo "Project structure initialized!"