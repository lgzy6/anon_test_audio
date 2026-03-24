#!/bin/bash
# 使用 spkanon 框架评估匿名化效果

ANON_DIR="/root/autodl-tmp/anon_test/outputs/persudo_target"
SPKANON_DIR="/root/autodl-tmp/spkanon/spane"

cd $SPKANON_DIR

# 运行评估
python run.py config/config.yaml \
  --device cuda \
  data.anon_data_dir=$ANON_DIR
