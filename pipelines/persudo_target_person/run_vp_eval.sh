#!/bin/bash
# VoicePrivacy 评估完整流程

# 1. 准备数据
python pipelines/persudo_target_person/prepare_vp_data.py \
  --anon_dir outputs/persudo_target \
  --original_dir /root/autodl-tmp/datasets/LibriSpeech/test-clean \
  --output_dir outputs/voiceprivacy_data \
  --vp_original_data /root/autodl-tmp/Voice-Privacy-Challenge-2024/data/libri_test

# 2. 复制到 VoicePrivacy 目录
VP_DIR=/root/autodl-tmp/Voice-Privacy-Challenge-2024
mkdir -p $VP_DIR/data
cp -r outputs/voiceprivacy_data $VP_DIR/data/libri_test_persudo

# 3. 创建评估配置
cat > /tmp/eval_persudo.yaml << 'EOF'
data_dir: data
exp_dir: exp
anon_data_suffix: _persudo

datasets:
  - name: libri_test
    data: libri_test_persudo
    enrolls: [_enrolls]
    trials: [_trials_f, _trials_m]

eval_steps:
  privacy:
    - asv
  utility:
    - asr

privacy:
  asv:
    model_type: ecapa
    dataset_name: [libri_test]

utility:
  asr:
    dataset_name: [libri_test]
EOF

# 4. 运行评估
cd $VP_DIR
python run_evaluation.py --config /tmp/eval_persudo.yaml
