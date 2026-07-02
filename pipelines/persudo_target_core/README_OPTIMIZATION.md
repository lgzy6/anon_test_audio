# 性别分组优化方案

## 改进内容

### 1. 按性别计算混合熵
- 男性和女性分别计算混合熵
- 提高同性别内的区分度

### 2. 双层Bank (L6 + L24)
- Bank存储格式: `{phone_id: {'l6': tensor, 'l24': tensor}}`
- L24用于计算相似度（query）
- L6用于返回特征（value）

### 3. 去除skip_top3
- 简化检索逻辑
- 直接从top-20中随机采样

### 4. 性别路由
- 支持同性别合成（默认）
- 支持跨性别合成

## 使用方法

### 步骤1: 构建性别独立Bank
```bash
cd /root/autodl-tmp/anon_test/pipelines/persudo_target_core
./build_gender_banks.sh
```

### 步骤2: 匿名化合成

**同性别合成（默认）:**
```bash
python synthesize_pseudo.py \
    --audio input.wav \
    --output output.wav \
    --bank checkpoints/trainother500_with_phones/pseudo_bank.gender-m.pt
```

**跨性别合成:**
```bash
# 男声 -> 女声
python synthesize_pseudo.py \
    --audio male_input.wav \
    --output output.wav \
    --bank checkpoints/trainother500_with_phones/pseudo_bank.gender-f.pt \
    --target-gender f

# 女声 -> 男声
python synthesize_pseudo.py \
    --audio female_input.wav \
    --output output.wav \
    --bank checkpoints/trainother500_with_phones/pseudo_bank.gender-m.pt \
    --target-gender m
```

## 文件说明

- `compute_entropy.py` - 支持按性别计算混合熵
- `build_bank.py` - 构建L6+L24双层Bank
- `synthesize_pseudo.py` - L24检索+性别路由
- `build_gender_banks.sh` - 批量构建脚本
