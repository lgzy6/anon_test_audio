# 迁移创新点所需的关键代码

## 问题根源分析

### 你的实现 vs 原始框架的关键差异

#### 1. **WavLM特征提取层不同** ⚠️ 最关键
**原始框架**: 使用两层WavLM特征
```python
# converter.py line 236-237
vc_feats, phone_feats = src_vecs  # 两个不同的层
batch_phones = self.phone_predictor(phone_feats).argmax(dim=2)
```

**你的实现**: 只用一层
```python
# test_e2e_with_duration.py line 132
wavlm_feats = models['wavlm'](waveform).squeeze(0).cpu().numpy()  # 单层
```

**影响**: phone预测和VC使用不同层的特征，你的实现混用了同一层，导致信息耦合。

#### 2. **Duration调整的插值方式**
**原始框架**: 使用整数索引
```python
# converter.py line 272-277
feat_indices = torch.linspace(
    feat_idx_start, feat_idx_end,
    n_frames[src_idx][distinct_idx].item(),
    dtype=torch.int64  # ← 整数索引
)
src_feats_dur.append(vc_feats[src_idx, feat_indices])
```

**你的实现**: 也用整数，但可能有边界问题
```python
# test_e2e_with_duration.py line 214
indices = torch.linspace(feat_idx_start, feat_idx_end, new_dur.item(), dtype=torch.long, device=device)
```

#### 3. **kNN匹配使用bmm而非mm**
**原始框架**: 批量矩阵乘法
```python
# converter.py line 298-300
dot_p = torch.bmm(
    conv_feats.unsqueeze(1), tgt_feats.transpose(1, 2)
).squeeze(1)
```

**你的实现**: 普通矩阵乘法
```python
# test_e2e_with_duration.py line 238
cos_sim = torch.mm(src_batch, tgt_batch.T)
```

**影响**: bmm保证每个源特征只与对应phone的目标特征比较，你的实现可能跨phone匹配。

## 必须修复的代码

### 修复1: 使用双层WavLM特征

在 `scripts/test_e2e_with_duration.py` 中修改WavLM初始化：

```python
# 修改 load_models() 函数
wavlm_phone = WavLMWrapper(
    ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
    layer=24,  # phone预测用layer 24
    device=device
)
wavlm_vc = WavLMWrapper(
    ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'),
    layer=6,   # VC用layer 6
    device=device
)
```

### 修复2: 分离phone预测和VC特征

```python
# 在 extract_source_features() 中
with torch.no_grad():
    wavlm_vc_feats = models['wavlm_vc'](waveform).squeeze(0).cpu().numpy()
    wavlm_phone_feats = models['wavlm_phone'](waveform).squeeze(0)

phones = models['phone_predictor'](wavlm_phone_feats).cpu().numpy()

return {
    'wavlm': wavlm_vc_feats,  # 用于VC
    'phones': phones,
    ...
}
```

### 修复3: 使用bmm进行phone-aware匹配

```python
# 在 anonymize_with_duration() 中替换kNN部分
h_anon = torch.zeros_like(src_feats_adj)

for phone_id in tqdm(unique_phone_ids, desc="kNN"):
    if phone_id == 0 or int(phone_id) not in phone_clusters:
        mask = src_phones_adj == phone_id
        h_anon[mask] = src_feats_adj[mask]
        continue

    tgt_feats = torch.from_numpy(phone_clusters[int(phone_id)]).float().to(device)
    mask = src_phones_adj == phone_id
    src_batch = src_feats_adj[mask]  # [N, D]

    # 使用bmm: [N, 1, D] x [N, D, M] -> [N, 1, M] -> [N, M]
    dot_p = torch.bmm(
        src_batch.unsqueeze(1),
        tgt_feats.unsqueeze(0).expand(src_batch.shape[0], -1, -1).transpose(1, 2)
    ).squeeze(1)

    src_norm = torch.norm(src_batch, dim=-1, keepdim=True)
    tgt_norm = torch.norm(tgt_feats, dim=-1)
    cos_sim = dot_p / (src_norm * tgt_norm.unsqueeze(0) + 1e-8)

    max_indices = torch.argmax(cos_sim, dim=1)
    h_anon[mask] = tgt_feats[max_indices]
```

## 诊断语义丢失和韵律卡顿的方法

创建对比脚本 `scripts/compare_with_original.py`:

```python
# 1. 提取原始框架的中间特征
# 2. 提取你的框架的中间特征
# 3. 对比每个阶段的差异
```
