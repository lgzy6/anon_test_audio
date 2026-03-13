# 音频质量差异分析

## 关键差异

### 1. kNN匹配策略不同

**原始spkanon框架 (converter.py:307-310)**
```python
max_indices = torch.argmax(cos_sim, dim=1)  # 选择最相似的1个
conv_feats = tgt_feats[torch.arange(tgt_feats.shape[0]), max_indices]
```

**当前实现 (test_e2e_with_duration.py:240-242)**
```python
topk_indices = cos_sim.topk(k=actual_k, dim=1)[1]  # 选择top-k
h_anon[mask] = tgt_clusters[topk_indices].mean(dim=1)  # 取平均
```

**问题**: 你的实现对top-k结果取平均，会导致特征模糊化，降低音频清晰度。

### 2. 相似度计算方式

**原始框架 (converter.py:298-304)**
```python
dot_p = torch.bmm(conv_feats.unsqueeze(1), tgt_feats.transpose(1, 2)).squeeze(1)
src_norm = torch.norm(conv_feats, dim=-1)
tgt_norm = torch.norm(tgt_feats, dim=-1)
cos_sim = torch.div(dot_p, src_norm.unsqueeze(1) * tgt_norm)
```
使用batch矩阵乘法，每个源特征与对应phone的所有目标特征计算相似度。

**当前实现 (test_e2e_with_duration.py:235-238)**
```python
src_batch = torch.nn.functional.normalize(src_feats_adj[mask], dim=-1)
tgt_batch = torch.nn.functional.normalize(tgt_clusters, dim=-1)
cos_sim = torch.mm(src_batch, tgt_batch.T)
```
先归一化再矩阵乘法，数值上等价但实现方式不同。

### 3. 目标特征来源

**原始框架**: 使用完整的目标话语特征 `tgt_feats[phone_id]`，保留了目标说话人的完整声学信息。

**当前实现**: 使用聚类中心 `phone_clusters[phone_id]`，是多个特征的平均，丢失了细节。

## 核心问题

**取平均操作导致特征退化**:
```python
h_anon[mask] = tgt_clusters[topk_indices].mean(dim=1)  # ← 这里
```

原始框架直接使用最相似的单个特征，而你对k个特征取平均，相当于做了二次平滑，导致：
- 音频细节丢失
- 清晰度下降
- 自然度降低

## 建议修复

将 `mean(dim=1)` 改为只选择最相似的特征：
```python
topk_indices = cos_sim.topk(k=1, dim=1)[1].squeeze(1)
h_anon[mask] = tgt_clusters[topk_indices]
```
