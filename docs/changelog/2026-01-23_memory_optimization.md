# 变更记录: Step 6 Pool Building 内存优化

**日期**: 2026-01-23
**问题**: OOM Kill (360GB 内存仍被杀死)
**影响模块**: `pipelines/offline/pool_building_dual_gpu.py`, `configs/production.yaml`

---

## 问题描述

运行 Step 6 (Target Pool 构建) 时，在构建 Phone 子索引阶段 (90% 进度) 被系统 OOM Kill:

```
Phone indices:  90%|███████████████████▊ | 37/41 [00:18<00:00, 24.06it/s]
Killed
```

即使系统有 360GB 内存也无法完成。

---

## 根因分析

### 内存消耗来源

1. **全局 FAISS 索引** (~55M 帧 × 1024 维 × 4 字节 ≈ 220GB)
2. **index_map 数组**:
   - `frame_to_utt`: 55M × 4 字节 = 220MB
   - `frame_to_phone`: 55M × 4 字节 = 220MB
   - `frame_to_gender`: 55M × 1 字节 = 55MB
   - `frame_to_symbol`: 55M × 4 字节 = 220MB
3. **Phone 子索引累积** (问题核心):
   - 每个 phone 的 `frame_indices` 数组 (平均 ~135 万帧 × 8 字节 = 10MB)
   - 每个 phone 的 FAISS 索引 (平均 ~135 万帧 × 1024 维 × 4 字节 = 5.5GB)
   - 41 个 phone 全部保存在内存 → 累计超过 200GB

### 原代码问题

```python
# 原代码: 所有 phone 索引累积在内存中
def _build_phone_indices(self, features_path, index_map):
    phone_indices = {}  # 所有索引都保存在这里

    for phone_id in range(self.num_phones):
        # ... 构建索引 ...

        # 问题: 一次性加载所有特征到内存
        if n_frames > 100000:
            phone_features_list = []
            for i in range(0, n_frames, 50000):
                batch_features = feats[batch_indices].astype(np.float32)
                phone_features_list.append(batch_features)
            phone_features = np.vstack(phone_features_list)  # 内存峰值!

        # 问题: 索引和 frame_indices 一直保留在内存
        phone_indices[phone_id] = {
            'index': sub_index,
            'frame_indices': phone_frames,  # 大数组
            'count': n_frames
        }

    return phone_indices  # 所有数据一起返回
```

---

## 解决方案

### 策略: 边构建边保存 (Streaming Save)

不再将所有 phone 索引累积在内存中，而是每构建完一个 phone 索引就立即保存到磁盘并释放内存。

### 修改内容

#### 1. 配置文件 `configs/production.yaml`

新增内存优化选项:

```yaml
pool_building:
  # ----- Phone 索引 -----
  build_phone_indices: true
  phone_index_type: "auto"
  phone_ivf_nlist: 256
  phone_save_immediately: true    # [新增] 立即保存释放内存

  # ----- 内存优化 (防止 OOM) -----
  low_memory_mode: true           # [新增] 启用低内存模式
```

#### 2. 数据类 `DualGPUPoolConfig`

新增配置字段:

```python
@dataclass
class DualGPUPoolConfig:
    # ... 原有字段 ...

    # 内存优化选项 [新增]
    phone_save_immediately: bool = True  # 立即保存 phone 索引到磁盘
    low_memory_mode: bool = True         # 低内存模式
```

#### 3. `_build_phone_indices` 方法重写

核心改动:

```python
def _build_phone_indices(self, features_path, index_map):
    phone_indices = {}
    save_immediately = self.pool_config.low_memory_mode

    if save_immediately:
        phone_dir = self.pool_dir / 'phone_indices'
        phone_dir.mkdir(exist_ok=True)
        phone_meta = {}

    for phone_id in range(self.num_phones):
        # 1. 创建索引并训练 (采样训练，不加载全部)
        if n_frames >= 1000:
            train_size = min(50000, n_frames)
            train_indices = phone_frames[::train_step][:train_size]
            train_data = feats[train_indices].astype(np.float32)
            sub_index.train(train_data)
            del train_data

        # 2. 分批添加到索引 (每批最多 30000 帧)
        batch_size = 30000
        for i in range(0, n_frames, batch_size):
            batch_features = feats[batch_indices].astype(np.float32)
            sub_index.add(batch_features)
            del batch_features  # 立即释放

        # 3. 立即保存并释放 [关键改动]
        if save_immediately:
            faiss.write_index(sub_index, str(phone_dir / f'{phone_name}.index'))
            np.save(phone_dir / f'{phone_name}_frames.npy', phone_frames)
            phone_meta[phone_name] = {'phone_id': phone_id, 'count': n_frames}
            del sub_index  # 释放索引内存

        del phone_frames
        gc.collect()

    if save_immediately:
        return {}  # 返回空字典，数据已保存到磁盘
```

#### 4. `_build_gender_indices` 方法优化

同样采用边构建边保存策略，并减小批次大小:

- 批次大小: 100000 → 50000
- 添加 `del batch_data` 立即释放
- 低内存模式下立即保存到磁盘

#### 5. `_save_pool` 方法适配

检测低内存模式下 phone_indices/gender_indices 为空时跳过保存 (已在构建时保存):

```python
def _save_pool(self, global_index, phone_indices, gender_indices, ...):
    # 低内存模式下，phone_indices 和 gender_indices 已经在构建时保存
    if phone_indices:  # 非低内存模式才执行
        # ... 保存逻辑 ...
```

---

## 内存使用对比

| 阶段 | 原方案内存峰值 | 优化后内存峰值 |
|------|---------------|---------------|
| 全局索引构建 | ~220GB | ~220GB (无变化) |
| Phone 索引构建 | ~220GB + 200GB = 420GB | ~220GB + 5.5GB = 225GB |
| Gender 索引构建 | +55GB | +27GB (已释放 phone) |
| 总峰值 | >400GB | <250GB |

---

## 文件变更清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `configs/production.yaml` | 修改 | 新增 `phone_save_immediately` 和 `low_memory_mode` 配置 |
| `pipelines/offline/pool_building_dual_gpu.py` | 修改 | 重写 phone/gender 索引构建逻辑，实现流式保存 |

---

## 验证方法

```bash
# 监控内存使用
watch -n 1 'free -h'

# 运行 Step 6
python scripts/run_offline_pipeline.py --config configs/production.yaml --steps 6
```

预期:
- 内存峰值不超过 250GB
- Phone indices 阶段平稳运行，无 OOM Kill
- 输出目录 `target_pool/phone_indices/` 中可看到逐个生成的索引文件

---

## 回滚方法

如需恢复原行为 (不推荐):

```yaml
# configs/production.yaml
pool_building:
  phone_save_immediately: false
  low_memory_mode: false
```
