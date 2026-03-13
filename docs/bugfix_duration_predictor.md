# Bug修复：Duration Predictor维度错误

## 问题描述

运行 `test_e2e_with_duration.py` 时报错：
```
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 256, 97]
```

## 根本原因

`PhonePredictor` 包装类的 `forward()` 方法会自动为2D输入添加batch维度：

```python
# predictor.py line 161-162
if x.dim() == 2:
    x = x.unsqueeze(0)  # [1, 97] -> [1, 1, 97]
```

这个逻辑是为WavLM特征 `[T, D]` 设计的，但duration predictor的输入是phone indices `[B, T]`，导致错误地添加了额外维度。

## 解决方案

直接调用底层的 `ConvDecoder` 模型，绕过 `PhonePredictor` 包装类：

```python
# 修改前
pred_durations = models['duration_predictor'](phones_tensor).squeeze(0)

# 修改后
pred_durations = models['duration_predictor'].model(phones_tensor).squeeze(0)
```

## 修改文件

- `scripts/test_e2e_with_duration.py` line 202

## 测试结果

✓ 成功处理 test-clean 数据集音频
✓ 输出维度正确：[521, 1024] -> [545, 1024]
