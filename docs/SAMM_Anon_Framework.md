# SAMM-Anon 语音匿名化框架

> **版本**: v3.0
> **创建时间**: 2026-01-26
> **框架名称**: SAMM-Anon (Self-supervised Anonymization with Masked Modeling)

---

## 一、项目概述

### 1.1 简介

SAMM-Anon 是一个基于深度学习的语音匿名化框架，融合了以下核心技术：

- **Eta-WavLM**: 说话人子空间投影，去除说话人身份信息
- **SAMM**: 自监督掩码建模，增强不可链接性
- **Private kNN-VC**: 约束近邻检索，实现声音转换

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| **隐私保护** | 去除说话人身份，防止重识别攻击 |
| **语义保留** | 保持语音内容的可识别性 (低 WER) |
| **自然度** | 生成自然流畅的匿名语音 (高 MOS) |
| **不可链接性** | 同一说话人的不同语音不可关联 |

---

## 二、项目结构

```
anon_test/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   ├── production.yaml        # 生产环境配置
│   └── test.yaml              # 测试配置
│
├── models/                     # 模型定义
│   ├── ssl/                   # 自监督学习模块
│   │   ├── wavlm_model.py     # WavLM 模型封装
│   │   └── wrappers.py        # SSL 特征提取器
│   │
│   ├── eta_wavlm/             # 说话人子空间投影
│   │   └── projector.py       # Eta-WavLM 投影器
│   │
│   ├── samm/                  # SAMM 模块
│   │   ├── codebook.py        # 离散码本
│   │   ├── pattern_matrix.py  # 转移概率矩阵
│   │   ├── masking.py         # 掩码策略
│   │   ├── prosody.py         # 韵律处理
│   │   └── target_selector.py # 目标选择器
│   │
│   ├── knn_vc/                # kNN 声音转换
│   │   ├── retriever.py       # 约束近邻检索器
│   │   └── duration.py        # 时长匿名化
│   │
│   ├── phone_predictor/       # 音素预测
│   │   └── predictor.py       # Phone/Duration 预测器
│   │
│   └── vocoder/               # 声码器
│       ├── hifigan.py         # HiFi-GAN 封装
│       └── hifigan_model.py   # HiFi-GAN 模型
│
├── pipelines/                  # 处理流水线
│   ├── offline/               # 离线构建
│   │   ├── runner.py          # 统一运行器
│   │   ├── feature_extraction.py    # Step 1: 特征提取
│   │   ├── subspace_learning.py     # Step 2: 子空间学习
│   │   ├── feature_cleaning.py      # Step 3: 特征清洗
│   │   ├── codebook_training.py     # Step 4: 码本训练
│   │   ├── pattern_learning.py      # Step 5: Pattern 学习
│   │   ├── pool_building.py         # Step 6: 目标池构建
│   │   ├── pool_building_sampled.py # Step 6: 采样版本
│   │   └── pool_building_low_mem.py # Step 6: 低内存版本
│   │
│   └── online/                # 在线推理
│       └── anonymizer.py      # 匿名化主入口
│
├── data/                       # 数据处理
│   ├── datasets/              # 数据集加载
│   │   └── librispeech.py     # LibriSpeech 数据集
│   └── io/                    # IO 工具
│       ├── hdf5.py            # HDF5 读写
│       └── kaldi.py           # Kaldi 格式
│
├── scripts/                    # 运行脚本
│   ├── run_offline.py         # 运行离线流程
│   ├── rebuild_target_pool.py # 重建目标池
│   └── test_anonymization.py  # 测试匿名化
│
├── tools/                      # 工具集
│   ├── diagnostic/            # 诊断工具
│   ├── testing/               # 测试工具
│   └── comparison/            # 对比工具
│
├── evaluation/                 # 评估模块
├── utils/                      # 通用工具
├── checkpoints/               # 预训练模型
└── docs/                      # 文档
```

---

## 三、核心模块详解

### 3.1 Eta-WavLM (说话人子空间投影)

**位置**: `models/eta_wavlm/projector.py`

**原理**: 通过 PCA 学习说话人子空间 $U_s$，将特征投影到其正交补空间：

$$h_{clean} = h \cdot P_{orth} = h \cdot (I - U_s U_s^T)$$

**作用**: 去除 WavLM 特征中的说话人身份信息，保留语言内容。

```python
class EtaWavLMProjector:
    def forward(self, h: Tensor) -> Tensor:
        """去除说话人成分"""
        return h @ self.P_orth
```

### 3.2 SAMM (自监督掩码建模)

**位置**: `models/samm/`

#### 3.2.1 Codebook (离散码本)

将连续特征量化为离散符号，学习语音的离散表示：

```python
class SAMMCodebook:
    def quantize(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """量化特征到最近的码本向量"""
        distances = torch.cdist(features, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook[indices]
        return quantized, indices
```

#### 3.2.2 Pattern Matrix (转移概率矩阵)

学习符号之间的转移概率，用于掩码填充：

```python
class PatternMatrix:
    # M[i,j] = P(symbol_t = j | symbol_{t-1} = i)
    def sample_next(self, current_symbol: int) -> int:
        """基于当前符号采样下一个符号"""
        probs = self.M[current_symbol]
        return np.random.choice(self.K, p=probs)
```

#### 3.2.3 Masking (掩码策略)

```python
@dataclass
class MaskConfig:
    token_mask_ratio: float = 0.10   # 随机 token 掩码
    span_mask_ratio: float = 0.15    # span 连续掩码
    min_span: int = 3
    max_span: int = 10
    duration_noise_std: float = 0.15 # 时长扰动
```

### 3.3 kNN-VC (约束近邻检索)

**位置**: `models/knn_vc/retriever.py`

**检索策略**:

1. **Phone 约束**: 只在同音素特征中检索
2. **Gender 约束**: 匹配目标性别
3. **Symbol 约束**: 匹配 SAMM 符号 (可选)

```python
class ConstrainedKNNRetriever:
    def retrieve(
        self,
        query: Tensor,           # 查询特征
        phone_ids: Tensor,       # 音素标签
        target_gender: str,      # 目标性别
        symbol_ids: Tensor,      # SAMM 符号
    ) -> Tensor:
        """约束近邻检索"""
        # 1. Phone 聚类检索 (粗粒度)
        # 2. kNN 精细检索 (可选)
        # 3. Top-1 或加权平均
```

### 3.4 Vocoder (声码器)

**位置**: `models/vocoder/hifigan.py`

使用 HiFi-GAN 将 WavLM 特征转换为波形：

```python
class HiFiGANVocoder:
    def synthesize(self, features: Tensor) -> Tensor:
        """特征转波形"""
        return self.generator(features)
```

---

## 四、处理流程

### 4.1 Offline Pipeline (离线构建)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Offline Pipeline (6 Steps)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Feature Extraction                                      │
│  ├── 输入: LibriTTS 音频                                         │
│  ├── 模型: WavLM-Large (Layer 6)                                │
│  └── 输出: features.h5 (WavLM 特征)                              │
│                                                                  │
│  Step 2: Subspace Learning                                       │
│  ├── 输入: WavLM 特征                                            │
│  ├── 方法: PCA 学习说话人子空间                                   │
│  └── 输出: speaker_subspace.pt (U_s 矩阵)                        │
│                                                                  │
│  Step 3: Feature Cleaning                                        │
│  ├── 输入: WavLM 特征 + U_s                                      │
│  ├── 操作: h_clean = h @ P_orth                                  │
│  └── 输出: cleaned/features.h5 (去说话人特征)                     │
│                                                                  │
│  Step 4: Codebook Training                                       │
│  ├── 输入: 清洗后特征                                            │
│  ├── 方法: K-Means 聚类                                          │
│  └── 输出: codebook.pt (512 个码本向量)                          │
│                                                                  │
│  Step 5: Pattern Learning                                        │
│  ├── 输入: 清洗后特征 + Codebook                                 │
│  ├── 方法: 统计符号转移概率                                       │
│  └── 输出: pattern_matrix.pt (转移矩阵 M)                        │
│                                                                  │
│  Step 6: Pool Building                                           │
│  ├── 输入: 清洗后特征 + 元数据                                    │
│  ├── 构建: FAISS 索引 + Phone/Gender 子索引                      │
│  └── 输出: target_pool/ (检索池)                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**运行命令**:

```bash
# 运行完整流程
python scripts/run_offline.py --config configs/production.yaml

# 运行指定步骤
python scripts/run_offline.py --config configs/production.yaml --step 6

# 查看状态
python scripts/run_offline.py --config configs/production.yaml --status
```

### 4.2 Online Pipeline (在线推理)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Online Pipeline (5 Stages)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  源语音                                                          │
│     │                                                            │
│     ▼                                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Stage 1: SSL Feature Extraction     │                        │
│  │ WavLM-Large → [T, 1024] 特征        │                        │
│  └─────────────────────────────────────┘                        │
│     │                                                            │
│     ▼                                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Stage 2: Eta-WavLM Projection       │                        │
│  │ h_clean = h @ P_orth (去说话人)      │                        │
│  └─────────────────────────────────────┘                        │
│     │                                                            │
│     ▼                                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Stage 3: SAMM Masking (可选)        │                        │
│  │ Token/Span 掩码 + Pattern 填充      │                        │
│  └─────────────────────────────────────┘                        │
│     │                                                            │
│     ▼                                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Stage 4: Constrained kNN Retrieval  │                        │
│  │ Phone/Gender 约束检索目标特征        │                        │
│  └─────────────────────────────────────┘                        │
│     │                                                            │
│     ▼                                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Stage 5: HiFi-GAN Vocoder           │                        │
│  │ 特征 → 波形                          │                        │
│  └─────────────────────────────────────┘                        │
│     │                                                            │
│     ▼                                                            │
│  匿名语音                                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**使用示例**:

```python
from pipelines.online.anonymizer import SpeechAnonymizer, AnonymizerConfig

# 加载配置
config = AnonymizerConfig.from_yaml("configs/production.yaml")

# 创建匿名化器
anonymizer = SpeechAnonymizer(config)

# 匿名化
anon_audio = anonymizer.anonymize(
    audio_path="input.wav",
    target_gender="cross"  # 跨性别转换
)
```

---

## 五、配置说明

### 5.1 主要配置项

```yaml
# configs/production.yaml

# 路径配置
paths:
  librispeech_root: "/path/to/LibriTTS"
  wavlm_checkpoint: "checkpoints/WavLM-Large.pt"
  cache_dir: "data/samm_anon/cache"
  checkpoints_dir: "data/samm_anon/checkpoints"

# SSL 配置
ssl:
  model: "wavlm-large"
  layer: 6                    # 使用第6层特征
  hidden_dim: 1024

# Eta-WavLM 配置
eta_wavlm:
  enabled: true
  speaker_subspace_dim: 64    # 说话人子空间维度

# SAMM 配置
samm:
  codebook_size: 512          # 码本大小
  masking:
    token_mask_ratio: 0.10    # Token 掩码比例
    span_mask_ratio: 0.15     # Span 掩码比例

# kNN-VC 配置
knn_vc:
  k: 4                        # 近邻数
  num_clusters: 8             # Phone 聚类数
  use_phone_constraint: true
  use_gender_constraint: true
  use_top1: true              # 使用 Top-1 而非平均
  use_cosine: true            # 使用余弦相似度
```

### 5.2 Pool Building 版本选择

```yaml
offline:
  pool_building:
    # 根据硬件环境选择合适的版本
    use_sampled_version: true     # 采样版本 (120GB 内存)
    use_low_mem_version: false    # 低内存流式版本
    use_dual_gpu_version: false   # 双 GPU 版本 (240GB+ 内存)

    # 采样参数
    target_frames: 15000000       # 采样 1500万帧
```

---

## 六、依赖与环境

### 6.1 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3090 (24GB) | RTX 4090 × 2 |
| 内存 | 64GB | 128GB+ |
| 存储 | 500GB SSD | 1TB NVMe SSD |

### 6.2 软件依赖

```txt
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
faiss-gpu>=1.7.3
h5py>=3.8.0
scikit-learn>=1.2.0
tqdm>=4.65.0
pyyaml>=6.0
soundfile>=0.12.0
```

### 6.3 预训练模型

| 模型 | 说明 | 来源 |
|------|------|------|
| WavLM-Large.pt | WavLM 预训练模型 | Microsoft |
| phone_decoder.pt | 音素预测器 | 自训练 |
| hifigan.pt | HiFi-GAN 声码器 | 开源 |

---

## 七、API 参考

### 7.1 SpeechAnonymizer

```python
class SpeechAnonymizer:
    """语音匿名化主类"""

    def __init__(self, config: AnonymizerConfig):
        """初始化匿名化器"""

    def anonymize(
        self,
        audio: Union[str, Tensor],
        target_gender: str = 'same',
        return_intermediate: bool = False,
    ) -> Tensor:
        """
        匿名化语音

        Args:
            audio: 音频路径或波形张量
            target_gender: 'same'|'cross'|'M'|'F'
            return_intermediate: 是否返回中间结果

        Returns:
            匿名化后的波形
        """
```

### 7.2 OfflineRunner

```python
class OfflineRunner:
    """离线流程运行器"""

    def run(
        self,
        start_step: int = 1,
        end_step: int = 6,
        steps: List[int] = None,
    ) -> Dict:
        """运行离线流程"""

    def get_status(self) -> Dict[int, bool]:
        """获取各步骤完成状态"""
```

---

## 八、评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **WER** | 词错误率 (语义保留) | < 15% |
| **EER** | 等错误率 (隐私保护) | > 40% |
| **MOS** | 平均意见分 (自然度) | > 3.5 |
| **Linkability** | 可链接性分数 | < 0.3 |

---

## 九、常见问题

### Q1: OOM (内存不足)

**解决方案**: 使用采样版本 `use_sampled_version: true`，将目标帧数降低：

```yaml
target_frames: 10000000  # 降到 1000万帧
```

### Q2: 构建速度慢

**解决方案**:
- 使用 GPU 加速 (`use_gpu: true`)
- 增加 batch_size
- 使用双 GPU 版本

### Q3: 音质下降

**排查方向**:
- 检查 WavLM 层数 (推荐 layer 6)
- 检查 Codebook 大小 (推荐 512)
- 检查 kNN k 值 (推荐 4)

---

## 十、版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.0 | 2025-01 | 基础框架，WavLM + kNN-VC |
| v2.0 | 2025-06 | 添加 Eta-WavLM，SAMM 掩码 |
| v2.1 | 2025-12 | Top-1 策略，余弦相似度优化 |
| v3.0 | 2026-01 | Pattern 分组，多版本 Pool Building |

---

## 十一、参考文献

1. **WavLM**: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
2. **kNN-VC**: Voice Conversion With Just Nearest Neighbors
3. **Private kNN-VC**: Towards Private Neural Voice Cloning
4. **HiFi-GAN**: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

---

*文档结束*
