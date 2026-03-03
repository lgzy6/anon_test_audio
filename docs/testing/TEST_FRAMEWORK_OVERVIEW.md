# SAMM-Anon v3.2 Test Framework Overview

## 1. Project Background

SAMM-Anon (Self-supervised Anonymization with Masked Modeling) is a privacy-preserving speech anonymization system. Its core goal is to **remove speaker identity from speech while preserving linguistic content and naturalness**.

The test framework is built to validate that the anonymization pipeline achieves three fundamental objectives:

| Objective | Description | Key Metric |
|-----------|-------------|------------|
| **Privacy** | Speaker identity is unrecoverable | EER > 40%, Speaker Similarity < 0.5 |
| **Utility** | Linguistic content is preserved | WER < 15% |
| **Quality** | Speech sounds natural | MOS > 3.5, Frame Continuity |

---

## 2. System Architecture Under Test

The anonymization pipeline consists of three core modules, tested both independently and end-to-end:

```
Input Audio
    |
    v
[WavLM SSL Extractor]  ──>  H_original (1024-dim features, Layer 15)
    |
    v
[Eta-WavLM Projection]  ──>  H_style (speaker subspace removed)
    |                         H_content (content component)
    v
[SAMM Masked Modeling]  ──>  Quantized style tokens + pattern masking
    |
    v
[Constrained kNN-VC]   ──>  H_anonymous (from target pool with phone/gender constraints)
    |
    v
[HiFi-GAN Vocoder]     ──>  Anonymized Audio Output
```

### Module Descriptions

**Eta-WavLM (Speaker Subspace Projection)**
- Learns a content subspace via PCA/LDA on phone centroids
- Projects features orthogonally to remove speaker identity
- Output: H_style = H - H_content

**SAMM (Masked Modeling)**
- Vector quantization with discrete codebook
- Learns pattern matrix for style characterization
- Token/span masking for unlinkability (prevents re-identification)

**Constrained kNN-VC (Voice Conversion)**
- Retrieves target speaker features from anonymized pool
- Constraints: same phone category, same gender
- FAISS-based cosine similarity Top-1 retrieval

---

## 3. Test Framework Design

### 3.1 Framework Type

The test suite uses a **custom standalone Python test framework** (not pytest/unittest). Each test is an independently executable Python script with `if __name__ == '__main__'` entry points.

Design rationale:
- Tests involve GPU-heavy ML computations (model training, inference)
- Tests produce visual outputs (t-SNE plots, cluster visualizations)
- Tests are iterative research experiments with evolving metrics
- Test orchestration requires pipeline ordering (disentanglement before clustering)

### 3.2 Test Architecture

```
tests/
├── run_v32_tests.py                  # Orchestrator: runs Test A + B sequentially
├── run_full_pipeline_test.py         # Full 4-part pipeline test (A→B→C→D)
├── quick_sanity_check.py             # Fast pre-flight validation
│
├── test_v32_disentanglement.py       # Test A: baseline (PCA + LDA)
├── test_v32_disentanglement_v3.py    # Test A v3: ExpandedSubspace approach
├── test_v32_disentanglement_v4.py    # Test A v4: variant
│
├── test_v32_style_clustering.py      # Test B: basic clustering
├── test_v32_style_clustering_v7.py   # Test B v7: improved VQ
├── test_v32_style_clustering_v72.py  # Test B v7.2: Sinkhorn-Knopp normalization
├── test_v32_style_clustering_v8.py   # Test B v8: contrastive learning
│
├── test_v32_semantic_reconstruction.py   # Test C: content preservation
├── hierarchical_vq_speaker_pool.py       # Experimental: hierarchical VQ
└── samm_75v_island_clustering.py         # Experimental: island clustering
```

### 3.3 Test Categories

The tests form a **progressive validation pipeline** with four stages:

```
[Test A: Disentanglement] → [Test B: Style Clustering] → [Test C: Reconstruction] → [Test D: End-to-End]
         ↓ projector              ↓ patterns                   ↓ quality             ↓ full eval
   "Is identity removed?"  "Are styles meaningful?"     "Is content intact?"    "Does it all work?"
```

---

## 4. Core Concepts

### 4.1 Linear Probe Methodology

The primary validation method is the **linear probe** paradigm:

1. Train a linear classifier on features to predict phone labels
2. High accuracy on H_original → features encode content (expected)
3. Low accuracy on H_style → content has been successfully removed (desired)
4. High accuracy on H_content → content is isolated (validation)

```python
# Concept: if a linear model can't classify phones from H_style,
# then speaker identity has been separated from content
probe = GPULinearProbe(input_dim=1024, n_classes=41)
probe.fit(H_style[train], phones[train], epochs=50)
acc = probe.score(H_style[test], phones[test])  # Should be LOW
```

### 4.2 Subspace Projection

Two methods are tested for learning the content subspace:

| Method | Description | Advantage |
|--------|-------------|-----------|
| **PCA** | SVD on phone centroids, dim = n_phones - 1 | Stable, deterministic |
| **LDA** | Fisher discriminant on between/within class scatter | Better class separation |
| **Expanded** (v3) | Multiple components per phone class | Richer subspace |

### 4.3 Vector Quantization for Style

Style clustering tests train a VQ codebook to discretize style features:

- **Codebook**: K learnable prototype vectors (typically K=8~64)
- **Assignment**: Each utterance maps to nearest codebook entry
- **Goal**: Clusters should capture speaking style, NOT speaker identity

Success criterion: Low ARI/NMI between cluster assignments and speaker labels.

### 4.4 Contrastive Learning (v8)

The v8 iteration introduces an **InfoNCE contrastive loss**:
- Positive pairs: utterances from the same speaker
- Negative pairs: utterances from different speakers
- Goal: Learn embeddings where style similarity doesn't leak speaker identity

---

## 5. Data Flow

### Input Data

| Dataset | Samples | Speakers | Purpose |
|---------|---------|----------|---------|
| LibriSpeech test-clean | 500 utterances | 5 speakers | Development/testing |
| IEMOCAP | 10,039 utterances | 10 speakers | Extended evaluation |

### Feature Cache

Pre-extracted features stored in HDF5 format:
```
cache/features/wavlm/
├── features.h5       # [N_frames, 1024] float32 tensor
└── metadata.json     # Utterance boundaries, speaker IDs, genders
```

### Target Pool

Pre-built anonymous speaker pool for kNN retrieval:
```
data/samm_anon/target_pool_v52/
├── features.npy     # Pool feature vectors
├── genders.npy      # Gender labels for constraints
└── *.faiss          # FAISS indices for efficient search
```

---

## 6. Configuration System

YAML-based configuration with environment-specific profiles:

| Config File | Purpose | Key Differences |
|-------------|---------|----------------|
| `base.yaml` | Development default | Full pipeline, moderate batch sizes |
| `production.yaml` | Production deployment | Memory optimization, dual-GPU |
| `test.yaml` | Testing profiles | Smaller data, configurable via TEST_PROFILE |
| `test_v32.yaml` | v3.2 specific settings | Minimal override |

Test profiles (via `TEST_PROFILE` env var):
- `no_eta`: Disable Eta-WavLM projection
- `minimal`: Disable SAMM masking
- `layer6`: Use WavLM Layer 6 instead of Layer 15

---

## 7. Evolution History

The test framework has evolved through multiple iterations:

| Version | Key Change | Problem Solved |
|---------|-----------|----------------|
| v1 (baseline) | PCA + LDA disentanglement | Initial validation |
| v3 | ExpandedSubspaceProjector | Richer content subspace |
| v4 | Disentanglement variant | Alternative projector |
| v7 | Improved VQ clustering | Basic style patterns |
| v7.2 | Sinkhorn-Knopp normalization | Balanced pattern usage |
| v7.3 | Island-state optimization | Feature space coverage |
| v8 | Contrastive learning + AMP | Pattern collapse fix, 40% memory reduction |

---

## 8. Key Design Decisions

1. **Standalone scripts over pytest**: Research-oriented tests benefit from direct execution, GPU control, and visual output generation.

2. **GPU-accelerated probes**: Linear probes run on CUDA for speed (50 epochs on 50K frames in seconds vs. minutes with sklearn).

3. **Progressive validation**: Each stage validates a prerequisite before proceeding. Disentanglement must pass before clustering training begins.

4. **Multiple algorithm variants**: Side-by-side comparison of PCA vs LDA, v7 vs v7.2 vs v8 allows empirical selection of the best approach.

5. **Balanced sampling**: All tests explicitly handle class imbalance (silence frames dominate ~50%+ of data).
