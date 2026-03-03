# Test Catalog: SAMM-Anon v3.2 Complete Test Reference

This document provides a detailed catalog of all test files in the `tests/` directory, including their purpose, inputs, outputs, success criteria, and key classes/functions.

---

## Test Orchestrators

### 1. `run_v32_tests.py` — Main Test Runner

**Purpose**: Sequentially orchestrate Test A (Disentanglement) and Test B (Style Clustering).

**Execution**:
```bash
python tests/run_v32_tests.py
python tests/run_v32_tests.py --config configs/base.yaml
```

**Flow**:
1. Import and call `test_v32_disentanglement.main()` → results['disentanglement']
2. Import and call `test_v32_style_clustering.main()` → results['style_clustering']
3. Print summary with PASSED/FAILED for each test

**Output**: Dictionary `{test_name: result_dict | error_dict}`

---

### 2. `run_full_pipeline_test.py` — Full Pipeline Test

**Purpose**: End-to-end validation across all 4 stages: Disentanglement → Clustering → Anonymization → Evaluation.

**Execution**:
```bash
python -m tests.run_full_pipeline_test
python -m tests.run_full_pipeline_test --audio /path/to/audio.flac
python -m tests.run_full_pipeline_test --skip-disentangle --skip-clustering
```

**CLI Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | Random from LibriSpeech | Input audio path |
| `--gender` | M | Source speaker gender (M/F) |
| `--output` | `outputs/full_test_{timestamp}/` | Output directory |
| `--skip-disentangle` | False | Skip Part 1 |
| `--skip-clustering` | False | Skip Part 2 |
| `--skip-anonymize` | False | Skip Part 3 |
| `--skip-eval` | False | Skip Part 4 |
| `--max-samples` | 50000 | Max frames for disentanglement |

**Parts**:

| Part | Function | What It Tests |
|------|----------|---------------|
| Part 1 | `run_disentanglement_test()` | Content removal from H_style via ExpandedSubspaceProjector |
| Part 2 | `run_style_clustering_test()` | KMeans clustering on style embeddings, ARI/NMI vs speaker |
| Part 3 | `run_anonymization()` | WavLM → Phone → kNN retrieval → HiFi-GAN synthesis |
| Part 4 | `run_evaluation()` | Frame continuity, speaker similarity, Whisper ASR WER |

**Key Functions**:
- `knn_retrieve_from_pool()`: Cosine Top-1 retrieval with gender constraint
- `calculate_wer()`: Edit-distance based WER computation
- `calculate_jaccard()`: Word-level Jaccard similarity

**Outputs**:
- `original.wav` / `anonymized.wav`: Audio files
- `style_clusters.png`: t-SNE visualization (3 panels: cluster/speaker/gender)
- `evaluation_results.json`: All metrics
- `full_results.json`: Aggregated results

---

## Test A: Disentanglement Tests

### 3. `test_v32_disentanglement.py` — Baseline Disentanglement (PCA + LDA)

**Purpose**: Validate that orthogonal projection removes phonemic content from features.

**Core Method**: GPU Linear Probe on H_original vs H_style vs H_content

**Key Classes**:
- `GPULinearProbe`: PyTorch linear classifier (replaces sklearn for speed)
  - `fit(X, y, epochs=50, batch_size=2048, lr=1e-3)`
  - `score(X, y) → float`
- `ContentSubspaceProjectorV2`: Content subspace via PCA or LDA
  - `fit(features, phones, max_per_class=2000)`: Learn subspace
  - `project_to_style(features)`: Remove content → H_style
  - `get_content_component(features)`: Extract content → H_content

**Test Flow**:
1. Load 100K frames from `features.h5`
2. Predict phones via `PhonePredictor`
3. Analyze phone distribution (check silence ratio)
4. Remove silence frames (phone_id=0)
5. Linear probe on H_original → baseline accuracy
6. Learn PCA subspace (dim = n_phones - 1 = 40) → probe H_style_PCA
7. Learn LDA subspace → probe H_style_LDA
8. Probe H_content (sanity check: should retain content)

**Success Criteria**:
| Metric | Target | Verdict |
|--------|--------|---------|
| H_style accuracy < 40% | Hard threshold | SUCCESS |
| H_style < H_original × 0.6 | Relative reduction | PARTIAL |
| Otherwise | — | FAILED |

**Technical Details**:
- PCA: SVD on phone centroids matrix, forced n_phones-1 dimensions
- LDA: Generalized eigenvalue problem S_b × v = λ × S_w × v, regularized S_w (1e-6)
- Balanced sampling: max 2000 frames per phone class

---

### 4. `test_v32_disentanglement_v3.py` — Expanded Subspace

**Purpose**: Test a richer subspace with multiple components per phone class.

**Key Class**:
- `ExpandedSubspaceProjector`: PCA with configurable `n_components` (default 100)
  - Builds subspace from multiple principal directions per phone
  - Used by `run_full_pipeline_test.py` as the default projector

**Difference from v1**: More subspace dimensions (100 vs 40), captures more content variance.

---

### 5. `test_v32_disentanglement_v4.py` — Variant Projector

**Purpose**: Alternative disentanglement approach for comparative evaluation.

---

## Test B: Style Clustering Tests

### 6. `test_v32_style_clustering.py` — Basic Style Clustering

**Purpose**: Verify that utterance-level style embeddings cluster by speaking style, not speaker identity.

**Method**:
1. Extract utterance-level embeddings (mean of frame-level H_style)
2. KMeans clustering (k=8)
3. Evaluate: ARI/NMI vs speaker labels, silhouette score
4. t-SNE visualization

**Success Criteria**:
| Metric | Target |
|--------|--------|
| ARI vs Speaker < 0.1 | Well decoupled |
| ARI vs Speaker < 0.3 | Partial |
| ARI vs Speaker >= 0.3 | Warning |

---

### 7. `test_v32_style_clustering_v7.py` — Improved VQ

**Purpose**: Train a vector quantization codebook on style features.

**Key Innovation**: Learnable codebook with EMA updates, distance-based assignment.

---

### 8. `test_v32_style_clustering_v72.py` — Sinkhorn-Knopp Normalization

**Purpose**: Address unbalanced pattern usage (some codebook entries never used).

**Key Innovation**:
- Sinkhorn-Knopp algorithm enforces uniform pattern assignment
- Island-state feature space optimization
- Aggressive temperature annealing schedule

---

### 9. `test_v32_style_clustering_v8.py` — Contrastive Learning (Latest)

**Purpose**: Fix pattern collapse and improve style-speaker decoupling.

**Key Innovations**:
- **InfoNCE contrastive loss**: Same-speaker samples → similar patterns
- **Improved separation loss**: Based on embedding distances, not just assignments
- **VQ-VAE commitment loss**: More stable quantization gradients
- **Temperature scheduling**: min temp raised to 0.5 (from 0.2)
- **Dead pattern re-initialization**: Random orthogonal vectors
- **Mixed precision (AMP)**: 40% GPU memory reduction
- **DataLoader multi-process**: 3x data loading speedup

**Key Classes**:
- `SpeakerUtteranceDataset`: PyTorch Dataset with speaker-to-index mapping
- Logging system: Timestamped training logs in `logs/` directory

**Training Pipeline**:
```
Load H5 features → Extract utterance embeddings → Build DataLoader
    → Train VQ codebook (with contrastive + separation + commitment losses)
    → Evaluate: ARI, NMI, Silhouette → Visualize t-SNE
```

---

## Test C: Semantic Reconstruction

### 10. `test_v32_semantic_reconstruction.py` — Content Preservation

**Purpose**: Verify that when no anonymization is applied (target = source pattern), content is perfectly preserved.

**Key Class**:
- `SemanticReconstructionTest`
  - `test_reconstruction_quality(features, phones, phone_clusters) → dict`

**Method**:
1. For each frame, find nearest neighbor in same-phone cluster
2. Replace feature with nearest neighbor
3. Measure reconstruction error (MSE, MAE, nearest distance)

**Success Criteria**:
- WER < 10% (approaching WavLM native reconstruction quality)

---

## Quick Validation

### 11. `quick_sanity_check.py` — Fast Pre-flight Check

**Purpose**: Rapid validation before committing to expensive training.

**Checks** (4 items):
| Check | What | Pass Criterion |
|-------|------|----------------|
| 1. Numerical Stability | NaN/Inf in H_style | No NaN or Inf |
| 2. Energy Retention | \|H_style\| / \|H_original\| | 30% < ratio < 95% |
| 3. Orthogonality | H_style · H_content ≈ 0 | Mean \|dot product\| < 0.01 |
| 4. Clustering Preview | Intra-class variance, inter-class distance | Qualitative |

**Verdicts**:
- `GO AHEAD`: style_ratio < 0.5
- `PROCEED WITH CAUTION`: 0.5 ≤ style_ratio < 0.8
- `RECONSIDER`: style_ratio ≥ 0.8 or numerical issues

**Fallback**: If no config/data available, uses synthetic random data with `DummyProjector`.

---

## Experimental Tests

### 12. `hierarchical_vq_speaker_pool.py` — Hierarchical VQ

**Purpose**: Explore multi-level vector quantization for the target speaker pool.

**Approach**: Hierarchical codebook structure with coarse-to-fine quantization.

---

### 13. `samm_75v_island_clustering.py` — Island Clustering

**Purpose**: Explore island-based clustering in feature space for better coverage.

**Approach**: Segment feature space into "islands" with guaranteed minimum utilization.

---

## Summary Table

| # | File | Category | GPU Required | Approx. Time | Key Output |
|---|------|----------|:---:|---|---|
| 1 | `run_v32_tests.py` | Orchestrator | Yes | ~10 min | Summary |
| 2 | `run_full_pipeline_test.py` | Full Pipeline | Yes | ~30 min | Audio + Metrics |
| 3 | `test_v32_disentanglement.py` | Disentanglement | Yes | ~5 min | Accuracy scores |
| 4 | `test_v32_disentanglement_v3.py` | Disentanglement | Yes | ~5 min | Accuracy scores |
| 5 | `test_v32_disentanglement_v4.py` | Disentanglement | Yes | ~5 min | Accuracy scores |
| 6 | `test_v32_style_clustering.py` | Clustering | Yes | ~3 min | ARI/NMI + plots |
| 7 | `test_v32_style_clustering_v7.py` | Clustering | Yes | ~10 min | Training logs |
| 8 | `test_v32_style_clustering_v72.py` | Clustering | Yes | ~10 min | Training logs |
| 9 | `test_v32_style_clustering_v8.py` | Clustering | Yes | ~15 min | Training logs + plots |
| 10 | `test_v32_semantic_reconstruction.py` | Reconstruction | No | ~1 min | MSE/MAE |
| 11 | `quick_sanity_check.py` | Validation | Optional | ~30 sec | Pass/Fail |
| 12 | `hierarchical_vq_speaker_pool.py` | Experimental | Yes | ~10 min | Codebook |
| 13 | `samm_75v_island_clustering.py` | Experimental | Yes | ~10 min | Clusters |
