# Test Execution Guide

How to set up the environment, configure, and run the SAMM-Anon v3.2 test suite.

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x NVIDIA GPU, 8GB VRAM | 1x RTX 4090 (24GB) |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB (models + cache) | 20 GB |

### 1.2 Software Dependencies

Install via conda + pip:

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate samm_anon

# Or install via pip
pip install -r requirements.txt
```

**Core Dependencies**:
- Python 3.10+
- PyTorch 2.1.0 + Torchaudio 2.1.0
- FAISS-GPU 1.7.4 (fallback: faiss-cpu)
- NumPy 1.24.3, SciPy 1.11.3, scikit-learn
- h5py, librosa, soundfile
- matplotlib, tqdm

**Optional Dependencies**:
- `openai-whisper`: Required for ASR evaluation in full pipeline test (auto-installed)
- `jiwer`: Alternative WER calculation
- `tensorboard`: Training visualization

### 1.3 Required Checkpoints

Ensure the following files exist in `checkpoints/`:

```
checkpoints/
├── WavLM-Large.pt          # ~1.2 GB — WavLM SSL model
├── phone_decoder.pt        # ~3.9 MB — Phone predictor
├── duration_decoder.pt     # ~1.6 MB — Duration predictor
├── hifigan.pt              # HiFi-GAN vocoder weights
└── hifigan.json            # HiFi-GAN config
```

### 1.4 Required Data Cache

Pre-extracted features must exist:

```
cache/features/wavlm/
├── features.h5             # Pre-extracted WavLM Layer 15 features
└── metadata.json           # Utterance boundaries and speaker metadata
```

If not available, run the offline pipeline Step 1 first:
```bash
python scripts/run_offline.py --step 1
```

---

## 2. Running Tests

### 2.1 Quick Sanity Check (Start Here)

Fast validation before running expensive tests:

```bash
cd /root/autodl-tmp/anon_test
python tests/quick_sanity_check.py
```

**Expected output**: 4 checks (Numerical Stability, Energy Retention, Orthogonality, Clustering Preview) with a final GO AHEAD / PROCEED WITH CAUTION / RECONSIDER verdict.

**Time**: ~30 seconds (GPU) / ~2 minutes (CPU)

### 2.2 Individual Tests

Run specific component tests:

```bash
# Test A: Disentanglement (baseline PCA + LDA)
python tests/test_v32_disentanglement.py

# Test A: Disentanglement (v3 Expanded Subspace)
python tests/test_v32_disentanglement_v3.py

# Test B: Style Clustering (latest v8 with contrastive learning)
python tests/test_v32_style_clustering_v8.py

# Test C: Semantic Reconstruction
python tests/test_v32_semantic_reconstruction.py
```

### 2.3 Test Suite (A + B)

Run disentanglement and clustering tests sequentially:

```bash
python tests/run_v32_tests.py
python tests/run_v32_tests.py --config configs/test.yaml
```

### 2.4 Full Pipeline Test

End-to-end test with audio anonymization:

```bash
# Default: random audio from LibriSpeech
python -m tests.run_full_pipeline_test

# Specify input audio
python -m tests.run_full_pipeline_test --audio /path/to/audio.flac --gender M

# Skip specific stages
python -m tests.run_full_pipeline_test --skip-disentangle --skip-clustering

# Custom output directory
python -m tests.run_full_pipeline_test --output ./my_test_output
```

### 2.5 Experimental Tests

```bash
# Hierarchical VQ approach
python tests/hierarchical_vq_speaker_pool.py

# Island clustering approach
python tests/samm_75v_island_clustering.py
```

---

## 3. Configuration

### 3.1 Config File Selection

Tests auto-detect `configs/base.yaml` by default. Override with `--config`:

```bash
python tests/run_v32_tests.py --config configs/test.yaml
```

### 3.2 Test Profiles

Set `TEST_PROFILE` environment variable for ablation studies:

```bash
# Disable Eta-WavLM projection (test without subspace projection)
TEST_PROFILE=no_eta python tests/run_v32_tests.py

# Minimal mode (disable SAMM masking)
TEST_PROFILE=minimal python tests/run_v32_tests.py

# Use WavLM Layer 6 instead of Layer 15
TEST_PROFILE=layer6 python tests/run_v32_tests.py
```

### 3.3 Key Configuration Parameters

In `configs/base.yaml`:

```yaml
ssl:
  model: WavLM-Large
  layer: 15           # WavLM layer to extract (6 or 15)

eta_wavlm:
  n_components: 100   # Subspace dimension

samm:
  codebook_size: 64   # VQ codebook entries
  mask_ratio: 0.3     # Token masking ratio

knn_vc:
  strategy: top1      # Retrieval strategy
  gender_constrained: true
```

---

## 4. Output Structure

### 4.1 Test Outputs

Each test run generates outputs in `outputs/`:

```
outputs/full_test_{timestamp}/
├── original.wav              # Input audio (16kHz mono)
├── anonymized.wav            # Anonymized output audio
├── style_clusters.png        # t-SNE visualization (3 panels)
├── evaluation_results.json   # Detailed metrics
└── full_results.json         # Aggregated results
```

### 4.2 Training Logs

Style clustering training logs are saved to `logs/`:

```
logs/
├── training_v7_{timestamp}.log
├── training_v72_{timestamp}.log
└── training_v8_{timestamp}.log
```

---

## 5. Interpreting Results

### 5.1 Disentanglement Test

```
SUMMARY: Disentanglement Validation
  H_original accuracy:    72.3%     ← Baseline (content is encoded)
  H_style (PCA) accuracy: 28.1%    ← SUCCESS: content removed
  H_style (LDA) accuracy: 31.5%    ← SUCCESS: content removed
  H_content accuracy:     65.8%    ← Content retained (validation)
  Random baseline:         2.4%
```

**Interpretation**: H_style accuracy should be much lower than H_original. A >50% reduction indicates successful disentanglement.

### 5.2 Style Clustering Test

```
Style Clustering Results:
  Silhouette Score:   0.15   ← Moderate cluster structure
  ARI vs Speaker:     0.05   ← Low = good (not correlated with speaker)
  NMI vs Speaker:     0.12   ← Low = good
  ARI vs Gender:      0.02   ← Low = good
```

**Interpretation**: Low ARI/NMI vs Speaker indicates clusters capture style, not identity.

### 5.3 Full Pipeline Test

```
FINAL SUMMARY
  Disentanglement:  content reduction 61.2%
  Style Clustering: ARI=0.05, silhouette=0.15
  Semantic:         WER=8.3% (Excellent)
  Privacy:          speaker_sim=0.42
```

**Grades**:
| WER | Grade | Speaker Similarity | Privacy Grade |
|-----|-------|--------------------|---------------|
| < 10% | Excellent | < 0.5 | Good |
| < 30% | Good | 0.5 - 0.8 | Moderate |
| < 50% | Fair | > 0.8 | Poor |
| >= 50% | Poor | — | — |

---

## 6. Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce max samples
python -m tests.run_full_pipeline_test --max-samples 20000
```

**Features not found**:
```
FileNotFoundError: Features not found: cache/features/wavlm/features.h5
```
Solution: Run offline pipeline Step 1 to extract features, or check `paths.cache_dir` in config.

**Phone decoder not found**:
```
WARNING: phone_decoder.pt not found, using random labels
```
Solution: Download or train the phone decoder. Tests will still run with random labels but results won't be meaningful.

**No audio files in LibriSpeech**:
```
ERROR: No audio files found in LibriSpeech test-clean
```
Solution: Use `--audio` to specify an audio file, or download LibriSpeech test-clean dataset.

---

## 7. Recommended Test Order

For a complete validation cycle:

```
1. quick_sanity_check.py              → Verify basic projection quality
2. test_v32_disentanglement.py        → Validate content removal
3. test_v32_style_clustering_v8.py    → Train and evaluate style patterns
4. run_full_pipeline_test.py          → End-to-end with audio output
```

If any step fails, fix the underlying issue before proceeding to the next.
