# Evaluation Metrics Reference

Complete reference for all metrics used in the SAMM-Anon v3.2 test framework, including formulas, interpretation, and target thresholds.

---

## 1. Privacy Metrics

### 1.1 Equal Error Rate (EER)

**What**: The point where False Acceptance Rate = False Rejection Rate in a speaker verification system.

**Why**: Measures how well anonymization prevents speaker re-identification.

**Interpretation**:
| EER | Meaning |
|-----|---------|
| 50% | Perfect anonymization (random chance) |
| > 40% | Good privacy protection |
| 20-40% | Partial protection |
| < 20% | Speaker still largely identifiable |

**Used in**: Full pipeline evaluation (external ASV system).

---

### 1.2 Speaker Similarity (Cosine)

**What**: Cosine similarity between sentence-level WavLM embeddings of original and anonymized speech.

**Formula**:
```
speaker_sim = cos(mean(H_original), mean(H_anonymized))
```

**Interpretation**:
| Value | Privacy Grade |
|-------|--------------|
| < 0.5 | Good — speaker identity changed significantly |
| 0.5 - 0.8 | Moderate — some identity leakage |
| > 0.8 | Poor — speaker still recognizable |

**Used in**: `run_full_pipeline_test.py` Part 4.

---

### 1.3 Linkability Score

**What**: Probability that two anonymized utterances from the same speaker can be linked.

**Target**: < 0.3 (hard to link same-speaker utterances).

**Used in**: External evaluation (VoicePrivacy benchmark).

---

## 2. Content Preservation Metrics

### 2.1 Word Error Rate (WER)

**What**: Minimum edit distance between reference and hypothesis transcriptions, normalized by reference length.

**Formula**:
```
WER = (Substitutions + Insertions + Deletions) / N_reference_words
```

**Implementation**: Dynamic programming edit distance in `run_full_pipeline_test.py`:
```python
def calculate_wer(ref, hyp):
    # Levenshtein distance on word sequences
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    # ... DP matrix computation
    return edit_distance / len(ref_words)
```

**Interpretation**:
| WER | Grade | Meaning |
|-----|-------|---------|
| < 10% | Excellent | Near-perfect content preservation |
| 10-30% | Good | Minor content loss |
| 30-50% | Fair | Noticeable degradation |
| > 50% | Poor | Severe content loss |

**ASR Model**: Whisper (base model, English).

**Used in**: `run_full_pipeline_test.py` Part 4.

---

### 2.2 Jaccard Similarity

**What**: Word-level overlap between original and anonymized transcriptions.

**Formula**:
```
Jaccard = |words_orig ∩ words_anon| / |words_orig ∪ words_anon|
```

**Interpretation**: 1.0 = identical word sets, 0.0 = no overlap. Complements WER by measuring vocabulary preservation regardless of word order.

**Used in**: `run_full_pipeline_test.py` Part 4.

---

### 2.3 Reconstruction Error (MSE / MAE)

**What**: Feature-level reconstruction quality when using same-pattern retrieval (no anonymization).

**Formulas**:
```
MSE = mean((features - reconstructed)^2)
MAE = mean(|features - reconstructed|)
```

**Target**: MSE should be low enough that vocoder produces intelligible speech (WER < 10%).

**Used in**: `test_v32_semantic_reconstruction.py`.

---

## 3. Disentanglement Metrics

### 3.1 Linear Probe Accuracy

**What**: Phone classification accuracy using a linear classifier on projected features.

**Method**:
1. Train a single linear layer (1024 → 41 classes) for 50 epochs
2. Evaluate on held-out 20% test split
3. Compare across feature spaces

**Key Comparisons**:
| Feature Space | Expected Accuracy | Reason |
|---------------|:-:|--------|
| H_original | High (60-80%) | Contains both content and speaker info |
| H_style | Low (< 40%) | Content should be removed |
| H_content | High (50-70%) | Content should be concentrated here |
| Random | 2.4% (1/41) | Chance level for 41 phone classes |

**Content Reduction**:
```
reduction = (1 - acc_style / acc_original) × 100%
```

A reduction > 50% indicates meaningful disentanglement.

**Implementation**: `GPULinearProbe` class — PyTorch nn.Linear trained with Adam optimizer, CrossEntropyLoss.

**Used in**: All `test_v32_disentanglement*.py` files and Part 1 of full pipeline.

---

### 3.2 Energy Retention Ratio

**What**: How much of the original feature energy is retained after projection.

**Formula**:
```
style_ratio = mean(||H_style||) / mean(||H_original||)
content_ratio = mean(||H_content||) / mean(||H_original||)
```

**Interpretation**:
| style_ratio | Verdict |
|:-:|---------|
| < 0.3 | Style component too weak |
| 0.3 - 0.5 | Good disentanglement |
| 0.5 - 0.8 | Moderate, proceed with caution |
| > 0.95 | Almost no content removed |

**Used in**: `quick_sanity_check.py` Check 2.

---

### 3.3 Orthogonality Score

**What**: Verifies that H_style and H_content are orthogonal (as designed).

**Formula**:
```
orthogonality = mean(|H_style · H_content|)
```

**Target**: < 0.01 (nearly orthogonal).

**Used in**: `quick_sanity_check.py` Check 3.

---

## 4. Clustering Metrics

### 4.1 Adjusted Rand Index (ARI)

**What**: Measures agreement between two clustering assignments, adjusted for chance.

**Formula**: Adjusted version of the Rand Index that accounts for random label agreement.

**Range**: [-1, 1]
- 1.0 = perfect agreement
- 0.0 = random agreement
- < 0 = worse than random

**In our context**: ARI(cluster_labels, speaker_labels)
- **Low ARI** (< 0.1) = clusters do NOT correspond to speakers = **SUCCESS**
- **High ARI** (> 0.3) = clusters align with speakers = **FAILURE** (identity leakage)

**Used in**: All style clustering tests.

---

### 4.2 Normalized Mutual Information (NMI)

**What**: Mutual information between cluster assignments and speaker labels, normalized to [0, 1].

**Range**: [0, 1]
- 0.0 = no mutual information (independent)
- 1.0 = perfect correlation

**In our context**: NMI(cluster_labels, speaker_labels)
- **Low NMI** (< 0.15) = good decoupling
- **High NMI** (> 0.3) = information leakage

**Used in**: All style clustering tests.

---

### 4.3 Silhouette Score

**What**: Measures how similar each point is to its own cluster vs. nearest neighbor cluster.

**Formula**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance.

**Range**: [-1, 1]
- 1.0 = dense, well-separated clusters
- 0.0 = overlapping clusters
- < 0 = misclassified points

**In our context**: Applied to style embeddings clustered by KMeans.
- **High silhouette** (> 0.2) = style patterns form meaningful clusters
- **Low silhouette** (< 0.1) = weak cluster structure

**Used in**: All style clustering tests.

---

## 5. Audio Quality Metrics

### 5.1 Frame Continuity

**What**: Cosine similarity between consecutive frames in WavLM feature space.

**Formula**:
```
continuity = mean(cos(H[t], H[t+1])) for t = 0..T-2
low_ratio = fraction of frames where cos < 0.5
```

**Interpretation**:
- High mean_cos (> 0.9) = smooth, natural transitions
- Low low_ratio (< 5%) = no abrupt discontinuities

**Used in**: `run_full_pipeline_test.py` Part 4.

---

### 5.2 Mean Opinion Score (MOS)

**What**: Subjective quality rating by human listeners on a 1-5 scale.

**Target**: > 3.5 (good quality).

**Note**: MOS requires human evaluation and is not computed automatically by the test suite. It is referenced as an external benchmark.

---

## 6. Metrics Summary Table

| Metric | Category | Range | Target | Higher = Better? |
|--------|----------|-------|--------|:-:|
| EER | Privacy | 0-50% | > 40% | Yes |
| Speaker Similarity | Privacy | 0-1 | < 0.5 | No |
| Linkability | Privacy | 0-1 | < 0.3 | No |
| WER | Content | 0-100% | < 15% | No |
| Jaccard Similarity | Content | 0-1 | > 0.8 | Yes |
| MSE | Reconstruction | 0-∞ | Low | No |
| Linear Probe (H_style) | Disentanglement | 0-100% | < 40% | No |
| Content Reduction | Disentanglement | 0-100% | > 50% | Yes |
| Energy Ratio (style) | Disentanglement | 0-1 | 0.3-0.5 | — |
| Orthogonality | Disentanglement | 0-∞ | < 0.01 | No |
| ARI vs Speaker | Clustering | -1 to 1 | < 0.1 | No |
| NMI vs Speaker | Clustering | 0-1 | < 0.15 | No |
| Silhouette | Clustering | -1 to 1 | > 0.2 | Yes |
| Frame Continuity | Quality | 0-1 | > 0.9 | Yes |
| MOS | Quality | 1-5 | > 3.5 | Yes |
