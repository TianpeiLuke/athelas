---
tags:
  - resource
  - active_learning
  - active_sample_selection
  - uncertainty_sampling
  - margin_entropy
keywords:
  - active learning
  - uncertainty sampling
  - margin
  - entropy
  - least confidence
  - GBDT
  - deep learning
topics:
  - active learning
  - sampling strategies
  - uncertainty estimation
language: python
date of note: 2025-11-17
---

# Margin / Entropy / Least-Confidence Uncertainty Sampling

## 1. Purpose and Scope

This document specifies the classic probabilistic uncertainty sampling family:

- Least confidence
- Margin sampling
- Entropy-based sampling

All three are treated as modes of a single implementation:
`UncertaintySampler(strategy_mode=...)`.

Scope:

- Deep learning (DL): any classifier that outputs class probabilities.
- GBDT: XGBoost / LightGBM / CatBoost on tabular data.
- Tasks: supervised, semi-supervised (via pseudo-labels), or self-supervised with a classifier head on top of an encoder.

## 2. Conceptual Overview

The core idea: query points whose predicted label is most uncertain under the current model. For probabilistic classifiers:

- If softmax probabilities are close to uniform, the model is unsure.
- For margin sampling, a small gap between the top-2 class probabilities indicates ambiguity.

These strategies are:

- Cheap: only require probabilities over the unlabeled pool.
- Generic: independent of architecture (DL vs GBDT).
- Strong baselines on many tabular and vision benchmarks.

They can fail when:

- The model is overconfident even on errors.
- The pool contains many outliers or distribution-shifted samples.

## 3. Mathematical Definition

Let:

- Labeled set: $\mathcal{L} = \{(x_i, y_i)\}$
- Unlabeled pool: $\mathcal{U} = \{x_j\}$
- Model predictive distribution: $p_\theta(y \mid x) \in \mathbb{R}^C$

For each $x \in \mathcal{U}$, compute a score:

1. Least confidence:

$$
s_{\text{LC}}(x) = 1 - \max_c p_\theta(y=c \mid x)
$$

2. Margin:

Let $p_{(1)}(x) \ge p_{(2)}(x) \ge \dots$ be sorted class probabilities:

$$
m(x) = p_{(1)}(x) - p_{(2)}(x), \quad
s_{\text{margin}}(x) = -m(x)
$$

(smaller margin ⇒ larger score)

3. Entropy:

$$
s_{\text{entropy}}(x) = -\sum_{c=1}^C p_\theta(y=c \mid x)\, \log p_\theta(y=c \mid x)
$$

Batch selection:

Given budget $k$, pick indices:

$$
\mathcal{B} = \operatorname{TopK}\big( \{(x, s(x)) : x \in \mathcal{U}\} \big)
$$

where `TopK` returns the k unlabeled points with highest score.

## 4. Model Interface and Data Contracts

### 4.1 Required model APIs

The sampler assumes the model (or wrapper) exposes:

- `predict_proba(X_unlabeled) -> np.ndarray [N, C]`  
  - For DL: typically softmax over logits.  
  - For GBDT: normalized class probabilities from `predict_proba` or equivalent.

Optional:

- `predict(X_unlabeled)` for regression (not used here).  
- Device info for DL to handle CPU/GPU moves (handled by wrappers).

### 4.2 Data formats

- `X_unlabeled`: `np.ndarray` or framework tensor; ultimately converted to `np.ndarray` for scoring.
- Supports:
  - Binary and multi-class classification.
  - Multi-label only if converted to a probability vector per example.

The sampler works with either indices or feature arrays; the core logic operates on `probs: np.ndarray [N, C]`.

## 5. Deep Learning Variant

### 5.1 Computational graph and hooks

- Use eval mode (`model.eval()`) and no gradients.
- For each unlabeled mini-batch:
  - Forward pass to obtain logits.
  - Apply softmax to get probabilities.
- No need for hidden-layer embeddings or gradients.

### 5.2 Algorithm Steps (DL)

1. Switch model to evaluation mode.
2. Iterate unlabeled data in mini-batches (to fit GPU memory).
3. For each batch:
   - Compute `probs = softmax(logits)`.
   - Compute scores according to selected `strategy_mode`.
4. Concatenate scores across all batches.
5. Select TopK indices by score.

## 6. GBDT Variant (XGBoost / LightGBM / CatBoost)

### 6.1 Required APIs

- XGBoost:
  - `predict(DMatrix, output_margin=False)` to get probabilities (`[N, C]`).
- LightGBM:
  - `predict(X, raw_score=False)` with `predict_proba` wrapper if needed.
- CatBoost:
  - `predict_proba(X)` or `predict(X, prediction_type='Probability')`.

### 6.2 Algorithm Steps (GBDT)

1. Build appropriate dataset object (e.g., `xgboost.DMatrix`).  
2. Call GBDT’s probability prediction API.  
3. Compute `s(x)` exactly as in the DL case.  
4. Select TopK indices.

Because GBDTs are typically CPU-bound and fast on tabular data, we can often score the entire unlabeled pool in one call.

### 6.3 Practical Notes

- For high-class-count problems, entropy is more discriminative than least confidence.
- For tabular setups, margin sampling is a strong default.
- Calibration of probabilities (e.g., via Platt scaling / temperature scaling) can improve behavior but is not required.

## 7. Configuration Parameters

| Name            | Type | Default  | Description                                           |
|-----------------|------|----------|-------------------------------------------------------|
| `strategy_mode` | str  | "margin" | "margin", "entropy", or "least_confidence".          |
| `batch_size`    | int  | 32       | Points to query per AL round.                        |
| `dl_batch_size` | int  | 1024     | DL-only: minibatch size when scoring unlabeled pool. |
| `normalize`     | bool | True     | Ensure probabilities sum to 1 before scoring.        |

## 8. Complexity and Performance

- DL:  
  - Time: one forward pass over the entire unlabeled pool.  
  - Complexity: $O(|\mathcal{U}| \cdot \text{cost\_forward})$.

- GBDT:  
  - Time: one `predict_proba` call; typically $O(|\mathcal{U}| \cdot T)$ where T is number of trees.

Memory: only the probability matrix (either per batch or entire pool if small).

## 9. Integration into ActiveSampleSelection API

Python class:

```python
class UncertaintySampler(ActiveSampler):
    def __init__(self, strategy_mode: str = "margin", dl_batch_size: int = 1024):
        ...

    def select_batch(self, model, unlabeled_x, labeled_x=None, labeled_y=None, batch_size=32):
        # 1. probs = model.predict_proba(unlabeled_x)
        # 2. compute scores based on strategy_mode
        # 3. return indices of TopK scores
```

## 10. Known Failure Modes and Diagnostics

- Overconfident model:
  - Scores collapse (most points low-entropy).
  - Diagnostic: histogram of entropy values is heavily skewed near 0.
- Noisy labels / outliers:
  - Method repeatedly selects hard/noisy examples.
  - Diagnostic: track disagreement between model and human labels on queried set; high noise fraction can indicate issues.

Mitigations:

- Combine with diversity (e.g., cluster-margin).  
- Use ensembles / MC-dropout for better uncertainty estimation.

## 11. Testing Strategy

- Synthetic 2D toy dataset:
  - Compare queried regions vs decision boundary visualization.
- Check invariants:
  - When all probabilities are equal, all scores identical.
  - If one example has strictly lower max-prob than all others, it must be selected under least confidence.
- Cross-backend tests:
  - Same logic applied to DL softmax model and XGBoost `predict_proba`.

## 12. Extensions and Variations

- BALD and other Bayesian acquisition functions build on the same probability interface but with MC sampling.
- Class-balanced uncertainty: apply uncertainty separately per predicted class and sample per-class TopK to avoid class collapse.
