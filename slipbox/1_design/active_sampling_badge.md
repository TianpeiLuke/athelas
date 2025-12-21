---
tags:
  - resource
  - active_learning
  - active_sample_selection
  - hybrid_sampling
  - badge
keywords:
  - active learning
  - BADGE
  - gradient embeddings
  - uncertainty and diversity
  - pseudo-labels
  - GBDT
  - deep learning
topics:
  - active learning
  - sampling strategies
  - hybrid methods
language: python
date of note: 2025-11-17
---

# BADGE: Batch Active Learning by Diverse Gradient Embeddings

## 1. Purpose and Scope

BADGE selects a batch of unlabeled points based on gradient embeddings:

> Points with large gradients are informative (uncertain), and selecting them in a diverse way improves coverage.

Scope:

- Primary: deep neural networks with differentiable classification heads.
- Secondary: GBDTs via pseudo-gradient approximations (see Section 6).

The implementation is provided as `BADGESampler`.

## 2. Conceptual Overview

BADGE constructs, for each unlabeled point, an approximation of the gradient of the loss w.r.t. classifier parameters:

- Large gradient norm ⇒ model could change significantly if the point were labeled and trained on.
- By clustering gradient vectors (via k-means++ or k-center), we pick a subset that spans diverse directions of change.

This combines:

- Uncertainty (through gradients / pseudo-labels).  
- Diversity (through clustering in gradient space).

## 3. Mathematical Definition

For a C-class classifier:

- Model outputs logits $f_\theta(x) \in \mathbb{R}^C$.  
- Probabilities $p(x) = \text{softmax}(f_\theta(x))$.

Let $\hat{y}(x) = \arg\max_c p_c(x)$ be the pseudo-label.

For each unlabeled point $x$, define gradient embedding:

$$
g(x) = \nabla_{\theta} \ell(\theta; x, \hat{y}(x))
$$

In practice, for a linear last layer $W \in \mathbb{R}^{C \times D}$ on top of feature $h(x) \in \mathbb{R}^D$, the gradient w.r.t. $W$ can be written as:

$$
g(x) = \big( p(x) - e_{\hat{y}(x)} \big) \otimes h(x) \in \mathbb{R}^{C \cdot D}
$$

where $e_{\hat{y}}$ is the one-hot vector of the pseudo-label.

BADGE:

1. Computes $g(x)$ for all $x \in \mathcal{U}$.  
2. Uses k-means++ or farthest-first to select k diverse gradient vectors.

## 4. Model Interface and Data Contracts

### 4.1 Required model APIs (DL)

We assume a classifier with a linear last layer:

- `encode(X_unlabeled) -> np.ndarray [N, D]` — feature extractor.
- `predict_proba(X_unlabeled) -> np.ndarray [N, C]`.
- Access to number of classes `C`.

We do not require full autograd; we compute the closed-form gradient embedding for the last layer.

### 4.2 Data formats

- For each unlabeled point, we need:
  - Feature vector $h(x)$: `[D]`.  
  - Probability vector $p(x)$: `[C]`.  
  - Pseudo-label index `y_hat`.

Gradient embedding is stored as `[C*D]` or `[C, D]` reshaped.

## 5. Deep Learning Variant

### 5.1 Algorithm Steps (DL)

1. Put model in `eval()` mode.  
2. For unlabeled mini-batches:
   - Compute `h = encode(X_batch)` → `[B, D]`.  
   - Compute `p = predict_proba(X_batch)` → `[B, C]`.  
   - Compute pseudo-labels: `y_hat = p.argmax(axis=1)`.  
   - Construct gradient embeddings:

      ```python
      # p: [B, C], y_hat: [B], h: [B, D]
      one_hot = np.eye(C)[y_hat]               # [B, C]
      diff = p - one_hot                       # [B, C]
      g = diff[..., None] * h[:, None, :]      # [B, C, D]
      g_flat = g.reshape(B, C * D)             # gradient embeddings
      ```

   - Store `g_flat` for all points.

3. Run k-means++-style farthest-first on the set of gradient embeddings to select k points.

### 5.2 Complexity and Performance

- Cost is dominated by:
  - Encoding (`encode`) + probability computation.
  - k-center on gradient embeddings dimension `C*D`.

- Memory: storing `[N, C*D]` can be heavy; we may:
  - Use float16, or  
  - Sample a subset of the unlabeled pool for the BADGE step.

## 6. GBDT Variant (Pseudo-BADGE)

BADGE is not naturally defined for GBDTs, but we can approximate.

### 6.1 Pseudo-gradient embeddings

For GBDTs trained with a differentiable loss:

- For each unlabeled point:
  - Predict probability vector $p(x)$ and pseudo-label $\hat{y}(x)$.  
  - Compute pseudo-gradient of loss w.r.t. logits:

$$
\delta(x) = p(x) - e_{\hat{y}(x)} \in \mathbb{R}^C
$$

- Combine with leaf embeddings $z(x) \in \mathbb{R}^{D_{\text{leaf}}}$:

$$
g_{\text{GBDT}}(x) = \delta(x) \otimes z(x) \in \mathbb{R}^{C \cdot D_{\text{leaf}}}
$$

Use the same k-center selection on $g_{\text{GBDT}}(x)$.

### 6.2 Algorithm Steps (GBDT)

1. Compute `leaf_embeddings(X_unlabeled) -> [N, D_leaf]`.  
2. Compute `p = predict_proba(X_unlabeled) -> [N, C]`.  
3. Compute pseudo-labels and $\delta(x)$ for each sample.  
4. Form $g_{\text{GBDT}}(x)$ via outer product and flatten to `[C * D_leaf]`.  
5. Run farthest-first / k-means++ over these embeddings to select k samples.

This yields a Pseudo-BADGE variant compatible with XGBoost / LightGBM / CatBoost.

## 7. Configuration Parameters

| Name              | Type | Default | Description                                                  |
|-------------------|------|---------|--------------------------------------------------------------|
| `batch_size`      | int  | 32      | Queried points per AL iteration.                            |
| `dl_batch_size`   | int  | 1024    | DL-only: minibatch size when computing gradient embeddings. |
| `metric`          | str  | "euclidean" | Distance metric for gradient embeddings.                 |
| `leaf_dim`        | int  | 256     | GBDT-only: hashed embedding dimension for leaf vectors.     |
| `use_subset`      | bool | False   | If True, run BADGE on a subsample of the unlabeled pool.    |

## 8. Complexity and Performance

- DL:
  - Encoding + probability pass: similar to uncertainty sampling.  
  - Additional cost: storing and clustering gradient embeddings.  

- GBDT:
  - `predict_proba` + `leaf_embeddings` are cheap.  
  - The main cost is clustering in `[C * D_leaf]` space.

For very large pools, BADGE can be run on a candidate subset, then refined with simpler uncertainty criteria.

## 9. Integration into ActiveSampleSelection API

```python
class BADGESampler(ActiveSampler):
    def __init__(self, dl_batch_size: int = 1024, metric: str = "euclidean", use_subset: bool = False):
        ...

    def select_batch(self, model, unlabeled_x, labeled_x=None, labeled_y=None, batch_size=32):
        # 1. Extract features and probabilities
        # 2. Build gradient embeddings (DL or GBDT pseudo-gradients)
        # 3. Run farthest-first / k-means++
        # 4. Return indices of selected points
```

The sampler relies on a model wrapper exposing consistent `encode`, `predict_proba`, and (optionally) `leaf_embeddings` methods.

## 10. Known Failure Modes and Diagnostics

- High-dimensional gradient embeddings:
  - Memory / time blow-up when C and D are large.
  - Mitigation: dimensionality reduction (PCA), candidate subset, or smaller feature layers.

- Noisy pseudo-labels:
  - When model is very early in training, pseudo-labels may be unreliable.
  - Mitigation: delay BADGE until initial warm-up; combine with basic uncertainty sampling in the first rounds.

Diagnostics:

- Monitor distribution of gradient norms.  
- Visualize clustering assignments over gradient embeddings.  
- Compare AL performance vs margin-only and core-set-only baselines.

## 11. Testing Strategy

- Small benchmark (e.g., MNIST / CIFAR subset):
  - Compare BADGE vs. margin and core-set baselines for early rounds.  
- Sanity check on synthetic data:
  - Ensure farthest-first over gradient embeddings is sensitive to both uncertainty and diversity.

## 12. Extensions and Variations

- Combine BADGE with class balancing: enforce per-class quotas in selected batch.  
- Use approximate nearest neighbors for clustering in very high dimensions.  
- Extend to multi-label classification via appropriate gradient definitions.
