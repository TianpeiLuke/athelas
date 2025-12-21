---
tags:
  - resource
  - active_learning
  - active_sample_selection
  - diversity_sampling
  - core_set
keywords:
  - active learning
  - core-set
  - k-center
  - diversity
  - leaf embeddings
  - GBDT
  - deep learning
topics:
  - active learning
  - sampling strategies
  - diversity and coverage
language: python
date of note: 2025-11-17
---

# Core-Set and Leaf Core-Set Sampling

## 1. Purpose and Scope

This document defines core-set style diversity sampling:

- DL Core-Set: k-center selection in representation space (penultimate-layer embeddings).
- GBDT Leaf Core-Set: same idea applied in leaf-embedding space (tree leaf indices).

Goal: choose a batch of unlabeled points whose embeddings cover the pool well, minimizing redundancy and improving representativeness.

Applicable to:

- DL encoders (vision, text, tabular).
- GBDT models (XGBoost, LightGBM, CatBoost) via leaf embeddings.

## 2. Conceptual Overview

Core-set selection approximates the k-center problem:

> Find a subset of points such that every unlabeled point is close to at least one selected point in embedding space.

Key ideas:

- Promotes diversity: avoids querying near-duplicate points.
- Complements uncertainty-based methods:
  - Uncertainty may focus on a small region.
  - Core-set spreads the selected points across the data manifold.

We implement a greedy farthest-first traversal, which is a standard 2-approximation to the k-center problem.

## 3. Mathematical Definition

Let:

- Unlabeled pool embeddings: $Z = \{ z_j \in \mathbb{R}^D : x_j \in \mathcal{U} \}$
- Distance function: typically Euclidean, $d(z_i, z_j)$.

We want a subset $\mathcal{B} \subset Z$ of size k minimizing:

$$
\max_{z \in Z} \min_{b \in \mathcal{B}} d(z, b)
$$

Greedy farthest-first:

1. Initialize $\mathcal{B} = \{ z_{j_0} \}$ with a random or seed point.  
2. At each step, choose:

$$
z^{*} = \arg\max_{z \in Z} \min_{b \in \mathcal{B}} d(z, b)
$$

3. Add $z^{*}$ to $\mathcal{B}$; repeat until $|\mathcal{B}| = k$.

Leaf Core-Set uses leaf embeddings instead of continuous embeddings (see Section 6).

## 4. Model Interface and Data Contracts

### 4.1 Required model APIs

DL variant:

- `encode(X_unlabeled) -> np.ndarray [N, D]`  
  (e.g., penultimate-layer activations, frozen encoder features.)

GBDT variant:

- `leaf_embeddings(X_unlabeled) -> np.ndarray [N, T]`  
  where each dimension is a tree leaf index; an internal wrapper one-hot/hashes these to `[N, D_leaf]`.

### 4.2 Data formats

- Embeddings are treated as `np.ndarray` on CPU.
- Optional: a `metric` parameter to choose between `"euclidean"` and `"cosine"` distances.

## 5. Deep Learning Variant

### 5.1 Embedding extraction

- Use encoder in `eval()` mode.
- For each unlabeled mini-batch:
  - Compute embeddings from a selected layer (configurable).
  - Move to CPU and store in an array `[N, D]`.

### 5.2 Algorithm Steps (DL)

1. Extract embeddings for all unlabeled points.  
2. Initialize:
   - If there are labeled points, optionally seed the set with their embeddings.
   - Otherwise, pick a random unlabeled point as the first center.
3. Maintain an array `min_dist[j]` = distance from each unlabeled embedding to the nearest selected center.
4. Iteratively:
   - Pick index `j* = argmax_j min_dist[j]`.
   - Add `j*` to selected set.
   - Update `min_dist` using the newly added center.
5. Return indices of selected points.

Complexity: $O(k \cdot |\mathcal{U}| \cdot D)$, but with vectorization and possible approximate nearest-neighbor speedups.

## 6. GBDT Variant: Leaf Core-Set

### 6.1 Leaf embeddings

For GBDTs:

- XGBoost:
  - `pred_leaf=True` returns leaf indices `[N, T]`.

We transform leaf indices to embeddings using one of:

- One-hot per tree (large but exact).  
- Hashing trick to a fixed dimension `D_leaf`.  
- Learned embedding table (optional future extension).

These embeddings capture structural similarity in tree-space.

### 6.2 Algorithm Steps (GBDT)

Same as DL, but using leaf embeddings instead of neural embeddings:

1. Compute `leaf_embeddings(X_unlabeled) -> [N, D_leaf]`.  
2. Optionally normalize (e.g., L2 normalization).  
3. Run farthest-first traversal to select k centers.  
4. Return corresponding sample indices.

This variant is named `LeafCoreSetSampler`.

## 7. Configuration Parameters

| Name               | Type | Default       | Description                                             |
|--------------------|------|---------------|---------------------------------------------------------|
| `batch_size`       | int  | 32            | Queried points per AL iteration.                       |
| `metric`           | str  | "euclidean"   | Distance metric for embeddings.                        |
| `seed_with_labeled`| bool | True          | Whether to seed centers with labeled set embeddings.   |
| `leaf_dim`         | int  | 256           | GBDT-only: hashed embedding dimension for leaves.      |
| `approximate`      | bool | False         | If True, allow approximate nearest neighbors later.    |

## 8. Complexity and Performance

- DL:
  - Embedding extraction dominates cost (one forward pass).
  - k-center selection is quadratic-ish in pool size; may need:
    - subsampling, or
    - approximate methods for very large pools.
- GBDT:
  - Leaf embedding extraction is cheap (traversing trees).
  - Same k-center complexity in leaf space.

## 9. Integration into ActiveSampleSelection API

Two concrete classes:

```python
class CoreSetSampler(ActiveSampler):
    # DL-focused core-set sampler over encoder embeddings.
    ...

class LeafCoreSetSampler(ActiveSampler):
    # GBDT-focused core-set sampler over leaf embeddings.
    ...
```

Both implement:

- `select_batch(model, unlabeled_x, labeled_x=None, labeled_y=None, batch_size=...)`

Difference lies only in how embeddings are produced.

## 10. Known Failure Modes and Diagnostics

- Outliers:
  - Farthest-first may pick extreme outliers early.
  - Mitigation: clip distances; reject points with distance above a quantile; or combine with uncertainty.
- High-dimensional embeddings:
  - Distances may become noisy.
  - Mitigation: PCA or smaller embedding layers.

Diagnostics:

- Plot distance distribution to nearest center vs. iteration.  
- Visualize embeddings (t-SNE/UMAP) and highlight selected points.

## 11. Testing Strategy

- Synthetic blobs:
  - Check that selected centers roughly align with distinct clusters.
- Compare DL vs Leaf Core-Set on a small tabular dataset:
  - Use a DL encoder and an XGBoost model trained on the same features.
  - Inspect overlap in selected points and coverage of feature space.

## 12. Extensions and Variations

- Core-set + uncertainty:
  - Score = $\lambda \cdot \text{uncertainty} + (1 - \lambda) \cdot \text{diversity}$.
- Cluster-based selection:
  - k-means clustering, then pick nearest points to cluster centroids (cheaper approximation).
