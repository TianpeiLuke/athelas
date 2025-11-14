---
tags:
  - paper
  - deep_learning
  - tabular
  - mlp_regularization
keywords:
  - regularization cocktail
  - simple neural nets
  - tabular MLP
  - hyperparameter search
  - GBDT comparison
topics:
  - tabular deep learning
  - regularization strategies
  - model selection
language: python
date of note: 2025-11-13
---

# Paper – Simple Neural Nets Can Excel on Tabular Data (Regularization Is All You Need)

## Core idea

This work argues that **plain MLPs can match or beat sophisticated tabular models and GBDTs** if you invest in a **strong regularization “cocktail” and proper hyperparameter search**.

The key claim: prior comparisons under-estimated neural nets because they were under-regularized and under-tuned.

## Key mechanisms

- Use a **standard MLP** (fully connected layers) but:
  - Combine many regularization techniques: dropout, stochastic depth, mixup, weight decay, etc.
  - Carefully tune width, depth, learning rate, and regularization coefficients.
- Apply a **large hyperparameter search** across multiple datasets to find robust configurations.

## Performance vs GBDTs

- Across dozens of tabular datasets:
  - These well-tuned MLPs:
    - **Outperform many specialized deep architectures** for tabular data.
    - **Can outperform XGBoost** and other GBDTs on average when given comparable tuning effort.
- The paper highlights that **“strong baselines matter”**: naive MLPs are weak, but optimized ones are highly competitive.

## When to use cocktail MLPs

- When you have:
  - Compute budget for **hyperparameter optimization**.
  - Infrastructure for large-scale tuning (e.g., Bayesian optimization, NAS).
- Useful as a **simple but powerful baseline**:
  - Architecture is trivial.
  - Most “magic” is in regularization/tuning.

## Caveats

- Requires significant **tuning effort** to reach the promised performance.
- Without that effort, GBDTs with modest tuning will often win.
- Interpretability is standard NN-level (black box, unless combined with separate explainers).

## Reference

- Kadra et al., **“Regularization is all you Need: Simple Neural Nets can Excel on Tabular Data”**, arXiv:2106.11189.
