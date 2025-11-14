---
tags:
  - paper
  - deep_learning
  - tabular
  - imn
keywords:
  - mesomorphic networks
  - IMN
  - interpretable neural networks
  - hypernetworks
  - local linear models
topics:
  - interpretable deep learning
  - tabular modeling
  - local explanations
language: python
date of note: 2025-11-13
---

# Paper – IMN: Interpretable Mesomorphic Networks for Tabular Data

## Related notes
- [[tabular_dl_vs_gbdt_production_strategy]] - Analysis of when to use IMN vs GBDTs in production
- [[tabular_dl_models_index]] - Entry point for all tabular deep learning models

## Core idea

Interpretable Mesomorphic Networks (IMN) are **both deep and linear at the same time**:  
they use **hypernetworks** to generate **per-instance linear models**, combining high predictive power with **“free-lunch” interpretability** in the original feature space.

## Key mechanisms

- **Mesomorphic architecture**:
  - A deep hypernetwork takes the input and outputs the weights of a **local linear model** for that input.
  - The final prediction is linear in the original features, so coefficients are directly interpretable.
- **Training**:
  - End-to-end supervised learning.
  - Optimize hypernetwork parameters so that generated local linear models fit the data.
- **Interpretability**:
  - For each instance, IMN provides:
    - A set of feature coefficients.
    - An additive decomposition of the prediction.
  - Comparable in spirit to local surrogate methods (LIME, SHAP) but **built into the model**.

## Performance vs GBDTs and other DL models

- Experiments show IMN:
  - Achieves **accuracy comparable to state-of-the-art black-box tabular models** on many datasets.
  - **Outperforms other inherently interpretable methods** (e.g., TabNet and various explainer-based approaches).
- It often closes much of the gap between interpretable and non-interpretable models, while remaining natively explainable.

## When to use IMN

- Domains where:
  - **Per-instance, feature-level explanations** are required (healthcare, finance, regulated settings).
  - You still want **near-SOTA predictive performance**.
- As a drop-in alternative to:
  - Post-hoc explanation stacks (GBDT/NN + SHAP).
  - Purely symbolic models with limited capacity.

## Caveats

- Hypernetwork architecture and training can be more complex than plain MLPs.
- Inference cost is higher than a fixed linear model (must generate weights per instance).
- Still relatively new; fewer production-grade implementations and benchmarks than GBDTs.

## Reference

- Kadra et al., **“Interpretable Mesomorphic Networks for Tabular Data”**, NeurIPS 2024, arXiv:2305.13072.
