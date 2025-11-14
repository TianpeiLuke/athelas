---
tags:
  - entry_point
  - paper
  - tabular
  - deep_learning
keywords:
  - tabular deep learning
  - GBDT comparison
  - paper notes
  - index card
  - model zoo
topics:
  - documentation index
  - model architecture overview
  - tabular model selection
language: python
date of note: 2025-11-13
---

# Entry – Deep Tabular Models vs GBDTs (Index)

## Purpose

This entry-point note is an **index card** for deep learning models that compete with or complement XGBoost/LightGBM/CatBoost on tabular data.  
It links to short Zettelkasten-style notes for each model and to a higher-level analysis of when these models actually beat GBDTs.

## Paper notes (one model per card)

- [[tabular_tabpfn_foundation_model]]  
  Foundation model for small–medium tabular tasks, using in-context Bayesian-style inference and outperforming GBDTs on many small datasets.

- [[tabular_ft_transformer_architecture]]  
  General-purpose Transformer baseline for tabular data; strong and universal deep baseline, competitive with GBDTs across many benchmarks.

- [[tabular_saint_row_column_attention]]  
  Row+column attention transformer with contrastive self-supervised pretraining; on their benchmarks, outperforms XGBoost/CatBoost/LightGBM on average.

- [[tabular_tabr_retrieval_augmented]]  
  Retrieval-augmented MLP with kNN-style attention over neighbors; state-of-the-art average performance on several tabular benchmarks and strong under temporal drift.

- [[tabular_tabnet_attentive_selection]]  
  Sequential attention model that selects sparse feature subsets per decision step; interpretable feature usage with competitive performance on many benchmarks.

- [[tabular_realmlp_default_baselines]]  
  Meta-tuned MLP architecture providing strong default performance on medium-to-large tabular datasets and pairing well with GBDTs in ensembles.

- [[tabular_node_differentiable_trees]]  
  Differentiable generalization of oblivious decision trees that can outperform leading GBDT packages on many tabular tasks.

- [[tabular_imn_interpretable_hypernetworks]]  
  Hypernetwork-based model that generates per-instance linear models, aiming for near-SOTA performance with native, instance-level interpretability.

## Synthesis / analysis

- [[tabular_dl_vs_gbdt_production_strategy]]  
  Discusses:
  - When deep tabular models actually beat GBDTs (dataset regimes, regularity, drift, unlabeled data).
  - When GBDTs still dominate in practice.
  - A pragmatic strategy to augment or replace XGBoost/LightGBM/CatBoost in production pipelines using TabPFN, RealMLP, FT-Transformer, TabR, and IMN.

## How to use this cluster

- Start with the **analysis note** to understand the **big picture** and production strategy.
- Jump into any **paper note** when you need details on:
  - Architectural choices.
  - Performance claims vs GBDTs.
  - When a given model is a good fit for your workload.
- Extend this index as you add new model cards (e.g., TabM, Mamba-based tabular models, new foundation models).
