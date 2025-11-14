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

- [[Paper – TabPFN: Tabular Prior-Data Fitted Networks]]  
  Foundation model for small–medium tabular tasks, using in-context Bayesian-style inference and outperforming GBDTs on many small datasets.

- [[Paper – FT-Transformer: Feature Tokenizer + Transformer for Tabular Data]]  
  General-purpose Transformer baseline for tabular data; strong and universal deep baseline, competitive with GBDTs across many benchmarks.

- [[Paper – SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training]]  
  Row+column attention transformer with contrastive self-supervised pretraining; on their benchmarks, outperforms XGBoost/CatBoost/LightGBM on average.

- [[Paper – TabR: Retrieval-Augmented Deep Learning for Tabular Data]]  
  Retrieval-augmented MLP with kNN-style attention over neighbors; state-of-the-art average performance on several tabular benchmarks and strong under temporal drift.

- [[Paper – TabNet: Attentive Interpretable Tabular Learning]]  
  Sequential attention model that selects sparse feature subsets per decision step; interpretable feature usage with competitive performance on many benchmarks.

- [[Paper – Simple Neural Nets Can Excel on Tabular Data (Regularization Is All You Need)]]  
  Shows that heavily regularized, well-tuned MLPs can match or outperform GBDTs, emphasizing the importance of strong baselines and hyperparameter search.

- [[Paper – RealMLP: Better by Default for Tabular Data]]  
  Meta-tuned MLP architecture providing strong default performance on medium-to-large tabular datasets and pairing well with GBDTs in ensembles.

- [[Paper – NODE: Neural Oblivious Decision Ensembles for Tabular Data]]  
  Differentiable generalization of oblivious decision trees that can outperform leading GBDT packages on many tabular tasks.

- [[Paper – IMN: Interpretable Mesomorphic Networks for Tabular Data]]  
  Hypernetwork-based model that generates per-instance linear models, aiming for near-SOTA performance with native, instance-level interpretability.

## Synthesis / analysis

- [[Analysis – Deep Tabular Models vs GBDTs and Production Strategy]]  
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
