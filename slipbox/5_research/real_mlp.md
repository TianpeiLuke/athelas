---
tags:
  - paper
  - deep_learning
  - tabular
  - realmlp
keywords:
  - RealMLP
  - pre-tuned defaults
  - tabular MLP
  - time-accuracy tradeoff
  - GBDT defaults
topics:
  - tabular deep learning
  - default hyperparameters
  - production-friendly baselines
language: python
date of note: 2025-11-13
---

# Paper – RealMLP: Better by Default for Tabular Data

## Core idea

RealMLP is a slightly refined MLP architecture whose **hyperparameters are meta-tuned once** over many datasets to provide strong **default settings**.  
The goal: make deep learning on tabular data *competitive with GBDTs* **without per-dataset tuning**, improving the “out-of-the-box” story.

## Key mechanisms

- Define an **improved MLP architecture** (RealMLP) with:
  - Particular choices for depth, width, activation, normalization, dropout, etc.
- Perform **meta-tuning**:
  - Optimize RealMLP and GBDT hyperparameters on a **meta-train set of 118 datasets**.
  - Evaluate on a **disjoint meta-test set of 90 datasets**, plus GBDT-friendly benchmarks.
- Provide **strong default configurations**:
  - For both RealMLP and major GBDT implementations.

## Performance vs GBDTs

- On **medium-to-large datasets (1k–500k rows)**:
  - RealMLP with default parameters has a **favorable time–accuracy tradeoff** vs other NNs.
  - RealMLP is **competitive with tuned GBDTs** in terms of benchmark scores.
- A **simple ensemble of RealMLP + GBDTs**, each with improved defaults, achieves **excellent performance without hyperparameter tuning**.

## When to use RealMLP

- When you want a **production-friendly NN** that:
  - Works well out of the box.
  - Avoids per-dataset tuning where GBDTs currently dominate.
- Great candidate to pair with GBDTs in an **ensemble baseline**:
  - Same preprocessing.
  - Different inductive biases.

## Caveats

- While defaults are strong, **some datasets still benefit from tuning**.
- On highly irregular datasets, even pre-tuned RealMLP can lag GBDTs.
- RealMLP is a **baseline**, not a sophisticated architecture; TabPFN/TabR/SAINT can still win on hard benchmarks.

## Reference

- Holzmüller et al., **“Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data”**, NeurIPS 2024, arXiv:2407.04491.
