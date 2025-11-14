---
tags:
  - paper
  - deep_learning
  - tabular
  - ft_transformer
keywords:
  - FT-Transformer
  - feature tokenizer
  - transformer encoder
  - tabular benchmarks
  - rtdl
topics:
  - tabular deep learning
  - architecture baselines
  - GBDT comparison
language: python
date of note: 2025-11-13
---

# Paper – FT-Transformer: Feature Tokenizer + Transformer for Tabular Data

## Core idea

FT-Transformer is a **simple but strong adaptation of the Transformer** to tabular data.  
Each feature becomes a “token” via a **feature tokenizer**, and a standard Transformer encoder operates over these tokens to produce a representation used for prediction.

## Key mechanisms

- **Feature Tokenizer (FT)**:
  - Categorical features → embeddings (as usual).
  - Numerical features → learned linear / piecewise encodings, then embedded as tokens.
- **Transformer over features**:
  - Self-attention across feature tokens captures complex feature interactions.
  - A `[CLS]` token aggregates information for final prediction.
- **Implementation**:
  - Provided in the **RTDL** (Revisiting Tabular Deep Learning) codebase.
  - Hyperparameters tuned similarly to other DL baselines for fair comparison.

## Performance vs GBDTs

- Across a wide benchmark of tabular datasets, FT-Transformer:
  - Becomes a **strong DL baseline** that outperforms previous tabular NN architectures (MLP, TabNet, NODE) on average.
  - Is often **competitive with CatBoost/XGBoost**, though not uniformly superior.
- The authors describe it as a **“more universal” architecture**:
  - Performs reasonably well across many datasets, including both “DL-friendly” and “GBDT-friendly” ones.

## When to use FT-Transformer

- Medium-scale datasets (thousands–hundreds of thousands of rows).
- When you want:
  - A **single strong NN baseline** for tabular tasks.
  - A model that can capture rich interactions but is still conceptually simple.
- Good candidate to **augment** CatBoost/LightGBM in ensembles.

## Caveats

- Still requires **non-trivial tuning** (depth, heads, embedding sizes, regularization).
- Training/inference typically more expensive than GBDTs.
- On many classic tabular benchmarks, tuned GBDTs still edge it out.

## Reference

- Gorishniy et al., **“Revisiting Deep Learning Models for Tabular Data”**, NeurIPS 2021 (FT-Transformer, RTDL).
