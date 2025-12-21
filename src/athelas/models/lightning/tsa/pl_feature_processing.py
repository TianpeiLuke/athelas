#!/usr/bin/env python3
"""
PyTorch Lightning Feature Processing Components

Feature aggregation and processing utilities for TSA models.

Phase 1: Algorithm-Preserving Refactoring
- Direct recreation of legacy components
- NO optimizations or modifications
- EXACT numerical behavior preservation

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/basic_blocks.py
"""

import torch
import torch.nn as nn


def compute_FM_parallel(feature_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute Factorization Machine in parallel.

    EXACT recreation of legacy compute_FM_parallel from basic_blocks.py.
    Phase 1: No modifications.

    FM formula: 0.5 * (sum(x)^2 - sum(x^2))

    This captures second-order feature interactions efficiently.

    Args:
        feature_embedding: Feature embeddings [B, L, D, E]
            - B: Batch size
            - L: Sequence length
            - D: Number of features
            - E: Embedding dimension

    Returns:
        FM output [B, L, E] - Aggregated feature interactions

    Example:
        >>> features = torch.randn(32, 50, 100, 128)  # [B, L, D, E]
        >>> fm_output = compute_FM_parallel(features)  # [32, 50, 128]
    """
    summed_features_emb = torch.sum(feature_embedding, dim=-2)
    summed_features_emb_square = torch.square(summed_features_emb)

    # Square then sum
    squared_features_emb = torch.square(feature_embedding)
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)

    # FM computation
    FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

    return FM


class FeatureAggregation(nn.Module):
    """
    MLP-based feature aggregation for dimensionality reduction.

    EXACT recreation of legacy FeatureAggregation from basic_blocks.py.
    Phase 1: No modifications - preserve exact behavior.

    Architecture:
        Progressive reduction: n → n/2 → n/4 → n/8 → n/16 → n/32 → 1
        LeakyReLU activation between layers

    This is critical for order attention - aggregates features BEFORE attention.
    Reduces the feature dimension from n to 1, creating a single representation
    per position in the sequence.

    Args:
        num_feature: Number of input features (must be divisible by 32)

    Forward:
        Input: [*, num_feature]
        Output: [*, 1]

    Example:
        >>> aggregator = FeatureAggregation(num_feature=128)
        >>> features = torch.randn(32, 50, 128, 64)  # [B, L, num_feature, E]
        >>> # Aggregates over num_feature dimension
        >>> agg_features = aggregator(features.permute(0, 1, 3, 2))  # [B, L, E, num_feature]
        >>> # Result: [B, L, E, 1]
    """

    def __init__(self, num_feature: int):
        super(FeatureAggregation, self).__init__()

        self.dim_embed = num_feature

        # Progressive dimensionality reduction
        # Each layer reduces by half until we reach dimension 1
        self.encoder = nn.Sequential(
            nn.Linear(num_feature, num_feature // 2),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 2, num_feature // 4),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 4, num_feature // 8),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 8, num_feature // 16),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 16, num_feature // 32),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features via MLP.

        Args:
            x: Input features [*, num_feature]

        Returns:
            Aggregated features [*, 1]
        """
        encode = self.encoder(x)
        return encode
