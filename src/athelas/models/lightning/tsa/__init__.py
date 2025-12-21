#!/usr/bin/env python3
"""
PyTorch Lightning Temporal Self-Attention Components

Modular, Lightning-compatible components for TSA models.

Module Organization:
- pl_temporal_encoding: Time encoding utilities
- pl_feature_processing: Feature aggregation and FM
- pl_attention_layers: Attention mechanisms (temporal and standard)
- pl_mixture_of_experts: MoE feedforward layers
- pl_order_attention: Order attention module

Phase 1: Algorithm-Preserving Refactoring
All components preserve EXACT numerical equivalence with legacy implementation.
"""

# Temporal encoding
from .pl_temporal_encoding import TimeEncode

# Feature processing
from .pl_feature_processing import (
    compute_FM_parallel,
    FeatureAggregation,
)

# Mixture of Experts
from .pl_mixture_of_experts import (
    MoE,
    Experts,
    Top2Gating,
)

# Attention layers
from .pl_attention_layers import (
    TemporalMultiheadAttention,
    AttentionLayer,
    AttentionLayerPreNorm,
)

# Order attention
from .pl_order_attention import (
    OrderAttentionModule,
    OrderAttentionConfig,
)

# Feature attention
from .pl_feature_attention import (
    FeatureAttentionModule,
    FeatureAttentionConfig,
)

# TSA models
from .pl_tsa_single_seq import (
    TSASingleSeq,
    TSASingleSeqConfig,
)
from .pl_tsa_dual_seq import (
    TSADualSeq,
    TSADualSeqConfig,
)

# Focal losses
from .pl_focal_losses import (
    create_loss_function,
    AsymmetricLoss,
    AsymmetricLossOptimized,
    ASLSingleLabel,
    CyclicalFocalLoss,
    ASLFocalLoss,
    FocalLoss,
    BinaryFocalLossWithLogits,
)


__all__ = [
    # Temporal encoding
    "TimeEncode",
    # Feature processing
    "compute_FM_parallel",
    "FeatureAggregation",
    # Mixture of Experts
    "MoE",
    "Experts",
    "Top2Gating",
    # Attention layers
    "TemporalMultiheadAttention",
    "AttentionLayer",
    "AttentionLayerPreNorm",
    # Order attention
    "OrderAttentionModule",
    "OrderAttentionConfig",
    # Feature attention
    "FeatureAttentionModule",
    "FeatureAttentionConfig",
    # TSA models
    "TSASingleSeq",
    "TSASingleSeqConfig",
    "TSADualSeq",
    "TSADualSeqConfig",
    # Focal losses
    "create_loss_function",
    "AsymmetricLoss",
    "AsymmetricLossOptimized",
    "ASLSingleLabel",
    "CyclicalFocalLoss",
    "ASLFocalLoss",
    "FocalLoss",
    "BinaryFocalLossWithLogits",
]
