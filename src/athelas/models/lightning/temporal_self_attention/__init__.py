#!/usr/bin/env python3
"""
PyTorch Lightning TSA Models Package

This package provides PyTorch Lightning implementations of Temporal Self-Attention (TSA) models
for fraud detection and sequential classification tasks.

The package is organized into modular components:
- Core TSA components (shared building blocks)
- Sequential attention module (temporal sequence processing)
- Feature attention module (current transaction processing)
- Main Lightning module (complete TSA classifier)
- Metrics, losses, and utilities
"""

# Main Lightning modules
from .pl_temporal_self_attention_classification import TemporalSelfAttentionClassification
from .pl_dual_sequence_tsa import DualSequenceTSA

# TSA component modules
from .pl_sequential_attention import SequentialAttentionModule
from .pl_feature_attention import FeatureAttentionModule

# Core shared components
from .pl_tsa_components import (
    TimeEncode,
    FeatureAggregation,
    MixtureOfExperts,
    TemporalMultiheadAttention,
    AttentionLayer,
    AttentionLayerPreNorm,
    compute_fm_parallel
)

# Metrics and evaluation
from .pl_tsa_metrics import (
    compute_tsa_metrics,
    TSAMetrics,
    plot_tsa_curves
)

# Loss functions
from .pl_tsa_losses import (
    FocalLoss,
    CyclicalFocalLoss,
    WeightedCrossEntropyLoss,
    LabelSmoothingCrossEntropyLoss,
    AsymmetricLoss,
    get_loss_function
)

# Distributed utilities
from .dist_utils import (
    all_gather,
    all_gather_object,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    reduce_tensor,
    broadcast_object,
    DistributedContext
)

# Native PyTorch schedulers
from .pl_schedulers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule_with_warmup,
    get_scheduler,
    SCHEDULER_REGISTRY
)

__all__ = [
    # Main Lightning modules
    "TemporalSelfAttentionClassification",
    "DualSequenceTSA",
    
    # TSA component modules
    "SequentialAttentionModule",
    "FeatureAttentionModule",
    
    # Core shared components
    "TimeEncode",
    "FeatureAggregation",
    "MixtureOfExperts",
    "TemporalMultiheadAttention",
    "AttentionLayer",
    "AttentionLayerPreNorm",
    "compute_fm_parallel",
    
    # Metrics and evaluation
    "compute_tsa_metrics",
    "TSAMetrics",
    "plot_tsa_curves",
    
    # Loss functions
    "FocalLoss",
    "CyclicalFocalLoss", 
    "WeightedCrossEntropyLoss",
    "LabelSmoothingCrossEntropyLoss",
    "AsymmetricLoss",
    "get_loss_function",
    
    # Distributed utilities
    "all_gather",
    "all_gather_object",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "reduce_tensor",
    "broadcast_object",
    "DistributedContext",
    
    # Native PyTorch schedulers
    "get_linear_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_inverse_sqrt_schedule_with_warmup",
    "get_scheduler",
    "SCHEDULER_REGISTRY",
]
