"""
Fusion Mechanisms for Multi-Modal Learning

This module provides atomic fusion components for combining multiple modalities
(e.g., text, tabular, images) in neural network architectures.

Components:
- ConcatenationFusion: Simple concatenation-based fusion (baseline)
- CrossAttentionFusion: Simple bidirectional cross-attention
- BidirectionalCrossAttention: Advanced cross-attention with FFN
- GateFusion: Learnable gating mechanism
- MixtureOfExperts: Routing-based expert fusion
"""

from .concatenation_fusion import ConcatenationFusion, validate_modality_features
from .cross_attention_fusion import CrossAttentionFusion
from .bidirectional_cross_attention import BidirectionalCrossAttention
from .gate_fusion import GateFusion
from .expert_routing_fusion import ExpertRoutingFusion

__all__ = [
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "BidirectionalCrossAttention",
    "GateFusion",
    "ExpertRoutingFusion",
    "validate_modality_features",
]
