"""
Fusion Mechanisms for Multi-Modal Learning

This module provides atomic fusion components for combining multiple modalities
(e.g., text, tabular, images) in neural network architectures.

Components:
- CrossAttentionFusion: Simple bidirectional cross-attention
- BidirectionalCrossAttention: Advanced cross-attention with FFN
- GateFusion: Learnable gating mechanism
- MixtureOfExperts: Routing-based expert fusion
"""

from .cross_attention_fusion import CrossAttentionFusion
from .bidirectional_cross_attention import BidirectionalCrossAttention
from .gate_fusion import GateFusion
from .mixture_of_experts import MixtureOfExperts

__all__ = [
    "CrossAttentionFusion",
    "BidirectionalCrossAttention",
    "GateFusion",
    "MixtureOfExperts",
]
