"""
PyTorch atomic components for neural network models.

This package provides reusable, atomic building blocks for constructing
neural network architectures following Zettelkasten principles:
- Atomicity: One module = one concept
- Connectivity: Explicit dependencies via imports
- Semantic naming: Names describe function, not origin

Components organized by function:
- attention: Attention mechanisms (multihead, cross-attention, etc.)
- blocks: Composite encoder blocks (transformer, LSTM, CNN, etc.)
- embeddings: Input embeddings (tabular, temporal, positional, etc.)
- feedforward: Feed-forward networks (MLP, residual blocks, etc.)
- fusion: Multi-modal fusion mechanisms (cross-attention, gating, MoE, etc.)
- pooling: Pooling operations (attention pooling, sequence pooling, etc.)
"""

from .attention import AttentionHead, MultiHeadAttention
from .pooling import AttentionPooling
from .feedforward import MLPBlock, ResidualBlock
from .embeddings import TabularEmbedding, combine_tabular_fields
from .blocks import (
    TransformerBlock,
    TransformerEncoder,
    LSTMEncoder,
    CNNEncoder,
    compute_cnn_output_length,
)
from .fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    BidirectionalCrossAttention,
    GateFusion,
    MixtureOfExperts,
    validate_modality_features,
)

__all__ = [
    # Attention mechanisms
    "AttentionHead",
    "MultiHeadAttention",
    # Pooling
    "AttentionPooling",
    # Feedforward networks
    "MLPBlock",
    "ResidualBlock",
    # Embeddings
    "TabularEmbedding",
    "combine_tabular_fields",
    # Composite blocks (encoders)
    "TransformerBlock",
    "TransformerEncoder",
    "LSTMEncoder",
    "CNNEncoder",
    "compute_cnn_output_length",
    # Fusion mechanisms
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "BidirectionalCrossAttention",
    "GateFusion",
    "MixtureOfExperts",
    "validate_modality_features",
]
