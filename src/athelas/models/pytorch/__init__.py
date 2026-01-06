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

from .attention import AttentionHead, MultiHeadAttention, TemporalMultiheadAttention
from .pooling import AttentionPooling, FeatureAggregation, compute_fm_parallel
from .feedforward import MLPBlock, ResidualBlock, MixtureOfExperts
from .embeddings import TimeEncode, TabularEmbedding, combine_tabular_fields
from .blocks import (
    TransformerBlock,
    TransformerEncoder,
    LSTMEncoder,
    CNNEncoder,
    compute_cnn_output_length,
    BertEncoder,
    get_bert_config,
    create_bert_optimizer_groups,
    # TSA-specific composite blocks
    AttentionLayer,
    AttentionLayerPreNorm,
    OrderAttentionModule,
    FeatureAttentionModule,
)
from .schedulers import (
    create_bert_scheduler,
    create_warmup_scheduler,
    get_scheduler_config_for_lightning,
    calculate_warmup_steps,
)
from .fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    BidirectionalCrossAttention,
    GateFusion,
    ExpertRoutingFusion,
    validate_modality_features,
)

__all__ = [
    # Attention mechanisms
    "AttentionHead",
    "MultiHeadAttention",
    "TemporalMultiheadAttention",
    # Pooling
    "AttentionPooling",
    "FeatureAggregation",
    "compute_fm_parallel",
    # Feedforward networks
    "MLPBlock",
    "ResidualBlock",
    "MixtureOfExperts",
    # Embeddings
    "TimeEncode",
    "TabularEmbedding",
    "combine_tabular_fields",
    # Composite blocks (encoders)
    "TransformerBlock",
    "TransformerEncoder",
    "LSTMEncoder",
    "CNNEncoder",
    "compute_cnn_output_length",
    "BertEncoder",
    "get_bert_config",
    "create_bert_optimizer_groups",
    # TSA-specific composite blocks
    "AttentionLayer",
    "AttentionLayerPreNorm",
    "OrderAttentionModule",
    "FeatureAttentionModule",
    # Schedulers
    "create_bert_scheduler",
    "create_warmup_scheduler",
    "get_scheduler_config_for_lightning",
    "calculate_warmup_steps",
    # Fusion mechanisms
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "BidirectionalCrossAttention",
    "GateFusion",
    "ExpertRoutingFusion",
    "validate_modality_features",
]
