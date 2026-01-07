"""
Composite Building Blocks for Neural Networks

This module provides composite encoder blocks that combine multiple atomic
components into complete encoding architectures.

Components:
- TransformerBlock: Self-attention + FFN block
- TransformerEncoder: Stack of transformer blocks
- LSTMEncoder: Bidirectional LSTM with attention pooling
- CNNEncoder: Multi-kernel 1D CNN for sequences (TextCNN)
- BertEncoder: Pretrained BERT-based encoder with utilities
"""

from .transformer_block import TransformerBlock
from .lstm_encoder import LSTMEncoder
from .transformer_encoder import TransformerEncoder
from .cnn_encoder import CNNEncoder, compute_cnn_output_length
from .bert_encoder import (
    BertEncoder,
    get_bert_config,
    create_bert_optimizer_groups,
)

# TSA-specific composite blocks (Phase 2)
from .attention_layer import AttentionLayer, AttentionLayerPreNorm
from .order_attention import OrderAttentionModule
from .feature_attention import FeatureAttentionModule
from .sequence_gate import DualSequenceGate

__all__ = [
    "TransformerBlock",
    "LSTMEncoder",
    "TransformerEncoder",
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
    "DualSequenceGate",
]
