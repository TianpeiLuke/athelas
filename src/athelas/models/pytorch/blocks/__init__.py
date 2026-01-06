"""
Composite Building Blocks for Neural Networks

This module provides composite encoder blocks that combine multiple atomic
components into complete encoding architectures.

Components:
- TransformerBlock: Self-attention + FFN block
- TransformerEncoder: Stack of transformer blocks
- LSTMEncoder: Bidirectional LSTM with attention pooling
- CNNEncoder: Multi-kernel 1D CNN for sequences (TextCNN)
"""

from .transformer_block import TransformerBlock
from .lstm_encoder import LSTMEncoder
from .transformer_encoder import TransformerEncoder
from .cnn_encoder import CNNEncoder, compute_cnn_output_length

__all__ = [
    "TransformerBlock",
    "LSTMEncoder",
    "TransformerEncoder",
    "CNNEncoder",
    "compute_cnn_output_length",
]
