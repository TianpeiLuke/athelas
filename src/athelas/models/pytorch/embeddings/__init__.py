"""
Embedding Components for Neural Networks

This module provides atomic embedding components for converting various input types
(tabular, temporal, positional, token) into dense vector representations.

Components:
- TabularEmbedding: General-purpose tabular feature embedding
- combine_tabular_fields: Utility to combine multiple tabular fields

Planned (from design document):
- TemporalEncoding: Time-based encodings (TimeEncode, TimeEncoder)
- PositionalEncoding: Position embeddings for transformers
- TokenEmbedding: Categorical/vocabulary embeddings
- FeatureEmbedding: Feature-level embeddings
"""

from .tabular_embedding import TabularEmbedding, combine_tabular_fields

__all__ = [
    "TabularEmbedding",
    "combine_tabular_fields",
]
