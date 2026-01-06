"""Pooling mechanisms for sequence aggregation."""

from .attention_pooling import AttentionPooling
from .feature_aggregation import FeatureAggregation, compute_fm_parallel

__all__ = ["AttentionPooling", "FeatureAggregation", "compute_fm_parallel"]
