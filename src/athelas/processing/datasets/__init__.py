"""
Dataset classes for PyTorch training pipelines.

Provides both regular (map-style) and streaming (iterable-style) datasets
with identical pipeline injection APIs.
"""

from .pipeline_datasets import PipelineDataset
from .pipeline_iterable_datasets import PipelineIterableDataset

__all__ = [
    "PipelineDataset",
    "PipelineIterableDataset",
]
