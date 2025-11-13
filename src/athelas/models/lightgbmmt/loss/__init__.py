"""
Loss functions for LightGBMMT multi-task learning.

This package provides refactored loss function implementations following
design patterns for clean architecture and reduced code duplication.
"""

from .base_loss_function import BaseLossFunction
from .fixed_weight_loss import FixedWeightLoss
from .adaptive_weight_loss import AdaptiveWeightLoss
from .knowledge_distillation_loss import KnowledgeDistillationLoss
from .weight_strategies import (
    WeightUpdateStrategy,
    StandardStrategy,
    TenItersStrategy,
    SqrtStrategy,
    DeltaStrategy,
)
from .loss_factory import LossFactory

__all__ = [
    "BaseLossFunction",
    "FixedWeightLoss",
    "AdaptiveWeightLoss",
    "KnowledgeDistillationLoss",
    "WeightUpdateStrategy",
    "StandardStrategy",
    "TenItersStrategy",
    "SqrtStrategy",
    "DeltaStrategy",
    "LossFactory",
]
