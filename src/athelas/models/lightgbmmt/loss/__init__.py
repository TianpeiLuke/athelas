"""
Loss functions for LightGBMMT multi-task learning.

This package provides refactored loss function implementations following
design patterns for clean architecture and reduced code duplication.
"""

from .base_loss_function import BaseLossFunction
from .fixed_weight_loss import FixedWeightLoss
from .adaptive_weight_loss import AdaptiveWeightLoss
from .knowledge_distillation_loss import KnowledgeDistillationLoss
from .loss_factory import LossFactory
from .weight_strategies import (
    BaseWeightStrategy,
    TenItersWeightStrategy,
    SqrtWeightStrategy,
    DeltaWeightStrategy,
    StandardWeightStrategy,
    WeightStrategyFactory,
)

__all__ = [
    "BaseLossFunction",
    "FixedWeightLoss",
    "AdaptiveWeightLoss",
    "KnowledgeDistillationLoss",
    "LossFactory",
    "BaseWeightStrategy",
    "TenItersWeightStrategy",
    "SqrtWeightStrategy",
    "DeltaWeightStrategy",
    "StandardWeightStrategy",
    "WeightStrategyFactory",
]
