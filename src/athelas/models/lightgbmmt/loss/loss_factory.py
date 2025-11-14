"""
Factory for creating loss function instances.

Provides centralized loss function creation with type safety and validation.
"""

from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from .base_loss_function import BaseLossFunction
from .fixed_weight_loss import FixedWeightLoss
from .adaptive_weight_loss import AdaptiveWeightLoss
from .knowledge_distillation_loss import KnowledgeDistillationLoss

if TYPE_CHECKING:
    from ..hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class LossFactory:
    """
    Factory for creating loss function instances.

    Provides centralized loss function creation with type safety and
    configuration validation.
    """

    _registry = {
        "fixed": FixedWeightLoss,
        "adaptive": AdaptiveWeightLoss,
        "adaptive_kd": KnowledgeDistillationLoss,
    }

    @classmethod
    def create(
        cls,
        loss_type: str,
        num_label: int,
        val_sublabel_idx: Dict[int, np.ndarray],
        trn_sublabel_idx: Optional[Dict[int, np.ndarray]] = None,
        hyperparams: Optional["LightGBMMtModelHyperparameters"] = None,
    ) -> BaseLossFunction:
        """
        Create a loss function instance.

        Parameters
        ----------
        loss_type : str
            Type of loss function ('fixed', 'adaptive', 'adaptive_kd')
        num_label : int
            Number of tasks
        val_sublabel_idx : dict
            Validation set indices for each task
        trn_sublabel_idx : dict, optional
            Training set indices for each task
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters containing loss parameters

        Returns
        -------
        loss_fn : BaseLossFunction
            Instantiated loss function

        Raises
        ------
        ValueError
            If loss_type is not registered or hyperparams not provided
        """
        if loss_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown loss_type: '{loss_type}'. Available types: {available}"
            )

        if hyperparams is None:
            raise ValueError("hyperparams is required")

        loss_class = cls._registry[loss_type]

        return loss_class(
            num_label=num_label,
            val_sublabel_idx=val_sublabel_idx,
            trn_sublabel_idx=trn_sublabel_idx,
            hyperparams=hyperparams,
        )

    @classmethod
    def register(cls, name: str, loss_class: type) -> None:
        """Register a new loss function type."""
        if not issubclass(loss_class, BaseLossFunction):
            raise TypeError(f"{loss_class} must inherit from BaseLossFunction")
        cls._registry[name] = loss_class

    @classmethod
    def get_available_losses(cls) -> list:
        """Get list of available loss types."""
        return list(cls._registry.keys())
