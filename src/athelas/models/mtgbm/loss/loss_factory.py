"""
Factory for creating loss function instances with adapter pattern.

ADAPTER PATTERN: Translates modern hyperparams-based API to legacy signatures.

This factory bridges the gap between:
- Modern API: Uses hyperparams object
- Legacy API: Uses positional args (val_sublabel_idx, num_label, etc.)

The adapter extracts legacy parameters from hyperparams and calls
legacy constructors with their expected signatures.
"""

from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from .base_loss_function import BaseLossFunction
from .fixed_weight_loss import FixedWeightLoss
from .adaptive_weight_loss import AdaptiveWeightLoss
from .knowledge_distillation_loss import KnowledgeDistillationLoss

if TYPE_CHECKING:
    from ...hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class LossFactory:
    """
    Factory with adapter pattern for loss function creation.

    Provides two APIs:
    1. create() - Modern hyperparams-based API
    2. create_legacy() - Direct legacy signature (for testing)

    The create() method adapts hyperparams to legacy signatures.
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
        Create a loss function using modern API (adapter method).

        Extracts legacy parameters from hyperparams and calls legacy constructors.

        Parameters
        ----------
        loss_type : str
            Type of loss function ('fixed', 'adaptive', 'adaptive_kd')
        num_label : int
            Number of tasks
        val_sublabel_idx : dict
            Validation set indices for each task {task_id: np.ndarray}
        trn_sublabel_idx : dict, optional
            Training set indices for each task {task_id: np.ndarray}
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters (only used for adaptive_kd loss)

        Returns
        -------
        loss_fn : BaseLossFunction
            Instantiated loss function with legacy signatures

        Raises
        ------
        ValueError
            If loss_type is not registered or required params missing
        """
        if loss_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown loss_type: '{loss_type}'. Available types: {available}"
            )

        loss_class = cls._registry[loss_type]

        # ADAPTER LOGIC: Call legacy constructors with correct signatures

        if loss_type == "fixed":
            # LEGACY: FixedWeightLoss(val_sublabel_idx, num_label, hyperparams)
            # Note: Parameter order swapped from modern API!
            return loss_class(
                val_sublabel_idx=val_sublabel_idx,
                num_label=num_label,
                hyperparams=hyperparams,
            )

        elif loss_type == "adaptive":
            # LEGACY: AdaptiveWeightLoss(num_label, val_sublabel_idx, trn_sublabel_idx, weight_method, hyperparams)
            # Extract weight_method from hyperparams or use None
            weight_method = None
            if hyperparams is not None:
                weight_method = getattr(hyperparams, "loss_weight_method", None)

            return loss_class(
                num_label=num_label,
                val_sublabel_idx=val_sublabel_idx,
                trn_sublabel_idx=trn_sublabel_idx or {},
                weight_method=weight_method,
                hyperparams=hyperparams,
            )

        elif loss_type == "adaptive_kd":
            # LEGACY: KnowledgeDistillationLoss(num_label, val_sublabel_idx, trn_sublabel_idx, patience, weight_method, hyperparams)
            # Extract patience and weight_method from hyperparams
            if hyperparams is None:
                raise ValueError(
                    "hyperparams required for adaptive_kd loss (need patience)"
                )

            patience = getattr(hyperparams, "loss_patience", 10)
            weight_method = getattr(hyperparams, "loss_weight_method", None)

            return loss_class(
                num_label=num_label,
                val_sublabel_idx=val_sublabel_idx,
                trn_sublabel_idx=trn_sublabel_idx or {},
                patience=patience,
                weight_method=weight_method,
                hyperparams=hyperparams,
            )

        else:
            # Should never reach here due to registry check above
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    @classmethod
    def create_legacy(cls, loss_type: str, **kwargs) -> BaseLossFunction:
        """
        Create a loss function using direct legacy signatures (for testing).

        Allows direct instantiation without adapter translation.

        Parameters
        ----------
        loss_type : str
            Type of loss function
        **kwargs
            Direct legacy constructor arguments

        Returns
        -------
        loss_fn : BaseLossFunction
            Instantiated loss function

        Examples
        --------
        >>> # Fixed weight
        >>> loss = LossFactory.create_legacy(
        ...     "fixed",
        ...     val_sublabel_idx={0: idx0, 1: idx1},
        ...     num_label=6
        ... )

        >>> # Adaptive weight
        >>> loss = LossFactory.create_legacy(
        ...     "adaptive",
        ...     num_label=6,
        ...     val_sublabel_idx={0: idx0, 1: idx1},
        ...     trn_sublabel_idx={0: idx0, 1: idx1},
        ...     weight_method="sqrt"
        ... )

        >>> # Adaptive KD
        >>> loss = LossFactory.create_legacy(
        ...     "adaptive_kd",
        ...     num_label=6,
        ...     val_sublabel_idx={0: idx0, 1: idx1},
        ...     trn_sublabel_idx={0: idx0, 1: idx1},
        ...     patience=10,
        ...     weight_method=None
        ... )
        """
        if loss_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown loss_type: '{loss_type}'. Available types: {available}"
            )

        loss_class = cls._registry[loss_type]
        return loss_class(**kwargs)

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
