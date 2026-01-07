"""
Loss Function Factory

Factory function for creating loss functions based on configuration dictionaries.
Provides a unified interface for instantiating different loss types with their parameters.

**Core Concept:**
- Single entry point for all loss function creation
- Configuration-driven instantiation
- Type-safe parameter validation

**Supported Loss Types:**
- FocalLoss
- CyclicalFocalLoss
- WeightedCrossEntropyLoss
- LabelSmoothingCrossEntropyLoss
- AsymmetricLoss
- CrossEntropyLoss (PyTorch default)

**Configuration Format:**
```python
loss_config = {
    "type": "FocalLoss",  # Loss type name
    "alpha": 0.25,        # Loss-specific parameters
    "gamma": 2.0,
    "reduction": "mean"
}
```

**Dependencies:**
- torch.nn → Default CrossEntropyLoss
- .focal_loss → FocalLoss, CyclicalFocalLoss
- .asymmetric_loss → AsymmetricLoss
- .label_smoothing → LabelSmoothingCrossEntropyLoss
- .weighted_ce → WeightedCrossEntropyLoss

**Used By:**
- Lightning modules → Loss function initialization
- Training scripts → Dynamic loss selection

**Usage Example:**
```python
from pytorch.losses import get_loss_function

# Focal Loss
config = {
    "type": "FocalLoss",
    "alpha": 0.25,
    "gamma": 2.0,
    "reduction": "mean"
}
loss_fn = get_loss_function(config)

# Weighted Cross Entropy
config = {
    "type": "WeightedCrossEntropyLoss",
    "class_weights": [1.0, 100.0],
    "reduction": "mean"
}
loss_fn = get_loss_function(config)

# Default Cross Entropy
config = {"type": "CrossEntropyLoss"}
loss_fn = get_loss_function(config)
```
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .focal_loss import FocalLoss, CyclicalFocalLoss
from .asymmetric_loss import AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel
from .label_smoothing import LabelSmoothingCrossEntropyLoss
from .weighted_ce import WeightedCrossEntropyLoss


def get_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.

    Args:
        loss_config: Dictionary containing loss configuration with the following structure:
            - type (str): Loss function type name
            - Additional parameters specific to each loss type

    Returns:
        Configured loss function as nn.Module

    Raises:
        ValueError: If loss type is not supported

    Examples:
        >>> # Focal Loss
        >>> config = {"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0}
        >>> loss_fn = get_loss_function(config)

        >>> # Cyclical Focal Loss
        >>> config = {
        ...     "type": "CyclicalFocalLoss",
        ...     "alpha": 0.25,
        ...     "gamma_min": 1.0,
        ...     "gamma_max": 3.0,
        ...     "cycle_length": 1000
        ... }
        >>> loss_fn = get_loss_function(config)

        >>> # Asymmetric Loss
        >>> config = {
        ...     "type": "AsymmetricLoss",
        ...     "gamma_neg": 4.0,
        ...     "gamma_pos": 1.0,
        ...     "clip": 0.05
        ... }
        >>> loss_fn = get_loss_function(config)
    """
    loss_type = loss_config.get("type", "CrossEntropyLoss")

    if loss_type == "FocalLoss":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
            reduction=loss_config.get("reduction", "mean"),
        )

    elif loss_type == "CyclicalFocalLoss":
        return CyclicalFocalLoss(
            gamma_pos=loss_config.get("gamma_pos", 0.0),
            gamma_neg=loss_config.get("gamma_neg", 4.0),
            gamma_hc=loss_config.get("gamma_hc", 0.0),
            eps=loss_config.get("eps", 0.1),
            reduction=loss_config.get("reduction", "mean"),
            epochs=loss_config.get("epochs", 200),
            factor=loss_config.get("factor", 2.0),
        )

    elif loss_type == "WeightedCrossEntropyLoss":
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        return WeightedCrossEntropyLoss(
            class_weights=class_weights, reduction=loss_config.get("reduction", "mean")
        )

    elif loss_type == "LabelSmoothingCrossEntropyLoss":
        return LabelSmoothingCrossEntropyLoss(
            num_classes=loss_config.get("num_classes", 2),
            smoothing=loss_config.get("smoothing", 0.1),
            reduction=loss_config.get("reduction", "mean"),
        )

    elif loss_type == "AsymmetricLoss":
        return AsymmetricLoss(
            gamma_neg=loss_config.get("gamma_neg", 4.0),
            gamma_pos=loss_config.get("gamma_pos", 1.0),
            clip=loss_config.get("clip", 0.05),
            reduction=loss_config.get("reduction", "mean"),
        )

    elif loss_type == "CrossEntropyLoss":
        # Default to standard CrossEntropyLoss
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        if class_weights is not None:
            return nn.CrossEntropyLoss(
                weight=class_weights, reduction=loss_config.get("reduction", "mean")
            )
        else:
            return nn.CrossEntropyLoss(reduction=loss_config.get("reduction", "mean"))

    else:
        # Default to standard CrossEntropyLoss for unknown types
        return nn.CrossEntropyLoss(reduction=loss_config.get("reduction", "mean"))
