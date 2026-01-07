"""
Weighted Cross Entropy Loss

Weighted Cross Entropy Loss handles class imbalance by applying different weights to each class.
A wrapper around PyTorch's CrossEntropyLoss that provides easy configuration of class weights.

**Core Concept:**
- Applies different weights to different classes
- Simple and effective for moderate class imbalance
- Commonly used baseline for imbalanced classification

**Parameters:**
- class_weights (Optional[torch.Tensor]): Weights for each class (default: None)
- reduction (str): 'mean', 'sum', or 'none' (default: 'mean')

**Forward Signature:**
Input:
  - inputs: (B, n_classes) - Model logits
  - targets: (B,) - Ground truth labels (class indices)

Output:
  - loss: scalar or (B,) - Loss value

**Dependencies:**
- torch.nn → CrossEntropyLoss
- torch → Tensor operations

**Used By:**
- Lightning modules → Classification training
- Any classification task with imbalanced classes

**Mathematical Formulation:**
L = - (1/N) * Σ w_y * log(p_y)

where:
- w_y is the weight for the true class y
- p_y is the predicted probability for class y
- N is the batch size (if reduction='mean')

**Typical Usage:**
- Binary classification: weights = [1.0, rare_class_ratio]
- Multi-class: weights inversely proportional to class frequencies

**Usage Example:**
```python
from pytorch.losses import WeightedCrossEntropyLoss
import torch

# For fraud detection with 1:100 class ratio
class_weights = torch.tensor([1.0, 100.0])  # [normal, fraud]
loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights)

logits = model(batch)
loss = loss_fn(logits, labels)
```
"""

import torch
import torch.nn as nn
from typing import Optional


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.

    This is a wrapper around PyTorch's CrossEntropyLoss that provides
    easy configuration of class weights for fraud detection scenarios
    and other imbalanced classification tasks.
    """

    def __init__(
        self, class_weights: Optional[torch.Tensor] = None, reduction: str = "mean"
    ):
        """
        Initialize Weighted Cross Entropy Loss.

        Args:
            class_weights: Tensor of weights for each class
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights, reduction=reduction)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross entropy loss.

        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]

        Returns:
            Weighted cross entropy loss value
        """
        return self.loss_fn(inputs, targets)
