"""
Label Smoothing Cross Entropy Loss

Label smoothing prevents overconfident predictions and improves generalization by smoothing
the hard target distribution. Instead of 1 for correct class and 0 for others, it uses
1-ε for correct class and ε/(n_classes-1) for others.

**Core Concept:**
- Softens hard target labels to prevent overconfidence
- Improves model calibration and generalization
- Particularly useful when labels might be noisy

**Parameters:**
- num_classes (int): Number of classes
- smoothing (float): Label smoothing factor, 0.0 = no smoothing (default: 0.1)
- reduction (str): 'mean', 'sum', or 'none' (default: 'mean')

**Forward Signature:**
Input:
  - inputs: (B, n_classes) - Model logits
  - targets: (B,) - Ground truth labels (class indices)

Output:
  - loss: scalar or (B,) - Loss value

**Dependencies:**
- torch.nn.functional → Log softmax
- torch → Tensor operations

**Used By:**
- Lightning modules → Classification training
- Any classification task → Improved generalization

**Mathematical Formulation:**
Target distribution:
- y_k = 1 - ε for correct class k
- y_i = ε / (K - 1) for incorrect classes i ≠ k

Loss:
L = - Σ y_i * log(p_i)

where:
- ε is the smoothing factor
- K is the number of classes
- p_i is the predicted probability for class i

**Typical Usage:**
- Small smoothing (0.05-0.1) for standard classification
- Larger smoothing (0.1-0.2) for noisy labels or when overfitting

**Usage Example:**
```python
from pytorch.losses import LabelSmoothingCrossEntropyLoss

# For fraud detection with potential label noise
loss_fn = LabelSmoothingCrossEntropyLoss(
    num_classes=2,
    smoothing=0.1
)

logits = model(batch)
loss = loss_fn(logits, labels)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    Label smoothing can help with overconfident predictions and improve
    generalization, which can be beneficial for fraud detection models
    and other classification tasks.
    """

    def __init__(
        self, num_classes: int, smoothing: float = 0.1, reduction: str = "mean"
    ):
        """
        Initialize Label Smoothing Cross Entropy Loss.

        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor (0.0 = no smoothing)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]

        Returns:
            Label smoothing cross entropy loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)

        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -torch.sum(true_dist * log_probs, dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
