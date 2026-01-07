"""
Focal Loss Functions

Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard examples.
Particularly useful for fraud detection, risk assessment, and other imbalanced classification tasks.

**Core Concept:**
- Automatic down-weighting of easy examples
- Configurable focusing parameter (gamma)
- Class balancing with alpha parameter

**Classes:**
- FocalLoss: Standard focal loss implementation
- CyclicalFocalLoss: Focal loss with dynamic gamma adjustment

**Parameters:**
- alpha (float): Weighting factor for rare class (default: 0.25)
- gamma (float): Focusing parameter (default: 2.0)
- reduction (str): 'mean', 'sum', or 'none' (default: 'mean')

**Forward Signature:**
Input:
  - inputs: (B, n_classes) - Model logits
  - targets: (B,) - Ground truth labels (class indices)

Output:
  - loss: scalar or (B,) - Loss value

**Dependencies:**
- torch.nn.functional → Cross entropy computation
- torch → Tensor operations

**Used By:**
- Lightning modules → Classification training
- Any project with class imbalance → Fraud detection, risk models

**Mathematical Formulation:**
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
- p_t is the model's estimated probability for the true class
- α_t is the weighting factor for class t
- γ is the focusing parameter (higher = more focus on hard examples)

**Reference:**
"Focal Loss for Dense Object Detection" - Lin et al., ICCV 2017

**Usage Example:**
```python
from pytorch.losses import FocalLoss, CyclicalFocalLoss

# Standard focal loss
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
logits = model(batch)
loss = loss_fn(logits, labels)

# Cyclical focal loss (gamma adjusts during training)
loss_fn = CyclicalFocalLoss(
    alpha=0.25,
    gamma_min=1.0,
    gamma_max=3.0,
    cycle_length=1000
)
```
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.

    Features:
    - Automatic down-weighting of easy examples
    - Configurable focusing parameter (gamma)
    - Class balancing with alpha parameter
    - Supports both binary and multiclass classification
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]

        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CyclicalFocalLoss(nn.Module):
    """
    Cyclical Focal Loss from legacy TSA implementation.

    This loss is intended for single-label classification problems with cyclical
    adjustment during training. It interpolates between asymmetric focal weights
    and positive weights based on epoch progress.

    Features:
    - Epoch-based cyclical adjustment
    - Asymmetric focusing for hard examples
    - Label smoothing support
    - Optimized for imbalanced classification

    The loss cycles between two weighting strategies:
    - Early training: Focus on hard negatives (asymmetric weights)
    - Late training: Focus on hard positives (positive weights)

    Reference: Legacy TSA implementation (asl_focal_loss.py)
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        gamma_hc: float = 0.0,
        eps: float = 0.1,
        reduction: str = "mean",
        epochs: int = 200,
        factor: float = 2.0,
    ):
        """
        Initialize Cyclical Focal Loss.

        Args:
            gamma_pos: Focusing parameter for positive samples (default: 0.0)
            gamma_neg: Focusing parameter for negative samples (default: 4.0)
            gamma_hc: Focusing parameter for hard positives (default: 0.0)
            eps: Label smoothing factor (default: 0.1)
            reduction: Specifies the reduction to apply to the output
            epochs: Total number of training epochs (default: 200)
            factor: Cyclical factor (2 for cyclical, 1 for modified) (default: 2.0)
        """
        super().__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        """
        Compute cyclical focal loss.

        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B] or [B, n_classes] (one-hot)
            epoch: Current training epoch (0-indexed)

        Returns:
            Cyclical focal loss value
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # Convert targets to class indices if one-hot encoded
        if len(list(targets.size())) > 1:
            targets = torch.argmax(targets, 1)

        # Create one-hot targets
        targets_classes = torch.zeros_like(inputs).scatter_(
            1, targets.long().unsqueeze(1), 1
        )

        # Compute cyclical eta parameter
        if self.factor * epoch < self.epochs:
            eta = 1 - self.factor * epoch / (self.epochs - 1)
        elif self.factor == 1.0:
            eta = 0
        else:
            eta = (self.factor * epoch / (self.epochs - 1) - 1.0) / (self.factor - 1.0)

        # Compute asymmetric and positive weights
        targets_binary = targets_classes
        anti_targets = 1 - targets_binary
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets_binary
        xs_neg = xs_neg * anti_targets

        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets_binary + self.gamma_neg * anti_targets,
        )
        positive_w = torch.pow(1 + xs_pos, self.gamma_hc * targets_binary)

        # Combine weights based on eta
        log_preds = log_preds * ((1 - eta) * asymmetric_w + eta * positive_w)

        # Apply label smoothing
        if self.eps > 0:
            targets_classes = targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # Compute loss
        loss = -targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CosineFocalLoss(nn.Module):
    """
    Focal Loss with Cosine-Scheduled Gamma.

    This is an alternative to CyclicalFocalLoss that uses step-based cosine
    scheduling for the gamma parameter. Unlike CyclicalFocalLoss which uses
    epoch-based interpolation between asymmetric and positive weights, this
    version applies a simple cosine schedule to the gamma focusing parameter.

    Features:
    - Step-based cosine schedule for gamma parameter
    - Automatic step counting during training
    - Simpler implementation than CyclicalFocalLoss
    - No epoch parameter needed in forward pass

    The gamma parameter cycles between gamma_min and gamma_max using a cosine schedule:
    gamma(t) = gamma_min + (gamma_max - gamma_min) * 0.5 * (1 + cos(π * t / cycle_length))

    This allows the model to focus on hard examples more or less during different
    phases of training, with smooth transitions.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma_min: float = 1.0,
        gamma_max: float = 3.0,
        cycle_length: int = 1000,
        reduction: str = "mean",
    ):
        """
        Initialize Cosine Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma_min: Minimum gamma value in the cycle (default: 1.0)
            gamma_max: Maximum gamma value in the cycle (default: 3.0)
            cycle_length: Number of steps for one complete cycle (default: 1000)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.cycle_length = cycle_length
        self.reduction = reduction
        self.step_count = 0

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine-scheduled focal loss.

        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]

        Returns:
            Focal loss value with cosine-scheduled gamma
        """
        # Compute dynamic gamma using cosine schedule
        cycle_position = (self.step_count % self.cycle_length) / self.cycle_length
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (
            0.5 * (1 + math.cos(math.pi * cycle_position))
        )

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute focal loss with dynamic gamma
        focal_loss = self.alpha * (1 - pt) ** gamma * ce_loss

        # Increment step count
        self.step_count += 1

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
