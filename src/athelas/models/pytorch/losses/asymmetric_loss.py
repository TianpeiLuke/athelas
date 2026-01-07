"""
Asymmetric Loss Functions

Asymmetric Loss handles extreme class imbalance by applying different focusing mechanisms
to positive and negative samples. From legacy TSA implementation (asl_focal_loss.py).

**Core Concept:**
- Different focusing parameters for positive and negative samples
- Probability clipping to prevent numerical instability
- Optimized for extreme class imbalance scenarios

**Classes:**
- AsymmetricLoss: Multi-label asymmetric loss (sigmoid-based)
- AsymmetricLossOptimized: Memory-optimized version for multi-label
- ASLSingleLabel: Single-label classification version

**Parameters:**
- gamma_neg (float): Focusing parameter for negative samples (default: 4.0)
- gamma_pos (float): Focusing parameter for positive samples (default: 1.0)
- clip (float): Probability clipping value (default: 0.05)
- eps (float): Numerical stability epsilon (default: 1e-8)

**Forward Signature:**
Multi-label (AsymmetricLoss, AsymmetricLossOptimized):
  Input:
    - x: (B, n_classes) - Model logits
    - y: (B, n_classes) - Multi-label binarized targets
  Output:
    - loss: scalar - Negative sum loss

Single-label (ASLSingleLabel):
  Input:
    - inputs: (B, n_classes) - Model logits
    - target: (B,) or (B, n_classes) - Class indices or one-hot
  Output:
    - loss: scalar - Loss value with specified reduction

**Dependencies:**
- torch.nn → Sigmoid, LogSoftmax
- torch → Tensor operations

**Used By:**
- Lightning modules → Fraud detection, rare event prediction
- Any project with extreme class imbalance

**Mathematical Formulation:**
ASL = - Σ [y * log(p) * (1 - p)^γ_pos + (1 - y) * log(1 - p) * p^γ_neg]

where:
- y is the ground truth label (0 or 1)
- p is the predicted probability
- γ_pos focuses on hard positive examples
- γ_neg focuses on hard negative examples

**Reference:**
"Asymmetric Loss For Multi-Label Classification" - Ridnik et al., 2021
https://github.com/Alibaba-MIIL/ASL

**Usage Example:**
```python
from pytorch.losses import AsymmetricLoss, ASLSingleLabel

# Multi-label classification
loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
logits = model(batch)
loss = loss_fn(logits, multi_label_targets)

# Single-label classification
loss_fn = ASLSingleLabel(gamma_neg=4, gamma_pos=0, eps=0.1)
loss = loss_fn(logits, class_indices)
```
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Applies different focusing mechanisms to positive and negative samples.
    Uses sigmoid activation, making it suitable for multi-label tasks where
    each class is treated independently.

    Reference: Legacy TSA implementation (asl_focal_loss.py)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        """
        Initialize Asymmetric Loss.

        Args:
            gamma_neg: Focusing parameter for negative samples (default: 4.0)
            gamma_pos: Focusing parameter for positive samples (default: 1.0)
            clip: Probability clipping value (default: 0.05)
            eps: Numerical stability epsilon (default: 1e-8)
            disable_torch_grad_focal_loss: Disable gradients during focal weight
                computation for memory efficiency (default: True)
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss for multi-label classification.

        Args:
            x: Model logits [B, n_classes]
            y: Multi-label binarized targets [B, n_classes]

        Returns:
            Negative sum of asymmetric loss (scalar)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (adds to negative probabilities)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """
    Memory-optimized Asymmetric Loss for multi-label classification.

    Optimized version that minimizes memory allocation and GPU uploading,
    favoring inplace operations. Use this for large-scale multi-label tasks
    when memory is a constraint.

    Reference: Legacy TSA implementation (asl_focal_loss.py)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        """
        Initialize Asymmetric Loss Optimized.

        Args:
            gamma_neg: Focusing parameter for negative samples (default: 4.0)
            gamma_pos: Focusing parameter for positive samples (default: 1.0)
            clip: Probability clipping value (default: 0.05)
            eps: Numerical stability epsilon (default: 1e-8)
            disable_torch_grad_focal_loss: Disable gradients during focal weight
                computation (default: False for optimized version)
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # Prevent memory allocation every iteration, encourage inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss with memory optimization.

        Args:
            x: Model logits [B, n_classes]
            y: Multi-label binarized targets [B, n_classes]

        Returns:
            Negative sum of asymmetric loss (scalar)
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping (inplace operations)
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    """
    Asymmetric Loss for single-label classification.

    This version is intended for single-label classification problems where
    each sample belongs to exactly one class. Uses softmax instead of sigmoid.

    Reference: Legacy TSA implementation (asl_focal_loss.py)
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        eps: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Initialize ASL Single Label.

        Args:
            gamma_pos: Focusing parameter for positive samples (default: 0.0)
            gamma_neg: Focusing parameter for negative samples (default: 4.0)
            eps: Label smoothing factor (default: 0.1)
            reduction: Specifies the reduction to apply: 'mean' or 'sum' (default: 'mean')
        """
        super().__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss for single-label classification.

        Args:
            inputs: Model logits [B, n_classes]
            target: Ground truth labels [B] (class indices) or [B, n_classes] (one-hot)

        Returns:
            Asymmetric loss value (scalar with specified reduction)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # Convert one-hot to class indices if needed
        if len(list(target.size())) > 1:
            target = torch.argmax(target, 1)

        # Create one-hot targets
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets

        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        # Label smoothing
        if self.eps > 0:
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # Loss calculation
        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = torch.sum(loss)

        return loss
