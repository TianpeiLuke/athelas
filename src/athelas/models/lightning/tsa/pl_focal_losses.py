"""
Focal Loss Implementations for Temporal Self-Attention Models

This module contains focal loss variants used in legacy TSA training.
All implementations preserve exact legacy behavior for algorithm-preserving refactoring.

Legacy Sources:
- projects/tsa/scripts/asl_focal_loss.py (AsymmetricLoss variants)
- projects/tsa/scripts/focalloss.py (Standard FocalLoss)

Design Principles:
- Zero behavioral changes from legacy implementations
- Type hints and comprehensive docstrings added
- Compatible with PyTorch Lightning training loop
- Supports all legacy loss configurations

Usage:
    from pl_focal_losses import create_loss_function

    loss_fn = create_loss_function(
        loss_type="Cyclical_FocalLoss",
        gamma_pos=0,
        gamma_neg=4,
        epochs=70
    )
"""

import warnings
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Asymmetric Loss (from legacy asl_focal_loss.py)
# ============================================================================


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Source: https://github.com/Alibaba-MIIL/ASL
    Legacy: projects/tsa/scripts/asl_focal_loss.py

    Applies different focal weights to positive and negative samples.
    Phase 1: EXACT copy of legacy implementation.

    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 1)
        clip: Asymmetric clipping value (default: 0.05)
        eps: Numerical stability epsilon (default: 1e-8)
        disable_torch_grad_focal_loss: Disable gradients during focal weight
            computation for memory efficiency (default: True)
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - EXACT legacy computation.

        Args:
            x: Input logits [batch_size, num_classes]
            y: Target multi-label binarized vector [batch_size, num_classes]

        Returns:
            Scalar loss value
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
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
    Memory-optimized version of AsymmetricLoss.

    Minimizes memory allocation and GPU uploading, favors inplace operations.
    Legacy: projects/tsa/scripts/asl_focal_loss.py

    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 1)
        clip: Asymmetric clipping value (default: 0.05)
        eps: Numerical stability epsilon (default: 1e-8)
        disable_torch_grad_focal_loss: Disable gradients during focal weight
            computation (default: False)
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

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
        Forward pass with inplace operations - EXACT legacy computation.

        Args:
            x: Input logits [batch_size, num_classes]
            y: Target multi-label binarized vector [batch_size, num_classes]

        Returns:
            Scalar loss value
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
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


# ============================================================================
# ASL Single Label (from legacy asl_focal_loss.py)
# ============================================================================


class ASLSingleLabel(nn.Module):
    """
    Asymmetric Loss for single-label classification.

    Legacy: projects/tsa/scripts/asl_focal_loss.py
    Phase 1: EXACT copy of legacy implementation.

    Args:
        gamma_pos: Focusing parameter for positive class (default: 0)
        gamma_neg: Focusing parameter for negative class (default: 4)
        eps: Label smoothing epsilon (default: 0.1)
        reduction: Loss reduction method (default: "mean")
    """

    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        eps: float = 0.1,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - EXACT legacy computation.

        Args:
            inputs: Input logits [batch_size, num_classes]
            target: Target class indices [batch_size] or one-hot [batch_size, num_classes]

        Returns:
            Scalar loss value
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # Convert target to class indices if one-hot
        if len(list(target.size())) > 1:
            target = torch.argmax(target, 1)

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


# ============================================================================
# Cyclical Focal Loss (from legacy asl_focal_loss.py)
# ============================================================================


class CyclicalFocalLoss(nn.Module):
    """
    Cyclical Focal Loss for single-label classification.

    Modulates focal loss weights cyclically based on training epoch.
    Legacy: projects/tsa/scripts/asl_focal_loss.py (Cyclical_FocalLoss)

    Phase 1: EXACT copy of legacy implementation.

    Args:
        gamma_pos: Focusing parameter for positive class (default: 0)
        gamma_neg: Focusing parameter for negative class (default: 4)
        gamma_hc: High confidence gamma (default: 0)
        eps: Label smoothing epsilon (default: 0.1)
        reduction: Loss reduction method (default: "mean")
        epochs: Total training epochs for cycle calculation (default: 200)
        factor: Cyclical factor (2 for cyclical, 1 for modified) (default: 2)

    Note:
        This loss requires the current epoch to be passed in forward().
        In Lightning: use self.current_epoch in training_step.
    """

    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        gamma_hc: float = 0,
        eps: float = 0.1,
        reduction: Literal["mean", "sum"] = "mean",
        epochs: int = 200,
        factor: float = 2,
    ):
        super(CyclicalFocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor  # factor=2 for cyclical, 1 for modified

    def forward(
        self, inputs: torch.Tensor, target: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        """
        Forward pass with epoch-based modulation - EXACT legacy computation.

        Args:
            inputs: Input logits [batch_size, num_classes]
            target: Target class indices [batch_size] or one-hot [batch_size, num_classes]
            epoch: Current training epoch (0-indexed)

        Returns:
            Scalar loss value
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # Convert target to class indices if one-hot
        if len(list(target.size())) > 1:
            target = torch.argmax(target, 1)

        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # Cyclical modulation factor (eta)
        if self.factor * epoch < self.epochs:
            eta = 1 - self.factor * epoch / (self.epochs - 1)
        elif self.factor == 1.0:
            eta = 0
        else:
            eta = (self.factor * epoch / (self.epochs - 1) - 1.0) / (self.factor - 1.0)

        # ASL weights with cyclical modulation
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
        positive_w = torch.pow(1 + xs_pos, self.gamma_hc * targets)
        log_preds = log_preds * ((1 - eta) * asymmetric_w + eta * positive_w)

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
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class ASLFocalLoss(nn.Module):
    """
    Standard ASL Focal Loss for single-label classification.

    Legacy: projects/tsa/scripts/asl_focal_loss.py (ASL_FocalLoss)
    Phase 1: EXACT copy of legacy implementation.

    Args:
        gamma: Focusing parameter (default: 2)
        eps: Label smoothing epsilon (default: 0.1)
        reduction: Loss reduction method (default: "mean")
    """

    def __init__(
        self,
        gamma: float = 2,
        eps: float = 0.1,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(ASLFocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - EXACT legacy computation.

        Args:
            inputs: Input logits [batch_size, num_classes]
            target: Target class indices [batch_size]

        Returns:
            Scalar loss value
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights (symmetric gamma)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg, self.gamma * targets + self.gamma * anti_targets
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

        return loss


# ============================================================================
# Standard Focal Loss (from legacy focalloss.py)
# ============================================================================


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: Optional[float] = None,
) -> torch.Tensor:
    """
    Functional interface for Focal Loss.

    Legacy: projects/tsa/scripts/focalloss.py

    According to Lin et al. (2018), the Focal loss is computed as:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        input: Input logits [batch_size, num_classes, ...]
        target: Target class indices [batch_size, ...]
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (default: 2.0)
        reduction: Loss reduction method: "none", "mean", or "sum" (default: "none")
        eps: Deprecated, kept for backward compatibility

    Returns:
        Computed focal loss
    """
    if eps is not None:
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(
            f"Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)})."
        )

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f"Expected target size {out_size}, got {target.size()}")

    if not input.device == target.device:
        raise ValueError(
            f"input and target must be in the same device. Got: {input.device} and {target.device}"
        )

    # Compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1)
    log_input_soft = F.log_softmax(input, dim=1)

    # Create one-hot target
    target_one_hot = F.one_hot(target, num_classes=input.shape[1])

    # Compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum("bc...,bc...->b...", (target_one_hot, focal))

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss


class FocalLoss(nn.Module):
    """
    Standard Focal Loss for classification.

    Legacy: projects/tsa/scripts/focalloss.py

    According to Lin et al. (2018), the Focal loss is computed as:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the class with label t.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (default: 2.0)
        reduction: Loss reduction method: "none", "mean", or "sum" (default: "none")
        eps: Deprecated, kept for backward compatibility

    Shape:
        - Input: [batch_size, num_classes, ...]
        - Target: [batch_size, ...]
        - Output: scalar if reduction != "none", else [batch_size, ...]
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = "none",
        eps: Optional[float] = None,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - EXACT legacy computation.

        Args:
            input: Input logits [batch_size, num_classes, ...]
            target: Target class indices [batch_size, ...]

        Returns:
            Computed focal loss
        """
        return focal_loss(
            input, target, self.alpha, self.gamma, self.reduction, self.eps
        )


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: Optional[float] = None,
) -> torch.Tensor:
    """
    Functional interface for Binary Focal Loss with logits.

    Legacy: projects/tsa/scripts/focalloss.py

    Args:
        input: Input logits [batch_size, ...]
        target: Target binary labels [batch_size, ...]
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Loss reduction method (default: "none")
        eps: Deprecated, kept for backward compatibility

    Returns:
        Computed binary focal loss
    """
    if eps is not None:
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not input.device == target.device:
        raise ValueError(
            f"input and target must be in the same device. Got: {input.device} and {target.device}"
        )

    # Compute probabilities
    probs = torch.sigmoid(input)

    # Compute focal loss
    weight = torch.pow(-probs + 1.0, gamma)
    focal_loss = -alpha * weight * F.logsigmoid(input)

    focal_loss_tmp = torch.einsum("b...,b...->b...", (target, focal_loss))

    if reduction == "none":
        loss = focal_loss_tmp
    elif reduction == "mean":
        loss = torch.mean(focal_loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(focal_loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with logits for binary classification.

    Legacy: projects/tsa/scripts/focalloss.py

    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Loss reduction method (default: "none")
        eps: Deprecated, kept for backward compatibility

    Shape:
        - Input: [batch_size, ...]
        - Target: [batch_size, ...]
        - Output: scalar if reduction != "none", else [batch_size, ...]
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
        eps: Optional[float] = None,
    ):
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - EXACT legacy computation.

        Args:
            input: Input logits [batch_size, ...]
            target: Target binary labels [batch_size, ...]

        Returns:
            Computed binary focal loss
        """
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction
        )


# ============================================================================
# Factory Function
# ============================================================================


def create_loss_function(loss_type: str = "CrossEntropyLoss", **kwargs) -> nn.Module:
    """
    Factory function to create loss functions matching legacy behavior.

    Args:
        loss_type: Type of loss function to create. Options:
            - "CrossEntropyLoss": Standard PyTorch CE loss
            - "BCEWithLogitsLoss": Binary cross-entropy with logits
            - "FocalLoss": Standard focal loss
            - "BinaryFocalLossWithLogits": Binary focal loss
            - "AsymmetricLoss": Asymmetric loss for multi-label
            - "AsymmetricLossOptimized": Memory-optimized asymmetric loss
            - "ASLSingleLabel": ASL for single-label
            - "CyclicalFocalLoss": Cyclical focal loss (requires epoch in forward)
            - "ASLFocalLoss": ASL focal loss variant
        **kwargs: Loss-specific parameters

    Returns:
        Initialized loss function

    Example:
        >>> loss_fn = create_loss_function(
        ...     loss_type="CyclicalFocalLoss",
        ...     gamma_pos=0,
        ...     gamma_neg=4,
        ...     epochs=70
        ... )
    """
    if loss_type == "CrossEntropyLoss":
        reduction = kwargs.get("reduction", "sum")
        return nn.CrossEntropyLoss(reduction=reduction)

    elif loss_type == "BCEWithLogitsLoss":
        reduction = kwargs.get("reduction", "sum")
        return nn.BCEWithLogitsLoss(reduction=reduction)

    elif loss_type == "FocalLoss":
        return FocalLoss(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            reduction=kwargs.get("reduction", "none"),
        )

    elif loss_type == "BinaryFocalLossWithLogits":
        return BinaryFocalLossWithLogits(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            reduction=kwargs.get("reduction", "none"),
        )

    elif loss_type == "AsymmetricLoss":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("gamma_neg", 4),
            gamma_pos=kwargs.get("gamma_pos", 1),
            clip=kwargs.get("clip", 0.05),
            eps=kwargs.get("eps", 1e-8),
            disable_torch_grad_focal_loss=kwargs.get(
                "disable_torch_grad_focal_loss", True
            ),
        )

    elif loss_type == "AsymmetricLossOptimized":
        return AsymmetricLossOptimized(
            gamma_neg=kwargs.get("gamma_neg", 4),
            gamma_pos=kwargs.get("gamma_pos", 1),
            clip=kwargs.get("clip", 0.05),
            eps=kwargs.get("eps", 1e-8),
            disable_torch_grad_focal_loss=kwargs.get(
                "disable_torch_grad_focal_loss", False
            ),
        )

    elif loss_type == "ASLSingleLabel":
        return ASLSingleLabel(
            gamma_pos=kwargs.get("gamma_pos", 0),
            gamma_neg=kwargs.get("gamma_neg", 4),
            eps=kwargs.get("eps", 0.1),
            reduction=kwargs.get("reduction", "mean"),
        )

    elif loss_type == "CyclicalFocalLoss":
        return CyclicalFocalLoss(
            gamma_pos=kwargs.get("gamma_pos", 0),
            gamma_neg=kwargs.get("gamma_neg", 4),
            gamma_hc=kwargs.get("gamma_hc", 0),
            eps=kwargs.get("eps", 0.1),
            reduction=kwargs.get("reduction", "mean"),
            epochs=kwargs.get("epochs", 200),
            factor=kwargs.get("factor", 2),
        )

    elif loss_type == "ASLFocalLoss":
        return ASLFocalLoss(
            gamma=kwargs.get("gamma", 2),
            eps=kwargs.get("eps", 0.1),
            reduction=kwargs.get("reduction", "mean"),
        )

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Valid options: CrossEntropyLoss, BCEWithLogitsLoss, FocalLoss, "
            f"BinaryFocalLossWithLogits, AsymmetricLoss, AsymmetricLossOptimized, "
            f"ASLSingleLabel, CyclicalFocalLoss, ASLFocalLoss"
        )
