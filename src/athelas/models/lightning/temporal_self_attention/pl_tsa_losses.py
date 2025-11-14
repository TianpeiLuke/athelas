#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fraud detection.
    
    Features:
    - Automatic down-weighting of easy examples
    - Configurable focusing parameter (gamma)
    - Class balancing with alpha parameter
    - Supports both binary and multiclass classification
    
    Reference: "Focal Loss for Dense Object Detection" - Lin et al.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
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
    Cyclical Focal Loss with dynamic gamma adjustment during training.
    
    Features:
    - Dynamic focusing parameter based on training progress
    - Cyclical adjustment for improved convergence
    - Automatic class balancing
    - Fraud detection optimization
    
    The gamma parameter cycles between gamma_min and gamma_max using a cosine schedule,
    allowing the model to focus on hard examples more or less during different phases
    of training.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma_min: float = 1.0,
        gamma_max: float = 3.0,
        cycle_length: int = 1000,
        reduction: str = "mean"
    ):
        """
        Initialize Cyclical Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma_min: Minimum gamma value in the cycle
            gamma_max: Maximum gamma value in the cycle
            cycle_length: Number of steps for one complete cycle
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
        Compute cyclical focal loss with dynamic gamma.
        
        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Cyclical focal loss value
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


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    
    This is a wrapper around PyTorch's CrossEntropyLoss that provides
    easy configuration of class weights for fraud detection scenarios.
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction: str = "mean"):
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


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing can help with overconfident predictions and improve
    generalization, which can be beneficial for fraud detection models.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = "mean"):
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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for handling extreme class imbalance.
    
    This loss function applies different focusing mechanisms to positive
    and negative samples, which can be particularly useful for fraud detection
    where the positive class (fraud) is much rarer than the negative class.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" - Ridnik et al.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean"
    ):
        """
        Initialize Asymmetric Loss.
        
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping value
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            inputs: Model logits [B, n_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Asymmetric loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:
            targets_one_hot = targets.float()
        
        # Probability clipping
        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, self.clip, 1 - self.clip)
        
        # Asymmetric focusing
        pt_pos = probs * targets_one_hot
        pt_neg = (1 - probs) * (1 - targets_one_hot)
        
        # Asymmetric loss computation
        loss_pos = -targets_one_hot * torch.log(probs) * (1 - pt_pos) ** self.gamma_pos
        loss_neg = -(1 - targets_one_hot) * torch.log(1 - probs) * pt_neg ** self.gamma_neg
        
        loss = loss_pos + loss_neg
        loss = loss.sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_config: Dictionary containing loss configuration
        
    Returns:
        Configured loss function
        
    Example:
        loss_config = {
            "type": "FocalLoss",
            "alpha": 0.25,
            "gamma": 2.0,
            "reduction": "mean"
        }
    """
    loss_type = loss_config.get("type", "CrossEntropyLoss")
    
    if loss_type == "FocalLoss":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
            reduction=loss_config.get("reduction", "mean")
        )
    
    elif loss_type == "CyclicalFocalLoss":
        return CyclicalFocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma_min=loss_config.get("gamma_min", 1.0),
            gamma_max=loss_config.get("gamma_max", 3.0),
            cycle_length=loss_config.get("cycle_length", 1000),
            reduction=loss_config.get("reduction", "mean")
        )
    
    elif loss_type == "WeightedCrossEntropyLoss":
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            reduction=loss_config.get("reduction", "mean")
        )
    
    elif loss_type == "LabelSmoothingCrossEntropyLoss":
        return LabelSmoothingCrossEntropyLoss(
            num_classes=loss_config.get("num_classes", 2),
            smoothing=loss_config.get("smoothing", 0.1),
            reduction=loss_config.get("reduction", "mean")
        )
    
    elif loss_type == "AsymmetricLoss":
        return AsymmetricLoss(
            gamma_neg=loss_config.get("gamma_neg", 4.0),
            gamma_pos=loss_config.get("gamma_pos", 1.0),
            clip=loss_config.get("clip", 0.05),
            reduction=loss_config.get("reduction", "mean")
        )
    
    else:
        # Default to standard CrossEntropyLoss
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            return nn.CrossEntropyLoss(
                weight=class_weights,
                reduction=loss_config.get("reduction", "mean")
            )
        else:
            return nn.CrossEntropyLoss(reduction=loss_config.get("reduction", "mean"))
