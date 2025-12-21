#!/usr/bin/env python3
"""
PyTorch Lightning Mixture of Experts Components

MoE feedforward layers with Top-2 gating for TSA models.

Phase 1: Algorithm-Preserving Refactoring
- Direct recreation of legacy components
- NO optimizations or modifications
- EXACT numerical behavior preservation

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/mixture_of_experts.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor


class MoE(nn.Module):
    """
    Mixture of Experts feedforward layer.

    EXACT recreation of legacy MoE from mixture_of_experts.py.
    Phase 1: No modifications - preserve exact behavior.

    Uses Top-2 gating to route inputs to multiple expert networks.
    Each input position is routed to its top 2 experts based on learned
    gating scores, enabling conditional computation and model capacity.

    Args:
        dim: Input/output dimension
        num_experts: Number of expert networks (default: 16)
        hidden_dim: Hidden dimension for experts (default: None, uses dim * 4)
        activation: Activation function (default: nn.ReLU)
        second_policy_train: Policy for second expert during training
            - "random": Probabilistic routing based on gate score
            - "all": Always use both experts
            - "none": Only use top expert
            - "threshold": Use if gate score > threshold
        second_policy_eval: Policy for second expert during eval
        second_threshold_train: Threshold for second expert training
        second_threshold_eval: Threshold for second expert eval
        capacity_factor_train: Capacity factor during training (default: 1.25)
        capacity_factor_eval: Capacity factor during eval (default: 2.0)
        loss_coef: Loss coefficient (not used in forward, kept for compatibility)

    Forward:
        Input: [B, N, D]
        Output: [B, N, D]

    Example:
        >>> moe = MoE(dim=256, num_experts=5, hidden_dim=64)
        >>> x = torch.randn(32, 50, 256)  # [B, N, D]
        >>> output = moe(x)  # [32, 50, 256]
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 16,
        hidden_dim: Optional[int] = None,
        activation=nn.ReLU,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
        second_threshold_train: float = 0.2,
        second_threshold_eval: float = 0.2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        loss_coef: float = 1e-2,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.loss_coef = loss_coef

        # Gating network
        self.gate = Top2Gating(
            dim,
            num_gates=num_experts,
            second_policy_train=second_policy_train,
            second_policy_eval=second_policy_eval,
            second_threshold_train=second_threshold_train,
            second_threshold_eval=second_threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
        )

        # Expert networks
        hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.experts = Experts(
            dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through mixture of experts.

        Args:
            inputs: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        b, n, d = inputs.shape
        e = self.num_experts

        # Route inputs to experts via gating
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum("bnd,bnec->ebcd", inputs, dispatch_tensor)

        # Feed through experts
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # Combine expert outputs
        output = torch.einsum("ebcd,bnec->bnd", expert_outputs, combine_tensor)
        return output


class Experts(nn.Module):
    """
    Expert networks for MoE.

    Each expert is a two-layer feedforward network with learned weights.
    All experts share the same architecture but have independent parameters.

    Args:
        dim: Input/output dimension
        num_experts: Number of experts
        hidden_dim: Hidden dimension (default: None, uses dim * 4)
        activation: Activation function (default: nn.GELU)

    Forward:
        Input: [E, *, D] where E is number of experts
        Output: [E, *, D]
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 16,
        hidden_dim: Optional[int] = None,
        activation=nn.GELU,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else dim * 4

        # Initialize expert weights
        # Shape: [num_experts, dim, hidden_dim] and [num_experts, hidden_dim, dim]
        w1 = torch.zeros(num_experts, dim, hidden_dim)
        w2 = torch.zeros(num_experts, hidden_dim, dim)

        # Initialize with uniform distribution
        std = 1 / math.sqrt(dim)
        w1 = w1.uniform_(-std, std)
        std = 1 / math.sqrt(hidden_dim)
        w2 = w2.uniform_(-std, std)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through expert networks.

        Uses einsum for efficient batched matrix multiplication across experts.

        Args:
            x: Input [E, *, D]

        Returns:
            Output [E, *, D]
        """
        hidden = torch.einsum("...nd,...dh->...nh", x, self.w1)
        hidden = self.act(hidden)
        out = torch.einsum("...nh,...hd->...nd", hidden, self.w2)
        return out


class Top2Gating(nn.Module):
    """
    Top-2 gating mechanism for MoE.

    Routes each input to its top 2 experts based on learned gating scores.
    Includes load balancing loss to encourage equal expert utilization.

    Args:
        dim: Input dimension
        num_gates: Number of gates (experts)
        eps: Small epsilon for numerical stability
        outer_expert_dims: Outer expert dimensions (for hierarchical MoE)
        second_policy_train: Policy for second expert during training
        second_policy_eval: Policy for second expert during eval
        second_threshold_train: Threshold for second expert training
        second_threshold_eval: Threshold for second expert eval
        capacity_factor_train: Capacity factor during training
        capacity_factor_eval: Capacity factor during eval

    Forward:
        Input: [B, N, D]
        Returns: (dispatch_tensor, combine_tensor, loss)
            - dispatch_tensor: [B, N, E, C] routing mask
            - combine_tensor: [B, N, E, C] combination weights
            - loss: Load balancing loss
    """

    def __init__(
        self,
        dim: int,
        num_gates: int,
        eps: float = 1e-9,
        outer_expert_dims: tuple = tuple(),
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
        second_threshold_train: float = 0.2,
        second_threshold_eval: float = 0.2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(
        self, x: Tensor, importance: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gating for experts.

        Args:
            x: Input [B, N, D]
            importance: Importance weights (optional, for hierarchical MoE)

        Returns:
            Tuple of (dispatch_tensor, combine_tensor, loss)
        """
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        # Select policy and parameters based on training mode
        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        # Compute gate scores
        raw_gates = torch.einsum("...bnd,...de->...bne", x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # Find top expert
        gate_1, index_1 = _top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.0).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]

        # Find second expert
        gates_without_top_1 = raw_gates * (1.0 - mask_1)
        gate_2, index_2 = _top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.0).float()
            mask_2 *= greater_zero_mask[..., None]

        # Normalize gates
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # Balancing loss
        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates**2)

        # Apply second expert policy
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0.0, 1.0)
            mask_2 *= (
                (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
            )
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Compute capacity
        expert_capacity = min(
            group_size, int((group_size * capacity_factor) / num_gates)
        )
        expert_capacity = max(expert_capacity, 4)  # MIN_EXPERT_CAPACITY
        expert_capacity_f = float(expert_capacity)

        # Assign to experts
        position_in_expert_1 = _cumsum_exclusive(mask_1, dim=-2) * mask_1
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 *= mask_1_flat

        position_in_expert_2 = _cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        # Create combine tensor
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * _safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :]
            + gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * _safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss


# ============================================================================
# Helper Functions
# ============================================================================


def _top1(t: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get top-1 values and indices.

    Args:
        t: Input tensor [*, N]

    Returns:
        Tuple of (values, indices) with shape [*]
    """
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def _cumsum_exclusive(t: Tensor, dim: int = -1) -> Tensor:
    """
    Cumulative sum exclusive (starts from 0).

    Like cumsum but shifts result by 1, starting from 0 instead of first element.

    Args:
        t: Input tensor
        dim: Dimension to cumsum over

    Returns:
        Exclusive cumsum tensor

    Example:
        >>> t = torch.tensor([1, 2, 3, 4])
        >>> _cumsum_exclusive(t)
        tensor([0, 1, 3, 6])
    """
    num_dims = len(t.shape)
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


def _safe_one_hot(indexes: Tensor, max_length: int) -> Tensor:
    """
    Safe one-hot encoding that handles out-of-bound indices.

    Unlike F.one_hot, this doesn't throw error for out-of-bound indices.
    Truncates to max_length if indices exceed it.

    Args:
        indexes: Index tensor
        max_length: Maximum length for one-hot encoding

    Returns:
        One-hot encoded tensor
    """
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]
