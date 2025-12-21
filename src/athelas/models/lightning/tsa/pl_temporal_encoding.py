#!/usr/bin/env python3
"""
PyTorch Lightning Temporal Encoding Components

Time encoding utilities for Temporal Self-Attention models.

Phase 1: Algorithm-Preserving Refactoring
- Direct recreation of legacy components
- NO optimizations or modifications
- EXACT numerical behavior preservation

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/basic_blocks.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class TimeEncode(nn.Module):
    """
    Learnable periodic time encoding for temporal sequences.

    EXACT recreation of legacy TimeEncode from basic_blocks.py.
    Phase 1: No modifications - preserve exact behavior.

    Architecture:
        - Learnable weight matrix: [time_dim, 1]
        - Learnable bias vector: [time_dim]
        - First dimension: Linear encoding
        - Remaining dimensions: Sinusoidal encoding

    Args:
        time_dim: Dimension of time encoding (output size)
        device: Device to place parameters on
        dtype: Data type for parameters

    Forward:
        Input: [B, L, 1] time values
        Output: [L, B, time_dim] time encodings

    Example:
        >>> time_encoder = TimeEncode(time_dim=128)
        >>> time_values = torch.randn(32, 50, 1)  # [B, L, 1]
        >>> time_enc = time_encoder(time_values)  # [L, B, time_dim] = [50, 32, 128]
    """

    def __init__(self, time_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TimeEncode, self).__init__()

        self.time_dim = time_dim

        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Encode time values with learnable periodic functions.

        Args:
            tt: Time values [B, L, 1]

        Returns:
            Time encodings [L, B, time_dim]
        """
        tt = tt.unsqueeze(-1)  # [B, L, 1, 1]

        # Sinusoidal encoding for dimensions 1 onwards
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))

        # Linear encoding for first dimension
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])

        # Concatenate
        t = torch.cat([out1, out2], -1)  # [B, L, 1, time_dim]
        t = t.squeeze(2)  # [B, L, time_dim]
        t = t.permute(1, 0, 2)  # [L, B, time_dim]

        return t
