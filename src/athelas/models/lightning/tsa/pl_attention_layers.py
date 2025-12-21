#!/usr/bin/env python3
"""
PyTorch Lightning Attention Layer Components

Attention mechanisms including temporal multi-head attention for TSA models.

Phase 1: Algorithm-Preserving Refactoring
- Direct recreation of legacy components
- NO optimizations or modifications
- EXACT numerical behavior preservation

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/basic_blocks.py, TemporalMultiheadAttentionDelta.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import warnings

from .pl_mixture_of_experts import MoE


class TemporalMultiheadAttention(nn.Module):
    """
    Multi-head attention with temporal encoding.

    EXACT recreation of legacy TemporalMultiheadAttention from TemporalMultiheadAttentionDelta.py.
    Phase 1: No modifications - preserve exact behavior.

    This adds temporal information to attention computation through time-based
    weighting (alpha_time) computed from time differences between positions.

    Key Innovation:
        Attention weights are multiplied by temporal decay factors based on
        time differences: alpha_time = exp((t_i - t_j) / 1000000)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Add bias as module parameter (default: True)
        add_bias_kv: Add bias to key and value sequences at dim=0
        add_zero_attn: Add new batch of zeros to key and value at dim=1
        kdim: Total number of features in key (default: None, uses embed_dim)
        vdim: Total number of features in value (default: None, uses embed_dim)

    Forward:
        Input: query [L, N, E], key [S, N, E], value [S, N, E], time [L, N, 1]
        Output: (output [L, N, E], attention_weights [N, L, S])

    Example:
        >>> attn = TemporalMultiheadAttention(embed_dim=256, num_heads=4)
        >>> query = torch.randn(50, 32, 256)  # [L, N, E]
        >>> time = torch.randn(50, 32, 1)     # [L, N, 1]
        >>> output, weights = attn(query, query, query, time)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        # Projection weights
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        time: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with temporal encoding.

        CRITICAL: Computes time differences and applies exponential temporal weighting.

        Args:
            query: Query tensor [L, N, E]
            key: Key tensor [S, N, E]
            value: Value tensor [S, N, E]
            time: Time sequence [L, N, 1] - relative time values
            key_padding_mask: Mask for padding [N, S]
            need_weights: Whether to return attention weights
            attn_mask: Attention mask [L, S]

        Returns:
            Tuple of (output, attention_weights)
            - output: [L, N, E]
            - attention_weights: [N, L, S] if need_weights else None
        """
        # CRITICAL: Compute temporal attention weights (alpha_time)
        # This is the key difference from standard multi-head attention
        tt = time.permute(1, 0, 2)  # [L, N, 1] -> [N, L, 1]

        # Compute time differences: tt - tt^T
        # Broadcasting: [N, L, 1] - [N, 1, L] = [N, L, L]
        alpha_time = tt - tt.view([tt.shape[0], tt.shape[2], tt.shape[1]])

        # Apply exponential decay based on time difference
        # Larger time differences get exponentially smaller weights
        alpha_time = torch.exp(alpha_time / 1000000)

        # Get dimensions
        tgt_len, bsz, embed_dim = query.size()

        assert embed_dim == self.embed_dim
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim
        scaling = float(head_dim) ** -0.5

        # Compute Q, K, V projections (standard attention computation)
        if not self._qkv_same_embed_dim:
            # Separate projection weights
            if self.in_proj_bias is not None:
                q = F.linear(query, self.q_proj_weight, self.in_proj_bias[0:embed_dim])
                k = F.linear(
                    key,
                    self.k_proj_weight,
                    self.in_proj_bias[embed_dim : (embed_dim * 2)],
                )
                v = F.linear(
                    value, self.v_proj_weight, self.in_proj_bias[(embed_dim * 2) :]
                )
            else:
                q = F.linear(query, self.q_proj_weight, self.in_proj_bias)
                k = F.linear(key, self.k_proj_weight, self.in_proj_bias)
                v = F.linear(value, self.v_proj_weight, self.in_proj_bias)
        else:
            # Combined projection weight
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                # Self-attention
                q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
                    3, dim=-1
                )
            elif key is value or torch.equal(key, value):
                # Encoder-decoder attention
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = F.linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:
                    _b = self.in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = self.in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
            else:
                # Different Q, K, V
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = F.linear(query, _w, _b)

                _b = self.in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = F.linear(key, _w, _b)

                _b = self.in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = F.linear(value, _w, _b)

        q = q * scaling

        # Handle attention mask
        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), (
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            )

            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # Handle key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        # Add bias to key and value
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Compute attention weights
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        # CRITICAL: Apply temporal weighting (alpha_time)
        # This is what makes it "temporal" attention
        alpha_time = (
            alpha_time.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, src_len)
        )
        attn_output_weights = alpha_time * attn_output_weights

        # Softmax and dropout
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # Compute output
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # Average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with optional temporal encoding and MoE.

    EXACT recreation of legacy AttentionLayer from basic_blocks.py.
    Phase 1: No modifications - preserve exact behavior.

    Architecture (post-norm):
        1. Multi-head attention (with optional temporal encoding)
        2. Add & norm
        3. Feedforward (optional MoE)
        4. Add & norm

    Args:
        dim_embed: Embedding dimension
        dim_attn_feedforward: Feedforward hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts (if MoE)
        use_time_seq: Whether to use temporal encoding

    Forward:
        Input: [L, B, E], time [L, B, time_dim]
        Output: [L, B, E]
    """

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
        use_time_seq: bool = True,
    ):
        super().__init__()

        # Parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.use_time_seq = use_time_seq

        # Multi-head attention (temporal or standard)
        if self.use_time_seq:
            self.multi_attn = TemporalMultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
        else:
            self.multi_attn = nn.MultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

        # Feedforward (MoE or standard)
        if use_moe:
            self.feedforward = MoE(
                dim=dim_embed,
                num_experts=num_experts,
                hidden_dim=dim_attn_feedforward,
                second_policy_train="random",
                second_policy_eval="random",
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with post-normalization.

        Args:
            x: Input [L, B, E]
            time: Time encoding [L, B, time_dim]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Padding mask [B, L]

        Returns:
            Output [L, B, E]
        """
        # Multi-head attention
        if self.use_time_seq:
            x2, _ = self.multi_attn(
                x, x, x, time, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        else:
            x2, _ = self.multi_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )

        # Add & norm
        x = x + self.dropout1(x2)
        x = self.layer_norm1(x)

        # Feedforward
        x2 = self.feedforward(x)

        # Add & norm
        x = x + self.dropout2(x2)
        x = self.layer_norm2(x)

        return x


class AttentionLayerPreNorm(nn.Module):
    """
    Multi-head attention layer with pre-normalization and optional MoE.

    EXACT recreation of legacy AttentionLayerPreNorm from basic_blocks.py.
    Phase 1: No modifications - preserve exact behavior.

    Architecture (pre-norm):
        1. Layer norm
        2. Multi-head attention
        3. Add (residual)
        4. Layer norm
        5. Feedforward (optional MoE)
        6. Add (residual)

    Args:
        dim_embed: Embedding dimension
        dim_attn_feedforward: Feedforward hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts (if MoE)

    Forward:
        Input: [L, B, E]
        Output: [L, B, E]
    """

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
    ):
        super().__init__()

        # Parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads

        # Multi-head attention (standard, no temporal)
        self.multi_attn = nn.MultiheadAttention(dim_embed, num_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

        # Feedforward (MoE or standard)
        if use_moe:
            self.feedforward = MoE(
                dim=dim_embed,
                num_experts=num_experts,
                hidden_dim=dim_attn_feedforward,
                second_policy_train="random",
                second_policy_eval="random",
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-normalization.

        Args:
            x: Input [L, B, E]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Padding mask [B, L]

        Returns:
            Output [L, B, E]
        """
        # Pre-norm
        x2 = self.layer_norm1(x)

        # Multi-head attention
        x2, _ = self.multi_attn(
            x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Add (residual)
        x = x + self.dropout1(x2)

        # Pre-norm
        x2 = self.layer_norm2(x)

        # Feedforward
        x2 = self.feedforward(x2)

        # Add (residual)
        x = x + self.dropout2(x2)

        return x
