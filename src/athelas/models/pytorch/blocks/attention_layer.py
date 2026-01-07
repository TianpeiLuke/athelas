"""
Attention Layer Blocks

Multi-head attention layers with temporal encoding and MoE feedforward.

**Core Concept:**
Composite attention blocks that combine temporal multi-head attention with
feedforward processing (MoE). Provides both post-normalization (AttentionLayer)
and pre-normalization (AttentionLayerPreNorm) variants for TSA models.

**Architecture:**
- Temporal multi-head attention with time encoding
- Layer normalization
- MoE-based feedforward network
- Residual connections
- Dropout regularization

**Components Used (Phase 1 Atomics):**
- TimeEncode: Temporal position encoding
- TemporalMultiheadAttention: Time-aware attention
- MixtureOfExperts: Feedforward processing with expert routing

**Parameters:**
- dim (int): Model dimension
- time_dim (int): Temporal encoding dimension
- num_heads (int): Number of attention heads
- dim_feedforward (int): Feedforward hidden dimension
- use_moe (bool): Whether to use MoE for feedforward
- num_experts (int): Number of experts if using MoE
- dropout (float): Dropout probability
- second_policy_train/eval (str): MoE routing policy

**Forward Signature:**
Input:
  - x: [L, B, E] - Input sequence
  - time_seq: [L, B, 1] - Temporal information
  - attn_mask: Optional attention mask
  - key_padding_mask: Optional padding mask

Output:
  - output: [L, B, E] - Processed sequence

**Used By:**
- temporal_self_attention_pytorch.pytorch.blocks.order_attention → OrderAttentionModule
- temporal_self_attention_pytorch.pytorch.blocks.feature_attention → FeatureAttentionModule

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.blocks import AttentionLayer, AttentionLayerPreNorm

# Post-norm variant
attn_layer = AttentionLayer(
    dim=128, time_dim=128, num_heads=4,
    dim_feedforward=512, use_moe=True, num_experts=5
)

# Pre-norm variant
attn_layer_pre = AttentionLayerPreNorm(
    dim=128, time_dim=128, num_heads=4,
    dim_feedforward=512, use_moe=True, num_experts=5
)

# Forward pass
x = torch.randn(50, 32, 128)  # [L, B, E]
time_seq = torch.randn(50, 32, 1)
output = attn_layer(x, time_seq)  # [50, 32, 128]
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) - Pre/post-norm
"""

import torch
import torch.nn as nn
from typing import Optional

# Import Phase 1 atomic components
from ..embeddings import TimeEncode
from ..attention import TemporalMultiheadAttention
from ..feedforward import MixtureOfExperts


class AttentionLayer(nn.Module):
    """
    Post-normalization attention layer with temporal encoding and MoE.

    Architecture: Attention → Add → Norm → FFN → Add → Norm
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        num_heads: int,
        dim_feedforward: int,
        use_moe: bool = False,
        num_experts: int = 3,
        dropout: float = 0.1,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
    ):
        """
        Initialize AttentionLayer (post-norm).

        Args:
            dim: Model dimension
            time_dim: Temporal encoding dimension
            num_heads: Number of attention heads
            dim_feedforward: Feedforward hidden dimension
            use_moe: Whether to use MoE for feedforward
            num_experts: Number of experts if using MoE
            dropout: Dropout probability
            second_policy_train: MoE training policy
            second_policy_eval: MoE evaluation policy
        """
        super().__init__()

        self.dim = dim
        self.use_moe = use_moe

        # Temporal encoding (Phase 1 atomic)
        self.time_encoder = TimeEncode(time_dim)

        # Temporal multi-head attention (Phase 1 atomic)
        self.self_attn = TemporalMultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )

        # Feedforward network (Phase 1 atomic - MoE or standard)
        if use_moe:
            self.feedforward = MixtureOfExperts(
                dim=dim,
                num_experts=num_experts,
                hidden_dim=dim_feedforward,
                second_policy_train=second_policy_train,
                second_policy_eval=second_policy_eval,
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim),
                nn.Dropout(dropout),
            )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with post-normalization.

        Args:
            x: Input tensor [L, B, E]
            time_seq: Time sequence [L, B, 1] or [B, L, 1]
            attn_mask: Attention mask (optional)
            key_padding_mask: Key padding mask (optional)

        Returns:
            output: [L, B, E] - Processed sequence
        """
        # Self-attention with temporal encoding
        attn_output, _ = self.self_attn(
            x, x, x, time_seq, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Add & Norm (post-norm pattern)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward
        ffn_output = self.feedforward(x)

        # Add & Norm (post-norm pattern)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class AttentionLayerPreNorm(nn.Module):
    """
    Pre-normalization attention layer with temporal encoding and MoE.

    Architecture: Norm → Attention → Add → Norm → FFN → Add
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        num_heads: int,
        dim_feedforward: int,
        use_moe: bool = False,
        num_experts: int = 3,
        dropout: float = 0.1,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
    ):
        """
        Initialize AttentionLayerPreNorm (pre-norm).

        Args:
            dim: Model dimension
            time_dim: Temporal encoding dimension
            num_heads: Number of attention heads
            dim_feedforward: Feedforward hidden dimension
            use_moe: Whether to use MoE for feedforward
            num_experts: Number of experts if using MoE
            dropout: Dropout probability
            second_policy_train: MoE training policy
            second_policy_eval: MoE evaluation policy
        """
        super().__init__()

        self.dim = dim
        self.use_moe = use_moe

        # Temporal encoding (Phase 1 atomic)
        self.time_encoder = TimeEncode(time_dim)

        # Temporal multi-head attention (Phase 1 atomic)
        self.self_attn = TemporalMultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )

        # Feedforward network (Phase 1 atomic - MoE or standard)
        if use_moe:
            self.feedforward = MixtureOfExperts(
                dim=dim,
                num_experts=num_experts,
                hidden_dim=dim_feedforward,
                second_policy_train=second_policy_train,
                second_policy_eval=second_policy_eval,
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim),
                nn.Dropout(dropout),
            )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-normalization.

        Args:
            x: Input tensor [L, B, E]
            time_seq: Time sequence [L, B, 1] or [B, L, 1]
            attn_mask: Attention mask (optional)
            key_padding_mask: Key padding mask (optional)

        Returns:
            output: [L, B, E] - Processed sequence
        """
        # Norm → Self-attention (pre-norm pattern)
        normed = self.norm1(x)
        attn_output, _ = self.self_attn(
            normed,
            normed,
            normed,
            time_seq,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        # Add
        x = x + self.dropout(attn_output)

        # Norm → Feedforward (pre-norm pattern)
        normed = self.norm2(x)
        ffn_output = self.feedforward(normed)

        # Add
        x = x + self.dropout(ffn_output)

        return x
