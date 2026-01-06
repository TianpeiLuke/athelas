"""
Transformer Block

Complete transformer layer combining self-attention and feedforward network.

**Core Concept:**
Standard transformer encoder layer with pre-norm architecture. Processes
sequential data by allowing tokens to attend to each other, then applying
position-wise feedforward transformation.

**Architecture (Pre-Norm):**
1. x + Self-Attention(LayerNorm(x))
2. x + FFN(LayerNorm(x))

**Parameters:**
- embedding_dim (int): Model dimension
- n_heads (int): Number of attention heads
- ff_hidden_dim (int): Feedforward hidden dimension (typically 4x embedding_dim)
- dropout (float): Dropout probability

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional)

Output:
  - output: (B, L, D)

**Dependencies:**
- names3risk_pytorch.pytorch.attention.multihead_attention → Self-attention
- names3risk_pytorch.pytorch.feedforward.mlp_block → Feedforward network

**Used By:**
- names3risk_pytorch.pytorch.blocks.transformer_encoder → Stacked transformer layers

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import TransformerBlock

block = TransformerBlock(
    embedding_dim=128,
    n_heads=8,
    ff_hidden_dim=512,
    dropout=0.2
)

x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()

output = block(x, attn_mask)  # (32, 50, 128)
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..attention import MultiHeadAttention
from ..feedforward import MLPBlock


class TransformerBlock(nn.Module):
    """Complete transformer encoder layer with pre-norm architecture."""

    def __init__(
        self, embedding_dim: int, n_heads: int, ff_hidden_dim: int, dropout: float = 0.0
    ):
        """
        Initialize TransformerBlock.

        Args:
            embedding_dim: Model dimension
            n_heads: Number of attention heads
            ff_hidden_dim: Feedforward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Self-attention with pre-norm
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.self_attn = MultiHeadAttention(embedding_dim, n_heads, dropout)

        # Feedforward with pre-norm
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = MLPBlock(embedding_dim, ff_hidden_dim, dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask (optional)

        Returns:
            output: (B, L, D)
        """
        # Self-attention with residual
        x = x + self.self_attn(self.ln1(x), attn_mask)

        # Feedforward with residual
        x = x + self.ffn(self.ln2(x))

        return x
