"""
Multi-Head Self-Attention

Combines multiple attention heads with output projection.

**Core Concept:**
Allows model to jointly attend to information from different representation subspaces.
For Names3Risk, different heads can focus on different aspects: character patterns,
email structure, name length, special characters, etc.

**Architecture:**
1. Split input into n_heads attention heads
2. Each head computes scaled dot-product attention
3. Concatenate head outputs
4. Project concatenated output back to embedding dimension

**Parameters:**
- embedding_dim (int): Input embedding dimension
- n_heads (int): Number of attention heads
- dropout (float): Dropout probability (default: 0.0)

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional)

Output:
  - output: (B, L, D)

**Dependencies:**
- names3risk_pytorch.pytorch.attention.attention_head → Individual attention heads

**Used By:**
- names3risk_pytorch.pytorch.blocks.transformer_block → Self-attention component

**Usage Example:**
```python
from names3risk_pytorch.pytorch.attention import MultiHeadAttention

mha = MultiHeadAttention(embedding_dim=128, n_heads=8, dropout=0.1)
x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()

output = mha(x, attn_mask)  # (32, 50, 128)
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize MultiHeadAttention.

        Args:
            embedding_dim: Input embedding dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights and output projection
        """
        super().__init__()

        # Create attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, n_heads, dropout) for _ in range(n_heads)]
        )

        # Output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask (optional)

        Returns:
            output: (B, L, D)
        """
        # Compute attention for each head and concatenate
        head_outputs = [head(x, attn_mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, L, D)

        # Project back to embedding dimension
        output = self.proj(concatenated)
        output = self.dropout(output)

        return output
