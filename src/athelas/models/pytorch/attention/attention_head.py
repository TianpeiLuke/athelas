"""
Single Attention Head

Computes scaled dot-product attention for one head in multi-head attention.

**Core Concept:**
Learns to focus on relevant parts of the input sequence via Query-Key-Value mechanism.
For Names3Risk, helps identify which characters/subwords in names/emails are most
indicative of fraud patterns.

**Architecture:**
1. Project input to Q, K, V via single linear layer
2. Compute attention scores: Q @ K^T / sqrt(d_k)
3. Apply optional mask (for padding or causal masking)
4. Softmax to get attention weights
5. Apply weights to V: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V

**Parameters:**
- embedding_dim (int): Input embedding dimension
- n_heads (int): Number of heads in parent MultiHeadAttention
- dropout (float): Dropout for attention weights (default: 0.0)

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional, True = attend, False = ignore)

Output:
  - output: (B, L, head_size) where head_size = embedding_dim // n_heads

**Dependencies:**
- torch.nn.Linear → Q, K, V projection
- torch.nn.functional.softmax → Attention weight normalization

**Used By:**
- names3risk_pytorch.pytorch.attention.multihead_attention → Combines multiple heads

**Alternative Approaches:**
- Additive attention (Bahdanau) → Concat Q,K then MLP instead of dot product
- Linear attention → Approximations for O(N) complexity

**Usage Example:**
```python
from names3risk_pytorch.pytorch.attention import AttentionHead

head = AttentionHead(embedding_dim=128, n_heads=8, dropout=0.1)
x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()  # All tokens are valid

output = head(x, attn_mask)  # (32, 50, 16) where 16 = 128 // 8
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Original scaled dot-product attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionHead(nn.Module):
    """
    Single scaled dot-product attention head.

    Computes Q, K, V projections and attention for one head.
    """

    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize AttentionHead.

        Args:
            embedding_dim: Input embedding dimension (must be divisible by n_heads)
            n_heads: Number of heads in parent MultiHeadAttention (for dimension calculation)
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        assert embedding_dim % n_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.head_size = embedding_dim // n_heads

        # Single linear layer projects to Q, K, V (concatenated)
        self.qkv = nn.Linear(embedding_dim, 3 * self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention for this head.

        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask, True = attend, False = ignore

        Returns:
            output: (B, L, head_size)
        """
        B, L, D = x.shape

        # Project to Q, K, V and split: (B, L, 3 * head_size)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Compute attention scores: (B, L, L)
        scores = q @ k.transpose(-2, -1) * (self.head_size**-0.5)

        # Apply mask if provided
        if attn_mask is not None:
            # Convert mask to boolean if needed (handles both int and bool masks)
            # attn_mask shape: (B, L) -> expand to (B, 1, L) for broadcasting
            mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
            scores = scores.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # Normalize and apply dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention to values: (B, L, head_size)
        output = weights @ v

        return output
