"""
Transformer-Based Text Encoder

Transformer encoder with attention pooling for text sequence encoding.

**Core Concept:**
Processes text sequences using stacked transformer layers with self-attention,
allowing model to capture long-range dependencies and character patterns.
Uses learned positional encodings and attention pooling for final representation.

**Architecture:**
1. Token + Position Embeddings
2. N x Transformer Blocks
3. Attention Pooling: Variable-length → Fixed-size
4. Linear Projection: Transform to desired output dimension

**Parameters:**
- vocab_size (int): Vocabulary size
- embedding_dim (int): Token/position embedding dimension
- hidden_dim (int): Output projection dimension
- n_blocks (int): Number of transformer blocks
- n_heads (int): Number of attention heads per block
- ff_hidden_dim (int): Feedforward hidden dimension
- max_seq_len (int): Maximum sequence length for positional encoding
- dropout (float): Dropout probability

**Forward Signature:**
Input:
  - tokens: (B, L) - Token IDs
  - attn_mask: (B, L) - Attention mask (optional, True = attend, False = ignore)

Output:
  - encoded: (B, 2*hidden_dim) - Encoded representations

**Dependencies:**
- names3risk_pytorch.pytorch.blocks.transformer_block → Transformer layers
- names3risk_pytorch.pytorch.pooling.attention_pooling → Sequence pooling

**Used By:**
- names3risk_pytorch.models.transformer2risk → Text encoder

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import TransformerEncoder

encoder = TransformerEncoder(
    vocab_size=4000,
    embedding_dim=128,
    hidden_dim=256,
    n_blocks=8,
    n_heads=8,
    ff_hidden_dim=512,
    max_seq_len=100,
    dropout=0.2
)

tokens = torch.randint(0, 4000, (32, 50))  # (batch, seq_len)
attn_mask = torch.ones(32, 50).bool()  # All tokens valid

encoded = encoder(tokens, attn_mask)  # (32, 512) = (32, 2*256)
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .transformer_block import TransformerBlock
from ..pooling import AttentionPooling


class TransformerEncoder(nn.Module):
    """
    Transformer-based text encoder with attention pooling.

    Stacks transformer blocks and pools sequence to fixed representation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_blocks: int = 8,
        n_heads: int = 8,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        """
        Initialize TransformerEncoder.

        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token/position embeddings
            hidden_dim: Output projection dimension
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads per block
            ff_hidden_dim: Feedforward hidden dim (default: 4 * embedding_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embedding_dim

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Attention pooling
        self.pooling = AttentionPooling(input_dim=embedding_dim)

        # Output projection to match desired hidden_dim
        self.proj = nn.Linear(embedding_dim, 2 * hidden_dim)

    def forward(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode token sequences.

        Args:
            tokens: (B, L) - Token IDs
            attn_mask: (B, L) - Attention mask (optional)

        Returns:
            encoded: (B, 2*hidden_dim) - Encoded representations
        """
        B, L = tokens.shape

        # Token embeddings: (B, L, embedding_dim)
        token_emb = self.token_embedding(tokens)

        # Position embeddings: (L, embedding_dim) broadcast to (B, L, embedding_dim)
        positions = torch.arange(L, device=tokens.device)
        pos_emb = self.position_embedding(positions)

        # Combined embeddings
        x = token_emb + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Attention pooling: (B, embedding_dim)
        # Note: attn_mask not used in pooling (could add if needed)
        pooled = self.pooling(x)

        # Project to output dimension: (B, 2*hidden_dim)
        encoded = self.proj(pooled)

        return encoded
