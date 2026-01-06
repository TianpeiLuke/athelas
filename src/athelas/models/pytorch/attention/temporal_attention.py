"""
Temporal Multi-head Attention

Multi-head attention with integrated temporal encoding for time-aware sequence processing.

**Core Concept:**
Extends standard multi-head attention by incorporating temporal information directly
into the attention mechanism. Essential for TSA to capture both content-based and
time-based relationships in transaction sequences.

**Architecture:**
- Standard multi-head attention base
- Integrated temporal encoder (TimeEncode)
- Temporal bias added to keys and values
- Supports all standard attention features (masking, dropout)

**Parameters:**
- embed_dim (int): Embedding dimension
- num_heads (int): Number of attention heads
- dropout (float): Dropout probability (default: 0.0)

**Forward Signature:**
Input:
  - query: [L, B, E] - Query tensor
  - key: [L, B, E] - Key tensor
  - value: [L, B, E] - Value tensor
  - time_seq: [L, B, 1] - Time sequence (deltas or timestamps)
  - attn_mask: [L, L] - Attention mask (optional)
  - key_padding_mask: [B, L] - Key padding mask (optional)

Output:
  - attention_output: [L, B, E] - Attended features
  - attention_weights: [B, num_heads, L, L] - Attention weights

**Dependencies:**
- torch.nn.MultiheadAttention → Base attention mechanism
- temporal_self_attention_pytorch.pytorch.embeddings.TimeEncode → Temporal encoding

**Used By:**
- temporal_self_attention_pytorch.pytorch.blocks.attention_layer → Temporal attention layers

**Alternative Approaches:**
- Concatenate time to features → Less principled
- Separate temporal attention → More parameters
- Relative position encoding → Not absolute time-aware

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.attention import TemporalMultiheadAttention

temporal_attn = TemporalMultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)

# Process sequence with temporal information
query = key = value = torch.randn(50, 32, 128)  # [L, B, E]
time_seq = torch.randn(50, 32, 1)  # [L, B, 1] - time deltas

output, weights = temporal_attn(query, key, value, time_seq)
# output: [50, 32, 128], weights: [32, 4, 50, 50]
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Multi-head attention
- "Time2Vec: Learning a Vector Representation of Time" (Kazemi et al., 2019)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..embeddings import TimeEncode


class TemporalMultiheadAttention(nn.Module):
    """
    Temporal Multi-head Attention with time encoding integration.

    Integrates temporal information into the attention mechanism
    by adding learned temporal encodings to keys and values.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize TemporalMultiheadAttention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Use standard MultiheadAttention as base
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False
        )

        # Time encoding integration
        self.time_encoder = TimeEncode(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal attention.

        Args:
            query: Query tensor [L, B, E]
            key: Key tensor [L, B, E]
            value: Value tensor [L, B, E]
            time_seq: Time sequence [L, B, 1] or [B, L, 1]
            attn_mask: Attention mask [L, L] (optional)
            key_padding_mask: Key padding mask [B, L] (optional)

        Returns:
            attention_output: [L, B, E] - Attended features
            attention_weights: [B, num_heads, L, L] - Attention weights
        """
        # Encode temporal information
        if time_seq is not None:
            # Ensure correct shape: [B, L, 1] -> [L, B, E]
            if time_seq.dim() == 3 and time_seq.size(0) != query.size(0):
                time_seq = time_seq.permute(1, 0, 2)  # [B, L, 1] -> [L, B, 1]

            time_encoding = self.time_encoder(time_seq.permute(1, 0, 2))  # [L, B, E]

            # Add temporal encoding to key and value
            key = key + time_encoding
            value = value + time_encoding

        # Apply standard multi-head attention
        return self.attention(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
