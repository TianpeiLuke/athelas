"""
Attention-Weighted Sequence Pooling

Pools variable-length sequences into fixed-size representations using learned attention weights.

**Core Concept:**
Instead of simple max/mean pooling, learns to weight each sequence element by importance.
Essential for fraud detection where specific name patterns or email characteristics may be
critical fraud indicators.

**Architecture:**
1. Project each sequence element to attention score via linear layer
2. Apply sequence mask to ignore padding tokens
3. Normalize scores with softmax
4. Compute weighted sum of sequence elements

**Parameters:**
- input_dim (int): Dimension of input sequence elements
- dropout (float): Dropout probability for attention scores (default: 0.0)

**Forward Signature:**
Input:
  - sequence: (B, L, D) - Batch of sequences
  - lengths: (B,) - Actual lengths before padding (optional)

Output:
  - pooled: (B, D) - Pooled representations

**Dependencies:**
- torch.nn.Linear → Attention score projection
- torch.nn.functional.softmax → Score normalization

**Used By:**
- names3risk_pytorch.pytorch.blocks.lstm_encoder → LSTM sequence summarization
- names3risk_pytorch.pytorch.blocks.transformer_encoder → Transformer sequence summarization

**Alternative Approaches:**
- Mean pooling → Simpler but weights all tokens equally
- Max pooling → Takes most salient token but ignores others
- Last token → Simple but loses sequence context
- [CLS] token (transformers) → Requires special token, less flexible

**Usage Example:**
```python
from names3risk_pytorch.pytorch.pooling import AttentionPooling

# Create attention pooling layer
pooling = AttentionPooling(input_dim=256, dropout=0.1)

# Pool variable-length sequences (e.g., customer names)
sequences = torch.randn(32, 50, 256)  # (batch=32, max_len=50, dim=256)
lengths = torch.tensor([30, 45, 20, ...])  # Actual lengths before padding

pooled = pooling(sequences, lengths)  # (32, 256)
```

**References:**
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Attention-weighted sequence pooling.

    Learns to weight sequence elements by importance, then computes weighted sum.
    Handles variable-length sequences via masking.
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        """
        Initialize AttentionPooling.

        Args:
            input_dim: Dimension of input sequence elements (e.g., 256 for LSTM output,
                      128 for transformer embeddings)
            dropout: Dropout probability for attention scores (0.0 = no dropout)
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence using learned attention weights.

        Args:
            sequence: (B, L, D) - Input sequences
            lengths: (B,) - Actual sequence lengths before padding (optional)
                    If None, assumes all sequences have same length

        Returns:
            pooled: (B, D) - Pooled representations
        """
        # Compute attention scores: (B, L, 1)
        scores = self.attention(sequence)

        # Apply mask to ignore padding if lengths provided
        if lengths is not None:
            # Create mask: (B, L) where True = valid token, False = padding
            mask = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            # Mask out padding tokens by setting their scores to -inf
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Normalize scores with softmax: (B, L, 1)
        weights = F.softmax(scores, dim=1)
        weights = self.dropout(weights)

        # Weighted sum: (B, D)
        pooled = torch.sum(weights * sequence, dim=1)

        return pooled
