"""
LSTM-Based Text Encoder

Bidirectional LSTM with attention pooling for text sequence encoding.

**Core Concept:**
Processes variable-length text sequences (customer names, emails) using LSTM
to capture sequential patterns, then pools to fixed-size representation using
learned attention weights.

**Architecture:**
1. Token Embedding: Vocabulary → Dense vectors
2. Bidirectional LSTM: Forward + Backward passes
3. Attention Pooling: Variable-length → Fixed-size
4. Layer Normalization: Output stabilization

**Parameters:**
- vocab_size (int): Vocabulary size for token embeddings
- embedding_dim (int): Token embedding dimension
- hidden_dim (int): LSTM hidden dimension
- num_layers (int): Number of LSTM layers
- dropout (float): Dropout probability
- bidirectional (bool): Use bidirectional LSTM (default: True)

**Forward Signature:**
Input:
  - tokens: (B, L) - Token IDs
  - lengths: (B,) - Actual sequence lengths (optional, for packing)

Output:
  - encoded: (B, 2*hidden_dim) - Encoded representations (2x for bidirectional)

**Dependencies:**
- names3risk_pytorch.pytorch.pooling.attention_pooling → Sequence pooling

**Used By:**
- names3risk_pytorch.models.lstm2risk → Text encoder for LSTM2Risk model

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import LSTMEncoder

encoder = LSTMEncoder(
    vocab_size=4000,
    embedding_dim=16,
    hidden_dim=128,
    num_layers=4,
    dropout=0.2
)

tokens = torch.randint(0, 4000, (32, 50))  # (batch, seq_len)
lengths = torch.randint(10, 50, (32,))  # Actual lengths

encoded = encoder(tokens, lengths)  # (32, 256) = (32, 2*128)
```

**References:**
- "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..pooling import AttentionPooling


class LSTMEncoder(nn.Module):
    """
    LSTM-based text sequence encoder with attention pooling.

    Encodes variable-length token sequences into fixed-size representations.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTMEncoder.

        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention pooling (input_dim = 2*hidden_dim for bidirectional)
        lstm_output_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.pooling = AttentionPooling(input_dim=lstm_output_dim)

        # Output normalization
        self.norm = nn.LayerNorm(lstm_output_dim)

    def forward(
        self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode token sequences.

        Args:
            tokens: (B, L) - Token IDs
            lengths: (B,) - Actual sequence lengths before padding (optional)

        Returns:
            encoded: (B, 2*hidden_dim) - Encoded representations
        """
        # Token embeddings: (B, L, embedding_dim)
        embeddings = self.token_embedding(tokens)

        # LSTM encoding
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            packed_output, _ = self.lstm(packed_emb)
            # Unpack
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, _ = self.lstm(embeddings)

        # Attention pooling: (B, 2*hidden_dim)
        pooled = self.pooling(lstm_output, lengths)

        # Layer normalization
        encoded = self.norm(pooled)

        return encoded
