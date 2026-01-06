"""
CNN-Based Sequence Encoder

Multi-kernel 1D CNN for encoding sequential data (text, time series, etc.).

**Core Concept:**
Processes sequences using multiple parallel 1D convolutional layers with different
kernel sizes to capture patterns at multiple scales. Classic TextCNN architecture
that's effective for text classification and sequence encoding.

**Architecture:**
1. Token/Feature Embedding (if vocab provided)
2. Multiple parallel Conv1d branches with different kernel sizes
3. Each branch: Conv1d → ReLU → MaxPool (repeated num_layers times)
4. Concatenate all branches
5. Optional projection to output dimension

**Parameters:**
- input_dim (int): Input feature dimension (e.g., embedding_dim)
- kernel_sizes (List[int]): List of kernel sizes for parallel convolutions (default: [3, 5, 7])
- num_channels (List[int]): Number of channels per layer (default: [100, 100])
- num_layers (int): Number of convolutional layers per kernel (default: 2)
- output_dim (int): Output dimension (optional, for projection)
- dropout (float): Dropout probability (default: 0.5)
- vocab_size (int): If provided, includes embedding layer
- embedding_dim (int): Embedding dimension if vocab_size provided

**Forward Signature:**
Input:
  - x: (B, L, D) - Sequence features, OR
  - x: (B, L) - Token IDs if vocab_size provided

Output:
  - encoded: (B, output_dim) or (B, num_channels[-1] * len(kernel_sizes)) - Encoded features

**Dependencies:**
- torch.nn.Conv1d → 1D convolution
- torch.nn.MaxPool1d → Pooling
- torch.nn.Embedding → Optional token embedding

**Used By:**
- athelas.models.lightning.text.pl_text_cnn → Text CNN classifier
- athelas.models.lightning.bimodal.pl_bimodal_cnn → Text encoder
- Any model requiring multi-scale sequence encoding

**Alternative Approaches:**
- athelas.models.pytorch.blocks.lstm_encoder → Recurrent encoding
- athelas.models.pytorch.blocks.transformer_encoder → Self-attention encoding
- Simple Conv1d → Single kernel size

**Usage Example:**
```python
from athelas.models.pytorch.blocks import CNNEncoder

# For text with embedding layer
encoder = CNNEncoder(
    vocab_size=10000,
    embedding_dim=128,
    kernel_sizes=[3, 5, 7],
    num_channels=[100, 100],
    num_layers=2,
    output_dim=256,
    dropout=0.5
)

tokens = torch.randint(0, 10000, (32, 50))  # (batch, seq_len)
encoded = encoder(tokens)  # (32, 256)

# For pre-embedded features (no vocab)
encoder = CNNEncoder(
    input_dim=128,
    kernel_sizes=[3, 5],
    num_channels=[64, 64],
    num_layers=2,
    output_dim=128
)

features = torch.randn(32, 50, 128)  # (batch, seq_len, dim)
encoded = encoder(features)  # (32, 128)
```

**Implementation Notes:**
- Each kernel size operates independently in parallel
- MaxPool reduces sequence length at each layer
- Final pooling aggregates to single vector per kernel
- Output is concatenation of all kernel outputs
- Effective for capturing local patterns (n-grams for text)

**Design Pattern:**
This is the classic Kim CNN / TextCNN architecture, proven effective for:
- Text classification (captures n-gram patterns)
- Time series encoding (captures temporal patterns at multiple scales)
- Any 1D sequential data

**References:**
- "Convolutional Neural Networks for Sentence Classification" (Kim, 2014) - TextCNN
- "Character-level Convolutional Networks for Text Classification" (Zhang et al., 2015)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CNNEncoder(nn.Module):
    """
    Multi-kernel 1D CNN encoder for sequential data.

    Encodes sequences using parallel convolutional branches with different
    kernel sizes, then concatenates results for multi-scale representation.
    """

    def __init__(
        self,
        input_dim: int = None,
        kernel_sizes: List[int] = None,
        num_channels: List[int] = None,
        num_layers: int = 2,
        output_dim: int = None,
        dropout: float = 0.5,
        vocab_size: int = None,
        embedding_dim: int = None,
        max_seq_len: int = 512,
    ):
        """
        Initialize CNNEncoder.

        Args:
            input_dim: Input feature dimension (required if vocab_size not provided)
            kernel_sizes: List of kernel sizes for parallel convs (default: [3, 5, 7])
            num_channels: Number of output channels per layer (default: [100, 100])
            num_layers: Number of conv layers per kernel (default: 2)
            output_dim: Optional output projection dimension
            dropout: Dropout probability (default: 0.5)
            vocab_size: If provided, includes embedding layer
            embedding_dim: Embedding dimension if vocab_size provided
            max_seq_len: Maximum sequence length for computing conv dimensions
        """
        super().__init__()

        # Handle embedding layer
        if vocab_size is not None:
            if embedding_dim is None:
                raise ValueError("embedding_dim required when vocab_size provided")
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.input_dim = embedding_dim
        else:
            if input_dim is None:
                raise ValueError("input_dim required when vocab_size not provided")
            self.embeddings = None
            self.input_dim = input_dim

        # Default kernel sizes and channels
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.num_channels = num_channels or [100, 100]
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout

        # Build parallel conv branches
        self.convs = nn.ModuleList(
            [self._build_conv_branch(k) for k in self.kernel_sizes]
        )

        # Output dimension after concatenating all branches
        self.features_dim = self.num_channels[-1] * len(self.kernel_sizes)

        # Optional output projection
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(self.features_dim, output_dim)
            )
            self.output_dim = output_dim
        else:
            self.projection = None
            self.output_dim = self.features_dim

    def _build_conv_branch(self, kernel_size: int) -> nn.Module:
        """
        Build a single convolutional branch with given kernel size.

        Each branch has num_layers of: Conv1d → ReLU → MaxPool
        Final layer pools to single value.
        """
        layers = []
        input_channels = [self.input_dim] + self.num_channels[:-1]

        # Compute output dimension after all convolutions
        seq_len = self.max_seq_len
        for i in range(self.num_layers):
            # Conv layer
            layers.append(
                nn.Conv1d(
                    in_channels=input_channels[i],
                    out_channels=self.num_channels[i],
                    kernel_size=kernel_size,
                )
            )
            layers.append(nn.ReLU())

            # Update sequence length after conv
            seq_len = seq_len - kernel_size + 1

            # MaxPool (adaptive on last layer to pool to single value)
            if i < self.num_layers - 1:
                # Intermediate layers: stride pooling
                layers.append(nn.MaxPool1d(kernel_size=kernel_size))
                seq_len = (seq_len - kernel_size) // kernel_size + 1
            else:
                # Last layer: pool to single value
                # Use adaptive pooling for variable sequence lengths
                layers.append(nn.AdaptiveMaxPool1d(1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequences.

        Args:
            x: (B, L, D) - Sequence features, OR
               (B, L) - Token IDs if embeddings provided

        Returns:
            encoded: (B, output_dim) - Encoded representations
        """
        # Apply embeddings if provided
        if self.embeddings is not None:
            x = self.embeddings(x)  # (B, L, D)

        # Reshape for Conv1d: (B, L, D) → (B, D, L)
        x = x.permute(0, 2, 1)

        # Apply parallel conv branches
        branch_outputs = []
        for conv in self.convs:
            out = conv(x)  # (B, C, 1)
            out = out.squeeze(2)  # (B, C)
            branch_outputs.append(out)

        # Concatenate all branches
        features = torch.cat(branch_outputs, dim=1)  # (B, C * num_kernels)

        # Optional projection
        if self.projection is not None:
            encoded = self.projection(features)
        else:
            encoded = features

        return encoded

    def __repr__(self) -> str:
        embed_str = (
            f"vocab={self.embeddings.num_embeddings}, " if self.embeddings else ""
        )
        return (
            f"CNNEncoder({embed_str}"
            f"kernels={self.kernel_sizes}, "
            f"channels={self.num_channels}, "
            f"layers={self.num_layers}, "
            f"output_dim={self.output_dim})"
        )


def compute_cnn_output_length(
    input_length: int,
    kernel_size: int,
    num_layers: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """
    Utility to compute output sequence length after CNN layers.

    Useful for understanding how sequence length changes through the network.

    Args:
        input_length: Initial sequence length
        kernel_size: Convolutional kernel size
        num_layers: Number of conv + pool layers
        stride: Convolution stride (default: 1)
        padding: Convolution padding (default: 0)

    Returns:
        output_length: Final sequence length

    Example:
        >>> # After 2 layers of conv(k=3) + maxpool(k=3)
        >>> compute_cnn_output_length(512, kernel_size=3, num_layers=2)
        56
    """
    length = input_length

    for i in range(num_layers):
        # After convolution
        length = (length + 2 * padding - kernel_size) // stride + 1

        # After max pooling (except last layer uses adaptive pooling)
        if i < num_layers - 1:
            length = (length - kernel_size) // kernel_size + 1

    return length
