"""
Multi-Layer Perceptron Block

Standard feedforward network: Linear → Activation → Dropout → Linear → Dropout

**Core Concept:**
Position-wise feedforward network applied to each token independently.
Standard component in transformer architectures. For Names3Risk, processes
each character/subword embedding to capture non-linear patterns.

**Parameters:**
- input_dim (int): Input dimension
- hidden_dim (int): Hidden layer dimension (typically 4x input_dim)
- dropout (float): Dropout probability (default: 0.0)
- activation (str): Activation function (default: 'relu')

**Forward Signature:**
Input:
  - x: (B, L, D) or (B, D) - Input features

Output:
  - output: Same shape as input

**Dependencies:**
- torch.nn.Linear → Feedforward layers
- torch.nn.Dropout → Regularization

**Used By:**
- names3risk_pytorch.pytorch.blocks.transformer_block → Transformer FFN component

**Alternative Approaches:**
- names3risk_pytorch.pytorch.feedforward.residual_block → With skip connection
- Gated feedforward (GLU, SwiGLU) → Learnable gating

**Usage Example:**
```python
from names3risk_pytorch.pytorch.feedforward import MLPBlock

mlp = MLPBlock(input_dim=128, hidden_dim=512, dropout=0.2)
x = torch.randn(32, 50, 128)  # (batch, seq_len, dim)
output = mlp(x)  # (32, 50, 128)
```
"""

import torch
import torch.nn as nn
from typing import Literal


class MLPBlock(nn.Module):
    """Multi-layer perceptron block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu"] = "relu",
    ):
        super().__init__()

        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
