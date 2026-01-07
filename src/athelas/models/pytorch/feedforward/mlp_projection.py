"""
MLP Projection Block

Feedforward network with dimension change: Linear → Activation → Dropout → Linear → Dropout

**Core Concept:**
Similar to MLPBlock but allows input_dim != output_dim.
Useful for projection layers that transform feature dimensions.

**Parameters:**
- input_dim (int): Input dimension
- hidden_dim (int): Hidden layer dimension
- output_dim (int): Output dimension (can differ from input_dim)
- dropout (float): Dropout probability (default: 0.0)
- activation (str): Activation function (default: 'relu')
- output_dropout (bool): Apply dropout after final linear layer (default: False)

**Forward Signature:**
Input:
  - x: (B, L, D_in) or (B, D_in) - Input features

Output:
  - output: (B, L, D_out) or (B, D_out) - Projected features

**Dependencies:**
- torch.nn.Linear → Projection layers
- torch.nn.Dropout → Regularization

**Used By:**
- TSA Lightning modules → Feature projection
- Classifier heads → Dimension reduction/expansion

**Relation to MLPBlock:**
- MLPBlock: input_dim → hidden_dim → input_dim (preserves dimension)
- MLPProjection: input_dim → hidden_dim → output_dim (changes dimension)

**Usage Example:**
```python
from names3risk_pytorch.pytorch.feedforward import MLPProjection

# Project from 256 to 128 dimensions
projection = MLPProjection(
    input_dim=256,
    hidden_dim=512,
    output_dim=128,
    dropout=0.1
)

x = torch.randn(32, 50, 256)  # (batch, seq_len, input_dim)
output = projection(x)  # (32, 50, 128)
```
"""

import torch
import torch.nn as nn
from typing import Literal


class MLPProjection(nn.Module):
    """
    MLP block with dimension projection.

    Transforms input from input_dim to output_dim through a hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu"] = "relu",
        output_dropout: bool = False,
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

        layers = [
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        ]

        # Optional final dropout (defaults to False for backward compatibility)
        if output_dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
