"""
Residual Feedforward Block

Feedforward network with residual/skip connection for training stability.

**Core Concept:**
Applies non-linear transformation while preserving input via residual connection.
Enables deeper networks by mitigating vanishing gradient problem. Critical for
Names3Risk multi-layer classifier to learn complex fraud patterns.

**Architecture:**
- Pre-norm variant: LayerNorm → FFN → Residual connection
- Post-norm variant: FFN → Dropout → Residual connection

**Parameters:**
- dim (int): Input/output dimension
- expansion_factor (int): Hidden layer size multiplier (default: 4)
- dropout (float): Dropout probability (default: 0.0)
- activation (str): Activation function - 'relu', 'gelu', 'silu' (default: 'relu')
- norm_first (bool): Apply normalization before FFN (default: True)

**Forward Signature:**
Input:
  - x: (B, D) - Input features

Output:
  - output: (B, D) - x + FFN(x) or x + FFN(LayerNorm(x))

**Dependencies:**
- torch.nn.Linear → Feedforward layers
- torch.nn.LayerNorm → Optional normalization
- torch.nn.Dropout → Optional dropout

**Used By:**
- names3risk_pytorch.models.lstm2risk → LSTM2Risk classifier
- names3risk_pytorch.models.transformer2risk → Transformer2Risk classifier

**Alternative Approaches:**
- Plain FFN → No residual, harder to train deep networks
- Highway networks → Learnable gating, more parameters
- DenseNet connections → Concatenation instead of addition

**Usage Example:**
```python
from names3risk_pytorch.pytorch.feedforward import ResidualBlock

# Pre-norm residual block (LSTM style)
block = ResidualBlock(
    dim=512,
    expansion_factor=4,  # 512 -> 2048 -> 512
    norm_first=True
)

# Post-norm residual block (Transformer style)
block = ResidualBlock(
    dim=512,
    expansion_factor=1,  # 512 -> 512 -> 512
    dropout=0.2,
    norm_first=False
)

x = torch.randn(32, 512)
output = block(x)  # (32, 512)
```

**References:**
- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
"""

import torch
import torch.nn as nn
from typing import Literal


class ResidualBlock(nn.Module):
    """
    Feedforward block with residual connection.

    Supports both pre-norm and post-norm variants, configurable activation
    and expansion factor.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu"] = "relu",
        norm_first: bool = True,
    ):
        """
        Initialize ResidualBlock.

        Args:
            dim: Input/output dimension
            expansion_factor: Hidden layer size = dim * expansion_factor
            dropout: Dropout probability after second linear layer
            activation: Activation function name
            norm_first: If True, apply LayerNorm before FFN (pre-norm)
        """
        super().__init__()
        self.norm_first = norm_first
        hidden_dim = dim * expansion_factor

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Optional pre-norm
        if norm_first:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: (B, D) - Input features

        Returns:
            output: (B, D) - x + FFN(x) or x + FFN(LayerNorm(x))
        """
        if self.norm_first:
            return x + self.ffn(self.norm(x))
        else:
            return x + self.ffn(x)
