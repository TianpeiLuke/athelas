"""
Temporal Position Encoding

Learnable temporal encoding using periodic functions for time-series data.

**Core Concept:**
Encodes temporal information via combination of linear transformation and sinusoidal
functions with learnable parameters. Essential for TSA models to capture both
absolute time values and periodic temporal patterns in transaction sequences.

**Architecture:**
- Linear component: Direct time representation
- Sinusoidal components: Periodic patterns at multiple frequencies
- Learnable parameters: Domain-specific adaptation

**Parameters:**
- time_dim (int): Dimension of temporal encoding
- device: Device for tensor allocation
- dtype: Data type for tensors

**Forward Signature:**
Input:
  - tt: Time tensor [B, L, 1] - Time deltas or timestamps

Output:
  - temporal_encoding: [L, B, time_dim] - Encoded temporal information

**Dependencies:**
- torch.nn.Linear → Linear transformation
- torch.nn.functional → Sinusoidal activation

**Used By:**
- temporal_self_attention_pytorch.pytorch.attention.temporal_attention → Time-aware attention
- temporal_self_attention_pytorch.pytorch.blocks.order_attention → Order attention module

**Alternative Approaches:**
- Fixed sinusoidal encoding (Transformer) → Not learnable
- Learned embeddings → No periodic structure
- RNN hidden states → Computational cost

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.embeddings import TimeEncode

time_encoder = TimeEncode(time_dim=128)

# Encode time deltas (days since last transaction)
time_deltas = torch.tensor([[[0.5]], [[1.2]], [[2.0]]])  # [B=3, L=1, 1]
time_encoding = time_encoder(time_deltas)  # [1, 3, 128]
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Positional encoding
- "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class TimeEncode(nn.Module):
    """
    Learnable temporal position encoding using periodic functions.

    Combines linear transformation with sinusoidal functions for
    temporal pattern modeling.
    """

    def __init__(self, time_dim: int, device=None, dtype=None):
        """
        Initialize TimeEncode.

        Args:
            time_dim: Dimension of temporal encoding
            device: Device for tensor allocation (default: None)
            dtype: Data type for tensors (default: None)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.time_dim = time_dim

        # Learnable weight matrix and bias
        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform."""
        # Kaiming uniform initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information.

        Args:
            tt: Time tensor [B, L, 1] - Time deltas or timestamps

        Returns:
            temporal_encoding: [L, B, time_dim] - Encoded temporal information
        """
        # Ensure 3D shape [B, L, 1]
        tt = tt.unsqueeze(-1) if tt.dim() == 2 else tt

        # Sinusoidal encoding (all dimensions except first)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))

        # Linear encoding (first dimension)
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])

        # Combine encodings
        t = torch.cat([out1, out2], -1)  # [B, L, time_dim]
        t = t.squeeze(-2)  # Remove extra dimension
        t = t.permute(1, 0, 2)  # [L, B, time_dim]

        return t
