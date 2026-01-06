"""
Tabular Embedding

General-purpose embedding layer for tabular/numerical features.

**Core Concept:**
Projects concatenated tabular features (numerical and/or categorical) into a
dense embedding space through LayerNorm → Linear → ReLU. Provides a standard
interface for encoding tabular data in neural networks.

**Architecture:**
1. Layer normalization to stabilize inputs
2. Linear projection to target dimension
3. ReLU activation for non-linearity

**Parameters:**
- input_dim (int): Dimension of concatenated tabular features
- hidden_dim (int): Target embedding dimension

**Forward Signature:**
Input:
  - x: (B, input_dim) - Concatenated tabular features

Output:
  - embeddings: (B, hidden_dim) - Embedded representations

**Dependencies:**
- torch.nn.LayerNorm → Input normalization
- torch.nn.Linear → Projection layer
- torch.nn.ReLU → Activation function

**Used By:**
- athelas.models.lightning.tabular.pl_tab_ae → Tabular autoencoder
- athelas.models.lightning.bimodal → Bimodal fusion models
- athelas.models.lightning.trimodal → Trimodal fusion models
- Any model processing tabular features

**Alternative Approaches:**
- athelas.models.pytorch.embeddings.token_embedding → For categorical features
- Simple Linear layer → Without normalization
- Deep MLP → Multiple layers for more capacity

**Usage Example:**
```python
from athelas.models.pytorch.embeddings import TabularEmbedding

# Create tabular embedding layer
embedding = TabularEmbedding(input_dim=50, hidden_dim=128)

# Embed concatenated tabular features
tabular_features = torch.randn(32, 50)  # (batch, input_dim)
embeddings = embedding(tabular_features)
# Output: (32, 128)
```

**Implementation Notes:**
- LayerNorm helps with features at different scales
- ReLU provides non-linearity while being computationally efficient
- Output dimension can be used directly or fed to downstream layers
- For multiple tabular fields, concatenate them first, then embed

**Utility Function:**
Use `combine_tabular_fields()` to prepare multi-field tabular data.

**References:**
- "Layer Normalization" (Ba et al., 2016) - Normalization technique
- "Rectified Linear Units Improve Restricted Boltzmann Machines" (Nair & Hinton, 2010) - ReLU
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union


class TabularEmbedding(nn.Module):
    """
    General-purpose tabular feature embedding.

    Projects concatenated tabular features through LayerNorm → Linear → ReLU
    to create dense embeddings suitable for downstream neural network processing.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize TabularEmbedding.

        Args:
            input_dim: Dimension of concatenated input features
            hidden_dim: Target embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Embedding: LayerNorm → Linear → ReLU
        self.embedding = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tabular features.

        Args:
            x: (B, input_dim) - Concatenated tabular features

        Returns:
            embeddings: (B, hidden_dim) - Embedded representations
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, got {x.shape[1]}"
            )

        return self.embedding(x)

    def __repr__(self) -> str:
        return (
            f"TabularEmbedding(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )


def combine_tabular_fields(
    batch: Dict[str, Union[torch.Tensor, List]],
    field_list: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Utility function to combine multiple tabular fields into a single tensor.

    Concatenates specified fields from a batch dictionary into a single tensor
    of shape (B, total_features), handling both list and tensor inputs.

    Args:
        batch: Dictionary containing field_name -> values mapping
        field_list: List of field names to concatenate
        device: Target device for output tensor

    Returns:
        combined: (B, total_features) - Concatenated tabular features

    Raises:
        KeyError: If a field in field_list is not in batch
        TypeError: If a field has unsupported type

    Example:
        >>> batch = {
        ...     'age': torch.tensor([25, 30, 35]),
        ...     'income': torch.tensor([50000, 60000, 55000]),
        ...     'score': [0.8, 0.9, 0.7]
        ... }
        >>> combined = combine_tabular_fields(
        ...     batch,
        ...     ['age', 'income', 'score'],
        ...     torch.device('cpu')
        ... )
        >>> combined.shape
        torch.Size([3, 3])
    """
    features = []

    for field in field_list:
        if field not in batch:
            raise KeyError(
                f"Missing field '{field}' in batch during tabular combination"
            )

        val = batch[field]

        # Convert to tensor if needed
        if isinstance(val, list):
            val = torch.tensor(val, dtype=torch.float32, device=device)
        elif isinstance(val, torch.Tensor):
            val = val.to(dtype=torch.float32, device=device)
        else:
            raise TypeError(
                f"Unsupported type for field '{field}': {type(val)}. "
                f"Expected list or torch.Tensor"
            )

        # Ensure 2D shape (B, features)
        if val.dim() == 1:
            val = val.unsqueeze(1)

        features.append(val)

    # Concatenate along feature dimension
    return torch.cat(features, dim=1)
