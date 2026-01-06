"""
Feature Aggregation with Factorization Machine Support

Progressive dimensionality reduction via deep MLP and FM-style feature interactions.

**Core Concept:**
Reduces features from n_features to 1 via progressive halving (n → n/2 → n/4 → ... → 1).
Essential for TSA to aggregate multiple categorical and numerical features before attention.
Includes FM computation for second-order feature interactions.

**Architecture:**
- Progressive MLP: Halves feature dimension at each layer
- Activation: LeakyReLU for non-linearity
- Termination: Stops when dimension reaches 1

**Parameters:**
- num_feature (int): Number of input features to aggregate

**Forward Signature:**
Input:
  - x: [..., num_feature] - Feature embeddings to aggregate

Output:
  - aggregated: [..., 1] - Aggregated features

**Dependencies:**
- torch.nn.Linear → Dimensionality reduction layers
- torch.nn.LeakyReLU → Non-linear activation

**Used By:**
- temporal_self_attention_pytorch.pytorch.blocks.order_attention → Feature aggregation before attention
- temporal_self_attention_pytorch.pytorch.blocks.feature_attention → Last order feature aggregation

**Alternative Approaches:**
- Mean pooling → Simpler but loses learned importance
- Max pooling → Takes most salient but discards others
- Single linear projection → Less expressive

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.pooling import FeatureAggregation, compute_fm_parallel

# Create feature aggregation
agg = FeatureAggregation(num_feature=20)

# Aggregate categorical and numerical features
features = torch.randn(32, 50, 20, 128)  # [B, L, n_features, embed_dim]
aggregated = agg(features.permute(0, 1, 3, 2))  # [B, L, embed_dim, n_features] -> [B, L, embed_dim]

# Compute FM interactions
fm_features = compute_fm_parallel(features)  # [B, L, embed_dim]
```

**References:**
- "Factorization Machines" (Rendle, 2010) - FM for feature interactions
- "Neural Factorization Machines" (He & Chua, 2017) - Deep learning + FM
"""

import torch
import torch.nn as nn


class FeatureAggregation(nn.Module):
    """
    Feature aggregation via progressive dimensionality reduction.

    Uses deep MLP to aggregate features with progressive halving:
    n → n/2 → n/4 → ... → 1
    """

    def __init__(self, num_feature: int):
        """
        Initialize FeatureAggregation.

        Args:
            num_feature: Number of input features to aggregate
        """
        super().__init__()

        self.dim_embed = num_feature

        # Build progressive reduction layers
        layers = []
        current_dim = num_feature

        while current_dim > 1:
            next_dim = max(1, current_dim // 2)
            layers.extend([nn.Linear(current_dim, next_dim), nn.LeakyReLU()])
            current_dim = next_dim

        # Remove the last LeakyReLU
        if layers:
            layers = layers[:-1]

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features across the feature dimension.

        Args:
            x: Input tensor [..., num_feature]

        Returns:
            aggregated: [..., 1] - Aggregated features
        """
        return self.encoder(x)


def compute_fm_parallel(feature_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute Factorization Machine (FM) style feature interactions in parallel.

    Computes second-order feature interactions using the factorization machine formula:
    FM(x) = 0.5 * [(Σ embeddings)² - Σ (embeddings²)]

    This captures pairwise feature interactions without computing all pairs explicitly,
    reducing computational complexity from O(n²) to O(n).

    Args:
        feature_embedding: Feature embeddings [B, n_features, embed_dim]

    Returns:
        fm_interaction: FM interaction features [B, embed_dim]

    Example:
        >>> features = torch.randn(32, 10, 128)  # 32 samples, 10 features, 128 dims
        >>> interactions = compute_fm_parallel(features)  # [32, 128]
    """
    # Sum of embeddings then square: (Σ x_i)²
    summed_features_emb = torch.sum(feature_embedding, dim=-2)  # [B, embed_dim]
    summed_features_emb_square = torch.square(summed_features_emb)  # [B, embed_dim]

    # Square of embeddings then sum: Σ (x_i²)
    squared_features_emb = torch.square(feature_embedding)  # [B, n_features, embed_dim]
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)  # [B, embed_dim]

    # FM interaction computation: 0.5 * [(Σ x_i)² - Σ (x_i²)]
    fm_interaction = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

    return fm_interaction
