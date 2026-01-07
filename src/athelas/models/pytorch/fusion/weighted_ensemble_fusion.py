"""
Weighted Ensemble Fusion

Weighted combination of multiple embeddings using pre-computed weights.

**Core Concept:**
Combines 2 or more embeddings using pre-computed weight scores, supporting
both binary (2-source) and general (N-source) fusion scenarios. Particularly
useful for ensemble models, multi-branch architectures, and models with
pre-computed importance scores.

**Architecture:**
1. Weighted combination using einsum for efficiency
2. Optional softmax normalization of weights
3. Optional layer normalization of fused output

**Parameters:**
- embed_dim (int): Dimension of embeddings to fuse
- num_sources (int): Number of embedding sources (default: 2)
- normalize (bool): Apply layer normalization (default: True)
- apply_softmax (bool): Apply softmax to weights (default: False)

**Forward Signature:**
Input:
  - embeddings: List[Tensor] - List of (B, embed_dim) embeddings
  - weights: Tensor - (B, num_sources) pre-computed weights

Output:
  - fused: (B, embed_dim) - Weighted combination of embeddings

**Dependencies:**
- torch.nn.LayerNorm → Output normalization (optional)
- torch.nn.functional.softmax → Weight normalization (optional)

**Used By:**
- DualSequenceTSA → Dual sequence temporal self-attention
- Any ensemble or multi-branch model with pre-computed weights

**Alternative Approaches:**
- GateFusion → Learnable gating (learns weights internally)
- ConcatFusion → Simple concatenation (no weighting)
- CrossAttentionFusion → Attention-based fusion

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.fusion import WeightedEnsembleFusion

# Binary case (2 sources)
fusion = WeightedEnsembleFusion(embed_dim=256, num_sources=2)

seq1_embed = torch.randn(32, 256)  # (batch, embed_dim)
seq2_embed = torch.randn(32, 256)
gate_scores = torch.randn(32, 2)   # Pre-computed weights

fused = fusion([seq1_embed, seq2_embed], gate_scores)
# Output: (32, 256)

# Multi-source case (3+ sources)
fusion_multi = WeightedEnsembleFusion(embed_dim=256, num_sources=3)

embeds = [torch.randn(32, 256) for _ in range(3)]
weights = torch.randn(32, 3)

fused = fusion_multi(embeds, weights)
# Output: (32, 256)
```

**Implementation Notes:**
- Binary case uses optimized einsum operations
- General case supports arbitrary number of sources
- Weights are assumed pre-computed (not learned internally)
- Use apply_softmax=True if weights don't sum to 1

**References:**
- DualSequenceTSA implementation
- "Ensemble Methods in Machine Learning" (Dietterich, 2000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class WeightedEnsembleFusion(nn.Module):
    """
    Weighted fusion of multiple embeddings using pre-computed weights.

    Combines 2+ embeddings via weighted sum, optionally with normalization.
    Optimized for both binary (2-source) and general (N-source) cases.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int = 2,
        normalize: bool = True,
        apply_softmax: bool = False,
    ):
        """
        Initialize WeightedEnsembleFusion.

        Args:
            embed_dim: Dimension of embeddings to fuse
            num_sources: Number of embedding sources (default: 2)
            normalize: Apply layer normalization to output (default: True)
            apply_softmax: Apply softmax to weights before fusion (default: False)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.apply_softmax = apply_softmax

        # Optional layer normalization
        self.normalize = normalize
        if normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, embeddings: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply weighted fusion to combine multiple embeddings.

        Args:
            embeddings: List of embeddings to fuse, each (B, embed_dim)
            weights: Pre-computed weights (B, num_sources) or (B,) for binary

        Returns:
            fused: (B, embed_dim) weighted combination of embeddings

        Raises:
            ValueError: If number of embeddings doesn't match num_sources
        """
        if len(embeddings) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} embeddings, got {len(embeddings)}"
            )

        # Optional softmax normalization of weights
        if self.apply_softmax:
            weights = F.softmax(weights, dim=-1)

        # Weighted combination - optimized for binary case
        if self.num_sources == 2 and weights.dim() == 2:
            # Binary case: efficient einsum for 2 sources
            fused = torch.einsum(
                "bi,bj->bj", weights[:, 0:1], embeddings[0]
            ) + torch.einsum("bi,bj->bj", weights[:, 1:2], embeddings[1])
        else:
            # General case: support N sources
            fused = sum(
                torch.einsum("b,bd->bd", weights[:, i], embeddings[i])
                for i in range(self.num_sources)
            )

        # Optional layer normalization
        if self.normalize:
            fused = self.layer_norm(fused)

        return fused

    def __repr__(self) -> str:
        return (
            f"WeightedEnsembleFusion(embed_dim={self.embed_dim}, "
            f"num_sources={self.num_sources}, normalize={self.normalize}, "
            f"apply_softmax={self.apply_softmax})"
        )
