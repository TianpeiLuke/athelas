"""
Expert Routing Fusion

Simple mixture of experts with learned routing for bimodal fusion.

**Core Concept:**
Treats each modality as an "expert" and learns routing weights to combine them.
Unlike hard routing (selecting one expert), this uses soft routing where the
router learns continuous weights for each expert based on their combined features.

**Architecture:**
1. Project each modality to common fusion dimension (if needed)
2. Concatenate all expert features
3. Router network: Linear → Softmax to compute expert weights
4. Weighted combination: sum(weight_i * expert_i)

**Parameters:**
- text_dim (int): Dimension of text expert features
- tab_dim (int): Dimension of tabular expert features
- fusion_dim (int): Target fusion dimension

**Forward Signature:**
Input:
  - text_features: (B, text_dim) - Text expert features
  - tab_features: (B, tab_dim) - Tabular expert features

Output:
  - fused: (B, fusion_dim) - Weighted combination of experts

**Dependencies:**
- torch.nn.Linear → Projection and routing layers
- torch.nn.Softmax → Routing weight normalization
- torch.nn.Identity → Optional skip projection

**Used By:**
- athelas.models.lightning.bimodal.pl_bimodal_moe → Bimodal MoE fusion
- Any model requiring expert-based fusion

**Alternative Approaches:**
- athelas.models.pytorch.fusion.gate_fusion → Gating instead of routing
- athelas.models.pytorch.fusion.cross_attention_fusion → Attention-based fusion
- Legacy MoE in temporal_self_attention → More complex with top-k routing

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.fusion import ExpertRoutingFusion

# Create expert routing fusion module
fusion = ExpertRoutingFusion(text_dim=768, tab_dim=128, fusion_dim=256)

# Fuse text and tabular experts
text_features = torch.randn(32, 768)  # (batch, text_dim)
tab_features = torch.randn(32, 128)   # (batch, tab_dim)

fused = fusion(text_features, tab_features)
# Output: (32, 256)
```

**Implementation Notes:**
- This is a simple 2-expert MoE suitable for bimodal fusion
- Router learns to weight experts based on concatenated features
- All experts are always active (soft routing, not sparse)
- For more complex MoE with top-k routing, see legacy temporal_self_attention

**References:**
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017)
- "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2021)
"""

import torch
import torch.nn as nn
from typing import Optional


class ExpertRoutingFusion(nn.Module):
    """
    Simple 2-expert mixture of experts with learned routing for bimodal fusion.

    Treats text and tabular as experts, learns routing weights,
    and computes weighted combination.
    """

    def __init__(self, text_dim: int, tab_dim: int, fusion_dim: int):
        """
        Initialize MixtureOfExperts.

        Args:
            text_dim: Dimension of text expert features
            tab_dim: Dimension of tabular expert features
            fusion_dim: Target fusion dimension
        """
        super().__init__()
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.fusion_dim = fusion_dim

        # Project experts to common dimension (if needed)
        self.text_proj = (
            nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        )
        self.tab_proj = (
            nn.Linear(tab_dim, fusion_dim) if tab_dim != fusion_dim else nn.Identity()
        )

        # Router: learns to weight experts based on concatenated features
        self.router = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),  # 2 experts
            nn.Softmax(dim=-1),
        )

    def forward(
        self, text_features: torch.Tensor, tab_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mixture of experts fusion with learned routing.

        Args:
            text_features: (B, text_dim) - Text expert features
            tab_features: (B, tab_dim) - Tabular expert features

        Returns:
            fused: (B, fusion_dim) - Weighted combination of experts
        """
        # Project experts to common dimension
        txt_feat = self.text_proj(text_features)  # (B, fusion_dim)
        tab_feat = self.tab_proj(tab_features)  # (B, fusion_dim)

        # Concatenate experts for routing decision
        concat_experts = torch.cat([txt_feat, tab_feat], dim=1)  # (B, fusion_dim * 2)

        # Compute routing weights via softmax (sum to 1)
        weights = self.router(concat_experts)  # (B, 2)

        # Extract individual weights
        w_txt = weights[:, 0].unsqueeze(1)  # (B, 1)
        w_tab = weights[:, 1].unsqueeze(1)  # (B, 1)

        # Weighted combination of experts
        fused = w_txt * txt_feat + w_tab * tab_feat  # (B, fusion_dim)

        return fused

    def get_routing_weights(
        self, text_features: torch.Tensor, tab_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get routing weights without computing final fusion.

        Useful for analyzing which expert is preferred for different samples.

        Args:
            text_features: (B, text_dim) - Text expert features
            tab_features: (B, tab_dim) - Tabular expert features

        Returns:
            weights: (B, 2) - Routing weights [text_weight, tab_weight]
        """
        txt_feat = self.text_proj(text_features)
        tab_feat = self.tab_proj(tab_features)
        concat_experts = torch.cat([txt_feat, tab_feat], dim=1)
        weights = self.router(concat_experts)
        return weights

    def __repr__(self) -> str:
        return (
            f"ExpertRoutingFusion(text_dim={self.text_dim}, "
            f"tab_dim={self.tab_dim}, fusion_dim={self.fusion_dim})"
        )
