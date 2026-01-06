"""
Gate Fusion

Learnable gating mechanism for combining two modalities.

**Core Concept:**
Uses a learnable gate to dynamically weight the contribution of two modalities
(e.g., text and tabular). The gate learns which modality is more important
for each sample, computing: `gate * text + (1 - gate) * tabular`.

**Architecture:**
1. Project both modalities to common fusion dimension
2. Concatenate projected features
3. Gate network: Linear → LayerNorm → Sigmoid
4. Weighted combination using learned gate values

**Parameters:**
- text_dim (int): Dimension of text features
- tab_dim (int): Dimension of tabular features
- fusion_dim (int): Target fusion dimension

**Forward Signature:**
Input:
  - text_features: (B, text_dim) - Text modality features
  - tab_features: (B, tab_dim) - Tabular modality features
  
Output:
  - fused: (B, fusion_dim) - Gated fusion of both modalities

**Dependencies:**
- torch.nn.Linear → Projection and gating layers
- torch.nn.LayerNorm → Normalization for gate network
- torch.nn.Sigmoid → Gate activation

**Used By:**
- athelas.models.lightning.bimodal.pl_bimodal_gate_fusion → Bimodal gated fusion
- Any model requiring learnable modality weighting

**Alternative Approaches:**
- athelas.models.pytorch.fusion.cross_attention_fusion → Attention-based fusion
- athelas.models.pytorch.fusion.mixture_of_experts → Routing-based fusion
- athelas.models.pytorch.fusion.concat_fusion → Simple concatenation

**Usage Example:**
```python
from athelas.models.pytorch.fusion import GateFusion

# Create gate fusion module
fusion = GateFusion(text_dim=768, tab_dim=128, fusion_dim=256)

# Fuse text and tabular features
text_features = torch.randn(32, 768)  # (batch, text_dim)
tab_features = torch.randn(32, 128)   # (batch, tab_dim)

fused = fusion(text_features, tab_features)
# Output: (32, 256)
```

**Implementation Notes:**
The gate learns to weight modalities based on their projected representations.
For samples where text is more informative, the gate will be closer to 1.
For samples where tabular data is more informative, the gate will be closer to 0.

**References:**
- "Highway Networks" (Srivastava et al., 2015) - Gating mechanisms
- "Multimodal Fusion with Deep Neural Networks" (Ngiam et al., 2011)
"""

import torch
import torch.nn as nn


class GateFusion(nn.Module):
    """
    Learnable gating mechanism for dynamic modality fusion.
    
    Computes gate weights to combine two modalities:
    fused = gate * text + (1 - gate) * tabular
    """
    
    def __init__(self, text_dim: int, tab_dim: int, fusion_dim: int):
        """
        Initialize GateFusion.
        
        Args:
            text_dim: Dimension of text features
            tab_dim: Dimension of tabular features  
            fusion_dim: Target fusion dimension
        """
        super().__init__()
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.fusion_dim = fusion_dim
        
        # Project each modality to common fusion dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.tab_proj = nn.Linear(tab_dim, fusion_dim)
        
        # Gate network: learns to weight modalities
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, 
        text_features: torch.Tensor, 
        tab_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learnable gating to fuse text and tabular modalities.
        
        Args:
            text_features: (B, text_dim) - Text modality features
            tab_features: (B, tab_dim) - Tabular modality features
            
        Returns:
            fused: (B, fusion_dim) - Gated fusion output
        """
        # Project to common dimension
        txt_feat = self.text_proj(text_features)  # (B, fusion_dim)
        tab_feat = self.tab_proj(tab_features)    # (B, fusion_dim)
        
        # Concatenate for gate computation
        combined = torch.cat([txt_feat, tab_feat], dim=1)  # (B, fusion_dim * 2)
        
        # Compute gate weights (0 to 1)
        gate = self.gate_net(combined)  # (B, fusion_dim)
        
        # Weighted combination: gate * text + (1 - gate) * tabular
        fused = gate * txt_feat + (1 - gate) * tab_feat
        
        return fused
    
    def __repr__(self) -> str:
        return (
            f"GateFusion(text_dim={self.text_dim}, "
            f"tab_dim={self.tab_dim}, fusion_dim={self.fusion_dim})"
        )
