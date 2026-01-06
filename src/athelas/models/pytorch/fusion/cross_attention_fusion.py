"""
Cross-Attention Fusion

Simple bidirectional cross-attention mechanism for fusing two modalities.

**Core Concept:**
Enables two modalities (e.g., text and tabular) to exchange information through
bidirectional cross-attention. Each modality attends to the other, creating
enhanced representations that incorporate cross-modal information.

**Architecture:**
1. Text queries attend to tabular keys/values (text2tab attention)
2. Tabular queries attend to text keys/values (tab2text attention)
3. Residual connections with layer normalization
4. Both modalities get enhanced representations

**Parameters:**
- hidden_dim (int): Dimension of input features (must be same for both modalities)
- num_heads (int): Number of attention heads (default: 4)

**Forward Signature:**
Input:
  - text_seq: (B, 1, H) or (B, H) - Text features
  - tab_seq: (B, 1, H) or (B, H) - Tabular features

Output:
  - text_out: (B, 1, H) - Enhanced text features
  - tab_out: (B, 1, H) - Enhanced tabular features

**Dependencies:**
- torch.nn.MultiheadAttention → Core attention mechanism
- torch.nn.LayerNorm → Normalization for residual connections

**Used By:**
- athelas.models.lightning.bimodal.pl_bimodal_cross_attn → Bimodal fusion
- Any model requiring simple bidirectional cross-attention

**Alternative Approaches:**
- athelas.models.pytorch.fusion.bidirectional_cross_attention → Advanced version with FFN
- athelas.models.pytorch.fusion.gate_fusion → Gating-based fusion
- athelas.models.pytorch.fusion.mixture_of_experts → Routing-based fusion

**Usage Example:**
```python
from athelas.models.pytorch.fusion import CrossAttentionFusion

# Create fusion module
fusion = CrossAttentionFusion(hidden_dim=256, num_heads=4)

# Fuse text and tabular features
text_features = torch.randn(32, 1, 256)  # (batch, seq_len=1, dim)
tab_features = torch.randn(32, 1, 256)   # (batch, seq_len=1, dim)

text_enhanced, tab_enhanced = fusion(text_features, tab_features)
# Both outputs: (32, 1, 256)
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Multi-head attention
- "Cross-Attention in Coupled Unmixing Nets" (Dou et al., 2018) - Cross-attention concepts
"""

import torch
import torch.nn as nn
from typing import Tuple


class CrossAttentionFusion(nn.Module):
    """
    Simple bidirectional cross-attention for two modalities.

    Allows text and tabular modalities to exchange information through
    bidirectional attention. Each modality attends to the other, creating
    enhanced representations.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        Initialize CrossAttentionFusion.

        Args:
            hidden_dim: Dimension of input features (must be same for both modalities)
            num_heads: Number of attention heads (default: 4)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Text queries attend to tabular keys/values
        self.text2tab = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Tabular queries attend to text keys/values
        self.tab2text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Layer normalization for residual connections
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.tab_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, text_seq: torch.Tensor, tab_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention between text and tabular modalities.

        Args:
            text_seq: (B, L, H) - Text sequence features
            tab_seq: (B, L, H) - Tabular sequence features

        Returns:
            text_out: (B, L, H) - Enhanced text features
            tab_out: (B, L, H) - Enhanced tabular features
        """
        # Text attends to tabular
        t2t_out, _ = self.text2tab(query=text_seq, key=tab_seq, value=tab_seq)
        # Residual connection + layer norm
        text_out = self.text_norm(text_seq + t2t_out)

        # Tabular attends to enhanced text
        tab2_out, _ = self.tab2text(query=tab_seq, key=text_out, value=text_out)
        # Residual connection + layer norm
        tab_out = self.tab_norm(tab_seq + tab2_out)

        return text_out, tab_out

    def __repr__(self) -> str:
        return (
            f"CrossAttentionFusion(hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads})"
        )
