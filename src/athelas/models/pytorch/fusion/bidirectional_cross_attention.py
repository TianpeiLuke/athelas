"""
Bidirectional Cross-Attention

Advanced bidirectional cross-attention with feed-forward networks for multi-modal fusion.

**Core Concept:**
A sophisticated bidirectional cross-attention mechanism that allows two modalities
(e.g., primary text and secondary text) to exchange information. Unlike simple
cross-attention, this includes feed-forward networks for additional processing
and multiple normalization layers for better training stability.

**Architecture:**
1. Primary queries attend to secondary keys/values
2. Secondary queries attend to primary keys/values
3. Residual connections with layer normalization
4. Feed-forward networks with ReLU activation and dropout
5. Additional layer normalization after FFN
6. Returns attention weights for analysis/visualization

**Parameters:**
- d_model (int): Dimension of input features (default: 100)
- num_heads (int): Number of attention heads (default: 8)
- dropout (float): Dropout probability (default: 0.1)

**Forward Signature:**
Input:
  - primary: (B, d_model) - Primary modality features
  - secondary: (B, d_model) - Secondary modality features

Output:
  - primary_final: (B, d_model) - Enhanced primary features
  - secondary_final: (B, d_model) - Enhanced secondary features
  - attention_weights: Dict with 'primary_to_secondary' and 'secondary_to_primary' keys

**Dependencies:**
- torch.nn.MultiheadAttention → Core attention mechanism
- torch.nn.LayerNorm → Normalization for residual connections
- athelas.models.pytorch.feedforward.MLPBlock → Reusable feed-forward networks

**Used By:**
- athelas.models.lightning.trimodal.pl_trimodal_cross_attn → Trimodal text fusion
- Any model requiring sophisticated bidirectional cross-attention

**Alternative Approaches:**
- athelas.models.pytorch.fusion.cross_attention_fusion → Simpler version without FFN
- athelas.models.pytorch.fusion.gate_fusion → Gating-based fusion
- athelas.models.pytorch.fusion.concat_fusion → Simple concatenation

**Usage Example:**
```python
from athelas.models.pytorch.fusion import BidirectionalCrossAttention

# Create advanced fusion module
fusion = BidirectionalCrossAttention(
    d_model=256,
    num_heads=8,
    dropout=0.1
)

# Fuse two text modalities (e.g., chat and shiptrack)
primary_text = torch.randn(32, 256)    # (batch, dim)
secondary_text = torch.randn(32, 256)  # (batch, dim)

primary_enhanced, secondary_enhanced, attn_weights = fusion(
    primary_text,
    secondary_text
)
# Outputs: (32, 256), (32, 256), dict with attention weights
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019) - Pre-norm architecture
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from ..feedforward import MLPBlock


class BidirectionalCrossAttention(nn.Module):
    """
    Advanced bidirectional cross-attention with feed-forward processing.

    Allows two modalities to exchange information through bidirectional attention
    with additional feed-forward processing for enhanced representation learning.
    """

    def __init__(self, d_model: int = 100, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize BidirectionalCrossAttention.

        Args:
            d_model: Dimension of input features (default: 100)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_p = dropout

        # Cross-attention layers: primary attends to secondary
        self.attn_p2s = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention layers: secondary attends to primary
        self.attn_s2p = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for attention residuals
        self.norm_primary = nn.LayerNorm(d_model)
        self.norm_secondary = nn.LayerNorm(d_model)

        # Feed-forward networks for additional processing
        # Reusing MLPBlock from feedforward module
        self.ffn_primary = MLPBlock(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            dropout=dropout,
            activation="relu",
        )

        self.ffn_secondary = MLPBlock(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            dropout=dropout,
            activation="relu",
        )

        # Layer normalization for FFN residuals
        self.norm_ffn_primary = nn.LayerNorm(d_model)
        self.norm_ffn_secondary = nn.LayerNorm(d_model)

    def forward(
        self, primary: torch.Tensor, secondary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply bidirectional cross-attention with feed-forward processing.

        Args:
            primary: (B, d_model) - Primary modality features
            secondary: (B, d_model) - Secondary modality features

        Returns:
            primary_final: (B, d_model) - Enhanced primary features
            secondary_final: (B, d_model) - Enhanced secondary features
            attention_weights: Dict with attention weight tensors
        """
        # Add sequence dimension for attention (treating each sample as single token)
        primary_seq = primary.unsqueeze(1)  # (B, 1, d_model)
        secondary_seq = secondary.unsqueeze(1)  # (B, 1, d_model)

        # Primary attends to Secondary
        p_attended, p_attn_weights = self.attn_p2s(
            query=primary_seq,  # What primary wants to know
            key=secondary_seq,  # What secondary can provide
            value=secondary_seq,  # The actual secondary information
        )

        # Residual connection and layer norm
        primary_enhanced = self.norm_primary(primary + p_attended.squeeze(1))

        # Secondary attends to Primary
        s_attended, s_attn_weights = self.attn_s2p(
            query=secondary_seq,  # What secondary wants to know
            key=primary_seq,  # What primary can provide
            value=primary_seq,  # The actual primary information
        )

        # Residual connection and layer norm
        secondary_enhanced = self.norm_secondary(secondary + s_attended.squeeze(1))

        # Feed-forward processing with residuals
        primary_ffn = self.ffn_primary(primary_enhanced)
        primary_final = self.norm_ffn_primary(primary_enhanced + primary_ffn)

        secondary_ffn = self.ffn_secondary(secondary_enhanced)
        secondary_final = self.norm_ffn_secondary(secondary_enhanced + secondary_ffn)

        # Store attention weights for analysis/visualization
        # Shape: (B, num_heads, 1, 1) -> squeeze to (B, num_heads)
        attention_weights = {
            "primary_to_secondary": p_attn_weights.squeeze(1)
            if p_attn_weights.dim() > 2
            else p_attn_weights,
            "secondary_to_primary": s_attn_weights.squeeze(1)
            if s_attn_weights.dim() > 2
            else s_attn_weights,
        }

        return primary_final, secondary_final, attention_weights

    def __repr__(self) -> str:
        return (
            f"BidirectionalCrossAttention(d_model={self.d_model}, "
            f"num_heads={self.num_heads}, dropout={self.dropout_p})"
        )
