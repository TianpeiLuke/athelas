"""
Order Attention Module

Stacked attention layers for processing order-based transaction sequences.

**Core Concept:**
Processes transaction sequences in temporal order using multiple stacked attention
layers. Aggregates categorical and numerical features, applies temporal attention,
and produces order-aware representations for fraud detection.

**Architecture:**
- Embedding layers for categorical features
- Feature aggregation for dimensionality reduction
- Multiple stacked AttentionLayer blocks
- Temporal encoding throughout
- Final aggregation to fixed-size representation

**Components Used:**
Phase 1 Atomics:
  - TimeEncode: Temporal position encoding
  - FeatureAggregation: Progressive feature reduction

Phase 2 Composites:
  - AttentionLayer: Multi-head attention with temporal encoding

**Parameters:**
- n_cat_features (int): Number of categorical features
- n_num_features (int): Number of numerical features
- n_embedding (int): Size of embedding vocabulary
- dim_embedding_table (int): Embedding dimension
- dim_attn_feedforward (int): Attention feedforward dimension
- n_layers_order (int): Number of attention layers
- num_heads (int): Number of attention heads
- use_moe (bool): Whether to use MoE in attention layers
- num_experts (int): Number of experts if using MoE
- dropout (float): Dropout probability

**Forward Signature:**
Input:
  - x_cat: [B, L, n_cat] - Categorical features
  - x_num: [B, L, n_num] - Numerical features
  - time_seq: [B, L, 1] - Time sequence
  
Output:
  - output: [B, 2*dim_embedding_table] - Order-aware representation

**Used By:**
- Lightning modules for TSA single/dual sequence models

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.blocks import OrderAttentionModule

order_attn = OrderAttentionModule(
    n_cat_features=10,
    n_num_features=5,
    n_embedding=1000,
    dim_embedding_table=64,
    dim_attn_feedforward=256,
    n_layers_order=2,
    num_heads=4,
    use_moe=True,
    num_experts=5
)

# Process transaction sequence
x_cat = torch.randint(0, 1000, (32, 50, 10))  # [B, L, n_cat]
x_num = torch.randn(32, 50, 5)  # [B, L, n_num]
time_seq = torch.randn(32, 50, 1)  # [B, L, 1]

output = order_attn(x_cat, x_num, time_seq)  # [32, 128]
```

**References:**
- "Temporal Self-Attention for Fraud Detection" - Order attention concept
"""

import torch
import torch.nn as nn
from typing import Optional

# Import Phase 1 atomic components
from ..embeddings import TimeEncode
from ..pooling import FeatureAggregation

# Import Phase 2 composite components
from .attention_layer import AttentionLayer


class OrderAttentionModule(nn.Module):
    """
    Order attention module for temporal transaction sequence processing.
    
    Stacks multiple attention layers with temporal encoding for
    order-aware feature learning.
    """
    
    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_embedding: int,
        dim_embedding_table: int,
        dim_attn_feedforward: int,
        n_layers_order: int,
        num_heads: int,
        use_moe: bool = False,
        num_experts: int = 3,
        dropout: float = 0.1,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
    ):
        """
        Initialize OrderAttentionModule.
        
        Args:
            n_cat_features: Number of categorical features
            n_num_features: Number of numerical features
            n_embedding: Size of embedding vocabulary
            dim_embedding_table: Embedding dimension
            dim_attn_feedforward: Attention feedforward dimension
            n_layers_order: Number of attention layers
            num_heads: Number of attention heads
            use_moe: Whether to use MoE in attention layers
            num_experts: Number of experts if using MoE
            dropout: Dropout probability
            second_policy_train: MoE training policy
            second_policy_eval: MoE evaluation policy
        """
        super().__init__()
        
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.dim_embedding_table = dim_embedding_table
        self.n_layers_order = n_layers_order
        
        # Embedding for categorical features
        self.cat_embedding = nn.Embedding(n_embedding, dim_embedding_table)
        
        # Linear projection for numerical features
        self.num_linear = nn.Linear(n_num_features, dim_embedding_table)
        
        # Feature aggregation (Phase 1 atomic)
        total_features = n_cat_features + 1  # +1 for numerical features
        self.feature_aggregation = FeatureAggregation(total_features)
        
        # Temporal encoding (Phase 1 atomic)
        self.time_encoder = TimeEncode(dim_embedding_table)
        
        # Stack of attention layers (Phase 2 composite)
        self.attention_layers = nn.ModuleList([
            AttentionLayer(
                dim=dim_embedding_table,
                time_dim=dim_embedding_table,
                num_heads=num_heads,
                dim_feedforward=dim_attn_feedforward,
                use_moe=use_moe,
                num_experts=num_experts,
                dropout=dropout,
                second_policy_train=second_policy_train,
                second_policy_eval=second_policy_eval,
            )
            for _ in range(n_layers_order)
        ])
        
        # Final linear layers for output
        self.output_linear1 = nn.Linear(dim_embedding_table, dim_embedding_table)
        self.output_linear2 = nn.Linear(dim_embedding_table, dim_embedding_table)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        time_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for order attention.
        
        Args:
            x_cat: Categorical features [B, L, n_cat]
            x_num: Numerical features [B, L, n_num]
            time_seq: Time sequence [B, L, 1]
            
        Returns:
            output: [B, 2*dim_embedding_table] - Order-aware representation
        """
        batch_size, seq_len, _ = x_cat.shape
        
        # Embed categorical features
        cat_emb = self.cat_embedding(x_cat)  # [B, L, n_cat, dim]
        
        # Project numerical features
        num_emb = self.num_linear(x_num)  # [B, L, dim]
        num_emb = num_emb.unsqueeze(2)  # [B, L, 1, dim]
        
        # Concatenate categorical and numerical embeddings
        all_features = torch.cat([cat_emb, num_emb], dim=2)  # [B, L, n_cat+1, dim]
        
        # Aggregate features (Phase 1 atomic)
        # Permute for aggregation: [B, L, dim, n_features]
        all_features = all_features.permute(0, 1, 3, 2)
        aggregated = self.feature_aggregation(all_features)  # [B, L, dim, 1]
        aggregated = aggregated.squeeze(-1)  # [B, L, dim]
        
        # Permute to [L, B, dim] for attention layers
        x = aggregated.permute(1, 0, 2)  # [L, B, dim]
        
        # Apply stacked attention layers (Phase 2 composite)
        for attn_layer in self.attention_layers:
            x = attn_layer(x, time_seq)  # [L, B, dim]
        
        # Permute back to [B, L, dim]
        x = x.permute(1, 0, 2)  # [B, L, dim]
        
        # Take first and last timesteps
        first_step = x[:, 0, :]  # [B, dim]
        last_step = x[:, -1, :]  # [B, dim]
        
        # Apply output projections
        first_out = self.output_linear1(first_step)  # [B, dim]
        last_out = self.output_linear2(last_step)  # [B, dim]
        
        # Concatenate first and last representations
        output = torch.cat([first_out, last_out], dim=-1)  # [B, 2*dim]
        
        return output
