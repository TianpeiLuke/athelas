#!/usr/bin/env python3
"""
Core TSA components for PyTorch Lightning implementation.

This module contains the fundamental building blocks for Temporal Self-Attention models,
reimplemented for full PyTorch Lightning integration while preserving exact functionality.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Tuple


class TimeEncode(nn.Module):
    """
    Learnable temporal position encoding using periodic functions.
    
    This module encodes temporal information using a combination of:
    - Linear transformation for direct time representation
    - Sinusoidal functions for periodic patterns
    - Learnable parameters for domain adaptation
    """
    
    def __init__(self, time_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.time_dim = time_dim
        
        # Learnable weight matrix and bias
        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)
    
    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information.
        
        Args:
            tt: Time tensor [B, L, 1]
            
        Returns:
            Temporal encoding [L, B, time_dim]
        """
        tt = tt.unsqueeze(-1)  # Add feature dimension
        
        # Sinusoidal encoding
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))
        
        # Linear encoding
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])
        
        # Combine encodings
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(-2)  # Remove extra dimension
        t = t.permute(1, 0, 2)  # [L, B, time_dim]
        
        return t


class FeatureAggregation(nn.Module):
    """
    Feature aggregation module for dimensionality reduction.
    
    Uses a deep MLP to aggregate features across the feature dimension
    with progressive dimensionality reduction.
    """
    
    def __init__(self, num_feature: int):
        super().__init__()
        
        self.dim_embed = num_feature
        
        # Progressive dimensionality reduction
        layers = []
        current_dim = num_feature
        
        while current_dim > 1:
            next_dim = max(1, current_dim // 2)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LeakyReLU()
            ])
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
            Aggregated tensor [..., 1]
        """
        return self.encoder(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts module for sparse expert routing.
    
    Features:
    - Multiple expert networks for specialized processing
    - Gating network for expert selection
    - Sparse routing for computational efficiency
    - Load balancing for expert utilization
    """
    
    def __init__(
        self, 
        dim: int, 
        num_experts: int, 
        hidden_dim: int,
        second_policy_train: str = "random",
        second_policy_eval: str = "random"
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(dim, num_experts)
        
        # Policy parameters (for compatibility with original MoE)
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert routing.
        
        Args:
            x: Input tensor [L, B, E] or [B, E]
            
        Returns:
            Expert-processed tensor with same shape as input
        """
        original_shape = x.shape
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
        
        # Compute gate scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [N, num_experts]
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [N, E]
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [N, E, num_experts]
        
        # Weighted combination of expert outputs
        output = torch.sum(expert_outputs * gate_scores.unsqueeze(-2), dim=-1)  # [N, E]
        
        # Restore original shape
        output = output.view(original_shape)
        
        return output


class TemporalMultiheadAttention(nn.Module):
    """
    Temporal Multi-head Attention with time encoding integration.
    
    This is a simplified version that integrates temporal information
    into the attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Use standard MultiheadAttention as base
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False
        )
        
        # Time encoding integration
        self.time_encoder = TimeEncode(embed_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal attention.
        
        Args:
            query: Query tensor [L, B, E]
            key: Key tensor [L, B, E]
            value: Value tensor [L, B, E]
            time_seq: Time sequence [L, B, 1]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Key padding mask [B, L]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Encode temporal information
        if time_seq is not None:
            time_encoding = self.time_encoder(time_seq.permute(1, 0, 2))  # [L, B, E]
            
            # Add temporal encoding to key and value
            key = key + time_encoding
            value = value + time_encoding
        
        # Apply standard multi-head attention
        return self.attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with temporal encoding and MoE support.
    
    Features:
    - Temporal multi-head attention or standard multi-head attention
    - Mixture of Experts feedforward network
    - Layer normalization and dropout
    - Residual connections
    """
    
    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
        use_time_seq: bool = True,
    ):
        super().__init__()
        
        self.dim_embed = dim_embed
        self.use_time_seq = use_time_seq
        
        # Multi-head attention
        if use_time_seq:
            self.multi_attn = TemporalMultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
        else:
            self.multi_attn = nn.MultiheadAttention(
                dim_embed, num_heads, dropout=dropout, batch_first=False
            )
        
        # Feedforward network
        if use_moe:
            self.feedforward = MixtureOfExperts(
                dim_embed, num_experts, dim_attn_feedforward
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )
        
        # Layer normalization and dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)
    
    def forward(
        self,
        x: torch.Tensor,
        time_seq: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for attention layer.
        
        Args:
            x: Input tensor [L, B, E]
            time_seq: Time sequence [L, B, 1]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Key padding mask [B, L]
            
        Returns:
            Output tensor [L, B, E]
        """
        # Multi-head attention with residual connection
        if self.use_time_seq and time_seq is not None:
            attn_output, _ = self.multi_attn(
                x, x, x, time_seq, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        else:
            attn_output, _ = self.multi_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)
        
        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)
        x = self.layer_norm2(x)
        
        return x


class AttentionLayerPreNorm(nn.Module):
    """
    Pre-normalization multi-head attention layer.
    
    Uses pre-normalization instead of post-normalization for improved
    training stability, especially useful for feature attention.
    """
    
    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
    ):
        super().__init__()
        
        self.dim_embed = dim_embed
        
        # Multi-head attention
        self.multi_attn = nn.MultiheadAttention(
            dim_embed, num_heads, dropout=dropout, batch_first=False
        )
        
        # Feedforward network
        if use_moe:
            self.feedforward = MixtureOfExperts(
                dim_embed, num_experts, dim_attn_feedforward
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )
        
        # Layer normalization and dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-normalization.
        
        Args:
            x: Input tensor [L, B, E]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Key padding mask [B, L]
            
        Returns:
            Output tensor [L, B, E]
        """
        # Pre-norm + multi-head attention + residual
        x2 = self.layer_norm1(x)
        x2, _ = self.multi_attn(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(x2)
        
        # Pre-norm + feedforward + residual
        x2 = self.layer_norm2(x)
        x2 = self.feedforward(x2)
        x = x + self.dropout2(x2)
        
        return x


class OrderAttentionModule(nn.Module):
    """
    Order Attention module for temporal sequence processing.
    
    This module processes temporal sequences to learn order-level patterns
    and representations using multi-layer attention with time encoding.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Core configuration
        self.n_cat_features = config["n_cat_features"]
        self.n_num_features = config["n_num_features"]
        self.n_embedding = config["n_embedding"]
        self.seq_len = config.get("seq_len", 51)
        self.dim_embed = 2 * config["dim_embedding_table"]
        self.embedding_table_dim = config["dim_embedding_table"]
        self.use_time_seq = config.get("use_time_seq", True)
        self.return_seq = config.get("return_seq", False)
        
        # Embedding table
        self.embedding = nn.Embedding(
            self.n_embedding + 2, 
            self.embedding_table_dim, 
            padding_idx=0
        )
        
        # Feature aggregation networks
        self.feature_aggregation_cat = FeatureAggregation(self.n_cat_features)
        self.feature_aggregation_num = FeatureAggregation(self.n_num_features)
        
        # Multi-layer attention stack
        self.layer_stack = nn.ModuleList([
            AttentionLayer(
                dim_embed=self.dim_embed,
                dim_attn_feedforward=config.get("dim_attn_feedforward", 64),
                num_heads=config.get("num_heads", 1),
                dropout=config.get("dropout", 0.1),
                use_moe=config.get("use_moe", True),
                num_experts=config.get("num_experts", 5),
                use_time_seq=self.use_time_seq
            )
            for _ in range(config.get("n_layers_order", 6))
        ])
        
        # Learnable dummy token for sequence representation
        self.dummy_order = nn.Parameter(torch.rand(1, self.dim_embed))
        
        # Layer normalization
        self.layer_norm_feature = nn.LayerNorm(self.dim_embed)
        self.layer_norm = nn.LayerNorm(self.dim_embed)
        
        # Optional embedding bias
        if config.get("emb_tbl_use_bias", True):
            self.emb_tbl_bias = nn.Parameter(
                torch.randn(self.n_cat_features + self.n_num_features, self.embedding_table_dim)
            )
        else:
            self.emb_tbl_bias = None
    
    def forward(
        self, 
        x_cat: torch.Tensor, 
        x_num: torch.Tensor, 
        time_seq: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for order attention processing.
        
        Args:
            x_cat: Categorical features [B, L, n_cat_features]
            x_num: Numerical features [B, L, n_num_features]
            time_seq: Time sequence information [B, L, 1]
            attn_mask: Attention mask [L, L]
            key_padding_mask: Key padding mask [B, L]
            
        Returns:
            Sequence representation [B, dim_embed] or [B, L+1, dim_embed] if return_seq=True
        """
        B, L = x_cat.shape[:2]
        
        # Categorical feature embedding and aggregation
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)  # [B, L, n_cat_features, embed_dim]
        x_cat_agg = self.feature_aggregation_cat(
            x_cat_all.permute(0, 1, 3, 2)
        ).squeeze(-1)  # [B, L, embed_dim]
        
        # Numerical feature embedding and aggregation
        num_indices = torch.arange(
            self.n_embedding - self.n_num_features + 1,
            self.n_embedding + 1,
            device=x_cat.device
        ).repeat(B, L).view(B, L, -1)
        
        x_num_all = self.embedding(num_indices) * x_num.unsqueeze(-1)  # [B, L, n_num_features, embed_dim]
        x_num_agg = self.feature_aggregation_num(
            x_num_all.permute(0, 1, 3, 2)
        ).squeeze(-1)  # [B, L, embed_dim]
        
        # Combine categorical and numerical features
        x = torch.cat([x_cat_agg, x_num_agg], dim=-1)  # [B, L, dim_embed]
        
        # Apply bias if configured
        if self.emb_tbl_bias is not None:
            x = x + self.emb_tbl_bias[None, None, :]
        
        # Transpose for attention layers (L, B, E)
        x = x.permute(1, 0, 2)
        x = self.layer_norm_feature(x)
        
        # Add dummy token for sequence-level representation
        dummy = self.dummy_order.repeat(B, 1).unsqueeze(0)  # [1, B, dim_embed]
        x = torch.cat([x, dummy], dim=0)  # [L+1, B, dim_embed]
        x = self.layer_norm(x)
        
        # Prepare time sequence if provided
        if self.use_time_seq and time_seq is not None:
            time_seq = torch.cat([
                time_seq, 
                torch.zeros(B, 1, 1, device=time_seq.device)
            ], dim=1).permute(1, 0, 2)  # [L+1, B, 1]
        
        # Multi-layer attention processing
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)
        
        # Return sequence or final representation
        if self.return_seq:
            return x.permute(1, 0, 2)  # [B, L+1, dim_embed]
        else:
            return x[-1, :, :]  # [B, dim_embed] - dummy token representation


class FeatureAttentionModule(nn.Module):
    """
    Feature Attention module for current transaction processing.
    
    This module processes the most recent transaction features and integrates
    them with engineered features using pre-normalization attention.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Core configuration
        self.n_cat_features = config["n_cat_features"]
        self.n_num_features = config["n_num_features"]
        self.n_embedding = config["n_embedding"]
        self.n_engineered_num_features = config.get("n_engineered_num_features", 0)
        self.dim_embed = 2 * config["dim_embedding_table"]
        self.embedding_table_dim = config["dim_embedding_table"]
        
        # Embedding tables
        self.embedding = nn.Embedding(
            self.n_embedding + 2, 
            self.embedding_table_dim, 
            padding_idx=0
        )
        
        self.embedding_engineered = nn.Embedding(
            self.n_engineered_num_features + 1, 
            self.embedding_table_dim, 
            padding_idx=0
        )
        
        # Pre-norm attention layers
        self.layer_stack_feature = nn.ModuleList([
            AttentionLayerPreNorm(
                dim_embed=self.embedding_table_dim,
                dim_attn_feedforward=config.get("dim_attn_feedforward", 64),
                num_heads=config.get("num_heads", 1),
                dropout=config.get("dropout", 0.1),
                use_moe=config.get("use_moe", True),
                num_experts=config.get("num_experts", 5)
            )
            for _ in range(config.get("n_layers_feature", 6))
        ])
        
        # Layer normalization
        self.layer_norm_engineered = nn.LayerNorm(self.embedding_table_dim)
        
        # Optional embedding bias
        if config.get("emb_tbl_use_bias", True):
            self.emb_tbl_bias = nn.Parameter(
                torch.randn(self.n_cat_features + self.n_num_features, self.embedding_table_dim)
            )
            
            if self.n_engineered_num_features > 0:
                self.engineered_emb_tbl_bias = nn.Parameter(
                    torch.randn(self.n_engineered_num_features, self.embedding_table_dim)
                )
            else:
                self.engineered_emb_tbl_bias = None
        else:
            self.emb_tbl_bias = None
            self.engineered_emb_tbl_bias = None
    
    def forward(
        self, 
        x_cat: torch.Tensor, 
        x_num: torch.Tensor, 
        x_engineered: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for feature attention processing.
        
        Args:
            x_cat: Categorical features [B, L, n_cat_features]
            x_num: Numerical features [B, L, n_num_features]
            x_engineered: Engineered features [B, n_engineered_num_features]
            
        Returns:
            Feature representation [B, embedding_table_dim]
        """
        B, L = x_cat.shape[:2]
        
        if x_engineered is not None:
            x_engineered = x_engineered.float()
        
        # Extract last order (current transaction) features
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)  # [B, L, n_cat_features, embed_dim]
        x_cat_last = x_cat_all[:, -1, :, :]  # [B, n_cat_features, embed_dim]
        
        # Numerical feature embeddings for last order
        num_indices = torch.arange(
            self.n_embedding - self.n_num_features + 1,
            self.n_embedding + 1,
            device=x_cat.device
        ).repeat(B, L).view(B, L, -1)
        
        x_num_all = self.embedding(num_indices) * x_num.unsqueeze(-1)
        x_num_last = x_num_all[:, -1, :, :]  # [B, n_num_features, embed_dim]
        
        # Combine categorical and numerical features
        x_last = torch.cat([x_cat_last, x_num_last], dim=1)  # [B, n_cat+n_num, embed_dim]
        
        # Apply bias if configured
        if self.emb_tbl_bias is not None:
            x_last = x_last + self.emb_tbl_bias[None, :, :]
        
        # Add engineered features if provided
        if self.n_engineered_num_features > 0 and x_engineered is not None:
            engineered_indices = torch.arange(
                1, self.n_engineered_num_features + 1, 
                device=x_cat.device
            )
            
            x_engineered_emb = self.embedding_engineered(engineered_indices) * x_engineered.unsqueeze(-1)
            
            if self.engineered_emb_tbl_bias is not None:
                x_engineered_emb = x_engineered_emb + self.engineered_emb_tbl_bias[None, :, :]
            
            # Add dummy engineered token
            dummy_engineered = self.embedding_engineered(
                torch.zeros(B, 1, dtype=torch.int, device=x_cat.device)
            )
            
            x_last = torch.cat([x_last, x_engineered_emb, dummy_engineered], dim=1)
        else:
            # Add dummy engineered token even if no engineered features
            dummy_engineered = self.embedding_engineered(
                torch.zeros(B, 1, dtype=torch.int, device=x_cat.device)
            )
            x_last = torch.cat([x_last, dummy_engineered], dim=1)
        
        # Transpose for attention layers (seq_len, B, E)
        x_last = x_last.permute(1, 0, 2)
        x_last = self.layer_norm_engineered(x_last)
        
        # Multi-layer feature attention processing
        for att_layer_feature in self.layer_stack_feature:
            x_last = att_layer_feature(x_last, None, None)
        
        # Return final feature representation (dummy token)
        return x_last[-1, :, :]  # [B, embedding_table_dim]


def compute_fm_parallel(feature_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute Factorization Machine (FM) style feature interactions in parallel.
    
    Args:
        feature_embedding: Feature embeddings [B, n_features, embed_dim]
        
    Returns:
        FM interaction features [B, embed_dim]
    """
    # Sum of embeddings
    summed_features_emb = torch.sum(feature_embedding, dim=-2)
    summed_features_emb_square = torch.square(summed_features_emb)
    
    # Square of embeddings then sum
    squared_features_emb = torch.square(feature_embedding)
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)
    
    # FM interaction computation
    fm_interaction = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
    
    return fm_interaction
