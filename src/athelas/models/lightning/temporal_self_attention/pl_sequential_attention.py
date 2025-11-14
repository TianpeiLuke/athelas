#!/usr/bin/env python3
"""
Sequential Attention Module for PyTorch Lightning TSA implementation.

This module contains the SequentialAttentionModule and its closely associated components
for temporal sequence processing in TSA models.
"""

import torch
import torch.nn as nn
from typing import Optional

from .pl_tsa_components import (
    TimeEncode,
    FeatureAggregation,
    MixtureOfExperts,
    TemporalMultiheadAttention,
    AttentionLayer
)


class SequentialAttentionModule(nn.Module):
    """
    Sequential Attention module for temporal sequence processing.
    
    This module processes temporal sequences to learn sequential patterns
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
        Forward pass for sequential attention processing.
        
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
