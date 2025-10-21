#!/usr/bin/env python3
"""
Feature Attention Module for PyTorch Lightning TSA implementation.

This module contains the FeatureAttentionModule and its closely associated components
for current transaction processing in TSA models.
"""

import torch
import torch.nn as nn
from typing import Optional

from .pl_tsa_components import AttentionLayerPreNorm


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
