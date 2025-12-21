#!/usr/bin/env python3
"""
PyTorch Lightning-compatible Order Attention Module

Recreates OrderAttentionLayer functionality using modular Lightning-compatible components
while preserving EXACT numerical equivalence with legacy implementation.

Phase 1: Algorithm-Preserving Refactoring
- Recreation using Lightning components from modular files
- NO optimizations or algorithmic changes
- EXACT legacy behavior preservation
- Goal: rtol ≤ 1e-6 numerical equivalence

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/basic_blocks.py (OrderAttentionLayer)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

# Import Lightning-compatible components from modular files
from .pl_attention_layers import AttentionLayer
from .pl_feature_processing import FeatureAggregation


# Configuration class (optional, requires Pydantic V2+)
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict

    class OrderAttentionConfig(BaseModel):
        """
        Configuration schema for OrderAttentionModule.

        Provides validation and defaults for order attention parameters.
        Requires Pydantic V2+.
        """

        # Pydantic V2 configuration
        model_config = ConfigDict(extra="allow")

        # Core architecture parameters
        n_cat_features: int = Field(ge=1, description="Number of categorical features")
        n_num_features: int = Field(ge=0, description="Number of numerical features")
        n_embedding: int = Field(ge=1, description="Size of embedding table")
        seq_len: int = Field(ge=1, description="Maximum sequence length")
        dim_embed: int = Field(ge=1, description="Embedding dimension (output size)")
        dim_attn_feedforward: int = Field(ge=1, description="Feedforward dimension")

        # Attention configuration
        num_heads: int = Field(default=1, ge=1, description="Number of attention heads")
        n_layers_order: int = Field(
            default=6, ge=1, description="Number of attention layers"
        )
        dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")

        # Advanced features
        use_moe: bool = Field(default=True, description="Use Mixture of Experts")
        num_experts: int = Field(
            default=5, ge=1, description="Number of experts (if MoE)"
        )
        use_time_seq: bool = Field(default=True, description="Use temporal encoding")
        emb_tbl_use_bias: bool = Field(default=True, description="Use embedding bias")
        return_seq: bool = Field(default=False, description="Return full sequence")

        @field_validator("dim_embed")
        @classmethod
        def validate_dim_embed(cls, v: int) -> int:
            """Ensure dim_embed is even for splitting between cat and num."""
            if v % 2 != 0:
                raise ValueError(f"dim_embed must be even, got {v}")
            return v

except ImportError:
    # Pydantic not available - use Dict as fallback
    OrderAttentionConfig = Dict


class OrderAttentionModule(nn.Module):
    """
    Lightning-compatible Order Attention module.

    Recreates OrderAttentionLayer functionality using Lightning components
    while preserving EXACT numerical equivalence with legacy implementation.

    Architecture:
        1. Categorical feature embedding via shared embedding table
        2. Numerical feature linear embedding (embedding * value)
        3. Feature-level aggregation via MLP (n_features → 1)
        4. Dummy token appended to sequence
        5. Multi-layer temporal attention (typically 6 layers)
        6. Extract dummy token representation

    Critical Implementation Details:
        - Embedding table is shared with feature attention (MUST use same object)
        - Padding index = 0
        - Dummy token is learnable parameter
        - Time encoding concatenated with zeros for dummy position
        - Feature aggregation BEFORE attention (not after)
        - Post-norm attention layers with temporal encoding

    Phase 1 Constraints:
        - NO caching or optimizations
        - NO formula modifications
        - NO architectural changes
        - EXACT legacy computation preserved

    Args:
        config: Configuration dictionary with parameters:
            - n_cat_features: Number of categorical features
            - n_num_features: Number of numerical features
            - n_embedding: Size of embedding table
            - seq_len: Maximum sequence length
            - dim_embed: Embedding dimension (output dimension)
            - dim_attn_feedforward: Feedforward dimension in attention
            - num_heads: Number of attention heads (default: 1)
            - dropout: Dropout rate (default: 0.1)
            - n_layers_order: Number of attention layers (default: 6)
            - emb_tbl_use_bias: Whether to use embedding bias (default: True)
            - use_moe: Whether to use Mixture of Experts (default: True)
            - num_experts: Number of experts in MoE (default: 5)
            - use_time_seq: Whether to use temporal encoding (default: True)
            - return_seq: Return full sequence or just last token (default: False)

    Example:
        >>> config = {
        ...     "n_cat_features": 53,
        ...     "n_num_features": 47,
        ...     "n_embedding": 1352,
        ...     "seq_len": 51,
        ...     "dim_embed": 256,
        ...     "dim_attn_feedforward": 64,
        ... }
        >>> module = OrderAttentionModule(config)
        >>>
        >>> # Forward pass
        >>> x_cat = torch.randint(0, 100, (32, 51, 53))  # [B, L, D_cat]
        >>> x_num = torch.randn(32, 51, 47)              # [B, L, D_num]
        >>> time_seq = torch.randn(32, 51, 1)            # [B, L, 1]
        >>>
        >>> output = module(x_cat, x_num, time_seq)      # [B, dim_embed]
    """

    def __init__(self, config: Union[Dict, "OrderAttentionConfig"]):
        super().__init__()

        # Convert config to dict if needed
        if hasattr(config, "model_dump"):
            config = config.model_dump()
        elif hasattr(config, "dict"):
            config = config.dict()

        self.config = config

        # Extract configuration parameters
        self.n_cat_features = config["n_cat_features"]
        self.n_num_features = config["n_num_features"]
        self.n_embedding = config["n_embedding"]
        self.seq_len = config["seq_len"]
        self.dim_embed = config["dim_embed"]
        self.dim_attn_feedforward = config["dim_attn_feedforward"]
        self.num_heads = config.get("num_heads", 1)
        self.return_seq = config.get("return_seq", False)
        self.use_time_seq = config.get("use_time_seq", True)

        # Embedding table dimension (half of dim_embed for concatenation)
        self.embedding_table_dim = self.dim_embed // 2

        # CRITICAL: Learnable dummy token for sequence representation
        # This will be appended to the sequence after embedding
        self.dummy_order = nn.Parameter(torch.rand(1, self.dim_embed))

        # Create shared embedding table
        # CRITICAL: This will be shared with FeatureAttentionModule
        # Using padding_idx=0 to match legacy behavior
        self.embedding = nn.Embedding(
            self.n_embedding + 2,  # +2 for padding and extra
            self.embedding_table_dim,
            padding_idx=0,
        )

        # Layer normalization for features (before attention)
        self.layer_norm_feature = nn.LayerNorm(int(self.embedding_table_dim * 2))

        # Stack multiple attention layers
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayer(
                    dim_embed=self.dim_embed,
                    dim_attn_feedforward=self.dim_attn_feedforward,
                    num_heads=self.num_heads,
                    dropout=config.get("dropout", 0.1),
                    use_moe=config.get("use_moe", True),
                    num_experts=config.get("num_experts", 5),
                    use_time_seq=self.use_time_seq,
                )
                for _ in range(config.get("n_layers_order", 6))
            ]
        )

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.layer_norm = nn.LayerNorm(self.dim_embed)

        # Optional embedding bias
        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(
                    self.n_cat_features + self.n_num_features, self.embedding_table_dim
                )
            )
            if config.get("emb_tbl_use_bias", True)
            else None
        )

        # Feature aggregation modules
        # CRITICAL: These aggregate features BEFORE attention
        self.feature_aggregation_cat = FeatureAggregation(self.n_cat_features)
        self.feature_aggregation_num = FeatureAggregation(self.n_num_features)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        time_seq: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through order attention.

        CRITICAL: This preserves EXACT legacy behavior - no modifications.

        Args:
            x_cat: Categorical features [B, L, D_cat]
                - Integer tensor with category indices
                - Padding should use index 0
            x_num: Numerical features [B, L, D_num]
                - Float tensor with numerical values
                - Will be multiplied with embeddings (linear embedding)
            time_seq: Time sequence [B, L, 1]
                - Relative time to last transaction
                - Used for temporal encoding if use_time_seq=True
                - Zeros will be concatenated for dummy token
            attn_mask: Attention mask [L, L] (optional)
                - Mask for attention computation
                - Will be broadcast to [B, L, L]
            key_padding_mask: Padding mask [B, L] (optional)
                - Boolean mask where True indicates padding
                - Used to mask out padding positions

        Returns:
            torch.Tensor: Order attention output [B, dim_embed]
                - If return_seq=False (default): Last token (dummy) representation
                - If return_seq=True: Full sequence [B, L+1, dim_embed]

        Shape Transformations:
            Input: [B, L, D_cat], [B, L, D_num], [B, L, 1]
            ↓ Embedding + Feature Aggregation
            ↓ Concatenate: [B, L, embedding_table_dim * 2]
            ↓ Add Dummy Token: [L+1, B, dim_embed]
            ↓ Multi-layer Temporal Attention (6 layers)
            ↓ Extract Dummy Token or Full Sequence
            Output: [B, dim_embed] or [B, L+1, dim_embed]
        """
        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # ===== 1. Categorical Feature Embedding =====
        # Convert to int indices and embed
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)  # [B, L, D_cat, E]

        # Aggregate categorical features: [D_cat] → [1]
        # [B, L, D_cat, E] → [B, L, E, D_cat] → [B, L, E, 1] → [B, L, E]
        x_cat = self.feature_aggregation_cat(x_cat_all.permute(0, 1, 3, 2)).squeeze(-1)

        # ===== 2. Numerical Feature Embedding =====
        # Create indices for numerical features
        # These are the last n_num_features indices in the embedding table
        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )

        # Linear embedding: embedding * value
        x_num_all = self.embedding(num_indices) * (x_num[..., None])  # [B, L, D_num, E]

        # Aggregate numerical features: [D_num] → [1]
        x_num = self.feature_aggregation_num(x_num_all.permute(0, 1, 3, 2)).squeeze(-1)

        # ===== 3. Concatenate and Normalize =====
        # Combine categorical and numerical embeddings
        x = torch.cat([x_cat, x_num], dim=-1)  # [B, L, embedding_table_dim * 2]

        # Permute for attention: [B, L, E] → [L, B, E]
        x = x.permute(1, 0, 2)

        # Layer norm
        x = self.layer_norm_feature(x)

        # ===== 4. Add Dummy Token =====
        # CRITICAL: Dummy token is appended to sequence
        dummy = (
            self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        )  # [B, 1, dim_embed]
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)  # [L+1, B, dim_embed]
        x = self.layer_norm(x)

        # ===== 5. Prepare Time Encoding =====
        if self.use_time_seq:
            # CRITICAL: Concatenate zeros for dummy token position
            time_seq = torch.cat(
                [time_seq, torch.zeros([B, 1, 1]).to(x.device)], dim=1
            )  # [B, L+1, 1]
            time_seq = time_seq.permute(1, 0, 2)  # [L+1, B, 1]
        else:
            time_seq = None

        # ===== 6. Multi-Layer Attention =====
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)

        # ===== 7. Extract Output =====
        if not self.return_seq:
            # Extract dummy token (last position): [L+1, B, E] → [B, E]
            x = torch.transpose(x, 0, 1)[:, -1, :]
        else:
            # Return full sequence: [L+1, B, E] → [B, L+1, E]
            x = torch.transpose(x, 0, 1)

        return x

    def get_dummy_token(self) -> torch.Tensor:
        """
        Get the learnable dummy token parameter.

        The dummy token is appended to the sequence and serves as a
        representation of the entire sequence after attention.

        Returns:
            torch.Tensor: Dummy token [1, dim_embed]
        """
        return self.dummy_order

    def get_embedding_table(self) -> nn.Embedding:
        """
        Get the shared embedding table.

        CRITICAL: This embedding table MUST be shared with FeatureAttentionModule.
        Use the same object reference, not a copy.

        Returns:
            nn.Embedding: Shared embedding table
        """
        return self.embedding

    def __repr__(self) -> str:
        return (
            f"OrderAttentionModule(\n"
            f"  n_cat_features={self.n_cat_features},\n"
            f"  n_num_features={self.n_num_features},\n"
            f"  n_embedding={self.n_embedding},\n"
            f"  seq_len={self.seq_len},\n"
            f"  dim_embed={self.dim_embed},\n"
            f"  n_layers={len(self.layer_stack)},\n"
            f"  use_time_seq={self.use_time_seq},\n"
            f"  use_moe={self.config.get('use_moe', True)}\n"
            f")"
        )
