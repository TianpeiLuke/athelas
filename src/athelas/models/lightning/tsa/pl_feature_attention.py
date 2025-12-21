#!/usr/bin/env python3
"""
PyTorch Lightning-compatible Feature Attention Module

Recreates FeatureAttentionLayer functionality using modular Lightning-compatible components
while preserving EXACT numerical equivalence with legacy implementation.

Phase 1: Algorithm-Preserving Refactoring
- Recreation using Lightning components from modular files
- NO optimizations or algorithmic changes
- EXACT legacy behavior preservation
- Goal: rtol ≤ 1e-6 numerical equivalence

Related Documents:
- Design: slipbox/1_design/tsa_lightning_refactoring_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/basic_blocks.py (FeatureAttentionLayer)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

# Import Lightning-compatible components from modular files
from .pl_attention_layers import AttentionLayerPreNorm
from .pl_feature_processing import compute_FM_parallel


# Configuration class (optional, requires Pydantic V2+)
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict

    class FeatureAttentionConfig(BaseModel):
        """
        Configuration schema for FeatureAttentionModule.

        Provides validation and defaults for feature attention parameters.
        Requires Pydantic V2+.
        """

        # Pydantic V2 configuration
        model_config = ConfigDict(extra="allow")

        # Core architecture parameters
        n_cat_features: int = Field(ge=1, description="Number of categorical features")
        n_num_features: int = Field(ge=0, description="Number of numerical features")
        n_embedding: int = Field(ge=1, description="Size of embedding table")
        n_engineered_num_features: int = Field(
            default=0, ge=0, description="Number of engineered features"
        )
        dim_embed: int = Field(ge=1, description="Embedding dimension (output size)")
        dim_attn_feedforward: int = Field(ge=1, description="Feedforward dimension")

        # Attention configuration
        num_heads: int = Field(default=1, ge=1, description="Number of attention heads")
        n_layers_feature: int = Field(
            default=6, ge=1, description="Number of attention layers"
        )
        dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")

        # Advanced features
        use_moe: bool = Field(default=True, description="Use Mixture of Experts")
        num_experts: int = Field(
            default=5, ge=1, description="Number of experts (if MoE)"
        )
        use_fm: bool = Field(default=False, description="Use Factorization Machine")
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
    FeatureAttentionConfig = Dict


class FeatureAttentionModule(nn.Module):
    """
    Lightning-compatible Feature Attention module.

    Recreates FeatureAttentionLayer functionality using Lightning components
    while preserving EXACT numerical equivalence with legacy implementation.

    Architecture:
        1. Extract last order (current transaction) features
        2. Categorical feature embedding via shared embedding table
        3. Numerical feature linear embedding (embedding * value)
        4. Optional FM (Factorization Machine) computation
        5. Combine with engineered features
        6. Multi-layer pre-norm attention (typically 6 layers)
        7. Extract final token representation

    Critical Implementation Details:
        - Embedding table is shared with order attention (MUST use same object)
        - Padding index = 0
        - Uses PRE-NORM attention (different from order attention's POST-NORM)
        - Engineered features have separate embedding
        - FM computation is optional
        - Extracts LAST order for feature processing

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
            - n_engineered_num_features: Number of engineered features (default: 0)
            - dim_embed: Embedding dimension (output dimension)
            - dim_attn_feedforward: Feedforward dimension in attention
            - num_heads: Number of attention heads (default: 1)
            - dropout: Dropout rate (default: 0.1)
            - n_layers_feature: Number of attention layers (default: 6)
            - emb_tbl_use_bias: Whether to use embedding bias (default: True)
            - use_moe: Whether to use Mixture of Experts (default: True)
            - num_experts: Number of experts in MoE (default: 5)
            - use_fm: Whether to use Factorization Machine (default: False)
            - return_seq: Return full sequence or just last token (default: False)

    Example:
        >>> config = {
        ...     "n_cat_features": 53,
        ...     "n_num_features": 47,
        ...     "n_embedding": 1352,
        ...     "n_engineered_num_features": 100,
        ...     "dim_embed": 256,
        ...     "dim_attn_feedforward": 64,
        ... }
        >>> module = FeatureAttentionModule(config)
        >>>
        >>> # Forward pass
        >>> x_cat = torch.randint(0, 100, (32, 51, 53))  # [B, L, D_cat]
        >>> x_num = torch.randn(32, 51, 47)              # [B, L, D_num]
        >>> x_engineered = torch.randn(32, 100)          # [B, D_eng]
        >>>
        >>> output = module(x_cat, x_num, x_engineered)  # [B, dim_embed // 2]
    """

    def __init__(self, config: Union[Dict, "FeatureAttentionConfig"]):
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
        self.n_engineered_num_features = config.get("n_engineered_num_features", 0)
        self.dim_embed = config["dim_embed"]
        self.dim_attn_feedforward = config["dim_attn_feedforward"]
        self.num_heads = config.get("num_heads", 1)
        self.return_seq = config.get("return_seq", False)
        self.use_fm = config.get("use_fm", False)

        # Embedding table dimension (half of dim_embed for concatenation)
        self.embedding_table_dim = self.dim_embed // 2

        # Create shared embedding table
        # CRITICAL: This will be shared with OrderAttentionModule
        # Using padding_idx=0 to match legacy behavior
        self.embedding = nn.Embedding(
            self.n_embedding + 2,  # +2 for padding and extra
            self.embedding_table_dim,
            padding_idx=0,
        )

        # Engineered features embedding (separate from main embedding)
        if self.n_engineered_num_features > 0:
            self.embedding_engineered = nn.Embedding(
                self.n_engineered_num_features, self.embedding_table_dim, padding_idx=0
            )

        # Stack multiple pre-norm attention layers
        # CRITICAL: Uses PRE-NORM (AttentionLayerPreNorm), not post-norm
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayerPreNorm(
                    dim_embed=self.embedding_table_dim,  # Note: uses half of dim_embed
                    dim_attn_feedforward=self.dim_attn_feedforward,
                    num_heads=self.num_heads,
                    dropout=config.get("dropout", 0.1),
                    use_moe=config.get("use_moe", True),
                    num_experts=config.get("num_experts", 5),
                )
                for _ in range(config.get("n_layers_feature", 6))
            ]
        )

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.layer_norm = nn.LayerNorm(self.embedding_table_dim)

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

        # Engineered features bias (if engineered features are used)
        if self.n_engineered_num_features > 0:
            self.emb_tbl_bias_engineered = (
                nn.Parameter(
                    torch.randn(
                        self.n_engineered_num_features, self.embedding_table_dim
                    )
                )
                if config.get("emb_tbl_use_bias", True)
                else None
            )

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        x_engineered: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through feature attention.

        CRITICAL: This preserves EXACT legacy behavior - no modifications.

        Args:
            x_cat: Categorical features [B, L, D_cat]
                - Integer tensor with category indices
                - Padding should use index 0
            x_num: Numerical features [B, L, D_num]
                - Float tensor with numerical values
                - Will be multiplied with embeddings (linear embedding)
            x_engineered: Engineered features [B, D_eng] (optional)
                - Static engineered features for current transaction
                - If None, creates zero tensor
            attn_mask: Attention mask [N, N] (optional)
                - Mask for attention computation
                - Will be broadcast to [B, N, N]
            key_padding_mask: Padding mask [B, N] (optional)
                - Boolean mask where True indicates padding
                - Used to mask out padding positions

        Returns:
            torch.Tensor: Feature attention output [B, embedding_table_dim]
                - If return_seq=False (default): Last token representation
                - If return_seq=True: Full sequence [B, N, embedding_table_dim]

        Shape Transformations:
            Input: [B, L, D_cat], [B, L, D_num], [B, D_eng]
            ↓ Extract Last Order: [B, D_cat], [B, D_num]
            ↓ Embedding: [B, D_cat, E], [B, D_num, E]
            ↓ Optional FM: [B, E]
            ↓ Combine with Engineered: [B, D_cat+D_num+D_eng+1, E]
            ↓ Multi-layer Pre-Norm Attention (6 layers)
            ↓ Extract Last Token or Full Sequence
            Output: [B, E] or [B, N, E]
        """
        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # ===== 1. Extract Last Order (Current Transaction) =====
        # CRITICAL: Feature attention processes ONLY the last order
        x_cat_last = x_cat[:, -1, :]  # [B, D_cat]
        x_num_last = x_num[:, -1, :]  # [B, D_num]

        # ===== 2. Categorical Feature Embedding =====
        # Convert to int indices and embed
        cat_indices = x_cat_last.int()  # [B, D_cat]
        x_cat_emb = self.embedding(cat_indices)  # [B, D_cat, E]

        # Apply bias if present
        if self.emb_tbl_bias is not None:
            x_cat_emb = x_cat_emb + self.emb_tbl_bias[: self.n_cat_features].unsqueeze(
                0
            )

        # ===== 3. Numerical Feature Embedding =====
        # Create indices for numerical features
        # These are the last n_num_features indices in the embedding table
        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, 1)
            .to(x_cat.device)
        )

        # Linear embedding: embedding * value
        x_num_emb = (
            self.embedding(num_indices) * (x_num_last[..., None])
        )  # [B, D_num, E]

        # Apply bias if present
        if self.emb_tbl_bias is not None:
            x_num_emb = x_num_emb + self.emb_tbl_bias[self.n_cat_features :].unsqueeze(
                0
            )

        # ===== 4. Optional Factorization Machine =====
        if self.use_fm:
            # Combine cat and num embeddings
            feature_emb = torch.cat(
                [x_cat_emb, x_num_emb], dim=1
            )  # [B, D_cat+D_num, E]

            # Compute FM
            fm_output = compute_FM_parallel(feature_emb.unsqueeze(1))  # [B, 1, E]
            fm_output = fm_output.squeeze(1).unsqueeze(1)  # [B, 1, E]

            # Concatenate FM with features
            x = torch.cat([feature_emb, fm_output], dim=1)  # [B, D_cat+D_num+1, E]
        else:
            # Just concatenate cat and num embeddings
            x = torch.cat([x_cat_emb, x_num_emb], dim=1)  # [B, D_cat+D_num, E]

        # ===== 5. Engineered Features Integration =====
        if x_engineered is not None and self.n_engineered_num_features > 0:
            # Create indices for engineered features
            eng_indices = (
                torch.arange(self.n_engineered_num_features).repeat(B, 1).to(x.device)
            )

            # Embed engineered features
            x_eng_emb = self.embedding_engineered(eng_indices)  # [B, D_eng, E]

            # Multiply by engineered values (linear embedding)
            x_eng_emb = x_eng_emb * x_engineered.unsqueeze(-1)  # [B, D_eng, E]

            # Apply bias if present
            if self.emb_tbl_bias_engineered is not None:
                x_eng_emb = x_eng_emb + self.emb_tbl_bias_engineered.unsqueeze(0)

            # Concatenate with sequence features
            x = torch.cat([x, x_eng_emb], dim=1)  # [B, D_cat+D_num(+1)+D_eng, E]

        # ===== 6. Add Final Dummy Token =====
        # CRITICAL: Add a learnable dummy token at the end
        # This serves as the final representation after attention
        dummy = torch.zeros(B, 1, self.embedding_table_dim).to(x.device)
        x = torch.cat(
            [x, dummy], dim=1
        )  # [B, N, E] where N = D_cat+D_num(+1)(+D_eng)+1

        # Permute for attention: [B, N, E] → [N, B, E]
        x = x.permute(1, 0, 2)

        # Layer norm
        x = self.layer_norm(x)

        # ===== 7. Multi-Layer Pre-Norm Attention =====
        # CRITICAL: Uses PRE-NORM attention layers
        for att_layer in self.layer_stack:
            x = att_layer(x, attn_mask, key_padding_mask)

        # ===== 8. Extract Output =====
        if not self.return_seq:
            # Extract last token (dummy token): [N, B, E] → [B, E]
            x = torch.transpose(x, 0, 1)[:, -1, :]
        else:
            # Return full sequence: [N, B, E] → [B, N, E]
            x = torch.transpose(x, 0, 1)

        return x

    def get_embedding_table(self) -> nn.Embedding:
        """
        Get the shared embedding table.

        CRITICAL: This embedding table MUST be shared with OrderAttentionModule.
        Use the same object reference, not a copy.

        Returns:
            nn.Embedding: Shared embedding table
        """
        return self.embedding

    def __repr__(self) -> str:
        return (
            f"FeatureAttentionModule(\n"
            f"  n_cat_features={self.n_cat_features},\n"
            f"  n_num_features={self.n_num_features},\n"
            f"  n_engineered_num_features={self.n_engineered_num_features},\n"
            f"  n_embedding={self.n_embedding},\n"
            f"  dim_embed={self.dim_embed},\n"
            f"  n_layers={len(self.layer_stack)},\n"
            f"  use_fm={self.use_fm},\n"
            f"  use_moe={self.config.get('use_moe', True)}\n"
            f")"
        )
