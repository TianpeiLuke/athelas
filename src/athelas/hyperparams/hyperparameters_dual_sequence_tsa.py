from pydantic import Field, model_validator, ConfigDict
from typing import Dict, Any, Optional

from .hyperparameters_tsa import TemporalSelfAttentionHyperparameters


class DualSequenceTSAHyperparameters(TemporalSelfAttentionHyperparameters):
    """
    Hyperparameters for Dual-Sequence Temporal Self-Attention (TSA) model training,
    extending TemporalSelfAttentionHyperparameters.

    Adds support for dual-sequence processing with a gate function that dynamically
    weights the importance of primary vs secondary sequences.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Additional Tier 2 Fields for Dual-Sequence =====
    # These extend the base TSA hyperparameters with dual-sequence specific configuration

    # Override model_class from parent
    model_class: str = Field(
        default="dual_sequence_tsa", description="Model class identifier"
    )

    # Dual sequence field key names (consistent seq* naming with seq1/seq2)
    seq1_cat_key: str = Field(
        default="x_seq_cat_primary",
        description="Key for primary sequence categorical features in batch dictionary",
    )

    seq1_num_key: str = Field(
        default="x_seq_num_primary",
        description="Key for primary sequence numerical features in batch dictionary",
    )

    seq1_time_key: str = Field(
        default="time_seq_primary",
        description="Key for primary sequence temporal features in batch dictionary",
    )

    seq2_cat_key: str = Field(
        default="x_seq_cat_secondary",
        description="Key for secondary sequence categorical features in batch dictionary",
    )

    seq2_num_key: str = Field(
        default="x_seq_num_secondary",
        description="Key for secondary sequence numerical features in batch dictionary",
    )

    seq2_time_key: str = Field(
        default="time_seq_secondary",
        description="Key for secondary sequence temporal features in batch dictionary",
    )

    # Gate function parameters
    gate_embedding_dim: int = Field(
        default=16,
        ge=1,
        description="Embedding dimension for gate function (typically smaller than main embedding)",
    )

    gate_hidden_dim: int = Field(
        default=256,
        ge=1,
        description="Hidden dimension for gate score computation network",
    )

    gate_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Threshold for secondary sequence filtering (sequences with gate score below this are skipped)",
    )

    @property
    def model_config_dict(self) -> Dict[str, Any]:
        """
        Get complete model configuration including dual-sequence params.
        Extends parent's model_config_dict with dual-sequence specific fields.
        """
        # Get parent config first (includes all base TSA configurations)
        config = super().model_config_dict

        # Add dual-sequence specific fields
        config.update(
            {
                # Dual sequence field keys
                "seq1_cat_key": self.seq1_cat_key,
                "seq1_num_key": self.seq1_num_key,
                "seq1_time_key": self.seq1_time_key,
                "seq2_cat_key": self.seq2_cat_key,
                "seq2_num_key": self.seq2_num_key,
                "seq2_time_key": self.seq2_time_key,
                # Gate function parameters
                "gate_embedding_dim": self.gate_embedding_dim,
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_threshold": self.gate_threshold,
            }
        )

        return config

    @model_validator(mode="after")
    def validate_dual_sequence_hyperparameters(
        self,
    ) -> "DualSequenceTSAHyperparameters":
        """
        Validate dual-sequence specific hyperparameters.
        Calls parent validator first, then adds dual-sequence specific checks.
        """
        # Call parent validator first (validates all base TSA parameters)
        super().validate_tsa_hyperparameters()

        # Dual-sequence specific validations
        if self.gate_embedding_dim > self.dim_embedding_table:
            raise ValueError(
                f"gate_embedding_dim ({self.gate_embedding_dim}) should not exceed "
                f"dim_embedding_table ({self.dim_embedding_table}) - gate function uses simpler embeddings"
            )

        if self.gate_hidden_dim < self.gate_embedding_dim:
            raise ValueError(
                f"gate_hidden_dim ({self.gate_hidden_dim}) should be at least "
                f"gate_embedding_dim ({self.gate_embedding_dim})"
            )

        # Validate gate threshold is in valid range
        if not (0.0 <= self.gate_threshold <= 1.0):
            raise ValueError(
                f"gate_threshold must be between 0.0 and 1.0, got {self.gate_threshold}"
            )

        return self
