from pydantic import Field, model_validator, PrivateAttr, ConfigDict
from typing import List, Dict, Any, Optional

from ...core.base.hyperparameters_base import ModelHyperparameters


class TemporalSelfAttentionHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for Temporal Self-Attention (TSA) model training,
    extending the base ModelHyperparameters.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Sequence dimensions
    n_embedding: int = Field(
        description="Vocabulary size for categorical embeddings (number of unique tokens)"
    )

    n_cat_features: int = Field(
        description="Number of categorical features per timestep in the sequence"
    )

    n_num_features: int = Field(
        description="Number of numerical/continuous features per timestep in the sequence"
    )

    seq_len: int = Field(description="Maximum sequence length (number of timesteps)")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Override model_class from base
    model_class: str = Field(
        default="temporal_self_attention", description="Model class identifier"
    )

    # Data field key names (consistent seq_* naming)
    seq_cat_key: str = Field(
        default="x_seq_cat",
        description="Key name for categorical sequence features in batch dictionary",
    )

    seq_num_key: str = Field(
        default="x_seq_num",
        description="Key name for numerical sequence features in batch dictionary",
    )

    seq_time_key: str = Field(
        default="time_seq",
        description="Key name for temporal/time sequence features in batch dictionary",
    )

    engineered_key: str = Field(
        default="x_engineered",
        description="Key name for engineered features in batch dictionary",
    )

    # Architecture - Embedding dimensions
    dim_embedding_table: int = Field(
        default=128, gt=0, description="Embedding dimension for categorical features"
    )

    dim_attn_feedforward: int = Field(
        default=512, gt=0, description="Feedforward dimension in attention layers"
    )

    num_heads: int = Field(
        default=8, ge=1, description="Number of attention heads in multi-head attention"
    )

    # Architecture - Model depth
    n_layers_order: int = Field(
        default=2,
        ge=1,
        description="Number of order attention layers (temporal sequence processing)",
    )

    n_layers_feature: int = Field(
        default=2,
        ge=1,
        description="Number of feature attention layers (current transaction processing)",
    )

    # Regularization
    dropout: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Dropout rate for regularization"
    )

    # Mixture of Experts (MoE) parameters
    use_moe: bool = Field(
        default=False, description="Enable Mixture of Experts in feedforward layers"
    )

    num_experts: int = Field(
        default=4,
        ge=1,
        description="Number of experts in MoE layer (when use_moe=True)",
    )

    expert_capacity_factor: float = Field(
        default=1.25,
        ge=1.0,
        description="Capacity factor for expert assignment (affects load balancing)",
    )

    expert_dropout: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Dropout rate for expert outputs"
    )

    # Temporal encoding
    use_time_seq: bool = Field(
        default=True, description="Enable temporal encoding for sequences"
    )

    time_encoding_dim: int = Field(
        default=32, ge=1, description="Dimension for temporal encoding"
    )

    # Attention output control
    return_seq: bool = Field(
        default=False,
        description="Return full sequence from order attention (True) or pooled output (False)",
    )

    # Padding
    use_key_padding_mask: bool = Field(
        default=True, description="Use padding mask for variable-length sequences"
    )

    # Loss function configuration
    loss: str = Field(
        default="CrossEntropyLoss",
        description="Loss function type: CrossEntropyLoss, FocalLoss, or CyclicalFocalLoss",
    )

    loss_alpha: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Alpha parameter for Focal Loss (class balance weight)",
    )

    loss_gamma: float = Field(
        default=2.0,
        ge=0.0,
        description="Gamma parameter for Focal Loss (focusing parameter)",
    )

    loss_gamma_min: float = Field(
        default=1.0, ge=0.0, description="Minimum gamma for Cyclical Focal Loss"
    )

    loss_gamma_max: float = Field(
        default=3.0, ge=0.0, description="Maximum gamma for Cyclical Focal Loss"
    )

    loss_cycle_length: int = Field(
        default=1000,
        ge=1,
        description="Cycle length for Cyclical Focal Loss (in steps)",
    )

    loss_reduction: str = Field(
        default="mean", description="Loss reduction method: mean, sum, or none"
    )

    # Training and Optimization parameters
    weight_decay: float = Field(
        default=0.0, ge=0.0, description="Weight decay for optimizer (L2 penalty)"
    )

    adam_epsilon: float = Field(
        default=1e-8,
        gt=0.0,
        description="Epsilon for Adam optimizer (numerical stability)",
    )

    warmup_steps: int = Field(
        default=300,
        ge=0,
        le=10000,
        description="Warmup steps for learning rate scheduler",
    )

    run_scheduler: bool = Field(
        default=True, description="Run learning rate scheduler flag"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _model_config_dict: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # Explicitly define the model_config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def model_config_dict(self) -> Dict[str, Any]:
        """Get complete model configuration dictionary derived from hyperparameters."""
        if self._model_config_dict is None:
            self._model_config_dict = {
                # Sequence dimensions
                "n_embedding": self.n_embedding,
                "n_cat_features": self.n_cat_features,
                "n_num_features": self.n_num_features,
                "seq_len": self.seq_len,
                # Data field key names
                "seq_cat_key": self.seq_cat_key,
                "seq_num_key": self.seq_num_key,
                "seq_time_key": self.seq_time_key,
                "engineered_key": self.engineered_key,
                # Architecture
                "dim_embedding_table": self.dim_embedding_table,
                "dim_attn_feedforward": self.dim_attn_feedforward,
                "num_heads": self.num_heads,
                "n_layers_order": self.n_layers_order,
                "n_layers_feature": self.n_layers_feature,
                "dropout": self.dropout,
                # Mixture of Experts
                "use_moe": self.use_moe,
                "num_experts": self.num_experts,
                "expert_capacity_factor": self.expert_capacity_factor,
                "expert_dropout": self.expert_dropout,
                # Temporal encoding
                "use_time_seq": self.use_time_seq,
                "time_encoding_dim": self.time_encoding_dim,
                "return_seq": self.return_seq,
                "use_key_padding_mask": self.use_key_padding_mask,
                # Loss function
                "loss": self.loss,
                "loss_alpha": self.loss_alpha,
                "loss_gamma": self.loss_gamma,
                "loss_gamma_min": self.loss_gamma_min,
                "loss_gamma_max": self.loss_gamma_max,
                "loss_cycle_length": self.loss_cycle_length,
                "loss_reduction": self.loss_reduction,
                # Training
                "weight_decay": self.weight_decay,
                "adam_epsilon": self.adam_epsilon,
                "warmup_steps": self.warmup_steps,
                "run_scheduler": self.run_scheduler,
                # From base class (inherited from ModelHyperparameters)
                "id_name": self.id_name,
                "label_name": self.label_name,
                "is_binary": self.is_binary,
                "num_classes": self.num_classes,
                "class_weights": self.class_weights,
                "metric_choices": self.metric_choices,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "device": self.device,
                "model_class": self.model_class,
            }
        return self._model_config_dict

    @model_validator(mode="after")
    def validate_tsa_hyperparameters(self) -> "TemporalSelfAttentionHyperparameters":
        """Validate TSA-specific hyperparameters and initialize derived fields."""
        # Call the base model validator first to initialize its derived fields
        super().validate_dimensions()

        # Initialize derived fields
        self._model_config_dict = None

        # Perform TSA-specific validations
        if self.num_heads > self.dim_embedding_table:
            raise ValueError(
                f"num_heads ({self.num_heads}) cannot exceed dim_embedding_table ({self.dim_embedding_table})"
            )

        if self.dim_embedding_table % self.num_heads != 0:
            raise ValueError(
                f"dim_embedding_table ({self.dim_embedding_table}) must be divisible by num_heads ({self.num_heads})"
            )

        # Validate loss function parameters
        if self.loss == "FocalLoss" or self.loss == "CyclicalFocalLoss":
            if self.loss_alpha < 0.0 or self.loss_alpha > 1.0:
                raise ValueError(
                    f"loss_alpha must be between 0.0 and 1.0, got {self.loss_alpha}"
                )

        if self.loss == "CyclicalFocalLoss":
            if self.loss_gamma_min >= self.loss_gamma_max:
                raise ValueError(
                    f"loss_gamma_min ({self.loss_gamma_min}) must be less than loss_gamma_max ({self.loss_gamma_max})"
                )

        # Validate MoE parameters
        if self.use_moe:
            if self.num_experts < 2:
                raise ValueError(
                    f"When use_moe=True, num_experts must be at least 2, got {self.num_experts}"
                )

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include TSA-specific derived fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add TSA-specific fields that should be exposed
        # (Currently all fields are already captured by parent's logic,
        # but we can add custom derived fields here if needed)
        derived_fields = {
            # Add any TSA-specific derived fields that should be exposed
        }

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}

    def get_trainer_config(self) -> Dict[str, Any]:
        """
        Get trainer configuration dictionary for PyTorch Lightning.
        This combines various trainer-related settings.

        Returns:
            Dict[str, Any]: Configuration dictionary for trainer
        """
        return {
            "max_epochs": self.max_epochs,
            "gpus": self.device if self.device >= 0 else 0,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "run_scheduler": self.run_scheduler,
            "weight_decay": self.weight_decay,
            "adam_epsilon": self.adam_epsilon,
        }
