from pydantic import Field, model_validator, PrivateAttr, ConfigDict
from typing import List, Dict, Any, Optional, Union

from ...core.base.hyperparameters_base import ModelHyperparameters


class TriModalHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for tri-modal model training with dual text and tabular modalities.
    Extends ModelHyperparameters to support multiple text inputs.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Override model_class for tri-modal
    model_class: str = Field(
        default="trimodal_bert", description="Model class identifier for tri-modal BERT"
    )

    # Dual text field specification
    primary_text_name: str = Field(
        description="Name of the primary text field (e.g., chat conversation)"
    )

    secondary_text_name: str = Field(
        description="Name of the secondary text field (e.g., shiptrack events)"
    )

    # Backward compatibility field for bi-modal models
    text_name: Optional[str] = Field(
        default=None,
        description="Legacy text field name for backward compatibility with bi-modal models",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Training and Optimization parameters
    lr_decay: float = Field(default=0.05, description="Learning rate decay")

    momentum: float = Field(
        default=0.9, description="Momentum for SGD optimizer (if SGD is chosen)"
    )

    weight_decay: float = Field(
        default=0.0, description="Weight decay for optimizer (L2 penalty)"
    )

    adam_epsilon: float = Field(default=1e-08, description="Epsilon for Adam optimizer")

    warmup_steps: int = Field(
        default=300,
        gt=0,
        le=1000,
        description="Warmup steps for learning rate scheduler",
    )

    run_scheduler: bool = Field(
        default=True, description="Run learning rate scheduler flag"
    )

    val_check_interval: float = Field(
        default=0.25,
        description="Validation check interval during training (float for fraction of epoch, int for steps)",
    )

    early_stop_metric: str = Field(
        default="val_loss", description="Metric for early stopping"
    )

    early_stop_patience: int = Field(
        default=3, gt=0, le=10, description="Patience for early stopping"
    )

    load_ckpt: bool = Field(default=False, description="Load checkpoint flag")

    gradient_clip_val: float = Field(
        default=1.0,
        description="Value for gradient clipping to prevent exploding gradients",
    )

    fp16: bool = Field(
        default=False,
        description="Enable 16-bit mixed precision training (requires compatible hardware)",
    )

    use_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage at the cost of ~20% slower training",
    )

    # Preprocessing parameters
    smooth_factor: float = Field(
        default=0.0, description="Risk table smoothing factor for categorical encoding"
    )

    count_threshold: int = Field(
        default=0, description="Risk table count threshold for categorical encoding"
    )

    # BERT/Text specific fields
    tokenizer: str = Field(
        default="bert-base-cased",
        description="Tokenizer name or path (e.g., from Hugging Face)",
    )

    max_sen_len: int = Field(
        default=512, description="Maximum sentence length for tokenizer"
    )

    fixed_tokenizer_length: bool = Field(
        default=True, description="Use fixed tokenizer length"
    )

    hidden_common_dim: int = Field(
        default=256, description="Common hidden dimension for encoders"
    )

    reinit_pooler: bool = Field(
        default=True, description="Reinitialize BERT pooler layer"
    )

    reinit_layers: int = Field(
        default=2, description="Number of BERT layers to reinitialize"
    )

    # Text processing parameters
    chunk_trancate: bool = Field(
        default=True, description="Chunk truncation flag for long texts"
    )

    max_total_chunks: int = Field(
        default=3, description="Maximum total chunks for processing long texts"
    )

    # Tokenizer output keys (unified for both text modalities with single tokenizer)
    text_input_ids_key: str = Field(
        default="input_ids", description="Key name for input_ids from tokenizer output"
    )

    text_attention_mask_key: str = Field(
        default="attention_mask",
        description="Key name for attention_mask from tokenizer output",
    )

    # Processing pipeline configuration
    primary_text_processing_steps: List[str] = Field(
        default=[
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ],
        description="Processing steps for primary text (e.g., chat with HTML/emoji)",
    )

    secondary_text_processing_steps: List[str] = Field(
        default=[
            "dialogue_splitter",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ],
        description="Processing steps for secondary text (e.g., structured shiptrack events)",
    )

    # Optional separate hidden dimensions (fallback to main hidden_common_dim)
    primary_hidden_common_dim: Optional[int] = Field(
        default=None,
        description="Hidden dimension for primary text encoder (falls back to hidden_common_dim if None)",
    )

    secondary_hidden_common_dim: Optional[int] = Field(
        default=None,
        description="Hidden dimension for secondary text encoder (falls back to hidden_common_dim if None)",
    )

    # Fusion network configuration
    fusion_hidden_dim: Optional[int] = Field(
        default=None,
        description="Hidden dimension for fusion network (auto-calculated if None)",
    )

    fusion_dropout: float = Field(
        default=0.1, description="Dropout rate for fusion network"
    )

    # Optional separate BERT fine-tuning settings
    primary_reinit_pooler: Optional[bool] = Field(
        default=None,
        description="Reinitialize primary BERT pooler (falls back to reinit_pooler if None)",
    )

    primary_reinit_layers: Optional[int] = Field(
        default=None,
        description="Number of primary BERT layers to reinitialize (falls back to reinit_layers if None)",
    )

    secondary_reinit_pooler: Optional[bool] = Field(
        default=None,
        description="Reinitialize secondary BERT pooler (falls back to reinit_pooler if None)",
    )

    secondary_reinit_layers: Optional[int] = Field(
        default=None,
        description="Number of secondary BERT layers to reinitialize (falls back to reinit_layers if None)",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _trimodal_model_config_dict: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # Explicitly define the model_config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def trimodal_model_config_dict(self) -> Dict[str, Any]:
        """Get complete tri-modal model configuration dictionary."""
        if self._trimodal_model_config_dict is None:
            # Get base config from parent's get_config method
            base_config = self.get_config()
            self._trimodal_model_config_dict = {
                **base_config,
                # Tri-modal specific configuration
                "chat_text_name": self.primary_text_name,
                "shiptrack_text_name": self.secondary_text_name,
                "chat_tokenizer": self.tokenizer,
                "shiptrack_tokenizer": self.tokenizer,
                "chat_hidden_common_dim": self.primary_hidden_common_dim
                or self.hidden_common_dim,
                "shiptrack_hidden_common_dim": self.secondary_hidden_common_dim
                or self.hidden_common_dim,
                # Single tokenizer means unified output keys for both text modalities (inherited from BimodalModelHyperparameters)
                "chat_text_input_ids_key": self.text_input_ids_key,
                "chat_text_attention_mask_key": self.text_attention_mask_key,
                "shiptrack_text_input_ids_key": self.text_input_ids_key,
                "shiptrack_text_attention_mask_key": self.text_attention_mask_key,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "fusion_dropout": self.fusion_dropout,
                "chat_reinit_pooler": self.primary_reinit_pooler
                if self.primary_reinit_pooler is not None
                else self.reinit_pooler,
                "chat_reinit_layers": self.primary_reinit_layers
                if self.primary_reinit_layers is not None
                else self.reinit_layers,
                "shiptrack_reinit_pooler": self.secondary_reinit_pooler
                if self.secondary_reinit_pooler is not None
                else self.reinit_pooler,
                "shiptrack_reinit_layers": self.secondary_reinit_layers
                if self.secondary_reinit_layers is not None
                else self.reinit_layers,
                # Add text processing fields
                "max_sen_len": self.max_sen_len,
                "chunk_trancate": self.chunk_trancate,
                "max_total_chunks": self.max_total_chunks,
            }
        return self._trimodal_model_config_dict

    @model_validator(mode="after")
    def validate_trimodal_hyperparameters(self) -> "TriModalHyperparameters":
        """Validate tri-modal specific hyperparameters and initialize derived fields."""
        # Call parent validator first
        super().validate_dimensions()

        # Tri-modal specific validations
        if self.primary_text_name == self.secondary_text_name:
            raise ValueError(
                "primary_text_name and secondary_text_name must be different"
            )

        # Initialize derived fields
        self._trimodal_model_config_dict = None

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include tri-modal specific derived fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add tri-modal derived fields that should be exposed
        derived_fields = {
            "trimodal_model_config_dict": self.trimodal_model_config_dict,
        }

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
