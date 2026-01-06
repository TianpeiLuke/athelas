from pydantic import Field, model_validator, PrivateAttr, ConfigDict
from typing import List, Dict, Any, Optional, Union

from .hyperparameters_base import ModelHyperparameters


class BimodalModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for bimodal model training (text + tabular),
    extending the base ModelHyperparameters.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # For BERT model configuration
    tokenizer: str = Field(
        description="Tokenizer name or path (e.g., from Hugging Face)"
    )

    # For text field specification
    text_name: str = Field(description="Name of the primary text field to be processed")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Override model_class from base
    model_class: str = Field(
        default="multimodal_bert", description="Model class identifier"
    )

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

    # Early stopping and Checkpointing parameters
    early_stop_metric: str = Field(
        default="val_loss", description="Metric for early stopping"
    )

    early_stop_patience: int = Field(
        default=3, gt=0, le=10, description="Patience for early stopping"
    )

    load_ckpt: bool = Field(default=False, description="Load checkpoint flag")

    # Preprocessing parameters
    smooth_factor: float = Field(
        default=0.0, description="Risk table smoothing factor for categorical encoding"
    )

    count_threshold: int = Field(
        default=0, description="Risk table count threshold for categorical encoding"
    )

    # Text Preprocessing and Tokenization parameters
    text_field_overwrite: bool = Field(
        default=False,
        description="Overwrite text field if it exists (e.g. during feature engineering)",
    )

    # For chunking long texts
    chunk_trancate: bool = Field(
        default=True, description="Chunk truncation flag for long texts"
    )  # Typo 'trancate' kept as per original

    max_total_chunks: int = Field(
        default=3, description="Maximum total chunks for processing long texts"
    )

    # For tokenizer settings
    max_sen_len: int = Field(
        default=512, description="Maximum sentence length for tokenizer"
    )

    fixed_tokenizer_length: bool = Field(
        default=True, description="Use fixed tokenizer length"
    )

    text_input_ids_key: str = Field(
        default="input_ids", description="Key name for input_ids from tokenizer output"
    )

    text_attention_mask_key: str = Field(
        default="attention_mask",
        description="Key name for attention_mask from tokenizer output",
    )

    # Text processing pipeline configuration
    text_processing_steps: List[str] = Field(
        default=[
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ],
        description="Processing steps for text preprocessing pipeline",
    )

    # Model structure parameters
    # For Convolutional layers
    num_channels: List[int] = Field(
        default=[100, 100], description="Number of channels for convolutional layers"
    )

    num_layers: int = Field(
        default=2,
        description="Number of layers in the model (e.g., BiLSTM, Transformer encoders)",
    )

    dropout_keep: float = Field(default=0.1, description="Dropout keep probability")

    kernel_size: List[int] = Field(
        default=[3, 5, 7], description="Kernel sizes for convolutional layers"
    )

    is_embeddings_trainable: bool = Field(
        default=True, description="Trainable embeddings flag"
    )

    # For BERT fine-tuning
    pretrained_embedding: bool = Field(
        default=True, description="Use pretrained embeddings"
    )

    reinit_layers: int = Field(
        default=2, description="Number of layers to reinitialize from pretrained model"
    )

    reinit_pooler: bool = Field(
        default=True, description="Reinitialize pooler layer flag"
    )

    # For Multimodal BERT
    hidden_common_dim: int = Field(
        default=100, description="Common hidden dimension for multimodal model"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _model_config_dict: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _tokenizer_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)

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
                "hidden_common_dim": self.hidden_common_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout_keep,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "trainable_embeddings": self.is_embeddings_trainable,
                "pretrained": self.pretrained_embedding,
                "reinit_layers": self.reinit_layers,
                "reinit_pooler": self.reinit_pooler,
            }
        return self._model_config_dict

    @property
    def tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration dictionary derived from hyperparameters."""
        if self._tokenizer_config is None:
            self._tokenizer_config = {
                "name": self.tokenizer,
                "max_length": self.max_sen_len,
                "fixed_length": self.fixed_tokenizer_length,
                "text_field": self.text_name,
                "input_ids_key": self.text_input_ids_key,
                "attention_mask_key": self.text_attention_mask_key,
            }
        return self._tokenizer_config

    @model_validator(mode="after")
    def validate_bimodal_hyperparameters(self) -> "BimodalModelHyperparameters":
        """Validate bimodal model-specific hyperparameters and initialize derived fields."""
        # Call the base model validator first to initialize its derived fields
        super().validate_dimensions()

        # Initialize derived fields
        self._model_config_dict = None
        self._tokenizer_config = None

        # Perform bimodal-specific validations
        if len(self.num_channels) != self.num_layers:
            raise ValueError(
                f"Length of num_channels ({len(self.num_channels)}) must match num_layers ({self.num_layers})"
            )

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include bimodal-specific derived fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add derived fields that should be exposed
        derived_fields = {
            # If you need to expose any derived fields, add them here
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
            "gradient_clip_val": self.gradient_clip_val,
            "val_check_interval": self.val_check_interval,
            "precision": 16 if self.fp16 else 32,
            "early_stop_metric": self.early_stop_metric,
            "early_stop_patience": self.early_stop_patience,
        }
