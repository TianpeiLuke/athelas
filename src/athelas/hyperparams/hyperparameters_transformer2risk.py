from pydantic import Field, model_validator
from .hyperparameters_base import ModelHyperparameters


class Transformer2RiskHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for Transformer2Risk bimodal fraud detection model.

    This class extends the base ModelHyperparameters with Transformer-specific
    architecture parameters needed for the Transformer2Risk model which combines:
    - Transformer encoder with self-attention for text sequence encoding
    - MLP for tabular feature encoding
    - Bimodal fusion for fraud prediction

    Key architectural differences from LSTM2Risk:
    - Uses self-attention mechanism instead of recurrent connections
    - Larger embedding dimensions (128 vs 16) for richer representations
    - Fixed-length sequences with positional embeddings (vs variable-length LSTM)
    - Multi-head attention for parallel attention to different aspects

    Inherits all base fields including:
    - Data field management (full_field_list, cat_field_list, tab_field_list)
    - Training parameters (lr, batch_size, max_epochs, optimizer)
    - Classification parameters (multiclass_categories, class_weights)
    - Derived properties (input_tab_dim, num_classes, is_binary)

    Example Usage:
    ```python
    hyperparam = Transformer2RiskHyperparameters(
        # Essential fields (Tier 1) - required
        full_field_list=["name", "email", "age", "income", "label"],
        cat_field_list=["name", "email"],
        tab_field_list=["age", "income"],
        id_name="customer_id",
        label_name="label",
        multiclass_categories=[0, 1],

        # Transformer-specific fields (Tier 2) - optional, using defaults
        embedding_size=128,
        hidden_size=256,
        n_embed=4000,
        n_blocks=8,
        n_heads=8,
        block_size=100,
        dropout_rate=0.2,

        # Can also override base fields
        lr=3e-5,
        batch_size=32,
        max_epochs=5
    )

    # Access derived properties
    print(f"Input tabular dimension: {hyperparam.input_tab_dim}")
    print(f"Number of classes: {hyperparam.num_classes}")
    print(f"Is binary classification: {hyperparam.is_binary}")

    # Serialize for SageMaker
    config = hyperparam.serialize_config()
    ```
    """

    # ===== System Inputs with Defaults (Tier 2) =====
    # Override model_class from base to identify this as Transformer2Risk

    model_class: str = Field(
        default="transformer2risk",
        description="Model class identifier for this hyperparameter configuration",
    )

    # ===== Transformer-Specific Architecture Parameters (Tier 2) =====
    # These parameters define the Transformer2Risk model architecture

    embedding_size: int = Field(
        default=128,
        gt=0,
        le=512,
        description="Token and position embedding dimension. "
        "Significantly larger than LSTM (128 vs 16) since transformers "
        "benefit from higher-dimensional embeddings for effective self-attention. "
        "Must be divisible by n_heads.",
    )

    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Dropout probability for regularization throughout the model. "
        "Applied in attention layers, feedforward networks, tabular projection, "
        "and classifier. Higher values provide more regularization but may underfit.",
    )

    hidden_size: int = Field(
        default=256,
        gt=0,
        le=1024,
        description="Hidden dimension for tabular feature projection. "
        "Text encoder projects embedding_size to 2*hidden_size. "
        "Combined bimodal representation is 4*hidden_size. "
        "Larger than LSTM (256 vs 128) to match increased model capacity.",
    )

    n_embed: int = Field(
        default=4000,
        gt=0,
        le=100000,
        description="Vocabulary size for token embeddings. "
        "Must match the tokenizer vocabulary size. "
        "Typically determined by BPE tokenizer training.",
    )

    n_blocks: int = Field(
        default=8,
        gt=0,
        le=24,
        description="Number of stacked transformer encoder blocks. "
        "Each block contains multi-head self-attention and feedforward network. "
        "More blocks increase model capacity but also computational cost. "
        "Typical range: 6-12 for medium-sized models.",
    )

    n_heads: int = Field(
        default=8,
        gt=0,
        le=16,
        description="Number of attention heads per transformer block. "
        "Must divide embedding_size evenly (head_size = embedding_size / n_heads). "
        "Multiple heads allow model to attend to different representation subspaces. "
        "Common values: 8, 12, 16 for standard architectures.",
    )

    block_size: int = Field(
        default=100,
        gt=0,
        le=512,
        description="Maximum sequence length for positional embeddings. "
        "Sequences longer than this will be truncated. "
        "Determines the size of the learned position embedding table. "
        "Should be set based on typical input text lengths (names/emails).",
    )

    # ===== Training and Optimization Parameters (Tier 2) =====
    # These parameters control the optimization process

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

    @model_validator(mode="after")
    def validate_transformer_hyperparameters(self) -> "Transformer2RiskHyperparameters":
        """Validate transformer-specific constraints."""
        # Call base validator first
        super().validate_dimensions()

        # Validate embedding_size is divisible by n_heads
        if self.embedding_size % self.n_heads != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible by "
                f"n_heads ({self.n_heads}) for multi-head attention. "
                f"Current head_size would be {self.embedding_size / self.n_heads:.2f}"
            )

        return self
