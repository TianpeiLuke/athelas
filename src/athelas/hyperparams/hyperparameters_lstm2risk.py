from pydantic import Field
from .hyperparameters_base import ModelHyperparameters


class LSTM2RiskHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for LSTM2Risk bimodal fraud detection model.

    This class extends the base ModelHyperparameters with LSTM-specific
    architecture parameters needed for the LSTM2Risk model which combines:
    - Bidirectional LSTM for text sequence encoding (names, emails)
    - MLP for tabular feature encoding
    - Bimodal fusion for fraud prediction

    Inherits all base fields including:
    - Data field management (full_field_list, cat_field_list, tab_field_list)
    - Training parameters (lr, batch_size, max_epochs, optimizer)
    - Classification parameters (multiclass_categories, class_weights)
    - Derived properties (input_tab_dim, num_classes, is_binary)

    Example Usage:
    ```python
    hyperparam = LSTM2RiskHyperparameters(
        # Essential fields (Tier 1) - required
        full_field_list=["name", "email", "age", "income", "label"],
        cat_field_list=["name", "email"],
        tab_field_list=["age", "income"],
        id_name="customer_id",
        label_name="label",
        multiclass_categories=[0, 1],

        # LSTM-specific fields (Tier 2) - optional, using defaults
        embedding_size=16,
        hidden_size=128,
        n_embed=4000,
        n_lstm_layers=4,
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
    # Override model_class from base to identify this as LSTM2Risk

    model_class: str = Field(
        default="lstm2risk",
        description="Model class identifier for this hyperparameter configuration",
    )

    # ===== LSTM-Specific Architecture Parameters (Tier 2) =====
    # These parameters define the LSTM2Risk model architecture

    embedding_size: int = Field(
        default=16,
        gt=0,
        le=512,
        description="Token embedding dimension for text encoding. "
        "Controls the size of learned embeddings for vocabulary tokens. "
        "Larger values capture more semantic information but increase parameters.",
    )

    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Dropout probability for regularization throughout the model. "
        "Applied in LSTM layers, tabular projection, and classifier. "
        "Higher values provide more regularization but may underfit.",
    )

    hidden_size: int = Field(
        default=128,
        gt=0,
        le=1024,
        description="LSTM hidden state dimension. "
        "Bidirectional LSTM outputs 2*hidden_size features. "
        "This dimension is also used for tabular feature projection. "
        "Combined bimodal representation is 4*hidden_size.",
    )

    n_embed: int = Field(
        default=4000,
        gt=0,
        le=100000,
        description="Vocabulary size for token embeddings. "
        "Must match the tokenizer vocabulary size. "
        "Typically determined by BPE tokenizer training.",
    )

    n_lstm_layers: int = Field(
        default=4,
        gt=0,
        le=10,
        description="Number of stacked LSTM layers. "
        "More layers can capture more complex patterns but increase training time. "
        "Dropout is applied between layers when n_lstm_layers > 1.",
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
