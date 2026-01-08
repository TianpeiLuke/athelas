from pydantic import Field, model_validator, PrivateAttr
from typing import List, Optional, Dict, Any, Union

from ...core.base.hyperparameters_base import ModelHyperparameters


class LightGBMModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for the LightGBM model training, extending the base ModelHyperparameters.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Most essential LightGBM hyperparameters
    num_leaves: int = Field(description="Maximum number of leaves in one tree.")

    learning_rate: float = Field(description="Learning rate for boosting.")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Override model_class from base
    model_class: str = Field(
        default="lightgbm", description="Model class identifier, set to LightGBM."
    )

    # Core LightGBM Parameters
    boosting_type: str = Field(
        default="gbdt",
        description="Boosting type: gbdt, rf, dart, goss.",
    )

    num_iterations: int = Field(
        default=100,
        ge=1,
        description="Number of boosting iterations.",
    )

    max_depth: int = Field(
        default=-1,
        description="Maximum depth of tree. -1 means no limit.",
    )

    min_data_in_leaf: int = Field(
        default=20,
        ge=1,
        description="Minimum number of data points in one leaf.",
    )

    min_sum_hessian_in_leaf: float = Field(
        default=1e-3,
        ge=0.0,
        description="Minimum sum of hessians in one leaf.",
    )

    # Feature Selection Parameters
    feature_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Feature fraction for each iteration.",
    )

    bagging_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Bagging fraction for each iteration.",
    )

    bagging_freq: int = Field(
        default=0,
        ge=0,
        description="Frequency for bagging. 0 means disable bagging.",
    )

    # Regularization Parameters
    lambda_l1: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 regularization term on weights.",
    )

    lambda_l2: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization term on weights.",
    )

    min_gain_to_split: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum gain to perform split.",
    )

    # Advanced Parameters
    categorical_feature: Optional[str] = Field(
        default=None,
        description="Categorical features specification.",
    )

    early_stopping_rounds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Early stopping rounds. None to disable.",
    )

    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility."
    )

    # Categorical Feature Parameters
    min_data_per_group: int = Field(
        default=100,
        ge=1,
        description="Minimum number of data per categorical group. Used for dealing with overfitting when #data is small or #category is large.",
    )

    cat_smooth: float = Field(
        default=10.0,
        ge=0.0,
        description="Categorical smoothing parameter. Used for reducing noise in categorical features. Larger values lead to stronger smoothing.",
    )

    max_cat_threshold: int = Field(
        default=32,
        ge=1,
        description="Maximum number of categories to consider for splitting. For categories with cardinality > max_cat_threshold, treat as numeric.",
    )

    use_native_categorical: bool = Field(
        default=True,
        description="Whether to use LightGBM's native categorical feature handling. If False, uses risk table mapping (XGBoost-style preprocessing).",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _objective: Optional[str] = PrivateAttr(default=None)
    _metric: Optional[Union[str, List[str]]] = PrivateAttr(default=None)

    model_config = ModelHyperparameters.model_config.copy()
    model_config.update({"extra": "allow"})

    # Public read-only properties for derived fields

    @property
    def objective(self) -> str:
        """Get objective derived from is_binary."""
        if self._objective is None:
            self._objective = "binary" if self.is_binary else "multiclass"
        return self._objective

    @property
    def metric(self) -> List[str]:
        """Get evaluation metrics derived from is_binary."""
        if self._metric is None:
            self._metric = (
                ["binary_logloss", "auc"]
                if self.is_binary
                else ["multi_logloss", "multi_error"]
            )
        return self._metric

    @model_validator(mode="after")
    def validate_lightgbm_hyperparameters(self) -> "LightGBMModelHyperparameters":
        """Validate LightGBM-specific hyperparameters"""
        # Call the base model validator first
        super().validate_dimensions()

        # Initialize derived fields
        self._objective = "binary" if self.is_binary else "multiclass"
        self._metric = (
            ["binary_logloss", "auc"]
            if self.is_binary
            else ["multi_logloss", "multi_error"]
        )

        # Validate multiclass parameters
        if self._objective == "multiclass" and self.num_classes < 2:
            raise ValueError(
                f"For multiclass objective '{self._objective}', 'num_classes' must be >= 2. "
                f"Current num_classes: {self.num_classes}"
            )

        # Validate early stopping configuration
        if self.early_stopping_rounds is not None and not self._metric:
            raise ValueError("'early_stopping_rounds' requires 'metric' to be set.")

        # Validate boosting type
        valid_boosting_types = ["gbdt", "rf", "dart", "goss"]
        if self.boosting_type not in valid_boosting_types:
            raise ValueError(
                f"Invalid boosting_type: {self.boosting_type}. Must be one of: {valid_boosting_types}"
            )

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include LightGBM-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add derived fields that should be exposed
        derived_fields = {"objective": self.objective, "metric": self.metric}

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
