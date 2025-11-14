from pydantic import Field, model_validator, PrivateAttr
from typing import List, Optional, Dict, Any, Union

from .hyperparameters_base import ModelHyperparameters


class XGBoostModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for the XGBoost model training, extending the base ModelHyperparameters.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Most essential XGBoost hyperparameters
    num_round: int = Field(description="The number of boosting rounds for XGBoost.")

    max_depth: int = Field(description="Maximum depth of a tree.")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Override model_class from base
    model_class: str = Field(
        default="xgboost", description="Model class identifier, set to XGBoost."
    )

    min_child_weight: float = Field(
        default=1.0,
        description="Minimum sum of instance weight (hessian) needed in a child.",
    )

    # General Parameters
    booster: str = Field(
        default="gbtree",
        description="Specify which booster to use: gbtree, gblinear or dart.",
    )

    # Booster Parameters
    eta: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Step size shrinkage used in update to prevents overfitting. Alias: learning_rate.",
    )

    gamma: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum loss reduction required to make a further partition on a leaf node of the tree.",
    )

    max_delta_step: float = Field(
        default=0.0,
        description="Maximum delta step we allow each tree's weight estimation to be. If 0, no constraint.",
    )

    subsample: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of the training instances.",
    )

    colsample_bytree: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns when constructing each tree.",
    )

    colsample_bylevel: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns for each level.",
    )

    colsample_bynode: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns for each split.",
    )

    lambda_xgb: float = Field(
        default=1.0,
        ge=0.0,
        description="L2 regularization term on weights. Alias: reg_lambda.",
    )

    alpha_xgb: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 regularization term on weights. Alias: reg_alpha.",
    )

    tree_method: str = Field(
        default="auto", description="The tree construction algorithm used in XGBoost."
    )

    sketch_eps: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="For tree_method 'approx'. Approximately (1 / sketch_eps) buckets are made.",
    )

    scale_pos_weight: float = Field(
        default=1.0,
        description="Control the balance of positive and negative weights, useful for unbalanced classes.",
    )

    num_parallel_tree: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of parallel trees constructed during each iteration. Used for random forests.",
    )

    # Learning Task Parameters
    base_score: Optional[float] = Field(
        default=None,
        description="The initial prediction score of all instances, global bias.",
    )

    seed: Optional[int] = Field(default=None, description="Random number seed.")

    early_stopping_rounds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Activates early stopping. Requires eval_metric.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _objective: Optional[str] = PrivateAttr(default=None)
    _eval_metric: Optional[Union[str, List[str]]] = PrivateAttr(default=None)

    class Config(ModelHyperparameters.Config):
        # Inherit Config from base and modify extra to allow
        extra = "allow"  # Changed from "forbid" to "allow" to fix circular reference handling

    # Public read-only properties for derived fields

    @property
    def objective(self) -> str:
        """Get objective derived from is_binary."""
        if self._objective is None:
            self._objective = "binary:logistic" if self.is_binary else "multi:softmax"
        return self._objective

    @property
    def eval_metric(self) -> List[str]:
        """Get evaluation metrics derived from is_binary."""
        if self._eval_metric is None:
            self._eval_metric = (
                ["logloss", "auc"] if self.is_binary else ["mlogloss", "merror"]
            )
        return self._eval_metric

    @model_validator(mode="after")
    def validate_xgboost_hyperparameters(self) -> "XGBoostModelHyperparameters":
        """Validate XGBoost-specific hyperparameters"""
        # Call the base model validator first
        super().validate_dimensions()

        # Initialize derived fields
        self._objective = "binary:logistic" if self.is_binary else "multi:softmax"
        self._eval_metric = (
            ["logloss", "auc"] if self.is_binary else ["mlogloss", "merror"]
        )

        # Validate multiclass parameters
        if self._objective.startswith("multi:") and self.num_classes < 2:
            raise ValueError(
                f"For multiclass objective '{self._objective}', 'num_classes' must be >= 2. "
                f"Current num_classes: {self.num_classes}"
            )

        # Validate early stopping configuration
        if self.early_stopping_rounds is not None and not self._eval_metric:
            raise ValueError(
                "'early_stopping_rounds' requires 'eval_metric' to be set."
            )

        # Validate GPU usage
        if self.tree_method == "gpu_hist" and self.device == -1:
            print(
                f"Warning: tree_method is '{self.tree_method}' but device is CPU (-1). "
                f"Ensure SageMaker instance is GPU for gpu_hist."
            )

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include XGBoost-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add derived fields that should be exposed
        derived_fields = {"objective": self.objective, "eval_metric": self.eval_metric}

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
