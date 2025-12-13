"""
Hyperparameters for LightGBMMT (Multi-Task) model training.

Extends ModelHyperparameters directly with complete LightGBM and multi-task parameters.
"""

from pydantic import Field, model_validator, PrivateAttr
from typing import Optional, Literal
import warnings

from .hyperparameters_base import ModelHyperparameters


class LightGBMMtModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for LightGBMMT (Multi-Task) model training.

    Extends ModelHyperparameters directly (not LightGBMModelHyperparameters).
    Includes complete LightGBM parameters plus multi-task specific parameters.
    All loss function parameters are prefixed with 'loss_' to avoid naming conflicts.

    Follows three-tier hyperparameter pattern:
    - Tier 1: Essential User Inputs (from ModelHyperparameters + LightGBM essentials)
    - Tier 2: System Inputs with Defaults (LightGBM + MT-specific parameters)
    - Tier 3: Derived Fields (enable_kd computed from loss_type, num_tasks from task_label_names)

    Design Notes:
    - No separate LossConfig class - all loss parameters integrated here
    - Loss functions receive this hyperparameters object directly
    - Training parameters (max_epochs, batch_size) inherited from base
    - No TrainingConfig class - only TrainingState for runtime tracking
    """

    # ========================================================================
    # TIER 1: ESSENTIAL USER INPUTS (Required, no defaults)
    # ========================================================================

    # Multi-Task Configuration
    task_label_names: list[str] = Field(
        description=(
            "List of task/label column names for multi-task learning (REQUIRED). "
            "Each column represents one task's binary labels. "
            "Aligns with label_config['output_label_name'] from ruleset generation. "
            "Example: ['isFraud', 'isCCfrd', 'isDDfrd']"
        ),
    )

    main_task_index: int = Field(
        ge=0,
        description="Index of the main task in the task list (e.g., 0 = first task, 1 = second task)",
    )

    # ========================================================================
    # TIER 2: SYSTEM INPUTS WITH DEFAULTS
    # ========================================================================

    # Override model_class
    model_class: str = Field(
        default="lightgbmmt",
        description="Model class identifier for multi-task LightGBM",
    )

    # Essential LightGBM Parameters
    num_leaves: int = Field(
        default=31,
        description="Maximum number of leaves in one tree (LightGBM default: 31)",
    )

    learning_rate: float = Field(
        default=0.1,
        description="Boosting learning rate / shrinkage_rate (LightGBM default: 0.1)",
    )

    # LightGBM Core Parameters
    boosting_type: str = Field(
        default="gbdt", description="Boosting type: gbdt, rf, dart, goss"
    )

    num_iterations: int = Field(
        default=100, ge=1, description="Number of boosting iterations (num_boost_round)"
    )

    max_depth: int = Field(
        default=-1, description="Maximum tree depth (-1 means no limit)"
    )

    min_data_in_leaf: int = Field(
        default=20, ge=1, description="Minimum number of data points in one leaf"
    )

    min_sum_hessian_in_leaf: float = Field(
        default=1e-3, ge=0.0, description="Minimum sum of hessians in one leaf"
    )

    # Feature Selection Parameters
    feature_fraction: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Feature fraction for each iteration"
    )

    bagging_fraction: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Bagging fraction for each iteration"
    )

    bagging_freq: int = Field(
        default=0, ge=0, description="Frequency for bagging (0 means disable bagging)"
    )

    # Regularization Parameters
    lambda_l1: float = Field(
        default=0.0, ge=0.0, description="L1 regularization term on weights"
    )

    lambda_l2: float = Field(
        default=0.0, ge=0.0, description="L2 regularization term on weights"
    )

    min_gain_to_split: float = Field(
        default=0.0, ge=0.0, description="Minimum gain to perform split"
    )

    # Advanced LightGBM Parameters
    categorical_feature: Optional[str] = Field(
        default=None, description="Categorical features specification"
    )

    early_stopping_rounds: Optional[int] = Field(
        default=None, ge=1, description="Early stopping rounds. None to disable"
    )

    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )

    # Loss Function Selection
    loss_type: Literal["fixed", "adaptive", "adaptive_kd"] = Field(
        default="adaptive",
        description="Loss function type: 'fixed' (static weights), 'adaptive' (similarity-based), 'adaptive_kd' (with knowledge distillation)",
    )

    # Loss Configuration Parameters (all prefixed with 'loss_')
    # Numerical stability
    loss_epsilon: float = Field(
        default=1e-15,
        gt=0,
        description="Small constant for numerical stability in sigmoid clipping",
    )

    loss_epsilon_norm: float = Field(
        default=1e-10,
        gt=0,
        description="Epsilon for safe division in normalization operations",
    )

    loss_clip_similarity_inverse: float = Field(
        default=1e10,
        gt=0,
        description="Maximum value for inverse similarity to prevent inf (adaptive loss)",
    )

    # Weight configuration
    loss_beta: float = Field(
        default=0.2,
        ge=0,
        description="Subtask weight scaling factor: subtask_weight = main_weight * beta (fixed loss)",
    )

    loss_main_task_weight: float = Field(
        default=1.0, gt=0, description="Weight for main task in fixed weight loss"
    )

    loss_weight_lr: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description=(
            "Learning rate for adaptive weight updates using Exponential Moving Average (EMA). "
            "Controls how quickly task weights adapt to similarity changes during training.\n\n"
            "Algorithm: w_new = (1 - lr) * w_old + lr * w_raw\n"
            "- w_raw: Raw similarity-based weights computed from Jensen-Shannon divergence\n"
            "- w_old: Previous iteration weights\n"
            "- w_new: Updated weights for current iteration\n\n"
            "Impact on Training:\n"
            "- lr = 1.0 (default): No smoothing, direct weight updates. Matches legacy behavior. "
            "Fast adaptation but may oscillate. Best for stable similarity patterns.\n"
            "- lr = 0.1: Typical smoothing. 10% new weights + 90% old weights. "
            "Balanced between stability and responsiveness. Recommended for most use cases.\n"
            "- lr = 0.01: Heavy smoothing. Very stable weight trajectories but slow adaptation. "
            "Use when similarity patterns are noisy or unstable.\n\n"
            "Trade-offs:\n"
            "- Higher lr (→1.0): Faster adaptation, responsive to changes, but may oscillate/overshoot\n"
            "- Lower lr (→0.0): Smoother trajectories, stable training, but slower to adapt to shifts\n\n"
            "Recommendation: Start with default 1.0 (legacy). If weight oscillations observed, "
            "try 0.1 for improved stability. Use <0.1 only for highly volatile similarity patterns."
        ),
    )

    # Knowledge distillation
    loss_patience: int = Field(
        default=100,
        ge=1,
        description="Number of consecutive performance declines before triggering KD label replacement",
    )

    # Weight update strategy
    loss_weight_method: Optional[Literal["tenIters", "sqrt", "delta"]] = Field(
        default=None,
        description="Weight update strategy: None (every iteration), 'tenIters' (periodic), 'sqrt' (sqrt transform), 'delta' (incremental)",
    )

    loss_weight_update_frequency: int = Field(
        default=50,
        ge=1,
        description="Iterations between weight updates (used with 'tenIters' method)",
    )

    loss_delta_lr: float = Field(
        default=0.01,
        gt=0,
        le=1,
        description=(
            "Learning rate for incremental (delta) weight updates when loss_weight_method='delta'. "
            "Controls the magnitude of weight adjustments based on similarity changes between iterations.\n\n"
            "Algorithm: w_new = w_old + delta_lr * (w_raw - w_cached)\n"
            "- w_raw: Current raw similarity-based weights\n"
            "- w_cached: Previous raw weights from last iteration\n"
            "- delta: Change in raw weights (w_raw - w_cached)\n"
            "- w_new: Updated weights after applying delta\n\n"
            "Impact on Training:\n"
            "- delta_lr = 0.01 (default): Very gradual weight updates. Conservative adaptation with strong "
            "memory of previous weights. Highly stable but slow to respond to changes.\n"
            "- delta_lr = 0.1: Moderate updates. Balances stability with responsiveness. "
            "Good for moderately changing similarity patterns.\n"
            "- delta_lr = 0.5: Aggressive updates. Fast adaptation to similarity changes. "
            "May be unstable if patterns fluctuate rapidly.\n\n"
            "Comparison with loss_weight_lr:\n"
            "- delta method: Focuses on changes (incremental updates based on differences)\n"
            "- standard method: Focuses on absolute values (EMA of raw weights)\n"
            "- delta method provides stronger memory effect and smoother trajectories\n\n"
            "Trade-offs:\n"
            "- Higher delta_lr (→1.0): Faster response to changes, less weight memory, may be unstable\n"
            "- Lower delta_lr (→0.0): Slower adaptation, stronger weight memory, very stable\n\n"
            "Recommendation: Use default 0.01 for stable incremental updates. "
            "Increase to 0.1 if faster adaptation needed. Only use >0.1 for rapidly changing tasks."
        ),
    )

    loss_sqrt_normalize: bool = Field(
        default=True,
        description=(
            "Apply L2 normalization after square root dampening when loss_weight_method='sqrt'. "
            "Controls numerical stability vs exact legacy behavior trade-off.\n\n"
            "Algorithm impact on sqrt method:\n"
            "- True (default): w = normalize(sqrt(w_raw)) - Ensures weights sum to 1.0\n"
            "- False (legacy): w = sqrt(w_raw) - No re-normalization after sqrt\n\n"
            "When True (enhanced, default):\n"
            "- Maintains consistent weight interpretation across iterations\n"
            "- Prevents numerical drift over many iterations\n"
            "- More numerically stable for long training runs\n"
            "- sqrt dampening effect preserved, just normalized to sum=1.0\n\n"
            "When False (exact legacy behavior):\n"
            "- Matches original implementation exactly\n"
            "- Weights may not sum to 1.0 after sqrt transformation\n"
            "- Potential for gradual numerical drift\n"
            "- Use only for exact legacy reproduction or validation\n\n"
            "Mathematical difference:\n"
            "- Both apply sqrt dampening: reduces extreme weight values\n"
            "- True additionally ensures: ||w|| = 1 (L2 norm equals 1)\n"
            "- False allows: ||w|| != 1 (weights may have different magnitudes)\n\n"
            "Recommendation: Use True (default) for improved numerical stability. "
            "Set False only when exact legacy behavior required for validation/migration."
        ),
    )

    # Performance optimization
    loss_cache_predictions: bool = Field(
        default=True,
        description="Enable prediction caching for 30-50% performance improvement",
    )

    loss_precompute_indices: bool = Field(
        default=True,
        description="Precompute index arrays for faster task-specific access",
    )

    # Logging
    loss_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level for loss function operations"
    )

    # ===== Derived Fields (Tier 3) =====
    _enable_kd: Optional[bool] = PrivateAttr(default=None)
    _objective: Optional[str] = PrivateAttr(default=None)
    _metric: Optional[list] = PrivateAttr(default=None)
    _num_tasks: Optional[int] = PrivateAttr(default=None)

    @property
    def num_tasks(self) -> int:
        """Get number of tasks derived from task_label_names."""
        if self._num_tasks is None:
            self._num_tasks = len(self.task_label_names)
        return self._num_tasks

    @property
    def enable_kd(self) -> bool:
        """Whether knowledge distillation is enabled (derived from loss_type)."""
        if self._enable_kd is None:
            self._enable_kd = self.loss_type == "adaptive_kd"
        return self._enable_kd

    @property
    def objective(self) -> str:
        """Get objective derived from is_binary."""
        if self._objective is None:
            self._objective = "binary" if self.is_binary else "multiclass"
        return self._objective

    @property
    def metric(self) -> list:
        """Get evaluation metrics derived from is_binary."""
        if self._metric is None:
            self._metric = (
                ["binary_logloss", "auc"]
                if self.is_binary
                else ["multi_logloss", "multi_error"]
            )
        return self._metric

    @model_validator(mode="after")
    def validate_mt_hyperparameters(self) -> "LightGBMMtModelHyperparameters":
        """Validate multi-task and LightGBM-specific hyperparameters."""
        # Call base validator first
        super().validate_dimensions()

        # Initialize derived fields
        self._num_tasks = len(self.task_label_names)
        self._enable_kd = self.loss_type == "adaptive_kd"
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

        # Validate boosting type
        valid_boosting_types = ["gbdt", "rf", "dart", "goss"]
        if self.boosting_type not in valid_boosting_types:
            raise ValueError(
                f"Invalid boosting_type: {self.boosting_type}. "
                f"Must be one of: {valid_boosting_types}"
            )

        # Validate loss_type
        valid_loss_types = ["fixed", "adaptive", "adaptive_kd"]
        if self.loss_type not in valid_loss_types:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. "
                f"Must be one of: {valid_loss_types}"
            )

        # Validate weight_method
        valid_methods = [None, "tenIters", "sqrt", "delta"]
        if self.loss_weight_method not in valid_methods:
            raise ValueError(
                f"Invalid loss_weight_method: {self.loss_weight_method}. "
                f"Must be one of: {valid_methods}"
            )

        # Validate beta
        if self.loss_beta > 1.0:
            warnings.warn(
                f"loss_beta > 1.0 ({self.loss_beta}) gives subtasks higher weight than main task",
                UserWarning,
                stacklevel=2,
            )

        # Validate patience with KD
        if self.enable_kd and self.loss_patience < 10:
            warnings.warn(
                f"Small patience ({self.loss_patience}) with KD enabled may cause "
                f"premature label replacement",
                UserWarning,
                stacklevel=2,
            )

        # Validate num_tasks if provided
        if self.num_tasks is not None:
            if self.num_tasks < 2:
                raise ValueError(
                    f"num_tasks must be >= 2 (1 main + at least 1 subtask), got {self.num_tasks}"
                )
            if self.main_task_index >= self.num_tasks:
                raise ValueError(
                    f"main_task_index ({self.main_task_index}) must be < num_tasks ({self.num_tasks})"
                )

        # Validate early stopping configuration
        if self.early_stopping_rounds is not None and not self._metric:
            raise ValueError("'early_stopping_rounds' requires 'metric' to be set")

        return self

    def get_public_init_fields(self) -> dict:
        """Override to include MT-specific and LightGBM-specific derived fields."""
        base_fields = super().get_public_init_fields()
        derived_fields = {
            "num_tasks": self.num_tasks,
            "enable_kd": self.enable_kd,
            "objective": self.objective,
            "metric": self.metric,
        }
        return {**base_fields, **derived_fields}
