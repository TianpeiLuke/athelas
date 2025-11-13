"""
Runtime training state for checkpointing and resumption.

Tracks dynamic training progress without configuration parameters.
Configuration lives in LightGBMMtModelHyperparameters.
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Dict, Optional, Any
import numpy as np


class TrainingState(BaseModel):
    """
    Runtime state for training (enables checkpointing and resumption).

    This tracks dynamic training state, NOT configuration parameters.
    Configuration parameters are in LightGBMMtModelHyperparameters.

    Design Pattern: State pattern for tracking runtime progress
    Uses Pydantic v2 BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )

    # Training progress
    current_epoch: int = Field(default=0, ge=0, description="Current training epoch")

    current_iteration: int = Field(
        default=0, ge=0, description="Current iteration within epoch"
    )

    # Best performance tracking
    best_metric: float = Field(default=0.0, description="Best metric value achieved")

    best_epoch: int = Field(
        default=0, ge=0, description="Epoch when best metric was achieved"
    )

    best_iteration: int = Field(
        default=0, ge=0, description="Iteration when best metric was achieved"
    )

    # Training history
    training_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Training metrics history"
    )

    validation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation metrics history"
    )

    # Multi-task specific
    weight_evolution: List[np.ndarray] = Field(
        default_factory=list, description="Evolution of task weights over training"
    )

    per_task_metrics: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-task metrics history"
    )

    # Early stopping
    epochs_without_improvement: int = Field(
        default=0, ge=0, description="Number of consecutive epochs without improvement"
    )

    patience_triggered: bool = Field(
        default=False, description="Whether early stopping patience has been triggered"
    )

    # KD state (for adaptive_kd loss)
    kd_active: bool = Field(
        default=False, description="Whether knowledge distillation is currently active"
    )

    kd_trigger_epoch: Optional[int] = Field(
        default=None, description="Epoch when KD was triggered"
    )

    def should_stop_early(self, patience: int) -> bool:
        """
        Check if early stopping should be triggered.

        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping

        Returns
        -------
        should_stop : bool
            True if patience exceeded
        """
        return self.epochs_without_improvement >= patience

    def update_best(self, metric: float, epoch: int, iteration: int) -> bool:
        """
        Update best metric if current is better.

        Parameters
        ----------
        metric : float
            Current metric value
        epoch : int
            Current epoch
        iteration : int
            Current iteration

        Returns
        -------
        improved : bool
            True if metric improved
        """
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_iteration = iteration
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """
        Serialize state for checkpointing.

        Uses model_dump() for base fields and custom serialization for numpy arrays.

        Returns
        -------
        checkpoint : dict
            Serializable checkpoint dictionary
        """
        checkpoint = self.model_dump()
        # Convert numpy arrays to lists for JSON serialization
        checkpoint["weight_evolution"] = [w.tolist() for w in self.weight_evolution]
        return checkpoint

    @classmethod
    def from_checkpoint_dict(cls, checkpoint: Dict[str, Any]) -> "TrainingState":
        """
        Deserialize state from checkpoint.

        Converts list representations back to numpy arrays.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary

        Returns
        -------
        state : TrainingState
            Restored training state
        """
        # Convert weight evolution back to numpy arrays
        if "weight_evolution" in checkpoint:
            checkpoint["weight_evolution"] = [
                np.array(w) for w in checkpoint["weight_evolution"]
            ]

        return cls(**checkpoint)

    @model_validator(mode="after")
    def validate_consistency(self) -> "TrainingState":
        """Validate state consistency."""
        # Ensure best_epoch/iteration are not greater than current
        if self.best_epoch > self.current_epoch:
            raise ValueError(
                f"best_epoch ({self.best_epoch}) cannot be greater than "
                f"current_epoch ({self.current_epoch})"
            )

        return self
