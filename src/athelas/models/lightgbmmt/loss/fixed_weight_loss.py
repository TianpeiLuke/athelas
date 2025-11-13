"""
Fixed weight loss function for LightGBMMT.

Uses static task weights: main task weight and subtask weights scaled by beta.
"""

from typing import Optional, Any, Tuple
import numpy as np

from .base_loss_function import BaseLossFunction


class FixedWeightLoss(BaseLossFunction):
    """
    Fixed weight loss with dynamic weight generation.

    Generates weight vector: [main_weight, β, β, ..., β]
    where β = main_weight * beta parameter

    Supports any number of tasks (not hardcoded to specific count).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = self._generate_weights()
        self.logger.info(f"Fixed weights: {self.weights}")

    def _generate_weights(self) -> np.ndarray:
        """Generate weight vector dynamically based on num_col and main_task_index."""
        weights = np.zeros(self.num_col)

        # Get main task index from hyperparameters (defaults to 0 for backward compatibility)
        main_idx = getattr(self.hyperparams, "main_task_index", 0)

        # Set main task weight
        weights[main_idx] = self.main_task_weight

        # Set subtask weights (all tasks except main task)
        subtask_weight = self.main_task_weight * self.beta
        for i in range(self.num_col):
            if i != main_idx:
                weights[i] = subtask_weight

        return weights

    def compute_weights(
        self, labels_mat: np.ndarray, preds_mat: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Return fixed weights (no adaptation)."""
        return self.weights

    def objective(
        self, preds: np.ndarray, train_data: Any, ep: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute weighted gradients and hessians."""
        # Preprocess
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)

        # Compute per-task gradients and hessians
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)

        # Weight and aggregate
        weights = self.weights.reshape(1, -1)
        grad = (grad_i * weights).sum(axis=1)
        hess = (hess_i * weights).sum(axis=1)

        return grad, hess, grad_i, hess_i
