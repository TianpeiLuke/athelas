"""
Adaptive weight loss function for LightGBMMT.

Uses similarity-based dynamic task weighting based on JS divergence.
"""

from typing import Optional, Any, Tuple
import numpy as np
from scipy.spatial.distance import jensenshannon

from .base_loss_function import BaseLossFunction


class AdaptiveWeightLoss(BaseLossFunction):
    """
    Adaptive weight loss with similarity-based weighting.

    Computes task weights based on Jensen-Shannon divergence between
    main task and subtasks, with optional weight update strategies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize weights
        self.weights = self._init_weights()

        # Track weight history
        self.weight_history = [self.weights.copy()]

        self.logger.info(f"Initialized adaptive weights: {self.weights}")

    def _init_weights(self) -> np.ndarray:
        """Initialize weights uniformly."""
        weights = np.ones(self.num_col) / self.num_col
        return weights

    def compute_weights(
        self, labels_mat: np.ndarray, preds_mat: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Compute adaptive weights based on task similarity.

        Parameters
        ----------
        labels_mat : np.ndarray
            Label matrix [N_samples, N_tasks]
        preds_mat : np.ndarray
            Prediction matrix [N_samples, N_tasks]
        iteration : int
            Current iteration number

        Returns
        -------
        weights : np.ndarray
            Computed task weights [N_tasks]
        """
        # Get main task index from hyperparameters (defaults to 0 for backward compatibility)
        main_idx = getattr(self.hyperparams, "main_task_index", 0)

        # Compute similarity between main task and subtasks
        main_pred = preds_mat[:, main_idx]
        similarities = np.zeros(self.num_col)
        similarities[main_idx] = 1.0  # Main task has similarity 1 with itself

        for i in range(self.num_col):
            if i == main_idx:
                continue  # Skip main task
            subtask_pred = preds_mat[:, i]

            # Compute Jensen-Shannon divergence
            js_div = jensenshannon(main_pred, subtask_pred)

            # Convert to similarity (inverse with clipping)
            if js_div < self.epsilon_norm:
                similarity = 1.0
            else:
                similarity = 1.0 / js_div
                similarity = min(similarity, self.clip_similarity_inverse)

            similarities[i] = similarity

        # Normalize similarities to get weights
        weights = self.normalize(similarities)

        # Apply weight learning rate
        if iteration > 0:
            weights = (1 - self.weight_lr) * self.weights + self.weight_lr * weights

        # Update stored weights
        self.weights = weights
        self.weight_history.append(weights.copy())

        return weights

    def objective(
        self, preds: np.ndarray, train_data: Any, ep: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute adaptive weighted gradients and hessians."""
        # Preprocess
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)

        # Compute per-task gradients and hessians
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)

        # Compute adaptive weights (pass iteration as 0 for now - will be updated in training loop)
        weights = self.compute_weights(labels_mat, preds_mat, iteration=0)

        # Weight and aggregate
        weights_reshaped = weights.reshape(1, -1)
        grad = (grad_i * weights_reshaped).sum(axis=1)
        hess = (hess_i * weights_reshaped).sum(axis=1)

        return grad, hess, grad_i, hess_i
