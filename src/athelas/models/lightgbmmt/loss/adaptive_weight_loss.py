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
    main task and subtasks, with optional weight update strategies:

    - None (default): Update at every iteration
    - 'tenIters': Update every 50 iterations
    - 'sqrt': Apply square root dampening to weights
    - 'delta': Incremental updates with delta learning rate
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize weights
        self.weights = self._init_weights()

        # Track weight history
        self.weight_history = [self.weights.copy()]

        # Iteration counter for weight update methods
        self.iteration_count = 0

        # Cached similarity for tenIters and delta methods
        self.cached_similarity = None

        self.logger.info(
            f"Initialized adaptive weights with method={self.weight_method}: {self.weights}"
        )

    def _init_weights(self) -> np.ndarray:
        """Initialize weights uniformly."""
        weights = np.ones(self.num_col) / self.num_col
        return weights

    def compute_weights(
        self, labels_mat: np.ndarray, preds_mat: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Compute adaptive weights based on task similarity.

        Supports multiple weight update strategies:
        - None (default): Update at every iteration
        - 'tenIters': Update every 50 iterations (more stable)
        - 'sqrt': Apply square root dampening (smoother weights)
        - 'delta': Incremental updates (memory of previous weights)

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
        self.iteration_count = iteration

        # Compute raw similarity-based weights
        raw_weights = self._compute_similarity_weights(labels_mat, preds_mat)

        # Apply weight update method
        if self.weight_method == "tenIters":
            weights = self._apply_ten_iters_method(raw_weights, iteration)
        elif self.weight_method == "sqrt":
            weights = self._apply_sqrt_method(raw_weights)
        elif self.weight_method == "delta":
            weights = self._apply_delta_method(raw_weights, iteration)
        else:
            # Standard method: direct use with learning rate
            weights = self._apply_standard_method(raw_weights, iteration)

        # Update stored weights and history
        self.weights = weights
        self.weight_history.append(weights.copy())

        return weights

    def _compute_similarity_weights(
        self, labels_mat: np.ndarray, preds_mat: np.ndarray
    ) -> np.ndarray:
        """
        Compute raw similarity-based weights using JS divergence.

        Returns normalized weights based on inverse JS divergence.
        """
        # Get main task index
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

        return weights

    def _apply_standard_method(
        self, raw_weights: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Standard adaptive weighting with learning rate smoothing.

        Updates at every iteration with exponential moving average.
        """
        if iteration > 0:
            weights = (1 - self.weight_lr) * self.weights + self.weight_lr * raw_weights
        else:
            weights = raw_weights

        return weights

    def _apply_ten_iters_method(
        self, raw_weights: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Update weights every 50 iterations for more stable training.

        Uses cached weights between updates to reduce computational overhead
        and provide smoother weight trajectories.
        """
        # Update every 50 iterations (frequency configurable via hyperparams)
        update_freq = self.weight_update_frequency or 50

        if iteration % update_freq == 0:
            # Compute and cache new weights
            self.cached_similarity = raw_weights
            weights = raw_weights
            self.logger.debug(f"Updated weights at iteration {iteration}")
        else:
            # Use cached weights
            if self.cached_similarity is not None:
                weights = self.cached_similarity
            else:
                # First iteration, use raw weights
                weights = raw_weights
                self.cached_similarity = raw_weights

        return weights

    def _apply_sqrt_method(self, raw_weights: np.ndarray) -> np.ndarray:
        """
        Apply square root dampening to similarity weights.

        Reduces extreme weight values for more stable training.
        Formula: w_dampened = sqrt(w_raw)
        """
        # Apply square root to dampen extreme values
        weights_dampened = np.sqrt(raw_weights)

        # Re-normalize after dampening
        weights = self.normalize(weights_dampened)

        return weights

    def _apply_delta_method(
        self, raw_weights: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Incremental weight updates based on changes (delta).

        Formula: w_new = w_old + delta_lr * (w_raw - w_old)

        Provides smooth adaptation with memory of previous weights.
        """
        if iteration == 0:
            # First iteration, use raw weights
            weights = raw_weights
            self.cached_similarity = raw_weights
        else:
            # Compute delta from previous weights
            if self.cached_similarity is not None:
                delta = raw_weights - self.cached_similarity
                # Apply delta with learning rate
                weights = self.weights + self.delta_lr * delta

                # Ensure weights remain positive and normalized
                weights = np.maximum(weights, self.epsilon_norm)
                weights = self.normalize(weights)
            else:
                weights = raw_weights

            # Cache current raw weights for next iteration
            self.cached_similarity = raw_weights

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
