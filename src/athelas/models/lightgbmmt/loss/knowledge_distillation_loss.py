"""
Knowledge Distillation loss function for LightGBMMT.

Extends adaptive weighting with knowledge distillation for struggling tasks.
"""

from typing import Optional, Any, Tuple
import numpy as np

from .adaptive_weight_loss import AdaptiveWeightLoss


class KnowledgeDistillationLoss(AdaptiveWeightLoss):
    """
    Knowledge Distillation loss extending adaptive weights.

    Monitors task performance and triggers KD (label replacement) when
    a task shows consistent performance decline.

    Uses BEST predictions (highest observed performance) for label replacement,
    following the design specification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # KD tracking state
        self.kd_active = False
        self.kd_trigger_iteration = None
        self.performance_history = {i: [] for i in range(self.num_col)}
        self.decline_count = {i: 0 for i in range(self.num_col)}

        # Track best predictions and scores for each task
        self.best_predictions = {i: None for i in range(self.num_col)}
        self.best_scores = {i: 0.0 for i in range(self.num_col)}
        self.best_iteration = {i: 0 for i in range(self.num_col)}

        # Track previous predictions to identify best model
        self.previous_predictions = {i: None for i in range(self.num_col)}

        # Track which tasks have been replaced
        self.replaced = {i: False for i in range(self.num_col)}

        # Current iteration counter
        self.current_iteration = 0

        self.logger.info(
            "Initialized KD loss with patience={} (best prediction tracking)".format(
                self.patience
            )
        )

    def _check_kd_trigger(self, task_scores: np.ndarray, iteration: int) -> None:
        """
        Check if KD should be triggered for any task.

        Tracks best scores and predictions for each task. When a task shows
        consistent decline (patience exceeded), marks it for KD replacement.

        Parameters
        ----------
        task_scores : np.ndarray
            Current per-task performance scores
        iteration : int
            Current iteration number
        """
        for task_id in range(self.num_col):
            # Skip if already replaced
            if self.replaced[task_id]:
                continue

            # Track performance history
            self.performance_history[task_id].append(task_scores[task_id])
            current_score = task_scores[task_id]

            # Update best score and predictions
            if current_score > self.best_scores[task_id]:
                self.best_scores[task_id] = current_score
                self.best_iteration[task_id] = iteration
                # Best predictions will be stored after this evaluation
                # (from previous_predictions at best_iteration)
                self.decline_count[task_id] = 0  # Reset counter on improvement
                self.logger.debug(
                    f"Task {task_id} new best score: {current_score:.4f} at iteration {iteration}"
                )
            else:
                # Performance did not improve
                self.decline_count[task_id] += 1

            # Trigger KD if patience exceeded
            if self.decline_count[task_id] >= self.patience:
                if not self.replaced[task_id]:
                    self.replaced[task_id] = True
                    self.logger.warning(
                        f"!TASK {task_id} replaced at iteration {iteration}, "
                        f"counter: {self.decline_count[task_id]}, "
                        f"best score: {self.best_scores[task_id]:.4f} "
                        f"from iteration {self.best_iteration[task_id]}"
                    )

    def _store_predictions(self, preds_mat: np.ndarray, iteration: int) -> None:
        """
        Store current predictions for best model tracking.

        If this iteration matches a task's best iteration, store these predictions
        as the best predictions for that task.

        Parameters
        ----------
        preds_mat : np.ndarray
            Current prediction matrix [N_samples, N_tasks]
        iteration : int
            Current iteration number
        """
        for task_id in range(self.num_col):
            # Store current predictions as previous for next iteration
            self.previous_predictions[task_id] = preds_mat[:, task_id].copy()

            # If we just found a new best in evaluation, store those predictions
            if iteration == self.best_iteration[task_id]:
                self.best_predictions[task_id] = preds_mat[:, task_id].copy()
                self.logger.debug(
                    f"Stored best predictions for task {task_id} at iteration {iteration}"
                )

    def _apply_kd(self, labels_mat: np.ndarray, preds_mat: np.ndarray) -> np.ndarray:
        """
        Apply knowledge distillation by replacing labels with BEST predictions.

        Uses the predictions from the iteration where each task achieved its
        highest validation score, not the current predictions.

        Parameters
        ----------
        labels_mat : np.ndarray
            Original label matrix [N_samples, N_tasks]
        preds_mat : np.ndarray
            Current prediction matrix [N_samples, N_tasks]

        Returns
        -------
        labels_kd : np.ndarray
            Modified label matrix with KD applied
        """
        labels_kd = labels_mat.copy()

        for task_id in range(self.num_col):
            if self.replaced[task_id] and self.best_predictions[task_id] is not None:
                # Use BEST predictions as soft labels (knowledge distillation)
                labels_kd[:, task_id] = self.best_predictions[task_id]
                self.logger.debug(
                    f"Applied KD to task {task_id} using best predictions "
                    f"from iteration {self.best_iteration[task_id]}"
                )

        return labels_kd

    def objective(
        self, preds: np.ndarray, train_data: Any, ep: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute adaptive weighted gradients with KD."""
        # Increment iteration counter
        self.current_iteration += 1

        # Preprocess
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)

        # Store predictions for best model tracking
        self._store_predictions(preds_mat, self.current_iteration)

        # Apply KD if any tasks have been replaced
        if any(self.replaced.values()):
            labels_mat = self._apply_kd(labels_mat, preds_mat)

        # Compute per-task gradients and hessians
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)

        # Compute adaptive weights
        weights = self.compute_weights(labels_mat, preds_mat, self.current_iteration)

        # Weight and aggregate
        weights_reshaped = weights.reshape(1, -1)
        grad = (grad_i * weights_reshaped).sum(axis=1)
        hess = (hess_i * weights_reshaped).sum(axis=1)

        return grad, hess, grad_i, hess_i

    def evaluate(self, preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, float]:
        """
        Evaluate with KD trigger checking.

        Returns
        -------
        task_scores : np.ndarray
            Per-task AUC scores
        mean_score : float
            Mean AUC across all tasks
        """
        # Call parent evaluation
        task_scores, mean_score = super().evaluate(preds, train_data)

        # Check KD trigger based on scores
        self._check_kd_trigger(task_scores, iteration=len(self.weight_history))

        return task_scores, mean_score
