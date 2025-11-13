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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # KD tracking state
        self.kd_active = False
        self.kd_trigger_iteration = None
        self.performance_history = {i: [] for i in range(self.num_col)}
        self.decline_count = {i: 0 for i in range(self.num_col)}

        self.logger.info("Initialized KD loss with patience={}".format(self.patience))

    def _check_kd_trigger(self, task_scores: np.ndarray, iteration: int) -> None:
        """
        Check if KD should be triggered for any task.

        Parameters
        ----------
        task_scores : np.ndarray
            Current per-task performance scores
        iteration : int
            Current iteration number
        """
        for task_id in range(self.num_col):
            # Track performance history
            self.performance_history[task_id].append(task_scores[task_id])

            # Check for decline
            if len(self.performance_history[task_id]) >= 2:
                current = self.performance_history[task_id][-1]
                previous = self.performance_history[task_id][-2]

                if current < previous:
                    self.decline_count[task_id] += 1
                else:
                    self.decline_count[task_id] = 0

                # Trigger KD if patience exceeded
                if self.decline_count[task_id] >= self.patience and not self.kd_active:
                    self.kd_active = True
                    self.kd_trigger_iteration = iteration
                    self.logger.warning(
                        f"KD triggered for task {task_id} at iteration {iteration} "
                        f"(decline count: {self.decline_count[task_id]})"
                    )

    def _apply_kd(self, labels_mat: np.ndarray, preds_mat: np.ndarray) -> np.ndarray:
        """
        Apply knowledge distillation by replacing labels with predictions.

        Parameters
        ----------
        labels_mat : np.ndarray
            Original label matrix [N_samples, N_tasks]
        preds_mat : np.ndarray
            Prediction matrix [N_samples, N_tasks]

        Returns
        -------
        labels_kd : np.ndarray
            Modified label matrix with KD applied
        """
        if not self.kd_active:
            return labels_mat

        # Replace labels with soft predictions for struggling tasks
        labels_kd = labels_mat.copy()

        for task_id in range(self.num_col):
            if self.decline_count[task_id] >= self.patience:
                # Use predictions as soft labels (knowledge distillation)
                labels_kd[:, task_id] = preds_mat[:, task_id]
                self.logger.debug(f"Applied KD to task {task_id}")

        return labels_kd

    def objective(
        self, preds: np.ndarray, train_data: Any, ep: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute adaptive weighted gradients with KD."""
        # Preprocess
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)

        # Apply KD if active
        if self.kd_active:
            labels_mat = self._apply_kd(labels_mat, preds_mat)

        # Compute per-task gradients and hessians
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)

        # Compute adaptive weights
        weights = self.compute_weights(labels_mat, preds_mat, iteration=0)

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
