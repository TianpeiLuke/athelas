"""
Base loss function for LightGBMMT multi-task learning.

MINIMAL REFACTORING DESIGN - Preserves legacy behavior.

Extracted from legacy implementations:
- projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/baseLoss.py
- projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossNoKD.py
- projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossKDswap.py

Philosophy: Extract ONLY code that is byte-for-byte identical across
all three legacy implementations. Everything else preserved in subclasses.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from ...hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class BaseLossFunction(ABC):
    """
    Minimal base class - ONLY truly shared operations.

    Provides:
    - grad() - Binary cross-entropy gradient (IDENTICAL in all legacy)
    - hess() - Binary cross-entropy hessian (IDENTICAL in all legacy)
    - _preprocess_labels() - Reshape operation (IDENTICAL in all legacy)
    - _preprocess_predictions() - Sigmoid + clip + reshape (IDENTICAL in all legacy)
    - _compute_auc() - Core AUC calculation loop (IDENTICAL in all legacy)

    Does NOT provide:
    - normalize() - Only in adaptive losses, not in baseLoss
    - unit_scale() - Only in adaptive losses, not in baseLoss
    - Weight computation - Different per loss type
    - Gradient aggregation - Different per loss type
    - Validation or logging - Legacy has none
    """

    def __init__(
        self,
        num_label,
        val_sublabel_idx,
        trn_sublabel_idx=None,
        hyperparams: Optional["LightGBMMtModelHyperparameters"] = None,
    ):
        """
        Minimal initialization - common to all legacy classes.

        Parameters
        ----------
        num_label : int
            Number of tasks
        val_sublabel_idx : dict
            Validation set indices for each task {task_id: np.ndarray}
        trn_sublabel_idx : dict, optional
            Training set indices for each task
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters for configurable constants
        """
        self.num_col = num_label
        self.val_label_idx = val_sublabel_idx
        self.trn_sublabel_idx = trn_sublabel_idx
        self.eval_mat = []
        self.hyperparams = hyperparams

        # Store last computed raw AUC scores for callback access
        self.last_raw_scores = None

    def grad(self, y_true, y_pred):
        """
        Binary cross-entropy gradient.

        LEGACY: Identical in baseLoss.py, customLossNoKD.py, customLossKDswap.py
        Formula: grad = y_pred - y_true

        Parameters
        ----------
        y_true : np.ndarray
            True labels [N_samples, N_tasks]
        y_pred : np.ndarray
            Predictions [N_samples, N_tasks]

        Returns
        -------
        grad : np.ndarray
            Gradients [N_samples, N_tasks]
        """
        return y_pred - y_true

    def hess(self, y_pred):
        """
        Binary cross-entropy hessian.

        LEGACY: Identical in baseLoss.py, customLossNoKD.py, customLossKDswap.py
        Formula: hess = y_pred * (1 - y_pred)

        Parameters
        ----------
        y_pred : np.ndarray
            Predictions [N_samples, N_tasks]

        Returns
        -------
        hess : np.ndarray
            Hessians [N_samples, N_tasks]
        """
        return y_pred * (1.0 - y_pred)

    def _preprocess_labels(self, train_data, num_col):
        """
        Reshape labels to matrix form.

        LEGACY: Identical in all three implementations
        Transform: labels [N*T] → labels_mat [N, T]

        Parameters
        ----------
        train_data : lightgbm.Dataset
            Training dataset
        num_col : int
            Number of tasks

        Returns
        -------
        labels_mat : np.ndarray
            Reshaped labels [N_samples, N_tasks]
        """
        labels = train_data.get_label()
        labels_mat = labels.reshape((num_col, -1)).transpose()
        return labels_mat

    def _preprocess_predictions(self, preds, num_col, epsilon=1e-15):
        """
        Transform and clip predictions.

        LEGACY: Identical in all three implementations
        Steps:
        1. Reshape: [N*T] → [T, N] → [N, T]
        2. Sigmoid: logits → probabilities
        3. Clip: [epsilon, 1-epsilon]

        Note: epsilon=1e-15 hardcoded in legacy

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions [N*T]
        num_col : int
            Number of tasks
        epsilon : float, optional
            Clipping constant (default: 1e-15, legacy value)

        Returns
        -------
        preds_mat : np.ndarray
            Preprocessed predictions [N_samples, N_tasks]
        """
        preds_mat = preds.reshape((num_col, -1)).transpose()
        preds_mat = expit(preds_mat)
        preds_mat = np.clip(preds_mat, epsilon, 1 - epsilon)
        return preds_mat

    def _compute_auc(self, labels_mat, preds_mat):
        """
        Compute per-task AUC scores with robust error handling.

        LEGACY: Core loop identical in all three implementations
        Uses validation indices to filter samples per task.

        Handles edge case where validation subset has only one class
        (can occur during early iterations with highly imbalanced tasks).

        Parameters
        ----------
        labels_mat : np.ndarray
            Labels [N_samples, N_tasks]
        preds_mat : np.ndarray
            Predictions [N_samples, N_tasks]

        Returns
        -------
        curr_score : np.ndarray
            Per-task AUC scores [N_tasks]
        """
        import logging

        logger = logging.getLogger(__name__)

        curr_score = []
        for j in range(self.num_col):
            try:
                s = roc_auc_score(
                    labels_mat[self.val_label_idx[j], j],
                    preds_mat[self.val_label_idx[j], j],
                )
                curr_score.append(s)
            except ValueError as e:
                if "Only one class present" in str(e):
                    # Single class in validation subset - use 0.5 (undefined AUC)
                    # This is common during early iterations with imbalanced tasks
                    subset_size = len(self.val_label_idx[j])
                    unique_classes = np.unique(labels_mat[self.val_label_idx[j], j])

                    logger.warning(
                        f"Task {j}: Only one class ({unique_classes[0]}) "
                        f"in validation subset (n={subset_size}). "
                        f"Setting AUC to 0.5 (undefined)."
                    )

                    curr_score.append(0.5)
                else:
                    # Re-raise other ValueErrors
                    raise
        return np.array(curr_score)

    @abstractmethod
    def objective(self, preds, train_data, ep=None):
        """
        Objective function - subclasses implement with legacy logic.

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions
        train_data : lightgbm.Dataset
            Training dataset
        ep : float, optional
            Epsilon parameter (legacy)

        Returns
        -------
        grad : np.ndarray
            Aggregated gradients
        hess : np.ndarray
            Aggregated hessians
        grad_i : np.ndarray
            Per-task gradients
        hess_i : np.ndarray
            Per-task hessians
        """
        pass

    @abstractmethod
    def evaluate(self, preds, train_data):
        """
        Evaluation function - subclasses implement with legacy logic.

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions
        train_data : lightgbm.Dataset
            Training dataset

        Returns
        -------
        metric_name : str
            Name of metric
        metric_value : float
            Metric value (negative for early stopping)
        is_higher_better : bool
            Whether higher is better
        """
        pass
