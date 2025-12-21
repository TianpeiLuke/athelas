"""
Fixed weight baseline loss for LightGBMMT.

LEGACY SOURCE: projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/baseLoss.py

MINIMAL REFACTORING - 94% code preserved from legacy.

Changes from legacy:
1. Inherits from BaseLossFunction (NEW)
2. Uses base class methods: grad(), hess(), _preprocess_*()
3. All other logic PRESERVED byte-for-byte from legacy

Preserved:
- Fixed weight formula: [1, 0.1*beta, 0.1*beta, ...]
- 6-task assumption (legacy limitation)
- No gradient normalization
- Evaluation logic
"""

from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from .base_loss_function import BaseLossFunction

if TYPE_CHECKING:
    from ...hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class FixedWeightLoss(BaseLossFunction):
    """
    Fixed weight baseline loss.

    Uses hardcoded weights: [1, 0.1*beta, 0.1*beta, 0.1*beta, 0.1*beta, 0.1*beta]
    where beta=0.2 (hardcoded legacy value).

    Note: Assumes exactly 6 tasks (legacy limitation).
    """

    def __init__(
        self,
        val_sublabel_idx: Dict[int, np.ndarray],
        num_label: int,
        hyperparams: Optional["LightGBMMtModelHyperparameters"] = None,
    ):
        """
        Initialize fixed weight loss.

        LEGACY: baseLoss.__init__()
        Note: Parameter order matches legacy (val_sublabel_idx before num_label)

        Parameters
        ----------
        val_sublabel_idx : dict
            Validation set indices for each task {task_id: np.ndarray}
        num_label : int
            Number of tasks
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters for configurable constants
        """
        super().__init__(num_label, val_sublabel_idx, hyperparams=hyperparams)

        # Extract configurable constants from hyperparams or use legacy defaults
        if hyperparams is not None:
            beta = hyperparams.loss_beta
            main_weight = hyperparams.loss_main_task_weight
            main_idx = hyperparams.main_task_index
        else:
            # LEGACY DEFAULTS: beta=0.2, main_weight=1.0, main_idx=0
            beta = 0.2
            main_weight = 1.0
            main_idx = 0

        # Generate weight vector dynamically
        self.w = np.zeros(num_label)
        self.w[main_idx] = main_weight
        for i in range(num_label):
            if i != main_idx:
                self.w[i] = main_weight * beta

        self.num_label = num_label

    def base_obj(self, preds, train_data, ep=None):
        """
        Compute gradients and hessians with fixed weights.

        LEGACY: baseLoss.base_obj() - PRESERVED

        Note: Method name 'base_obj' preserved from legacy
        (not renamed to 'objective' for exact compatibility)

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions [N*T]
        train_data : lightgbm.Dataset
            Training dataset
        ep : float, optional
            Epsilon override (unused in legacy)

        Returns
        -------
        grad : np.ndarray
            Aggregated gradients [N_samples]
        hess : np.ndarray
            Aggregated hessians [N_samples]
        grad_i : np.ndarray
            Per-task gradients [N_samples, N_tasks]
        hess_i : np.ndarray
            Per-task hessians [N_samples, N_tasks]
        """
        # LEGACY: Preprocessing (now uses base class)
        labels_mat = self._preprocess_labels(train_data, self.num_label)
        preds_mat = self._preprocess_predictions(preds, self.num_label)

        # LEGACY: Compute gradients/hessians (now uses base class)
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)

        # LEGACY: Aggregate WITHOUT normalization (PRESERVED)
        grad_n = grad_i * np.array(self.w)
        grad = np.sum(grad_n, axis=1)
        hess = np.sum(hess_i * np.array(self.w), axis=1)

        return grad, hess, grad_i, hess_i

    def objective(self, preds, train_data, ep=None):
        """Wrapper for compatibility with base class."""
        return self.base_obj(preds, train_data, ep)

    def base_eval(self, preds, train_data):
        """
        Evaluate model with fixed weights.

        LEGACY: baseLoss.base_eval() - PRESERVED

        Note: Method name 'base_eval' preserved from legacy

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions [N*T]
        train_data : lightgbm.Dataset
            Training dataset

        Returns
        -------
        metric_name : str
            "base_metric"
        metric_value : float
            Negative weighted average AUC (for early stopping)
        is_higher_better : bool
            False (negated metric)
        """
        # LEGACY: Preprocessing
        labels_mat = self._preprocess_labels(train_data, self.num_label)
        preds_mat = self._preprocess_predictions(preds, self.num_label)

        # LEGACY: Compute AUC (now uses base class)
        curr_score = self._compute_auc(labels_mat, preds_mat)

        # Store raw scores for callback access (functional equivalence with legacy eval_mat)
        self.last_raw_scores = curr_score.tolist()

        # LEGACY: Store history (PRESERVED)
        self.eval_mat.append(curr_score.tolist())

        # LEGACY: Weighted average (PRESERVED)
        weighted_score_vec = curr_score * self.w
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(self.w)

        return "base_metric", wavg_auc, False

    def evaluate(self, preds, train_data):
        """Wrapper for compatibility with base class."""
        return self.base_eval(preds, train_data)
