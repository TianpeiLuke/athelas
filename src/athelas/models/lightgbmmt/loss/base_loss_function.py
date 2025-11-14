"""
Base loss function for LightGBMMT multi-task learning.

Provides shared functionality and defines the interface for concrete loss implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, TYPE_CHECKING, Any
import numpy as np
import logging
from scipy.special import expit
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from ..hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class BaseLossFunction(ABC):
    """
    Abstract base class for MTGBM loss functions.

    Provides shared functionality:
    - Data preprocessing (sigmoid, clipping, reshaping)
    - Utility methods (normalization, gradient computation)
    - Input validation
    - Caching mechanisms
    - Logging infrastructure

    Design Pattern: Template Method + Strategy
    """

    def __init__(
        self,
        num_label: int,
        val_sublabel_idx: Dict[int, np.ndarray],
        trn_sublabel_idx: Optional[Dict[int, np.ndarray]] = None,
        hyperparams: Optional["LightGBMMtModelHyperparameters"] = None,
    ):
        """
        Initialize base loss function.

        Parameters
        ----------
        num_label : int
            Total number of tasks (main + subtasks)
        val_sublabel_idx : dict
            Validation set indices for each task {task_id: np.ndarray}
        trn_sublabel_idx : dict, optional
            Training set indices for each task
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters containing loss parameters
        """
        # Validate inputs
        if num_label < 2:
            raise ValueError(f"num_label must be >= 2, got {num_label}")

        if not val_sublabel_idx:
            raise ValueError("val_sublabel_idx cannot be empty")

        if hyperparams is None:
            raise ValueError("hyperparams is required")

        # Initialize attributes
        self.num_col = num_label
        self.val_sublabel_idx = val_sublabel_idx
        self.trn_sublabel_idx = trn_sublabel_idx or {}
        self.hyperparams = hyperparams

        # Extract loss parameters from hyperparams
        self.epsilon = hyperparams.loss_epsilon
        self.epsilon_norm = hyperparams.loss_epsilon_norm
        self.clip_similarity_inverse = hyperparams.loss_clip_similarity_inverse
        self.beta = hyperparams.loss_beta
        self.main_task_weight = hyperparams.loss_main_task_weight
        self.weight_lr = hyperparams.loss_weight_lr
        self.patience = hyperparams.loss_patience
        self.enable_kd = hyperparams.enable_kd
        self.weight_method = hyperparams.loss_weight_method
        self.weight_update_frequency = hyperparams.loss_weight_update_frequency
        self.delta_lr = hyperparams.loss_delta_lr
        self.cache_predictions = hyperparams.loss_cache_predictions
        self.precompute_indices = hyperparams.loss_precompute_indices
        self.log_level = hyperparams.loss_log_level

        # Setup caching if enabled
        self._pred_cache: Dict[int, np.ndarray] = {} if self.cache_predictions else {}
        self._label_cache: Dict[int, np.ndarray] = {} if self.cache_predictions else {}

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.log_level)

        self.logger.info(
            f"Initialized {self.__class__.__name__} with {num_label} tasks"
        )

    def _preprocess_predictions(
        self, preds: np.ndarray, num_col: int, epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Transform and clip predictions with caching.

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions from model
        num_col : int
            Number of tasks
        epsilon : float, optional
            Clipping constant (uses self.epsilon if None)

        Returns
        -------
        preds_mat : np.ndarray
            Preprocessed predictions [N_samples, N_tasks]
        """
        # Check cache
        cache_key = id(preds)
        if self.cache_predictions and cache_key in self._pred_cache:
            return self._pred_cache[cache_key]

        # Reshape
        preds_mat = preds.reshape(-1, num_col)

        # Apply sigmoid
        preds_mat = expit(preds_mat)

        # Clip for numerical stability
        eps = epsilon if epsilon is not None else self.epsilon
        preds_mat = np.clip(preds_mat, eps, 1 - eps)

        # Cache result
        if self.cache_predictions:
            self._pred_cache[cache_key] = preds_mat

        return preds_mat

    def _preprocess_labels(self, train_data: Any, num_col: int) -> np.ndarray:
        """
        Reshape label matrix with validation.

        Parameters
        ----------
        train_data : lightgbm.Dataset
            Training dataset containing labels
        num_col : int
            Number of tasks

        Returns
        -------
        labels_mat : np.ndarray
            Reshaped labels [N_samples, N_tasks]
        """
        # Check cache
        cache_key = id(train_data)
        if self.cache_predictions and cache_key in self._label_cache:
            return self._label_cache[cache_key]

        # Get labels
        labels = train_data.get_label()

        # Reshape
        labels_mat = labels.reshape(-1, num_col)

        # Validate
        if labels_mat.shape[1] != num_col:
            raise ValueError(f"Expected {num_col} tasks, got {labels_mat.shape[1]}")

        # Cache result
        if self.cache_predictions:
            self._label_cache[cache_key] = labels_mat

        return labels_mat

    def normalize(self, vec: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        """Standard normalization with NaN protection."""
        eps = epsilon if epsilon is not None else self.epsilon_norm
        total = vec.sum()
        if total < eps:
            return np.ones_like(vec) / len(vec)
        return vec / total

    def unit_scale(
        self, vec: np.ndarray, epsilon: Optional[float] = None
    ) -> np.ndarray:
        """L2 normalization with zero-norm protection."""
        eps = epsilon if epsilon is not None else self.epsilon_norm
        norm = np.linalg.norm(vec)
        if norm < eps:
            return np.ones_like(vec) / np.sqrt(len(vec))
        return vec / norm

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate gradients for binary cross-entropy."""
        return y_pred - y_true

    def hess(self, y_pred: np.ndarray) -> np.ndarray:
        """Calculate hessians for binary cross-entropy."""
        return y_pred * (1.0 - y_pred)

    def evaluate(self, preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, float]:
        """
        Standard evaluation function (per-task AUC).

        Returns
        -------
        task_scores : np.ndarray
            Per-task AUC scores
        mean_score : float
            Mean AUC across all tasks
        """
        # Preprocess
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col)

        # Compute per-task AUC
        task_scores = np.zeros(self.num_col)
        for i in range(self.num_col):
            try:
                task_scores[i] = roc_auc_score(labels_mat[:, i], preds_mat[:, i])
            except ValueError:
                # Handle case where only one class present
                task_scores[i] = 0.5

        return task_scores, task_scores.mean()

    def clear_cache(self) -> None:
        """Clear prediction and label caches."""
        if self.cache_predictions:
            self._pred_cache.clear()
            self._label_cache.clear()

    @abstractmethod
    def compute_weights(
        self, labels_mat: np.ndarray, preds_mat: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Compute task weights - must be implemented by subclasses.

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
            Task weights [N_tasks]
        """
        pass

    @abstractmethod
    def objective(
        self, preds: np.ndarray, train_data: Any, ep: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Objective function - must be implemented by subclasses.

        Parameters
        ----------
        preds : np.ndarray
            Raw predictions from model
        train_data : lightgbm.Dataset
            Training dataset
        ep : float, optional
            Override epsilon value

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
        pass
