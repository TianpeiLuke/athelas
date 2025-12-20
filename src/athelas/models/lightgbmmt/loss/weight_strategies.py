"""
Weight update strategies for adaptive multi-task learning.

EXTRACTED FROM: adaptive_weight_loss.py self_obj() method (lines 125-175)

STRATEGY PATTERN: Encapsulates weight computation algorithms while preserving
exact legacy behavior including state mutations.

Each strategy is a DIRECT COPY of original if/elif branches - zero logic changes.
"""

from typing import Dict, Callable
from abc import ABC, abstractmethod
import numpy as np


class BaseWeightStrategy(ABC):
    """
    Abstract base class for weight update strategies.

    CRITICAL: Strategies mutate state (similar, w_trn_mat) exactly as original.
    This is NOT pure functional - mutations are intentional for legacy equivalence.
    """

    @abstractmethod
    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute task weight vector for current iteration.

        LEGACY: This method MUST mutate self.similar and self.w_trn_mat
        to maintain exact equivalence with original inline code.

        Parameters
        ----------
        labels_mat : np.ndarray
            Label matrix [N_samples, N_tasks]
        preds_mat : np.ndarray
            Prediction matrix [N_samples, N_tasks]
        num_col : int
            Number of tasks
        trn_sublabel_idx : dict
            Training indices per task {task_id: np.ndarray}
        main_task_index : int
            Index of main task
        curr_iteration : int
            Current iteration number (1-indexed, matches curr_obj_round)
        similarity_vec_fn : callable
            Function to compute similarity vector (from parent class)

        Returns
        -------
        w : np.ndarray
            Weight vector [N_tasks]

        Side Effects
        ------------
        Mutates self.similar and self.w_trn_mat (intentional, legacy behavior)
        """
        pass


class TenItersWeightStrategy(BaseWeightStrategy):
    """
    Update weights every N iterations.

    LEGACY SOURCE: adaptive_weight_loss.py lines 125-135
    PRESERVED: Update frequency check, similarity caching, exact logic
    """

    def __init__(self, update_frequency: int = 10):
        """
        Initialize tenIters strategy.

        Parameters
        ----------
        update_frequency : int
            Update weights every N iterations (default 10 for adaptive, 50 for KD)
        """
        self.update_frequency = update_frequency
        # State containers (will be set by parent)
        self.similar = None
        self.w_trn_mat = None

    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute weights with periodic updates.

        LEGACY EQUIVALENT: Lines 125-135 from adaptive_weight_loss.py
        """
        # LEGACY: Exact copy of original logic
        i = curr_iteration - 1
        if i % self.update_frequency == 0:  # Configurable update frequency
            self.similar = similarity_vec_fn(
                labels_mat[:, main_task_index],
                preds_mat,
                num_col,
                trn_sublabel_idx,
                0.1,
            )
        w = self.similar
        self.w_trn_mat.append(w)

        return w


class SqrtWeightStrategy(BaseWeightStrategy):
    """
    Compute weights with square root transformation.

    LEGACY SOURCE: adaptive_weight_loss.py lines 137-147
    PRESERVED: Learning rate 0.5, sqrt transformation, exact logic
    """

    def __init__(self):
        """Initialize sqrt strategy."""
        # State containers (will be set by parent)
        self.similar = None
        self.w_trn_mat = None

    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute weights with sqrt transformation.

        LEGACY EQUIVALENT: Lines 137-147 from adaptive_weight_loss.py
        """
        # LEGACY: Exact copy of original logic
        w = similarity_vec_fn(
            labels_mat[:, main_task_index],
            preds_mat,
            num_col,
            trn_sublabel_idx,
            0.5,  # LEGACY: lr=0.5 for sqrt
        )
        w = np.sqrt(w)
        self.w_trn_mat.append(w)

        return w


class DeltaWeightStrategy(BaseWeightStrategy):
    """
    Update weights using delta (gradient-like) method.

    LEGACY SOURCE: adaptive_weight_loss.py lines 149-163
    PRESERVED: Delta computation, learning rate, first iteration special case
    """

    def __init__(self, delta_lr: float = 0.1):
        """
        Initialize delta strategy.

        Parameters
        ----------
        delta_lr : float
            Learning rate for delta updates (default 0.1 for adaptive, 0.01 for KD)
        """
        self.delta_lr = delta_lr
        # State containers (will be set by parent)
        self.similar = None
        self.w_trn_mat = None

    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute weights using delta method.

        LEGACY EQUIVALENT: Lines 149-163 from adaptive_weight_loss.py
        """
        # LEGACY: Exact copy of original logic
        simi = similarity_vec_fn(
            labels_mat[:, main_task_index],
            preds_mat,
            num_col,
            trn_sublabel_idx,
            0.1,
        )
        self.similar.append(simi)
        if curr_iteration == 1:
            w = self.similar[0]
        else:
            i = curr_iteration - 1
            diff = self.similar[i] - self.similar[i - 1]
            w = self.w_trn_mat[i - 1] + diff * self.delta_lr  # Configurable delta_lr
        self.w_trn_mat.append(w)

        return w


class EMAWeightStrategy(BaseWeightStrategy):
    """
    Exponential Moving Average (EMA) weight smoothing strategy.

    NEW STRATEGY: Not from legacy - designed to stabilize weight oscillations.

    Smooths weight updates over time using exponential moving average:
        w_new = (1 - lr) * w_old + lr * w_raw

    This reduces oscillations in adaptive weights when task similarities
    fluctuate rapidly during training.

    Parameters
    ----------
    weight_lr : float
        EMA learning rate (0 < weight_lr <= 1)
        - 1.0: No smoothing (equivalent to standard)
        - 0.1: 10% new weights + 90% old weights (typical smoothing)
        - 0.01: 1% new weights + 99% old weights (heavy smoothing)

    Use Cases
    ---------
    - Unstable training metrics
    - Weight oscillations between iterations
    - Task dominance switching rapidly
    """

    def __init__(self, weight_lr: float = 1.0):
        """
        Initialize EMA strategy.

        Parameters
        ----------
        weight_lr : float
            EMA learning rate for weight smoothing (default 1.0 = no smoothing)
        """
        self.weight_lr = weight_lr
        # State containers (will be set by parent)
        self.similar = None
        self.w_trn_mat = None

    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute weights with EMA smoothing.

        NEW: Applies exponential moving average to stabilize weights.

        Algorithm:
        - First iteration: w = w_raw (no history to smooth)
        - Subsequent: w = (1 - lr) * w_old + lr * w_raw
        """
        # Compute raw similarity-based weights
        w_raw = similarity_vec_fn(
            labels_mat[:, main_task_index],
            preds_mat,
            num_col,
            trn_sublabel_idx,
            0.1,  # Learning rate for similarity computation
        )

        # EMA smoothing
        if curr_iteration == 1:
            # First iteration: use raw weights (no history)
            w = w_raw
        else:
            # EMA: blend old and new weights
            w_old = self.w_trn_mat[-1]
            w = (1 - self.weight_lr) * np.array(w_old) + self.weight_lr * np.array(
                w_raw
            )
            w = w.tolist()

        # Update history
        self.w_trn_mat.append(w)

        return w


class StandardWeightStrategy(BaseWeightStrategy):
    """
    Standard weight computation (weight_method=None).

    LEGACY SOURCE: adaptive_weight_loss.py lines 165-175
    PRESERVED: Direct similarity computation, no caching
    """

    def __init__(self):
        """Initialize standard strategy."""
        # State containers (will be set by parent)
        self.similar = None
        self.w_trn_mat = None

    def compute_weight(
        self,
        labels_mat: np.ndarray,
        preds_mat: np.ndarray,
        num_col: int,
        trn_sublabel_idx: Dict[int, np.ndarray],
        main_task_index: int,
        curr_iteration: int,
        similarity_vec_fn: Callable,
    ) -> np.ndarray:
        """
        Compute weights using standard method.

        LEGACY EQUIVALENT: Lines 165-175 from adaptive_weight_loss.py
        """
        # LEGACY: Exact copy of original logic (else branch)
        w = similarity_vec_fn(
            labels_mat[:, main_task_index],
            preds_mat,
            num_col,
            trn_sublabel_idx,
            0.1,
        )
        self.w_trn_mat.append(w)

        return w


class WeightStrategyFactory:
    """
    Factory for creating weight update strategies.

    PATTERN: Factory Method - encapsulates strategy creation logic.
    """

    @staticmethod
    def create(
        weight_method: str,
        update_frequency: int = 10,
        delta_lr: float = 0.1,
        weight_lr: float = 1.0,
    ) -> BaseWeightStrategy:
        """
        Create weight strategy by name.

        Parameters
        ----------
        weight_method : str
            Strategy name: 'tenIters', 'sqrt', 'delta', 'ema', or None
        update_frequency : int
            Update frequency for tenIters (default 10)
        delta_lr : float
            Learning rate for delta method (default 0.1)
        weight_lr : float
            EMA learning rate for ema method (default 1.0)

        Returns
        -------
        strategy : BaseWeightStrategy
            Configured strategy instance

        Raises
        ------
        ValueError
            If weight_method is not recognized
        """
        if weight_method == "tenIters":
            return TenItersWeightStrategy(update_frequency=update_frequency)
        elif weight_method == "sqrt":
            return SqrtWeightStrategy()
        elif weight_method == "delta":
            return DeltaWeightStrategy(delta_lr=delta_lr)
        elif weight_method == "ema":
            return EMAWeightStrategy(weight_lr=weight_lr)
        elif weight_method is None or weight_method == "standard":
            return StandardWeightStrategy()
        else:
            raise ValueError(
                f"Unknown weight_method: {weight_method}. "
                f"Must be one of: 'tenIters', 'sqrt', 'delta', 'ema', None"
            )
