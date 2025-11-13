"""
Weight update strategies for adaptive multi-task learning.

Implements different strategies for when and how to update task weights.
"""

from abc import ABC, abstractmethod
import numpy as np


class WeightUpdateStrategy(ABC):
    """Abstract base for weight update strategies."""

    @abstractmethod
    def should_update(self, iteration: int) -> bool:
        """Determine if weights should be updated at this iteration."""
        pass

    @abstractmethod
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights before application."""
        pass


class StandardStrategy(WeightUpdateStrategy):
    """Update weights every iteration without transformation."""

    def should_update(self, iteration: int) -> bool:
        return True

    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return weights


class TenItersStrategy(WeightUpdateStrategy):
    """Update weights every N iterations."""

    def __init__(self, update_frequency: int = 50):
        self.update_frequency = update_frequency

    def should_update(self, iteration: int) -> bool:
        return iteration % self.update_frequency == 0

    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return weights


class SqrtStrategy(WeightUpdateStrategy):
    """Apply square root transformation to weights."""

    def should_update(self, iteration: int) -> bool:
        return True

    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return np.sqrt(weights)


class DeltaStrategy(WeightUpdateStrategy):
    """Incremental weight updates based on changes."""

    def __init__(self, delta_lr: float = 0.01):
        self.delta_lr = delta_lr
        self.previous_weights = None

    def should_update(self, iteration: int) -> bool:
        return True

    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.previous_weights is None:
            self.previous_weights = weights.copy()
            return weights

        diff = weights - self.previous_weights
        updated = self.previous_weights + diff * self.delta_lr
        self.previous_weights = updated.copy()
        return updated
