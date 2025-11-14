"""
Tests for TrainingState class.

Following pytest best practices:
- Read source code first (training_state.py analyzed)
- Test actual implementation behavior
- No mocking needed (self-contained Pydantic model)
- Test both happy path and edge cases
"""

import pytest
import numpy as np
from athelas.models.lightgbmmt.base.training_state import TrainingState


class TestTrainingStateInitialization:
    """Test TrainingState initialization and defaults."""

    def test_default_initialization(self):
        """Test that TrainingState initializes with correct defaults."""
        # Read source: All fields have default values
        state = TrainingState()

        # Verify default values from source
        assert state.current_epoch == 0
        assert state.current_iteration == 0
        assert state.best_metric == 0.0
        assert state.best_epoch == 0
        assert state.best_iteration == 0
        assert state.training_history == []
        assert state.validation_history == []
        assert state.weight_evolution == []
        assert state.per_task_metrics == []
        assert state.epochs_without_improvement == 0
        assert state.patience_triggered == False
        assert state.kd_active == False
        assert state.kd_trigger_epoch is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        state = TrainingState(
            current_epoch=5,
            current_iteration=100,
            best_metric=0.85,
            best_epoch=3,
            best_iteration=75,
        )

        assert state.current_epoch == 5
        assert state.current_iteration == 100
        assert state.best_metric == 0.85
        assert state.best_epoch == 3
        assert state.best_iteration == 75

    def test_initialization_with_numpy_arrays(self):
        """Test initialization with numpy arrays for weight_evolution."""
        # Read source: weight_evolution accepts List[np.ndarray]
        weights = [np.array([1.0, 0.5, 0.3]), np.array([0.9, 0.6, 0.4])]
        state = TrainingState(weight_evolution=weights)

        assert len(state.weight_evolution) == 2
        assert isinstance(state.weight_evolution[0], np.ndarray)
        assert np.array_equal(state.weight_evolution[0], weights[0])


class TestShouldStopEarly:
    """Test early stopping logic."""

    def test_should_not_stop_when_patience_not_exceeded(self):
        """Test that early stopping is not triggered when within patience."""
        # Read source: should_stop_early returns epochs_without_improvement >= patience
        state = TrainingState(epochs_without_improvement=3)

        # Patience = 5, epochs_without_improvement = 3 → False
        assert state.should_stop_early(patience=5) == False

    def test_should_stop_when_patience_exceeded(self):
        """Test that early stopping triggers when patience exceeded."""
        state = TrainingState(epochs_without_improvement=5)

        # Patience = 5, epochs_without_improvement = 5 → True (equal triggers)
        assert state.should_stop_early(patience=5) == True

    def test_should_stop_when_patience_greatly_exceeded(self):
        """Test early stopping with patience greatly exceeded."""
        state = TrainingState(epochs_without_improvement=10)

        # Patience = 3, epochs_without_improvement = 10 → True
        assert state.should_stop_early(patience=3) == True

    def test_should_not_stop_with_zero_epochs_without_improvement(self):
        """Test that early stopping doesn't trigger with recent improvement."""
        state = TrainingState(epochs_without_improvement=0)

        assert state.should_stop_early(patience=5) == False


class TestUpdateBest:
    """Test best metric update logic."""

    def test_update_best_with_improvement(self):
        """Test that update_best returns True and updates values when metric improves."""
        # Read source: Returns True if metric > best_metric, resets epochs_without_improvement
        # IMPORTANT: Validator checks best_epoch <= current_epoch, so set current_epoch
        state = TrainingState(
            current_epoch=5,  # Must be >= best_epoch for validation
            best_metric=0.70,
            epochs_without_improvement=3,
        )

        # Improved metric → should return True and update
        improved = state.update_best(metric=0.85, epoch=5, iteration=100)

        assert improved == True
        assert state.best_metric == 0.85
        assert state.best_epoch == 5
        assert state.best_iteration == 100
        assert state.epochs_without_improvement == 0  # Reset on improvement

    def test_update_best_without_improvement(self):
        """Test that update_best returns False and increments counter when no improvement."""
        # Read source: Returns False if metric <= best_metric, increments epochs_without_improvement
        state = TrainingState(best_metric=0.85, epochs_without_improvement=2)

        # No improvement → should return False and increment counter
        improved = state.update_best(metric=0.80, epoch=6, iteration=120)

        assert improved == False
        assert state.best_metric == 0.85  # Unchanged
        assert state.best_epoch == 0  # Unchanged (still default)
        assert state.best_iteration == 0  # Unchanged
        assert state.epochs_without_improvement == 3  # Incremented

    def test_update_best_with_equal_metric(self):
        """Test that equal metric is not considered improvement."""
        # Read source: condition is metric > best_metric (not >=)
        state = TrainingState(best_metric=0.85, epochs_without_improvement=1)

        # Equal metric → no improvement
        improved = state.update_best(metric=0.85, epoch=5, iteration=100)

        assert improved == False
        assert state.best_metric == 0.85
        assert state.epochs_without_improvement == 2  # Incremented

    def test_update_best_sequence(self):
        """Test sequence of updates with mixed improvements."""
        # IMPORTANT: Set current_epoch to allow validator to pass
        state = TrainingState(current_epoch=10)  # High enough for all test epochs

        # First improvement
        assert state.update_best(0.70, 1, 10) == True
        assert state.epochs_without_improvement == 0

        # No improvement
        assert state.update_best(0.65, 2, 20) == False
        assert state.epochs_without_improvement == 1

        # Another improvement
        assert state.update_best(0.80, 3, 30) == True
        assert state.epochs_without_improvement == 0


class TestCheckpointSerialization:
    """Test checkpoint serialization and deserialization."""

    def test_to_checkpoint_dict_basic(self):
        """Test serialization of basic state without numpy arrays."""
        # Read source: to_checkpoint_dict uses model_dump() + converts numpy to lists
        state = TrainingState(
            current_epoch=5, current_iteration=100, best_metric=0.85, best_epoch=3
        )

        checkpoint = state.to_checkpoint_dict()

        # Verify all fields are present and serializable
        assert isinstance(checkpoint, dict)
        assert checkpoint["current_epoch"] == 5
        assert checkpoint["current_iteration"] == 100
        assert checkpoint["best_metric"] == 0.85
        assert checkpoint["best_epoch"] == 3
        assert checkpoint["weight_evolution"] == []  # Empty list (no numpy arrays)

    def test_to_checkpoint_dict_with_numpy_arrays(self):
        """Test serialization with numpy arrays converts to lists."""
        # Read source: Converts np.ndarray to list for JSON serialization
        weights = [np.array([1.0, 0.5, 0.3]), np.array([0.9, 0.6, 0.4])]
        state = TrainingState(weight_evolution=weights)

        checkpoint = state.to_checkpoint_dict()

        # Verify numpy arrays are converted to lists
        assert isinstance(checkpoint["weight_evolution"], list)
        assert isinstance(checkpoint["weight_evolution"][0], list)
        assert checkpoint["weight_evolution"][0] == [1.0, 0.5, 0.3]
        assert checkpoint["weight_evolution"][1] == [0.9, 0.6, 0.4]

    def test_from_checkpoint_dict_basic(self):
        """Test deserialization of basic checkpoint."""
        # Read source: from_checkpoint_dict converts lists back to numpy arrays
        checkpoint = {
            "current_epoch": 5,
            "current_iteration": 100,
            "best_metric": 0.85,
            "best_epoch": 3,
            "best_iteration": 75,
        }

        state = TrainingState.from_checkpoint_dict(checkpoint)

        assert state.current_epoch == 5
        assert state.current_iteration == 100
        assert state.best_metric == 0.85
        assert state.best_epoch == 3
        assert state.best_iteration == 75

    def test_from_checkpoint_dict_with_weight_evolution(self):
        """Test deserialization converts lists back to numpy arrays."""
        # Read source: Converts weight_evolution lists to numpy arrays
        checkpoint = {"weight_evolution": [[1.0, 0.5, 0.3], [0.9, 0.6, 0.4]]}

        state = TrainingState.from_checkpoint_dict(checkpoint)

        # Verify lists are converted back to numpy arrays
        assert len(state.weight_evolution) == 2
        assert isinstance(state.weight_evolution[0], np.ndarray)
        assert isinstance(state.weight_evolution[1], np.ndarray)
        assert np.array_equal(state.weight_evolution[0], np.array([1.0, 0.5, 0.3]))
        assert np.array_equal(state.weight_evolution[1], np.array([0.9, 0.6, 0.4]))

    def test_checkpoint_roundtrip(self):
        """Test that serialization and deserialization preserve state."""
        # Create state with various data
        original_weights = [np.array([1.0, 0.5, 0.3]), np.array([0.9, 0.6, 0.4])]
        original_state = TrainingState(
            current_epoch=5,
            current_iteration=100,
            best_metric=0.85,
            best_epoch=3,
            best_iteration=75,
            weight_evolution=original_weights,
            epochs_without_improvement=2,
            kd_active=True,
        )

        # Serialize and deserialize
        checkpoint = original_state.to_checkpoint_dict()
        restored_state = TrainingState.from_checkpoint_dict(checkpoint)

        # Verify all fields match
        assert restored_state.current_epoch == original_state.current_epoch
        assert restored_state.current_iteration == original_state.current_iteration
        assert restored_state.best_metric == original_state.best_metric
        assert restored_state.best_epoch == original_state.best_epoch
        assert restored_state.best_iteration == original_state.best_iteration
        assert (
            restored_state.epochs_without_improvement
            == original_state.epochs_without_improvement
        )
        assert restored_state.kd_active == original_state.kd_active

        # Verify numpy arrays
        assert len(restored_state.weight_evolution) == len(
            original_state.weight_evolution
        )
        for i in range(len(original_weights)):
            assert np.array_equal(
                restored_state.weight_evolution[i], original_state.weight_evolution[i]
            )


class TestValidation:
    """Test Pydantic validation logic."""

    def test_validation_fails_when_best_epoch_greater_than_current(self):
        """Test that validator raises error when best_epoch > current_epoch."""
        # Read source: validate_consistency checks best_epoch <= current_epoch
        with pytest.raises(
            ValueError, match="best_epoch.*cannot be greater than.*current_epoch"
        ):
            TrainingState(
                current_epoch=5,
                best_epoch=10,  # Invalid: greater than current
            )

    def test_validation_passes_when_best_epoch_equals_current(self):
        """Test that validation allows best_epoch == current_epoch."""
        # Should not raise - equal is valid
        state = TrainingState(current_epoch=5, best_epoch=5)
        assert state.current_epoch == 5
        assert state.best_epoch == 5

    def test_validation_passes_when_best_epoch_less_than_current(self):
        """Test that validation allows best_epoch < current_epoch."""
        state = TrainingState(current_epoch=10, best_epoch=7)
        assert state.current_epoch == 10
        assert state.best_epoch == 7

    def test_negative_epochs_rejected(self):
        """Test that negative values are rejected by ge=0 constraint."""
        # Read source: Fields have ge=0 constraint
        with pytest.raises(ValueError):
            TrainingState(current_epoch=-1)

        with pytest.raises(ValueError):
            TrainingState(epochs_without_improvement=-5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_patience_early_stopping(self):
        """Test early stopping with zero patience."""
        # Read source: should_stop_early returns epochs_without_improvement >= patience
        # So 0 >= 0 is True (triggers immediately)
        state = TrainingState(epochs_without_improvement=0)
        assert state.should_stop_early(patience=0) == True  # 0 >= 0 → True

        state = TrainingState(epochs_without_improvement=1)
        assert state.should_stop_early(patience=0) == True  # 1 >= 0 → True

    def test_large_weight_evolution_list(self):
        """Test handling of large weight evolution lists."""
        # Create large list of weight arrays
        large_weights = [np.array([i, i + 0.1, i + 0.2]) for i in range(1000)]
        state = TrainingState(weight_evolution=large_weights)

        assert len(state.weight_evolution) == 1000

        # Test serialization/deserialization with large list
        checkpoint = state.to_checkpoint_dict()
        restored = TrainingState.from_checkpoint_dict(checkpoint)

        assert len(restored.weight_evolution) == 1000
        assert np.array_equal(restored.weight_evolution[0], large_weights[0])
        assert np.array_equal(restored.weight_evolution[-1], large_weights[-1])

    def test_empty_histories(self):
        """Test that empty history lists are handled correctly."""
        state = TrainingState()

        checkpoint = state.to_checkpoint_dict()
        assert checkpoint["training_history"] == []
        assert checkpoint["validation_history"] == []
        assert checkpoint["per_task_metrics"] == []

        restored = TrainingState.from_checkpoint_dict(checkpoint)
        assert restored.training_history == []
        assert restored.validation_history == []
        assert restored.per_task_metrics == []


class TestKnowledgeDistillationState:
    """Test knowledge distillation state tracking."""

    def test_kd_state_defaults(self):
        """Test KD state defaults to inactive."""
        state = TrainingState()

        assert state.kd_active == False
        assert state.kd_trigger_epoch is None

    def test_kd_state_activation(self):
        """Test KD state can be activated."""
        state = TrainingState(kd_active=True, kd_trigger_epoch=10)

        assert state.kd_active == True
        assert state.kd_trigger_epoch == 10

    def test_kd_state_in_checkpoint(self):
        """Test KD state is preserved in checkpoints."""
        state = TrainingState(kd_active=True, kd_trigger_epoch=15)

        checkpoint = state.to_checkpoint_dict()
        assert checkpoint["kd_active"] == True
        assert checkpoint["kd_trigger_epoch"] == 15

        restored = TrainingState.from_checkpoint_dict(checkpoint)
        assert restored.kd_active == True
        assert restored.kd_trigger_epoch == 15
