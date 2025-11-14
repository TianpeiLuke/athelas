"""
Tests for LossFactory implementation.

Tests factory pattern for creating loss function instances with type safety
and validation.

Following pytest best practices:
- Read source code first (loss_factory.py analyzed)
- Test actual implementation behavior
- Minimal mocking (only hyperparams)
- Test both happy path and error conditions
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import from actual source (following best practice)
from athelas.models.lightgbmmt.loss.loss_factory import LossFactory
from athelas.models.lightgbmmt.loss.base_loss_function import BaseLossFunction
from athelas.models.lightgbmmt.loss.fixed_weight_loss import FixedWeightLoss
from athelas.models.lightgbmmt.loss.adaptive_weight_loss import AdaptiveWeightLoss
from athelas.models.lightgbmmt.loss.knowledge_distillation_loss import KnowledgeDistillationLoss


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hyperparams():
    """Create mock hyperparameters with all required loss parameters."""
    mock = Mock()
    # Base loss parameters
    mock.loss_epsilon = 1e-15
    mock.loss_epsilon_norm = 1e-10
    mock.loss_clip_similarity_inverse = 1e10
    mock.loss_beta = 0.2
    mock.loss_main_task_weight = 1.0
    mock.loss_weight_lr = 0.1
    mock.loss_patience = 100
    mock.enable_kd = True
    mock.loss_weight_method = None
    mock.loss_weight_update_frequency = 50
    mock.loss_delta_lr = 0.01
    mock.loss_cache_predictions = True
    mock.loss_precompute_indices = True
    mock.loss_log_level = "INFO"
    # Adaptive-specific
    mock.main_task_index = 0
    return mock


@pytest.fixture
def sample_indices():
    """Create sample validation indices for 4 tasks."""
    return {
        0: np.array([0, 1, 2, 3, 4]),
        1: np.array([0, 1, 2, 3, 4]),
        2: np.array([0, 1, 2, 3, 4]),
        3: np.array([0, 1, 2, 3, 4]),
    }


@pytest.fixture
def factory_params(mock_hyperparams, sample_indices):
    """Create standard factory parameters."""
    return {
        "num_label": 4,
        "val_sublabel_idx": sample_indices,
        "hyperparams": mock_hyperparams,
    }


# ============================================================================
# Test Class 1: Loss Creation
# ============================================================================


class TestLossFactoryCreation:
    """Tests for creating loss function instances via factory."""

    def test_create_fixed_weight_loss(self, factory_params):
        """Test creating FixedWeightLoss via factory."""
        loss = LossFactory.create(loss_type="fixed", **factory_params)

        # Verify correct type
        assert isinstance(loss, FixedWeightLoss)
        assert isinstance(loss, BaseLossFunction)

        # Verify parameters passed correctly
        assert loss.num_col == 4
        assert loss.hyperparams == factory_params["hyperparams"]

    def test_create_adaptive_weight_loss(self, factory_params):
        """Test creating AdaptiveWeightLoss via factory."""
        loss = LossFactory.create(loss_type="adaptive", **factory_params)

        # Verify correct type
        assert isinstance(loss, AdaptiveWeightLoss)
        assert isinstance(loss, BaseLossFunction)

        # Verify parameters passed correctly
        assert loss.num_col == 4
        assert loss.hyperparams == factory_params["hyperparams"]

    def test_create_knowledge_distillation_loss(self, factory_params):
        """Test creating KnowledgeDistillationLoss via factory."""
        loss = LossFactory.create(loss_type="adaptive_kd", **factory_params)

        # Verify correct type
        assert isinstance(loss, KnowledgeDistillationLoss)
        assert isinstance(loss, AdaptiveWeightLoss)
        assert isinstance(loss, BaseLossFunction)

        # Verify parameters passed correctly
        assert loss.num_col == 4
        assert loss.hyperparams == factory_params["hyperparams"]

    def test_create_with_training_indices(self, factory_params, sample_indices):
        """Test creating loss with optional training indices."""
        trn_indices = {i: np.array([0, 1, 2]) for i in range(4)}

        loss = LossFactory.create(
            loss_type="fixed", trn_sublabel_idx=trn_indices, **factory_params
        )

        # Verify loss created successfully
        assert isinstance(loss, FixedWeightLoss)

    def test_create_returns_functional_loss(self, factory_params):
        """Test that created loss functions are functional."""
        loss = LossFactory.create(loss_type="fixed", **factory_params)

        # Verify loss has required methods (from BaseLossFunction)
        assert hasattr(loss, "objective")
        assert hasattr(loss, "evaluate")
        assert hasattr(loss, "compute_weights")

    def test_different_num_label_values(self, mock_hyperparams, sample_indices):
        """Test creating losses with different task counts."""
        # Test with 2 tasks
        indices_2 = {0: np.array([0, 1]), 1: np.array([0, 1])}
        loss_2 = LossFactory.create(
            loss_type="fixed",
            num_label=2,
            val_sublabel_idx=indices_2,
            hyperparams=mock_hyperparams,
        )
        assert loss_2.num_col == 2

        # Test with 10 tasks
        indices_10 = {i: np.array([0, 1, 2]) for i in range(10)}
        loss_10 = LossFactory.create(
            loss_type="adaptive",
            num_label=10,
            val_sublabel_idx=indices_10,
            hyperparams=mock_hyperparams,
        )
        assert loss_10.num_col == 10


# ============================================================================
# Test Class 2: Validation and Error Handling
# ============================================================================


class TestLossFactoryValidation:
    """Tests for factory validation and error handling."""

    def test_unknown_loss_type_raises_error(self, factory_params):
        """Test that unknown loss_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type: 'nonexistent'"):
            LossFactory.create(loss_type="nonexistent", **factory_params)

    def test_error_message_lists_available_types(self, factory_params):
        """Test error message includes available loss types."""
        with pytest.raises(ValueError, match="Available types:"):
            LossFactory.create(loss_type="invalid", **factory_params)

        # Verify all registered types mentioned
        try:
            LossFactory.create(loss_type="invalid", **factory_params)
        except ValueError as e:
            error_msg = str(e)
            assert "fixed" in error_msg
            assert "adaptive" in error_msg
            assert "adaptive_kd" in error_msg

    def test_missing_hyperparams_raises_error(self, sample_indices):
        """Test that missing hyperparams raises ValueError."""
        with pytest.raises(ValueError, match="hyperparams is required"):
            LossFactory.create(
                loss_type="fixed",
                num_label=4,
                val_sublabel_idx=sample_indices,
                hyperparams=None,
            )

    def test_invalid_num_label_propagates_error(self, mock_hyperparams, sample_indices):
        """Test that invalid num_label from loss class raises appropriate error."""
        # BaseLossFunction validates num_label >= 2
        with pytest.raises(ValueError):
            LossFactory.create(
                loss_type="fixed",
                num_label=1,  # Invalid: too small
                val_sublabel_idx=sample_indices,
                hyperparams=mock_hyperparams,
            )

    def test_empty_val_sublabel_idx_propagates_error(self, mock_hyperparams):
        """Test that empty val_sublabel_idx raises appropriate error."""
        with pytest.raises(ValueError):
            LossFactory.create(
                loss_type="fixed",
                num_label=4,
                val_sublabel_idx={},  # Invalid: empty
                hyperparams=mock_hyperparams,
            )


# ============================================================================
# Test Class 3: Registry Management
# ============================================================================


class TestLossFactoryRegistry:
    """Tests for factory registry management."""

    def test_get_available_losses_returns_list(self):
        """Test get_available_losses returns list of registered types."""
        available = LossFactory.get_available_losses()

        assert isinstance(available, list)
        assert len(available) > 0

    def test_get_available_losses_contains_default_types(self):
        """Test default loss types are available."""
        available = LossFactory.get_available_losses()

        # Check all expected default types present
        assert "fixed" in available
        assert "adaptive" in available
        assert "adaptive_kd" in available

    def test_registry_contains_correct_classes(self):
        """Test registry maps to correct loss classes."""
        # Access registry directly for verification
        registry = LossFactory._registry

        assert registry["fixed"] == FixedWeightLoss
        assert registry["adaptive"] == AdaptiveWeightLoss
        assert registry["adaptive_kd"] == KnowledgeDistillationLoss

    def test_all_registered_types_are_creatable(self, factory_params):
        """Test that all registered loss types can be created."""
        available = LossFactory.get_available_losses()

        for loss_type in available:
            loss = LossFactory.create(loss_type=loss_type, **factory_params)
            assert isinstance(loss, BaseLossFunction)


# ============================================================================
# Test Class 4: Extensibility
# ============================================================================


class TestLossFactoryExtensibility:
    """Tests for extending factory with custom loss functions."""

    def test_register_custom_loss_function(self, factory_params):
        """Test registering a custom loss function."""

        # Create custom loss class
        class CustomLoss(BaseLossFunction):
            def compute_weights(self, labels_mat, preds_mat, iteration):
                return np.ones(self.num_col) / self.num_col

            def objective(self, preds, train_data, ep=None):
                labels_mat = self._preprocess_labels(train_data, self.num_col)
                preds_mat = self._preprocess_predictions(preds, self.num_col, ep)
                grad_i = self.grad(labels_mat, preds_mat)
                hess_i = self.hess(preds_mat)
                return grad_i.sum(axis=1), hess_i.sum(axis=1), grad_i, hess_i

        # Register custom loss
        LossFactory.register("custom", CustomLoss)

        # Verify registration
        assert "custom" in LossFactory.get_available_losses()

        # Verify can create instance
        loss = LossFactory.create(loss_type="custom", **factory_params)
        assert isinstance(loss, CustomLoss)
        assert isinstance(loss, BaseLossFunction)

        # Cleanup: remove custom registration
        del LossFactory._registry["custom"]

    def test_register_requires_base_loss_inheritance(self):
        """Test that register enforces BaseLossFunction inheritance."""

        # Create class that doesn't inherit from BaseLossFunction
        class NotALoss:
            pass

        # Should raise TypeError
        with pytest.raises(TypeError, match="must inherit from BaseLossFunction"):
            LossFactory.register("invalid", NotALoss)

    def test_registered_loss_becomes_available(self):
        """Test registered loss appears in available losses."""

        class AnotherCustomLoss(BaseLossFunction):
            def compute_weights(self, labels_mat, preds_mat, iteration):
                return np.ones(self.num_col) / self.num_col

            def objective(self, preds, train_data, ep=None):
                labels_mat = self._preprocess_labels(train_data, self.num_col)
                preds_mat = self._preprocess_predictions(preds, self.num_col, ep)
                grad_i = self.grad(labels_mat, preds_mat)
                hess_i = self.hess(preds_mat)
                return grad_i.sum(axis=1), hess_i.sum(axis=1), grad_i, hess_i

        # Before registration
        available_before = LossFactory.get_available_losses()
        assert "another_custom" not in available_before

        # Register
        LossFactory.register("another_custom", AnotherCustomLoss)

        # After registration
        available_after = LossFactory.get_available_losses()
        assert "another_custom" in available_after

        # Cleanup
        del LossFactory._registry["another_custom"]

    def test_can_override_existing_registration(self, factory_params):
        """Test that registering with existing name overrides."""

        # Create alternative implementation
        class AlternativeFixedLoss(BaseLossFunction):
            def compute_weights(self, labels_mat, preds_mat, iteration):
                return np.ones(self.num_col) / self.num_col

            def objective(self, preds, train_data, ep=None):
                labels_mat = self._preprocess_labels(train_data, self.num_col)
                preds_mat = self._preprocess_predictions(preds, self.num_col, ep)
                grad_i = self.grad(labels_mat, preds_mat)
                hess_i = self.hess(preds_mat)
                return grad_i.sum(axis=1), hess_i.sum(axis=1), grad_i, hess_i

        # Save original
        original_class = LossFactory._registry["fixed"]

        try:
            # Override registration
            LossFactory.register("fixed", AlternativeFixedLoss)

            # Create instance - should use new class
            loss = LossFactory.create(loss_type="fixed", **factory_params)
            assert isinstance(loss, AlternativeFixedLoss)
            assert not isinstance(loss, FixedWeightLoss)

        finally:
            # Restore original
            LossFactory._registry["fixed"] = original_class


# ============================================================================
# Test Class 5: Integration Tests
# ============================================================================


class TestLossFactoryIntegration:
    """Integration tests for factory with actual loss functions."""

    def test_created_losses_have_different_weights(self, factory_params):
        """Test that different loss types produce different weight behaviors."""
        fixed_loss = LossFactory.create(loss_type="fixed", **factory_params)
        adaptive_loss = LossFactory.create(loss_type="adaptive", **factory_params)

        # Fixed loss should have fixed weights
        assert hasattr(fixed_loss, "weights")

        # Adaptive loss should have weight evolution
        assert hasattr(adaptive_loss, "weight_history")

    def test_factory_preserves_loss_functionality(self, factory_params):
        """Test that factory-created losses maintain full functionality."""
        loss = LossFactory.create(loss_type="adaptive", **factory_params)

        # Create mock data
        mock_data = Mock()
        mock_data.get_label.return_value = np.random.randint(0, 2, size=16).astype(
            float
        )
        preds = np.random.rand(16)

        # Test that loss methods work
        grad, hess, grad_i, hess_i = loss.objective(preds, mock_data)

        assert grad is not None
        assert hess is not None
        assert grad_i is not None
        assert hess_i is not None

    def test_all_loss_types_use_same_interface(self, factory_params):
        """Test that all loss types conform to BaseLossFunction interface."""
        loss_types = ["fixed", "adaptive", "adaptive_kd"]

        for loss_type in loss_types:
            loss = LossFactory.create(loss_type=loss_type, **factory_params)

            # All should have these methods from BaseLossFunction
            assert hasattr(loss, "objective")
            assert hasattr(loss, "evaluate")
            assert hasattr(loss, "compute_weights")
            assert hasattr(loss, "_preprocess_predictions")
            assert hasattr(loss, "_preprocess_labels")
