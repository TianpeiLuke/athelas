"""
Tests for ModelFactory implementation.

Tests factory pattern for creating model instances with type safety
and validation.

Following pytest best practices:
- Read source code first (model_factory.py analyzed)
- Test actual implementation behavior
- Minimal mocking (mock dependencies, test real factory logic)
- Test both happy path and error conditions
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import from actual source (following best practice)
from athelas.models.lightgbmmt.factory.model_factory import ModelFactory
from athelas.models.lightgbmmt.base.base_model import BaseMultiTaskModel
from athelas.models.lightgbmmt.implementations.mtgbm_model import MtgbmModel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_loss_function():
    """Create mock loss function."""
    mock = Mock()
    mock.num_col = 4
    return mock


@pytest.fixture
def mock_training_state():
    """Create mock training state."""
    mock = Mock()
    mock.current_epoch = 0
    mock.best_epoch = 0
    mock.best_metric = 0.0
    mock.epochs_without_improvement = 0
    return mock


@pytest.fixture
def mock_hyperparams():
    """Create mock hyperparameters."""
    mock = Mock()
    # Add common hyperparameter attributes
    mock.num_boost_round = 100
    mock.early_stopping_rounds = 10
    return mock


@pytest.fixture
def factory_params(mock_loss_function, mock_training_state, mock_hyperparams):
    """Create standard factory parameters."""
    return {
        "loss_function": mock_loss_function,
        "training_state": mock_training_state,
        "hyperparams": mock_hyperparams,
    }


# ============================================================================
# Test Class 1: Model Creation
# ============================================================================


class TestModelFactoryCreation:
    """Tests for creating model instances via factory."""

    def test_create_mtgbm_model(self, factory_params):
        """Test creating MtgbmModel via factory."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify correct type
        assert isinstance(model, MtgbmModel)
        assert isinstance(model, BaseMultiTaskModel)

    def test_create_passes_loss_function(self, factory_params):
        """Test that loss_function is passed to model."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify loss function passed
        assert model.loss_function == factory_params["loss_function"]

    def test_create_passes_training_state(self, factory_params):
        """Test that training_state is passed to model."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify training state passed
        assert model.training_state == factory_params["training_state"]

    def test_create_passes_hyperparams(self, factory_params):
        """Test that hyperparams is passed to model."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify hyperparams passed
        assert model.hyperparams == factory_params["hyperparams"]

    def test_create_returns_functional_model(self, factory_params):
        """Test that created models have required methods."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify model has required public methods (from BaseMultiTaskModel)
        assert hasattr(model, "train")
        assert hasattr(model, "save")
        assert hasattr(model, "load")

    def test_create_with_different_dependencies(
        self, mock_training_state, mock_hyperparams
    ):
        """Test creating model with different loss functions."""
        # Create different loss function mocks
        loss_fn_1 = Mock()
        loss_fn_1.num_col = 2

        loss_fn_2 = Mock()
        loss_fn_2.num_col = 10

        # Create models with different dependencies
        model_1 = ModelFactory.create(
            model_type="mtgbm",
            loss_function=loss_fn_1,
            training_state=mock_training_state,
            hyperparams=mock_hyperparams,
        )

        model_2 = ModelFactory.create(
            model_type="mtgbm",
            loss_function=loss_fn_2,
            training_state=mock_training_state,
            hyperparams=mock_hyperparams,
        )

        # Verify different loss functions
        assert model_1.loss_function.num_col == 2
        assert model_2.loss_function.num_col == 10


# ============================================================================
# Test Class 2: Validation and Error Handling
# ============================================================================


class TestModelFactoryValidation:
    """Tests for factory validation and error handling."""

    def test_unknown_model_type_raises_error(self, factory_params):
        """Test that unknown model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type: 'nonexistent'"):
            ModelFactory.create(model_type="nonexistent", **factory_params)

    def test_error_message_lists_available_types(self, factory_params):
        """Test error message includes available model types."""
        with pytest.raises(ValueError, match="Available types:"):
            ModelFactory.create(model_type="invalid", **factory_params)

        # Verify registered type mentioned
        try:
            ModelFactory.create(model_type="invalid", **factory_params)
        except ValueError as e:
            error_msg = str(e)
            assert "mtgbm" in error_msg

    def test_missing_loss_function_accepted_at_creation(
        self, mock_training_state, mock_hyperparams
    ):
        """Test that None loss_function is accepted at creation (fails later when used)."""
        # Model accepts None during initialization (validation happens during use)
        model = ModelFactory.create(
            model_type="mtgbm",
            loss_function=None,
            training_state=mock_training_state,
            hyperparams=mock_hyperparams,
        )
        assert model.loss_function is None

    def test_missing_training_state_accepted_at_creation(
        self, mock_loss_function, mock_hyperparams
    ):
        """Test that None training_state is accepted at creation (fails later when used)."""
        # Model accepts None during initialization (validation happens during use)
        model = ModelFactory.create(
            model_type="mtgbm",
            loss_function=mock_loss_function,
            training_state=None,
            hyperparams=mock_hyperparams,
        )
        assert model.training_state is None

    def test_missing_hyperparams_raises_error(
        self, mock_loss_function, mock_training_state
    ):
        """Test that None hyperparams raises AttributeError during initialization."""
        # Model tries to access hyperparams.loss_type during __init__, causing AttributeError
        with pytest.raises(AttributeError):
            ModelFactory.create(
                model_type="mtgbm",
                loss_function=mock_loss_function,
                training_state=mock_training_state,
                hyperparams=None,
            )


# ============================================================================
# Test Class 3: Registry Management
# ============================================================================


class TestModelFactoryRegistry:
    """Tests for factory registry management."""

    def test_get_available_models_returns_list(self):
        """Test get_available_models returns list of registered types."""
        available = ModelFactory.get_available_models()

        assert isinstance(available, list)
        assert len(available) > 0

    def test_get_available_models_contains_default_types(self):
        """Test default model types are available."""
        available = ModelFactory.get_available_models()

        # Check expected default type present
        assert "mtgbm" in available

    def test_registry_contains_correct_class(self):
        """Test registry maps to correct model class."""
        # Access registry directly for verification
        registry = ModelFactory._registry

        assert registry["mtgbm"] == MtgbmModel

    def test_all_registered_types_are_creatable(self, factory_params):
        """Test that all registered model types can be created."""
        available = ModelFactory.get_available_models()

        for model_type in available:
            model = ModelFactory.create(model_type=model_type, **factory_params)
            assert isinstance(model, BaseMultiTaskModel)


# ============================================================================
# Test Class 4: Extensibility
# ============================================================================


class TestModelFactoryExtensibility:
    """Tests for extending factory with custom models."""

    def test_register_custom_model(self, factory_params):
        """Test registering a custom model."""

        # Create custom model class (must implement all abstract methods)
        class CustomModel(BaseMultiTaskModel):
            def _prepare_data(self, train_df, val_df, test_df):
                return train_df, val_df, test_df

            def _initialize_model(self):
                pass

            def _train_model(self, train_data, val_data):
                return {}

            def _predict(self, data):
                return np.array([])

            def _save_model(self, output_path):
                pass

            def _load_model(self, model_path):
                pass

        # Register custom model
        ModelFactory.register("custom", CustomModel)

        # Verify registration
        assert "custom" in ModelFactory.get_available_models()

        # Verify can create instance
        model = ModelFactory.create(model_type="custom", **factory_params)
        assert isinstance(model, CustomModel)
        assert isinstance(model, BaseMultiTaskModel)

        # Cleanup: remove custom registration
        del ModelFactory._registry["custom"]

    def test_register_requires_base_model_inheritance(self):
        """Test that register enforces BaseMultiTaskModel inheritance."""

        # Create class that doesn't inherit from BaseMultiTaskModel
        class NotAModel:
            pass

        # Should raise TypeError
        with pytest.raises(TypeError, match="must inherit from BaseMultiTaskModel"):
            ModelFactory.register("invalid", NotAModel)

    def test_registered_model_becomes_available(self):
        """Test registered model appears in available models."""

        class AnotherCustomModel(BaseMultiTaskModel):
            def _prepare_data(self, train_df, val_df, test_df):
                return train_df, val_df, test_df

            def _initialize_model(self):
                pass

            def _train_model(self, train_data, val_data):
                return {}

            def _predict(self, data):
                return np.array([])

            def _save_model(self, output_path):
                pass

            def _load_model(self, model_path):
                pass

        # Before registration
        available_before = ModelFactory.get_available_models()
        assert "another_custom" not in available_before

        # Register
        ModelFactory.register("another_custom", AnotherCustomModel)

        # After registration
        available_after = ModelFactory.get_available_models()
        assert "another_custom" in available_after

        # Cleanup
        del ModelFactory._registry["another_custom"]

    def test_can_override_existing_registration(self, factory_params):
        """Test that registering with existing name overrides."""

        # Create alternative implementation (must implement all abstract methods)
        class AlternativeMtgbmModel(BaseMultiTaskModel):
            def __init__(self, loss_function, training_state, hyperparams):
                super().__init__(loss_function, training_state, hyperparams)
                self.is_alternative = True  # Distinguishing attribute

            def _prepare_data(self, train_df, val_df, test_df):
                return train_df, val_df, test_df

            def _initialize_model(self):
                pass

            def _train_model(self, train_data, val_data):
                return {}

            def _predict(self, data):
                return np.array([])

            def _save_model(self, output_path):
                pass

            def _load_model(self, model_path):
                pass

        # Save original
        original_class = ModelFactory._registry["mtgbm"]

        try:
            # Override registration
            ModelFactory.register("mtgbm", AlternativeMtgbmModel)

            # Create instance - should use new class
            model = ModelFactory.create(model_type="mtgbm", **factory_params)
            assert isinstance(model, AlternativeMtgbmModel)
            assert hasattr(model, "is_alternative")
            assert model.is_alternative == True

        finally:
            # Restore original
            ModelFactory._registry["mtgbm"] = original_class


# ============================================================================
# Test Class 5: Integration Tests
# ============================================================================


class TestModelFactoryIntegration:
    """Integration tests for factory with actual model classes."""

    def test_created_model_has_all_dependencies(self, factory_params):
        """Test that factory-created model has all required dependencies."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # Verify all dependencies present
        assert model.loss_function is not None
        assert model.training_state is not None
        assert model.hyperparams is not None

    def test_factory_preserves_model_interface(self, factory_params):
        """Test that factory-created models maintain BaseMultiTaskModel interface."""
        model = ModelFactory.create(model_type="mtgbm", **factory_params)

        # All models should have these public methods from BaseMultiTaskModel
        assert hasattr(model, "train")
        assert hasattr(model, "save")
        assert hasattr(model, "load")

        # Methods should be callable
        assert callable(model.train)
        assert callable(model.save)
        assert callable(model.load)

    def test_all_model_types_use_same_interface(self, factory_params):
        """Test that all model types conform to BaseMultiTaskModel interface."""
        model_types = ModelFactory.get_available_models()

        for model_type in model_types:
            model = ModelFactory.create(model_type=model_type, **factory_params)

            # All should inherit from BaseMultiTaskModel
            assert isinstance(model, BaseMultiTaskModel)

            # All should have required public methods
            assert hasattr(model, "train")
            assert hasattr(model, "save")
            assert hasattr(model, "load")

    def test_factory_pattern_consistency_with_loss_factory(self):
        """Test that ModelFactory follows same pattern as LossFactory."""
        # Both should have registry
        assert hasattr(ModelFactory, "_registry")

        # Both should have create method
        assert hasattr(ModelFactory, "create")

        # Both should have register method
        assert hasattr(ModelFactory, "register")

        # Both should have get_available method
        assert hasattr(ModelFactory, "get_available_models")

        # Registry should be a dict
        assert isinstance(ModelFactory._registry, dict)
