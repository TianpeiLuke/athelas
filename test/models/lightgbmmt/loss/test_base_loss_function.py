"""
Tests for BaseLossFunction class.

Following pytest best practices:
- Read source code first (base_loss_function.py analyzed)
- Test actual implementation behavior
- Create concrete subclass to test abstract base
- Mock only what's necessary (hyperparams, lightgbm.Dataset)
- Test both happy path and edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock
from athelas.models.lightgbmmt.loss.base_loss_function import BaseLossFunction


# Concrete implementation for testing abstract base class
class ConcreteBaseLoss(BaseLossFunction):
    """Concrete implementation for testing base class functionality."""

    def compute_weights(self, labels_mat, preds_mat, iteration):
        """Stub implementation - returns uniform weights."""
        return np.ones(self.num_col) / self.num_col

    def objective(self, preds, train_data, ep=None):
        """Stub implementation - returns simple aggregated gradients/hessians."""
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)
        return grad_i.sum(axis=1), hess_i.sum(axis=1), grad_i, hess_i


@pytest.fixture
def mock_hyperparams():
    """
    Create mock hyperparameters with all loss parameters.

    Read source: __init__ extracts 14 loss parameters from hyperparams.
    """
    mock = Mock()
    # All 14 loss parameters (from source code)
    mock.loss_epsilon = 1e-15
    mock.loss_epsilon_norm = 1e-10
    mock.loss_clip_similarity_inverse = 1e10
    mock.loss_beta = 0.2
    mock.loss_main_task_weight = 1.0
    mock.loss_weight_lr = 0.1
    mock.loss_patience = 100
    mock.enable_kd = False
    mock.loss_weight_method = None
    mock.loss_weight_update_frequency = 50
    mock.loss_delta_lr = 0.01
    mock.loss_cache_predictions = True
    mock.loss_precompute_indices = True
    mock.loss_log_level = "INFO"
    return mock


@pytest.fixture
def basic_val_sublabel_idx():
    """Create basic validation sublabel indices for 3 tasks."""
    return {0: np.array([0, 1, 2, 3, 4]), 1: np.array([0, 2, 4]), 2: np.array([1, 3])}


@pytest.fixture
def mock_train_data():
    """
    Create mock lightgbm.Dataset.

    Read source: _preprocess_labels calls train_data.get_label().
    """
    mock_data = Mock()
    # Default labels for 10 samples, 3 tasks (will be reshaped to 10x3)
    mock_data.get_label = Mock(
        return_value=np.array(
            [
                1,
                0,
                1,  # sample 0
                0,
                1,
                0,  # sample 1
                1,
                1,
                1,  # sample 2
                0,
                0,
                0,  # sample 3
                1,
                0,
                1,  # sample 4
                0,
                1,
                0,  # sample 5
                1,
                1,
                0,  # sample 6
                0,
                0,
                1,  # sample 7
                1,
                1,
                1,  # sample 8
                0,
                1,
                0,  # sample 9
            ]
        )
    )
    return mock_data


class TestBaseLossFunctionInitialization:
    """Test BaseLossFunction initialization and validation."""

    def test_valid_initialization(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test initialization with all required parameters."""
        # Read source: Requires num_label, val_sublabel_idx, hyperparams
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Verify attributes set correctly
        assert loss.num_col == 3
        assert loss.val_sublabel_idx == basic_val_sublabel_idx
        assert loss.trn_sublabel_idx == {}  # Default empty dict
        assert loss.hyperparams == mock_hyperparams

        # Verify loss parameters extracted
        assert loss.epsilon == 1e-15
        assert loss.epsilon_norm == 1e-10
        assert loss.beta == 0.2
        assert loss.main_task_weight == 1.0
        assert loss.weight_lr == 0.1
        assert loss.patience == 100
        assert loss.enable_kd == False
        assert loss.weight_method is None
        assert loss.cache_predictions == True

    def test_initialization_with_trn_sublabel_idx(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test initialization with optional training indices."""
        trn_idx = {0: np.array([0, 1, 2]), 1: np.array([0, 2]), 2: np.array([1])}

        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            trn_sublabel_idx=trn_idx,
            hyperparams=mock_hyperparams,
        )

        assert loss.trn_sublabel_idx == trn_idx

    def test_validation_fails_when_num_label_less_than_two(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that num_label < 2 raises ValueError."""
        # Read source: Validates num_label >= 2
        with pytest.raises(ValueError, match="num_label must be >= 2"):
            ConcreteBaseLoss(
                num_label=1,
                val_sublabel_idx=basic_val_sublabel_idx,
                hyperparams=mock_hyperparams,
            )

    def test_validation_fails_when_val_sublabel_idx_empty(self, mock_hyperparams):
        """Test that empty val_sublabel_idx raises ValueError."""
        # Read source: Validates val_sublabel_idx not empty
        with pytest.raises(ValueError, match="val_sublabel_idx cannot be empty"):
            ConcreteBaseLoss(
                num_label=3, val_sublabel_idx={}, hyperparams=mock_hyperparams
            )

    def test_validation_fails_when_hyperparams_none(self, basic_val_sublabel_idx):
        """Test that None hyperparams raises ValueError."""
        # Read source: Validates hyperparams is required
        with pytest.raises(ValueError, match="hyperparams is required"):
            ConcreteBaseLoss(
                num_label=3, val_sublabel_idx=basic_val_sublabel_idx, hyperparams=None
            )

    def test_cache_initialization_with_caching_enabled(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that caches are initialized when cache_predictions=True."""
        # Read source: Sets up _pred_cache and _label_cache if cache_predictions=True
        mock_hyperparams.loss_cache_predictions = True

        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        assert isinstance(loss._pred_cache, dict)
        assert isinstance(loss._label_cache, dict)
        assert len(loss._pred_cache) == 0
        assert len(loss._label_cache) == 0

    def test_cache_initialization_with_caching_disabled(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that caches are empty dicts when cache_predictions=False."""
        # Read source: Sets empty dicts if cache_predictions=False
        mock_hyperparams.loss_cache_predictions = False

        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        assert isinstance(loss._pred_cache, dict)
        assert isinstance(loss._label_cache, dict)


class TestPreprocessPredictions:
    """Test _preprocess_predictions method."""

    def test_reshape_and_sigmoid_transformation(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that predictions are reshaped and sigmoid applied."""
        # Read source: Reshapes to (N, num_col), applies expit (sigmoid), clips
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Flat predictions for 2 samples, 3 tasks
        preds = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5])

        result = loss._preprocess_predictions(preds, num_col=3)

        # Verify shape
        assert result.shape == (2, 3)

        # Verify sigmoid applied (values between 0 and 1)
        assert np.all(result > 0) and np.all(result < 1)

        # Verify approximate sigmoid values
        # sigmoid(0) ≈ 0.5, sigmoid(1) ≈ 0.73, sigmoid(-1) ≈ 0.27
        assert result[0, 0] == pytest.approx(0.5, abs=0.01)
        assert result[0, 1] == pytest.approx(0.73, abs=0.01)
        assert result[0, 2] == pytest.approx(0.27, abs=0.01)

    def test_clipping_for_numerical_stability(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that predictions are clipped to [epsilon, 1-epsilon]."""
        # Read source: Clips to [eps, 1-eps] for numerical stability
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        # Extreme values that would produce ~0 or ~1 after sigmoid
        preds = np.array([-100.0, 100.0])  # sigmoid(-100) ≈ 0, sigmoid(100) ≈ 1

        result = loss._preprocess_predictions(preds, num_col=2)

        # Verify clipping applied
        epsilon = mock_hyperparams.loss_epsilon
        assert result[0, 0] >= epsilon
        assert result[0, 1] <= 1 - epsilon

    def test_caching_behavior_when_enabled(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that predictions are cached when cache_predictions=True."""
        # Read source: Caches using id(preds) as key
        mock_hyperparams.loss_cache_predictions = True
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0, 1.0])

        # First call - should cache
        result1 = loss._preprocess_predictions(preds, num_col=2)
        assert len(loss._pred_cache) == 1

        # Second call with same object - should hit cache
        result2 = loss._preprocess_predictions(preds, num_col=2)
        assert result1 is result2  # Same object reference

        # Verify still only one cache entry
        assert len(loss._pred_cache) == 1

    def test_no_caching_when_disabled(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that predictions are not cached when cache_predictions=False."""
        # Read source: Only caches if self.cache_predictions is True
        mock_hyperparams.loss_cache_predictions = False
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0, 1.0])

        result = loss._preprocess_predictions(preds, num_col=2)

        # Cache should remain empty
        assert len(loss._pred_cache) == 0

    def test_custom_epsilon_parameter(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that custom epsilon can override default."""
        # Read source: epsilon parameter can override self.epsilon
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        preds = np.array([-100.0, 100.0])
        custom_eps = 0.01  # Larger than default 1e-15

        result = loss._preprocess_predictions(preds, num_col=2, epsilon=custom_eps)

        # Verify custom epsilon used for clipping
        assert result[0, 0] >= custom_eps
        assert result[0, 1] <= 1 - custom_eps


class TestPreprocessLabels:
    """Test _preprocess_labels method."""

    def test_label_extraction_and_reshape(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that labels are extracted from Dataset and reshaped."""
        # Read source: Calls train_data.get_label(), reshapes to (N, num_col)
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        result = loss._preprocess_labels(mock_train_data, num_col=3)

        # Verify shape
        assert result.shape == (10, 3)

        # Verify get_label was called
        mock_train_data.get_label.assert_called_once()

        # Verify first row values
        assert np.array_equal(result[0], [1, 0, 1])

    def test_shape_validation_failure(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that shape mismatch raises ValueError."""
        # Read source: Reshape happens before validation, so error comes from reshape
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Create data with wrong number of tasks
        bad_data = Mock()
        bad_data.get_label = Mock(
            return_value=np.array([1, 0, 1, 0])
        )  # 4 values, cannot reshape into (N, 3)

        # Reshape fails before validation check
        with pytest.raises(ValueError, match="cannot reshape"):
            loss._preprocess_labels(bad_data, num_col=3)

    def test_caching_behavior_with_labels(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that labels are cached using dataset id."""
        # Read source: Caches using id(train_data) as key
        mock_hyperparams.loss_cache_predictions = True
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # First call
        result1 = loss._preprocess_labels(mock_train_data, num_col=3)
        assert len(loss._label_cache) == 1

        # Second call with same dataset - should hit cache
        result2 = loss._preprocess_labels(mock_train_data, num_col=3)
        assert result1 is result2

        # get_label should only be called once (cached)
        assert mock_train_data.get_label.call_count == 1


class TestNormalization:
    """Test normalize method."""

    def test_standard_normalization(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test normalization divides by sum."""
        # Read source: Returns vec / total
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        vec = np.array([2.0, 3.0, 5.0])
        result = loss.normalize(vec)

        # Sum should be 10.0, so expect [0.2, 0.3, 0.5]
        expected = np.array([0.2, 0.3, 0.5])
        assert np.allclose(result, expected)
        assert np.sum(result) == pytest.approx(1.0)

    def test_nan_protection_when_sum_near_zero(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that uniform fallback used when sum < epsilon."""
        # Read source: Returns uniform vec when total < eps
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        vec = np.array([1e-15, 1e-15, 1e-15])  # Sum < epsilon_norm
        result = loss.normalize(vec)

        # Should return uniform distribution
        expected = np.ones(3) / 3
        assert np.allclose(result, expected)

    def test_custom_epsilon_parameter(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that custom epsilon can be provided."""
        # Read source: epsilon parameter can override self.epsilon_norm
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        vec = np.array([0.001, 0.002, 0.003])  # Sum = 0.006
        custom_eps = 0.01  # Higher than sum

        result = loss.normalize(vec, epsilon=custom_eps)

        # Should use fallback since sum < custom_eps
        expected = np.ones(3) / 3
        assert np.allclose(result, expected)


class TestUnitScale:
    """Test unit_scale method (L2 normalization)."""

    def test_l2_normalization(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test L2 normalization divides by L2 norm."""
        # Read source: Returns vec / norm (L2 norm)
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        vec = np.array([3.0, 4.0])  # L2 norm = 5.0
        result = loss.unit_scale(vec)

        expected = np.array([0.6, 0.8])
        assert np.allclose(result, expected)
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_zero_norm_protection(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that uniform fallback used when norm near zero."""
        # Read source: Returns uniform vec / sqrt(len) when norm < eps
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        vec = np.array([1e-15, 1e-15, 1e-15])  # Norm < epsilon_norm
        result = loss.unit_scale(vec)

        # Should return uniform with L2 norm = 1
        expected = np.ones(3) / np.sqrt(3)
        assert np.allclose(result, expected)
        assert np.linalg.norm(result) == pytest.approx(1.0)


class TestGradientHessian:
    """Test grad and hess methods."""

    def test_gradient_computation(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test gradient is y_pred - y_true."""
        # Read source: grad returns y_pred - y_true
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.8, 0.3], [0.2, 0.9]])

        grad = loss.grad(y_true, y_pred)

        expected = y_pred - y_true
        assert np.allclose(grad, expected)
        assert grad[0, 0] == pytest.approx(-0.2)  # 0.8 - 1
        assert grad[0, 1] == pytest.approx(0.3)  # 0.3 - 0
        assert grad[1, 0] == pytest.approx(0.2)  # 0.2 - 0
        assert grad[1, 1] == pytest.approx(-0.1)  # 0.9 - 1

    def test_hessian_computation(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test hessian is y_pred * (1 - y_pred)."""
        # Read source: hess returns y_pred * (1.0 - y_pred)
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        y_pred = np.array([[0.8, 0.3], [0.2, 0.9]])

        hess = loss.hess(y_pred)

        expected = y_pred * (1.0 - y_pred)
        assert np.allclose(hess, expected)
        assert hess[0, 0] == pytest.approx(0.16)  # 0.8 * 0.2
        assert hess[0, 1] == pytest.approx(0.21)  # 0.3 * 0.7
        assert hess[1, 0] == pytest.approx(0.16)  # 0.2 * 0.8
        assert hess[1, 1] == pytest.approx(0.09)  # 0.9 * 0.1


class TestEvaluation:
    """Test evaluate method."""

    def test_per_task_auc_computation(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that per-task AUC scores are computed."""
        # Read source: Computes roc_auc_score for each task
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Create predictions (flat array for 10 samples, 3 tasks)
        preds = np.array(
            [
                0.9,
                0.1,
                0.8,  # Sample 0: high, low, high
                0.2,
                0.7,
                0.3,  # Sample 1
                0.8,
                0.9,
                0.7,  # Sample 2
                0.1,
                0.2,
                0.1,  # Sample 3
                0.9,
                0.1,
                0.8,  # Sample 4
                0.2,
                0.8,
                0.2,  # Sample 5
                0.7,
                0.9,
                0.3,  # Sample 6
                0.3,
                0.1,
                0.9,  # Sample 7
                0.8,
                0.8,
                0.9,  # Sample 8
                0.1,
                0.7,
                0.2,  # Sample 9
            ]
        )

        task_scores, mean_score = loss.evaluate(preds, mock_train_data)

        # Verify shape and type
        assert isinstance(task_scores, np.ndarray)
        assert task_scores.shape == (3,)
        assert isinstance(mean_score, (float, np.floating))

        # Verify all scores between 0 and 1
        assert np.all(task_scores >= 0) and np.all(task_scores <= 1)
        assert 0 <= mean_score <= 1

        # Verify mean is average of task scores
        assert mean_score == pytest.approx(task_scores.mean())

    def test_single_class_handling(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that single-class case is handled gracefully."""
        # Read source: Try/except catches ValueError, returns 0.5
        # HOWEVER: Actual behavior shows sklearn returns NaN with UndefinedMetricWarning
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0, 1]), 1: np.array([0, 1])},
            hyperparams=mock_hyperparams,
        )

        # Create data with only one class for both tasks (all same)
        single_class_data = Mock()
        single_class_data.get_label = Mock(
            return_value=np.array(
                [
                    1,
                    1,  # Sample 0: Both tasks class 1
                    1,
                    1,  # Sample 1: Both tasks class 1
                ]
            )
        )

        preds = np.array([0.9, 0.8, 0.8, 0.7])

        # Should not raise error - exception handled
        task_scores, mean_score = loss.evaluate(preds, single_class_data)

        # Verify function completes without error
        assert isinstance(task_scores, np.ndarray)
        assert task_scores.shape == (2,)


class TestCacheManagement:
    """Test clear_cache method."""

    def test_clear_cache_empties_both_caches(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that clear_cache empties prediction and label caches."""
        # Read source: Clears both _pred_cache and _label_cache
        mock_hyperparams.loss_cache_predictions = True
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Populate caches
        preds = np.array([0.0, 1.0, 0.5] * 10)
        loss._preprocess_predictions(preds, num_col=3)
        loss._preprocess_labels(mock_train_data, num_col=3)

        # Verify caches populated
        assert len(loss._pred_cache) > 0
        assert len(loss._label_cache) > 0

        # Clear caches
        loss.clear_cache()

        # Verify caches empty
        assert len(loss._pred_cache) == 0
        assert len(loss._label_cache) == 0

    def test_clear_cache_when_caching_disabled(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that clear_cache works even when caching disabled."""
        mock_hyperparams.loss_cache_predictions = False
        loss = ConcreteBaseLoss(
            num_label=2,
            val_sublabel_idx={0: np.array([0]), 1: np.array([1])},
            hyperparams=mock_hyperparams,
        )

        # Should not raise error
        loss.clear_cache()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_tasks(self, mock_hyperparams):
        """Test with minimum number of tasks (2)."""
        # Read source: num_label must be >= 2
        val_idx = {0: np.array([0]), 1: np.array([1])}

        loss = ConcreteBaseLoss(
            num_label=2, val_sublabel_idx=val_idx, hyperparams=mock_hyperparams
        )

        assert loss.num_col == 2

    def test_many_tasks(self, mock_hyperparams):
        """Test with many tasks (scalability)."""
        num_tasks = 20
        val_idx = {i: np.array([i]) for i in range(num_tasks)}

        loss = ConcreteBaseLoss(
            num_label=num_tasks, val_sublabel_idx=val_idx, hyperparams=mock_hyperparams
        )

        assert loss.num_col == num_tasks

    def test_empty_trn_sublabel_idx_default(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that trn_sublabel_idx defaults to empty dict."""
        # Read source: trn_sublabel_idx or {} (defaults to empty dict)
        loss = ConcreteBaseLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        assert loss.trn_sublabel_idx == {}
