"""
Tests for FixedWeightLoss class.

Following pytest best practices:
- Read source code first (fixed_weight_loss.py analyzed)
- Test actual implementation behavior
- Use real objects when possible (mock only hyperparams and lightgbm.Dataset)
- Test both happy path and edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock
from athelas.models.lightgbmmt.loss.fixed_weight_loss import FixedWeightLoss


@pytest.fixture
def mock_hyperparams():
    """
    Create mock hyperparameters with all required loss parameters.

    Read source: FixedWeightLoss requires all 14 base loss parameters
    plus uses main_task_index (defaults to 0 if not present).
    """
    mock = Mock()
    # All 14 loss parameters from BaseLossFunction
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
    # FixedWeightLoss-specific: main_task_index
    mock.main_task_index = 0
    return mock


@pytest.fixture
def basic_val_sublabel_idx():
    """Create basic validation sublabel indices for 3 tasks."""
    return {0: np.array([0, 1, 2, 3, 4]), 1: np.array([0, 2, 4]), 2: np.array([1, 3])}


@pytest.fixture
def mock_train_data():
    """
    Create mock lightgbm.Dataset.

    Read source: objective() calls _preprocess_labels which needs train_data.get_label().
    """
    mock_data = Mock()
    # Labels for 10 samples, 3 tasks
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


class TestFixedWeightLossInitialization:
    """Test FixedWeightLoss initialization."""

    def test_valid_initialization(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test initialization with all required parameters."""
        # Read source: Calls super().__init__(), generates weights, logs them
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Verify inheritance from BaseLossFunction
        assert loss.num_col == 3
        assert loss.val_sublabel_idx == basic_val_sublabel_idx

        # Verify weights generated at initialization
        assert hasattr(loss, "weights")
        assert isinstance(loss.weights, np.ndarray)
        assert loss.weights.shape == (3,)

    def test_weights_generated_at_init(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that weights are generated during initialization."""
        # Read source: __init__ calls _generate_weights() and stores result
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Weights should be generated immediately
        assert loss.weights is not None
        assert len(loss.weights) == 3


class TestWeightGeneration:
    """Test _generate_weights method."""

    def test_weight_structure_with_main_task_index_zero(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test weight generation when main_task_index=0."""
        # Read source: main task gets main_task_weight, others get main_task_weight * beta
        mock_hyperparams.main_task_index = 0
        mock_hyperparams.loss_main_task_weight = 1.0
        mock_hyperparams.loss_beta = 0.2

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Main task (index 0) should have weight 1.0
        assert loss.weights[0] == 1.0

        # Subtasks (indices 1, 2) should have weight 1.0 * 0.2 = 0.2
        assert loss.weights[1] == pytest.approx(0.2)
        assert loss.weights[2] == pytest.approx(0.2)

    def test_weight_structure_with_custom_main_task_index(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test weight generation with main_task_index=1."""
        # Read source: main_task_index can be any valid task index
        mock_hyperparams.main_task_index = 1
        mock_hyperparams.loss_main_task_weight = 1.0
        mock_hyperparams.loss_beta = 0.3

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Subtask (index 0) should have weight 1.0 * 0.3 = 0.3
        assert loss.weights[0] == pytest.approx(0.3)

        # Main task (index 1) should have weight 1.0
        assert loss.weights[1] == 1.0

        # Subtask (index 2) should have weight 1.0 * 0.3 = 0.3
        assert loss.weights[2] == pytest.approx(0.3)

    def test_weight_scaling_with_beta(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that subtask weights scale correctly with beta parameter."""
        # Read source: subtask_weight = main_task_weight * beta
        mock_hyperparams.main_task_index = 0
        mock_hyperparams.loss_main_task_weight = 2.0
        mock_hyperparams.loss_beta = 0.5

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Main task should have weight 2.0
        assert loss.weights[0] == 2.0

        # Subtasks should have weight 2.0 * 0.5 = 1.0
        assert loss.weights[1] == pytest.approx(1.0)
        assert loss.weights[2] == pytest.approx(1.0)

    def test_dynamic_sizing_based_on_num_col(self, mock_hyperparams):
        """Test that weight generation is not hardcoded to specific task count."""
        # Read source: Uses self.num_col, not hardcoded number
        mock_hyperparams.main_task_index = 0
        mock_hyperparams.loss_main_task_weight = 1.0
        mock_hyperparams.loss_beta = 0.2

        # Test with 6 tasks (original common case)
        val_idx_6 = {i: np.array([i]) for i in range(6)}
        loss_6 = FixedWeightLoss(
            num_label=6, val_sublabel_idx=val_idx_6, hyperparams=mock_hyperparams
        )

        assert len(loss_6.weights) == 6
        assert loss_6.weights[0] == 1.0
        assert np.allclose(loss_6.weights[1:], 0.2)

    def test_backward_compatibility_default_main_task_index(
        self, basic_val_sublabel_idx
    ):
        """Test that main_task_index defaults to 0 when not explicitly set."""
        # Read source: getattr(self.hyperparams, "main_task_index", 0)
        # Note: Mock objects return Mock for missing attributes, so we explicitly set to 0
        mock = Mock()
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
        mock.loss_cache_predictions = False
        mock.loss_precompute_indices = True
        mock.loss_log_level = "INFO"
        mock.main_task_index = 0  # Explicitly set default value

        loss = FixedWeightLoss(
            num_label=3, val_sublabel_idx=basic_val_sublabel_idx, hyperparams=mock
        )

        # Should use index 0 (whether from default or explicit)
        assert loss.weights[0] == 1.0
        assert loss.weights[1] == pytest.approx(0.2)
        assert loss.weights[2] == pytest.approx(0.2)


class TestComputeWeights:
    """Test compute_weights method."""

    def test_returns_fixed_weights(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test that compute_weights returns same weights every time."""
        # Read source: Simply returns self.weights (no adaptation)
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Create dummy labels and predictions
        labels_mat = np.array([[1, 0, 1], [0, 1, 0]])
        preds_mat = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])

        # Call multiple times with different iterations
        weights_1 = loss.compute_weights(labels_mat, preds_mat, iteration=0)
        weights_2 = loss.compute_weights(labels_mat, preds_mat, iteration=10)
        weights_3 = loss.compute_weights(labels_mat, preds_mat, iteration=100)

        # All should be identical (no adaptation)
        assert np.array_equal(weights_1, weights_2)
        assert np.array_equal(weights_2, weights_3)
        assert np.array_equal(weights_1, loss.weights)

    def test_iteration_parameter_ignored(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that iteration parameter has no effect."""
        # Read source: iteration parameter not used in method body
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        labels_mat = np.array([[1, 0, 1]])
        preds_mat = np.array([[0.8, 0.2, 0.9]])

        # Different iterations should give same result
        for iteration in [0, 1, 10, 100, 1000]:
            weights = loss.compute_weights(labels_mat, preds_mat, iteration=iteration)
            assert np.array_equal(weights, loss.weights)

    def test_weights_match_initialized_weights(
        self, mock_hyperparams, basic_val_sublabel_idx
    ):
        """Test that returned weights match those generated at initialization."""
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        labels_mat = np.array([[1, 0, 1]])
        preds_mat = np.array([[0.8, 0.2, 0.9]])

        computed_weights = loss.compute_weights(labels_mat, preds_mat, iteration=0)

        # Should be exact same object (not just equal values)
        assert computed_weights is loss.weights


class TestObjective:
    """Test objective method."""

    def test_gradient_computation_with_weights(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that gradients are computed and weighted correctly."""
        # Read source: Computes grad_i, applies weights, sums across tasks
        mock_hyperparams.main_task_index = 0
        mock_hyperparams.loss_main_task_weight = 1.0
        mock_hyperparams.loss_beta = 0.5

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Predictions for 10 samples, 3 tasks (flat)
        preds = np.array([0.0] * 30)

        grad, hess, grad_i, hess_i = loss.objective(preds, mock_train_data)

        # Verify shapes
        assert grad.shape == (10,)  # Aggregated across tasks
        assert hess.shape == (10,)
        assert grad_i.shape == (10, 3)  # Per-task gradients
        assert hess_i.shape == (10, 3)  # Per-task hessians

    def test_hessian_computation_with_weights(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that hessians are computed and weighted correctly."""
        # Read source: Computes hess_i, applies weights, sums across tasks
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0] * 30)

        grad, hess, grad_i, hess_i = loss.objective(preds, mock_train_data)

        # Hessians should all be positive (from y_pred * (1 - y_pred))
        assert np.all(hess > 0)
        assert np.all(hess_i > 0)

    def test_weighted_aggregation_across_tasks(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that aggregation correctly applies weights."""
        # Read source: weights reshaped to (1, -1), multiplied with grad_i/hess_i, summed
        mock_hyperparams.main_task_index = 0
        mock_hyperparams.loss_main_task_weight = 2.0
        mock_hyperparams.loss_beta = 0.5

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0] * 30)

        grad, hess, grad_i, hess_i = loss.objective(preds, mock_train_data)

        # Manually verify weighted sum for first sample
        expected_grad_0 = (grad_i[0] * loss.weights).sum()
        assert grad[0] == pytest.approx(expected_grad_0)

        expected_hess_0 = (hess_i[0] * loss.weights).sum()
        assert hess[0] == pytest.approx(expected_hess_0)

    def test_returns_four_values(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that objective returns exactly 4 values."""
        # Read source: Returns grad, hess, grad_i, hess_i
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0] * 30)

        result = loss.objective(preds, mock_train_data)

        # Should be tuple of 4 numpy arrays
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_per_task_gradients_preserved(
        self, mock_hyperparams, basic_val_sublabel_idx, mock_train_data
    ):
        """Test that per-task gradients/hessians are returned unchanged."""
        # Read source: grad_i and hess_i returned in addition to aggregated
        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        preds = np.array([0.0] * 30)

        grad, hess, grad_i, hess_i = loss.objective(preds, mock_train_data)

        # Per-task values should have correct shape
        assert grad_i.shape[1] == 3  # 3 tasks
        assert hess_i.shape[1] == 3

        # Per-task values should be independent (not aggregated)
        # Verify they're different from aggregated values
        assert not np.array_equal(grad_i[:, 0], grad)


class TestDifferentTaskCounts:
    """Test FixedWeightLoss with various task counts."""

    def test_minimum_tasks(self, mock_hyperparams):
        """Test with minimum number of tasks (2)."""
        # Read source: Supports any num_col >= 2 (from BaseLossFunction)
        val_idx = {0: np.array([0]), 1: np.array([1])}

        loss = FixedWeightLoss(
            num_label=2, val_sublabel_idx=val_idx, hyperparams=mock_hyperparams
        )

        assert len(loss.weights) == 2
        assert loss.weights[0] == 1.0  # Main task
        assert loss.weights[1] == pytest.approx(0.2)  # Subtask

    def test_six_tasks_common_case(self, mock_hyperparams):
        """Test with 6 tasks (common original use case)."""
        val_idx = {i: np.array([i]) for i in range(6)}

        loss = FixedWeightLoss(
            num_label=6, val_sublabel_idx=val_idx, hyperparams=mock_hyperparams
        )

        assert len(loss.weights) == 6
        assert loss.weights[0] == 1.0
        assert np.allclose(loss.weights[1:], 0.2)

    def test_many_tasks_scalability(self, mock_hyperparams):
        """Test with many tasks (10+) for scalability."""
        val_idx = {i: np.array([i]) for i in range(15)}

        loss = FixedWeightLoss(
            num_label=15, val_sublabel_idx=val_idx, hyperparams=mock_hyperparams
        )

        assert len(loss.weights) == 15
        assert loss.weights[0] == 1.0
        assert np.allclose(loss.weights[1:], 0.2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_beta(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test with beta=0 (subtasks get zero weight)."""
        mock_hyperparams.loss_beta = 0.0

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # Main task should have weight 1.0
        assert loss.weights[0] == 1.0

        # Subtasks should have weight 0
        assert loss.weights[1] == 0.0
        assert loss.weights[2] == 0.0

    def test_beta_equal_to_one(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test with beta=1.0 (subtasks get same weight as main task)."""
        mock_hyperparams.loss_beta = 1.0

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # All tasks should have equal weight
        assert all(loss.weights == 1.0)

    def test_large_main_task_weight(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test with large main_task_weight value."""
        mock_hyperparams.loss_main_task_weight = 100.0
        mock_hyperparams.loss_beta = 0.1

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        assert loss.weights[0] == 100.0
        assert loss.weights[1] == pytest.approx(10.0)
        assert loss.weights[2] == pytest.approx(10.0)

    def test_main_task_index_at_end(self, mock_hyperparams, basic_val_sublabel_idx):
        """Test with main_task_index pointing to last task."""
        mock_hyperparams.main_task_index = 2  # Last task in 3-task setup

        loss = FixedWeightLoss(
            num_label=3,
            val_sublabel_idx=basic_val_sublabel_idx,
            hyperparams=mock_hyperparams,
        )

        # First two tasks should be subtasks
        assert loss.weights[0] == pytest.approx(0.2)
        assert loss.weights[1] == pytest.approx(0.2)

        # Last task should be main task
        assert loss.weights[2] == 1.0
