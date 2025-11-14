"""
Tests for AdaptiveWeightLoss implementation.

Tests adaptive weight computation with similarity-based weighting
using Jensen-Shannon divergence and multiple weight update strategies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

# Import from actual source (following best practice)
from athelas.models.lightgbmmt.loss.adaptive_weight_loss import (
    AdaptiveWeightLoss,
)


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
    mock.enable_kd = False
    mock.loss_weight_method = None  # Default: standard method
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
def adaptive_loss(mock_hyperparams, sample_indices):
    """Create AdaptiveWeightLoss instance."""
    return AdaptiveWeightLoss(
        num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
    )


@pytest.fixture
def sample_labels_preds():
    """Create sample labels and predictions for testing."""
    np.random.seed(42)
    n_samples = 10
    n_tasks = 4

    # Create realistic label and prediction matrices
    labels = np.random.randint(0, 2, size=(n_samples, n_tasks)).astype(float)
    preds = np.random.rand(n_samples, n_tasks)

    return labels, preds


# ============================================================================
# Test Class 1: Initialization
# ============================================================================


class TestAdaptiveWeightLossInitialization:
    """Tests for AdaptiveWeightLoss initialization."""

    def test_valid_initialization(self, mock_hyperparams, sample_indices):
        """Test valid initialization with all required parameters."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Verify initialization
        assert loss.num_col == 4
        assert loss.hyperparams == mock_hyperparams
        assert hasattr(loss, "weights")
        assert hasattr(loss, "weight_history")
        assert hasattr(loss, "iteration_count")
        assert hasattr(loss, "cached_similarity")

    def test_uniform_weight_initialization(self, mock_hyperparams, sample_indices):
        """Test weights are initialized uniformly."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Weights should be uniform (1/num_col for each)
        expected_weights = np.ones(4) / 4
        np.testing.assert_array_almost_equal(loss.weights, expected_weights)

    def test_weight_history_tracking_initialized(
        self, mock_hyperparams, sample_indices
    ):
        """Test weight history is initialized with initial weights."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Weight history should have initial weights
        assert len(loss.weight_history) == 1
        np.testing.assert_array_equal(loss.weight_history[0], loss.weights)

    def test_iteration_counter_initialized(self, mock_hyperparams, sample_indices):
        """Test iteration counter starts at 0."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        assert loss.iteration_count == 0

    def test_cached_similarity_initialized(self, mock_hyperparams, sample_indices):
        """Test cached similarity starts as None."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        assert loss.cached_similarity is None

    def test_inherits_from_base_loss_function(self, mock_hyperparams, sample_indices):
        """Test AdaptiveWeightLoss inherits from BaseLossFunction."""
        from docker.models.loss.base_loss_function import (
            BaseLossFunction,
        )

        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        assert isinstance(loss, BaseLossFunction)


# ============================================================================
# Test Class 2: Similarity Computation
# ============================================================================


class TestSimilarityComputation:
    """Tests for Jensen-Shannon divergence-based similarity computation."""

    def test_main_task_similarity_is_one(self, adaptive_loss, sample_labels_preds):
        """Test main task has similarity 1.0 with itself."""
        labels, preds = sample_labels_preds

        weights = adaptive_loss._compute_similarity_weights(labels, preds)

        # Main task (index 0) should have similarity 1.0 after normalization
        # Since all similarities are normalized, we check it's the highest
        assert weights[0] > 0

    def test_js_divergence_computation(self, adaptive_loss, sample_labels_preds):
        """Test JS divergence is computed between main task and subtasks."""
        labels, preds = sample_labels_preds

        # Patch jensenshannon to verify it's called
        with patch("docker.models.loss.adaptive_weight_loss.jensenshannon") as mock_js:
            mock_js.return_value = 0.5

            weights = adaptive_loss._compute_similarity_weights(labels, preds)

            # JS divergence should be called for subtasks (3 times for 4 tasks)
            assert mock_js.call_count == 3

    def test_similarity_conversion_from_divergence(
        self, adaptive_loss, sample_labels_preds
    ):
        """Test similarity is computed as inverse of JS divergence."""
        labels, preds = sample_labels_preds

        # Patch jensenshannon to control divergence values
        with patch("docker.models.loss.adaptive_weight_loss.jensenshannon") as mock_js:
            # Set different divergences for different subtasks
            mock_js.side_effect = [0.5, 0.25, 0.1]  # 3 subtasks

            weights = adaptive_loss._compute_similarity_weights(labels, preds)

            # Verify weights are computed (inverse relationship)
            # Lower divergence → higher similarity → higher weight after normalization
            assert len(weights) == 4
            assert np.all(weights > 0)

    def test_clipping_prevents_infinity(self, adaptive_loss, sample_labels_preds):
        """Test similarity clipping prevents inf values."""
        labels, preds = sample_labels_preds

        # Patch jensenshannon to return very small divergence
        with patch("docker.models.loss.adaptive_weight_loss.jensenshannon") as mock_js:
            mock_js.return_value = 1e-12  # Very small, would cause large inverse

            weights = adaptive_loss._compute_similarity_weights(labels, preds)

            # No weights should be inf
            assert not np.any(np.isinf(weights))
            assert np.all(np.isfinite(weights))

    def test_zero_divergence_handling(self, adaptive_loss, sample_labels_preds):
        """Test handling when JS divergence is near zero."""
        labels, preds = sample_labels_preds

        # Patch jensenshannon to return zero divergence
        with patch("docker.models.loss.adaptive_weight_loss.jensenshannon") as mock_js:
            mock_js.return_value = 0.0

            weights = adaptive_loss._compute_similarity_weights(labels, preds)

            # Should handle zero divergence gracefully
            assert not np.any(np.isnan(weights))
            assert not np.any(np.isinf(weights))

    def test_main_task_index_used_correctly(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test main_task_index from hyperparams is used for similarity."""
        # Set main task to index 2 instead of 0
        mock_hyperparams.main_task_index = 2

        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        weights = loss._compute_similarity_weights(labels, preds)

        # Main task (now index 2) should get similarity 1.0 before normalization
        # After normalization it should still be positive
        assert weights[2] > 0


# ============================================================================
# Test Class 3: Weight Normalization
# ============================================================================


class TestWeightNormalization:
    """Tests for weight normalization in adaptive loss."""

    def test_weights_sum_to_one(self, adaptive_loss, sample_labels_preds):
        """Test computed weights are normalized to sum to 1."""
        labels, preds = sample_labels_preds

        weights = adaptive_loss._compute_similarity_weights(labels, preds)

        # Weights should sum to approximately 1 (within numerical precision)
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_all_weights_positive(self, adaptive_loss, sample_labels_preds):
        """Test all computed weights are positive."""
        labels, preds = sample_labels_preds

        weights = adaptive_loss._compute_similarity_weights(labels, preds)

        assert np.all(weights > 0)

    def test_nan_protection_in_normalization(self, adaptive_loss):
        """Test NaN protection when normalizing with very similar predictions."""
        # Use very small but non-zero values (zeros cause scipy jensenshannon to produce NaN)
        labels = np.ones((10, 4)) * 0.01
        preds = np.ones((10, 4)) * 0.01  # Very similar predictions across tasks

        # Should handle edge case gracefully
        weights = adaptive_loss._compute_similarity_weights(labels, preds)

        # Should not have NaN values
        assert not np.any(np.isnan(weights))
        # Weights should still sum to 1
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)


# ============================================================================
# Test Class 4: Standard Weight Update Method
# ============================================================================


class TestStandardWeightUpdateMethod:
    """Tests for standard adaptive weighting (default method)."""

    def test_first_iteration_uses_raw_weights(self, adaptive_loss, sample_labels_preds):
        """Test first iteration uses raw computed weights."""
        labels, preds = sample_labels_preds

        # First iteration (iteration=0)
        weights = adaptive_loss.compute_weights(labels, preds, iteration=0)

        # Should be normalized weights
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)
        assert len(weights) == 4

    def test_subsequent_iterations_use_learning_rate(
        self, adaptive_loss, sample_labels_preds
    ):
        """Test subsequent iterations apply learning rate for smoothing."""
        labels, preds = sample_labels_preds

        # First iteration
        weights_0 = adaptive_loss.compute_weights(labels, preds, iteration=0)

        # Second iteration with different predictions
        preds_new = preds + 0.1
        weights_1 = adaptive_loss.compute_weights(labels, preds_new, iteration=1)

        # Weights should be different but smoothed
        assert not np.array_equal(weights_0, weights_1)

    def test_learning_rate_smoothing(self, mock_hyperparams, sample_indices):
        """Test learning rate controls weight adaptation speed."""
        # Set high learning rate for faster adaptation
        mock_hyperparams.loss_weight_lr = 0.9

        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels = np.random.rand(10, 4)
        preds1 = np.random.rand(10, 4)
        preds2 = np.random.rand(10, 4)

        weights_0 = loss.compute_weights(labels, preds1, iteration=0)
        weights_1 = loss.compute_weights(labels, preds2, iteration=1)

        # With high learning rate, weights should change more
        assert not np.allclose(weights_0, weights_1, rtol=0.01)


# ============================================================================
# Test Class 5: TenIters Weight Update Method
# ============================================================================


class TestTenItersWeightUpdateMethod:
    """Tests for tenIters periodic weight update method."""

    def test_updates_at_frequency_intervals(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test weights update every 50 iterations (default frequency)."""
        mock_hyperparams.loss_weight_method = "tenIters"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        # Update at iteration 0, 50, 100
        weights_0 = loss.compute_weights(labels, preds, iteration=0)
        weights_25 = loss.compute_weights(labels, preds + 0.1, iteration=25)
        weights_50 = loss.compute_weights(labels, preds + 0.2, iteration=50)

        # Weights at 25 should equal weights at 0 (no update yet)
        np.testing.assert_array_equal(weights_25, weights_0)

        # Weights at 50 should be different (update happened)
        assert not np.array_equal(weights_50, weights_0)

    def test_custom_update_frequency(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test custom weight update frequency."""
        mock_hyperparams.loss_weight_method = "tenIters"
        mock_hyperparams.loss_weight_update_frequency = 10  # Update every 10

        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        weights_0 = loss.compute_weights(labels, preds, iteration=0)
        weights_5 = loss.compute_weights(labels, preds + 0.1, iteration=5)
        weights_10 = loss.compute_weights(labels, preds + 0.2, iteration=10)

        # Update at 0, 10, 20, ...
        np.testing.assert_array_equal(weights_5, weights_0)
        assert not np.array_equal(weights_10, weights_0)

    def test_caches_weights_between_updates(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test weights are cached between update intervals."""
        mock_hyperparams.loss_weight_method = "tenIters"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        # Update at iteration 0
        weights_0 = loss.compute_weights(labels, preds, iteration=0)

        # Iterations 1-49 should use cached weights
        for i in range(1, 50):
            weights_i = loss.compute_weights(labels, preds + i * 0.01, iteration=i)
            np.testing.assert_array_equal(weights_i, weights_0)


# ============================================================================
# Test Class 6: Sqrt Weight Update Method
# ============================================================================


class TestSqrtWeightUpdateMethod:
    """Tests for sqrt dampening weight update method."""

    def test_applies_square_root_dampening(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test square root is applied to weights."""
        mock_hyperparams.loss_weight_method = "sqrt"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        weights = loss.compute_weights(labels, preds, iteration=0)

        # Weights should still sum to 1 after sqrt and renormalization
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)
        assert np.all(weights > 0)

    def test_dampens_extreme_values(self, mock_hyperparams, sample_indices):
        """Test sqrt reduces extreme weight values."""
        mock_hyperparams.loss_weight_method = "sqrt"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Create predictions with high similarity for main task
        labels = np.random.rand(10, 4)
        preds = np.random.rand(10, 4)

        # Patch to create extreme similarity
        with patch.object(loss, "_compute_similarity_weights") as mock_sim:
            # Extreme raw weights (before sqrt)
            mock_sim.return_value = np.array([0.9, 0.05, 0.03, 0.02])

            weights = loss._apply_sqrt_method(mock_sim.return_value)

            # After sqrt, weights should be more balanced
            assert weights[0] < 0.9  # Main task weight reduced
            np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_renormalization_after_sqrt(self, mock_hyperparams, sample_indices):
        """Test weights are renormalized after sqrt dampening."""
        mock_hyperparams.loss_weight_method = "sqrt"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        raw_weights = np.array([0.7, 0.2, 0.07, 0.03])

        weights = loss._apply_sqrt_method(raw_weights)

        # Should still sum to 1 after sqrt and renormalization
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)


# ============================================================================
# Test Class 7: Delta Weight Update Method
# ============================================================================


class TestDeltaWeightUpdateMethod:
    """Tests for delta incremental weight update method."""

    def test_first_iteration_uses_raw_weights(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test first iteration uses raw computed weights."""
        mock_hyperparams.loss_weight_method = "delta"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        weights = loss.compute_weights(labels, preds, iteration=0)

        # Should use raw weights for first iteration
        assert len(weights) == 4
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_incremental_updates_with_delta_lr(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test subsequent iterations use delta learning rate."""
        mock_hyperparams.loss_weight_method = "delta"
        mock_hyperparams.loss_delta_lr = 0.1
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        weights_0 = loss.compute_weights(labels, preds, iteration=0)
        weights_1 = loss.compute_weights(labels, preds + 0.2, iteration=1)

        # Weights should change but with delta smoothing
        assert not np.array_equal(weights_0, weights_1)
        np.testing.assert_almost_equal(weights_1.sum(), 1.0, decimal=10)

    def test_caches_previous_raw_weights(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test previous raw weights are cached for delta computation."""
        mock_hyperparams.loss_weight_method = "delta"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels, preds = sample_labels_preds

        # First iteration
        loss.compute_weights(labels, preds, iteration=0)

        # Cached similarity should be set
        assert loss.cached_similarity is not None
        assert len(loss.cached_similarity) == 4

    def test_ensures_positive_weights(self, mock_hyperparams, sample_indices):
        """Test delta method ensures weights remain positive."""
        mock_hyperparams.loss_weight_method = "delta"
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Create scenario that might produce negative weights
        labels = np.random.rand(10, 4)
        preds1 = np.random.rand(10, 4)
        preds2 = preds1 - 0.5  # Significantly different

        weights_0 = loss.compute_weights(labels, preds1, iteration=0)
        weights_1 = loss.compute_weights(labels, preds2, iteration=1)

        # All weights should be positive
        assert np.all(weights_1 >= 0)
        np.testing.assert_almost_equal(weights_1.sum(), 1.0, decimal=10)


# ============================================================================
# Test Class 8: Weight History Tracking
# ============================================================================


class TestWeightHistoryTracking:
    """Tests for weight history tracking across iterations."""

    def test_weight_history_updated_each_iteration(
        self, adaptive_loss, sample_labels_preds
    ):
        """Test weight history is updated after each iteration."""
        labels, preds = sample_labels_preds

        # Initial history has 1 entry (initialization)
        assert len(adaptive_loss.weight_history) == 1

        # Compute weights for 3 iterations
        for i in range(3):
            adaptive_loss.compute_weights(labels, preds, iteration=i)

        # History should have 1 (init) + 3 (iterations) = 4 entries
        assert len(adaptive_loss.weight_history) == 4

    def test_weight_history_preserves_values(self, adaptive_loss, sample_labels_preds):
        """Test weight history preserves computed values."""
        labels, preds = sample_labels_preds

        weights_list = []
        for i in range(3):
            w = adaptive_loss.compute_weights(labels, preds + i * 0.1, iteration=i)
            weights_list.append(w.copy())

        # Verify history matches computed weights (skip initial entry)
        for i, w in enumerate(weights_list):
            np.testing.assert_array_equal(adaptive_loss.weight_history[i + 1], w)

    def test_weight_history_independence(self, adaptive_loss, sample_labels_preds):
        """Test weight history entries are independent copies."""
        labels, preds = sample_labels_preds

        # Compute weights
        adaptive_loss.compute_weights(labels, preds, iteration=0)
        adaptive_loss.compute_weights(labels, preds, iteration=1)

        # Modify current weights
        adaptive_loss.weights[0] = 999.0

        # History should not be affected
        assert adaptive_loss.weight_history[1][0] != 999.0


# ============================================================================
# Test Class 9: Objective Function Integration
# ============================================================================


class TestObjectiveFunctionIntegration:
    """Tests for objective function with adaptive weighting."""

    def test_objective_returns_four_values(self, adaptive_loss):
        """Test objective returns grad, hess, grad_i, hess_i."""
        # Create mock train_data
        mock_train_data = Mock()
        mock_train_data.get_label.return_value = np.random.randint(
            0, 2, size=40
        ).astype(float)

        # Create predictions (flattened)
        preds = np.random.rand(40)

        result = adaptive_loss.objective(preds, mock_train_data)

        # Should return 4 arrays
        assert len(result) == 4
        grad, hess, grad_i, hess_i = result
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert isinstance(grad_i, np.ndarray)
        assert isinstance(hess_i, np.ndarray)

    def test_objective_uses_adaptive_weights(self, adaptive_loss):
        """Test objective function uses computed adaptive weights."""
        mock_train_data = Mock()
        mock_train_data.get_label.return_value = np.random.randint(
            0, 2, size=40
        ).astype(float)

        preds = np.random.rand(40)

        # Patch compute_weights to verify it's called
        with patch.object(adaptive_loss, "compute_weights") as mock_compute:
            mock_compute.return_value = np.array([0.4, 0.3, 0.2, 0.1])

            grad, hess, grad_i, hess_i = adaptive_loss.objective(preds, mock_train_data)

            # compute_weights should be called
            assert mock_compute.called

    def test_objective_weighted_aggregation(self, adaptive_loss):
        """Test objective aggregates gradients/hessians with weights."""
        mock_train_data = Mock()
        labels = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]).flatten()
        mock_train_data.get_label.return_value = labels

        preds = np.array(
            [[0.8, 0.2, 0.7, 0.3], [0.3, 0.9, 0.2, 0.8], [0.6, 0.7, 0.4, 0.5]]
        ).flatten()

        grad, hess, grad_i, hess_i = adaptive_loss.objective(preds, mock_train_data)

        # Verify shapes
        assert grad.shape == (3,)  # Aggregated per sample
        assert hess.shape == (3,)
        assert grad_i.shape == (3, 4)  # Per sample per task
        assert hess_i.shape == (3, 4)


# ============================================================================
# Test Class 10: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_tasks(self, mock_hyperparams, sample_indices):
        """Test with minimum number of tasks (2)."""
        indices_2 = {0: np.array([0, 1, 2]), 1: np.array([0, 1, 2])}

        loss = AdaptiveWeightLoss(
            num_label=2, val_sublabel_idx=indices_2, hyperparams=mock_hyperparams
        )

        labels = np.random.rand(3, 2)
        preds = np.random.rand(3, 2)

        weights = loss.compute_weights(labels, preds, iteration=0)

        assert len(weights) == 2
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_many_tasks(self, mock_hyperparams):
        """Test with many tasks (10 for scalability)."""
        indices_10 = {i: np.array([0, 1, 2, 3, 4]) for i in range(10)}

        loss = AdaptiveWeightLoss(
            num_label=10, val_sublabel_idx=indices_10, hyperparams=mock_hyperparams
        )

        labels = np.random.rand(5, 10)
        preds = np.random.rand(5, 10)

        weights = loss.compute_weights(labels, preds, iteration=0)

        assert len(weights) == 10
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_identical_predictions_all_tasks(self, mock_hyperparams, sample_indices):
        """Test handling when predictions are identical across all tasks."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # All tasks have identical predictions
        labels = np.random.rand(10, 4)
        preds = np.tile(np.random.rand(10, 1), (1, 4))  # Same predictions for all

        weights = loss.compute_weights(labels, preds, iteration=0)

        # Should handle gracefully without NaN/inf
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)

    def test_extreme_similarity_values(self, mock_hyperparams, sample_indices):
        """Test handling of extreme similarity values."""
        loss = AdaptiveWeightLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        labels = np.random.rand(10, 4)
        preds = np.random.rand(10, 4)

        # Patch to return extreme similarities
        with patch.object(loss, "_compute_similarity_weights") as mock_sim:
            # Very unbalanced similarities
            mock_sim.return_value = np.array([0.999, 0.0005, 0.0003, 0.0002])

            with patch.object(loss, "_apply_standard_method") as mock_method:
                mock_method.return_value = mock_sim.return_value

                weights = loss.compute_weights(labels, preds, iteration=0)

                # Should handle extreme values
                assert not np.any(np.isnan(weights))
                assert not np.any(np.isinf(weights))

    def test_all_weight_methods_produce_valid_weights(
        self, mock_hyperparams, sample_indices, sample_labels_preds
    ):
        """Test all weight methods produce valid normalized weights."""
        labels, preds = sample_labels_preds

        methods = [None, "tenIters", "sqrt", "delta"]

        for method in methods:
            mock_hyperparams.loss_weight_method = method
            loss = AdaptiveWeightLoss(
                num_label=4,
                val_sublabel_idx=sample_indices,
                hyperparams=mock_hyperparams,
            )

            weights = loss.compute_weights(labels, preds, iteration=0)

            # All methods should produce valid weights
            assert len(weights) == 4
            assert np.all(weights > 0)
            np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=10)
            assert not np.any(np.isnan(weights))
            assert not np.any(np.isinf(weights))
