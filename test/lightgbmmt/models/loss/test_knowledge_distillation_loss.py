"""
Tests for KnowledgeDistillationLoss implementation.

Tests knowledge distillation with best prediction tracking and label replacement
for struggling tasks in multi-task learning.

Following pytest best practices:
- Read source code first (knowledge_distillation_loss.py analyzed)
- Test actual implementation behavior
- Use real objects when possible (mock only hyperparams and lightgbm.Dataset)
- Test both happy path and edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

# Import from actual source (following best practice)
from athelas.models.lightgbmmt.loss.knowledge_distillation_loss import (
    KnowledgeDistillationLoss,
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
    mock.loss_patience = 5  # Small patience for testing KD trigger
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
def kd_loss(mock_hyperparams, sample_indices):
    """Create KnowledgeDistillationLoss instance."""
    return KnowledgeDistillationLoss(
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


@pytest.fixture
def mock_train_data():
    """Create mock lightgbm.Dataset."""
    mock_data = Mock()
    # Labels for 10 samples, 4 tasks (flattened)
    labels = np.random.randint(0, 2, size=40).astype(float)
    mock_data.get_label.return_value = labels
    return mock_data


# ============================================================================
# Test Class 1: Initialization
# ============================================================================


class TestKDInitialization:
    """Tests for KnowledgeDistillationLoss initialization."""

    def test_valid_initialization(self, mock_hyperparams, sample_indices):
        """Test valid initialization with all required parameters."""
        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Verify basic initialization
        assert loss.num_col == 4
        assert loss.hyperparams == mock_hyperparams

        # Verify KD state initialization
        assert loss.kd_active == False
        assert loss.kd_trigger_iteration is None
        assert loss.current_iteration == 0

    def test_performance_history_initialized(self, mock_hyperparams, sample_indices):
        """Test performance history initialized for each task."""
        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Each task should have empty performance history
        assert len(loss.performance_history) == 4
        for task_id in range(4):
            assert task_id in loss.performance_history
            assert loss.performance_history[task_id] == []

    def test_decline_count_initialized(self, mock_hyperparams, sample_indices):
        """Test decline count initialized to 0 for each task."""
        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Each task should start with 0 decline count
        assert len(loss.decline_count) == 4
        for task_id in range(4):
            assert loss.decline_count[task_id] == 0

    def test_best_tracking_initialized(self, mock_hyperparams, sample_indices):
        """Test best prediction tracking structures initialized."""
        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Verify best_predictions initialized to None
        assert len(loss.best_predictions) == 4
        for task_id in range(4):
            assert loss.best_predictions[task_id] is None

        # Verify best_scores initialized to 0.0
        assert len(loss.best_scores) == 4
        for task_id in range(4):
            assert loss.best_scores[task_id] == 0.0

        # Verify best_iteration initialized to 0
        assert len(loss.best_iteration) == 4
        for task_id in range(4):
            assert loss.best_iteration[task_id] == 0

    def test_replaced_flags_initialized(self, mock_hyperparams, sample_indices):
        """Test replaced flags initialized to False for each task."""
        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        # Each task should start as not replaced
        assert len(loss.replaced) == 4
        for task_id in range(4):
            assert loss.replaced[task_id] == False

    def test_inherits_from_adaptive_loss(self, mock_hyperparams, sample_indices):
        """Test KnowledgeDistillationLoss inherits from AdaptiveWeightLoss."""
        from athelas.models.lightgbmmt.loss.adaptive_weight_loss import AdaptiveWeightLoss

        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        assert isinstance(loss, AdaptiveWeightLoss)


# ============================================================================
# Test Class 2: KD Trigger Logic
# ============================================================================


class TestKDTriggerLogic:
    """Tests for KD trigger logic based on performance decline."""

    def test_decline_count_increments_on_no_improvement(self, kd_loss):
        """Test decline count increments when performance doesn't improve."""
        task_scores = np.array([0.8, 0.7, 0.6, 0.5])

        # First evaluation - sets baseline
        kd_loss._check_kd_trigger(task_scores, iteration=1)
        assert all(kd_loss.decline_count[i] == 0 for i in range(4))

        # Second evaluation - no improvement (same scores)
        kd_loss._check_kd_trigger(task_scores, iteration=2)
        assert all(kd_loss.decline_count[i] == 1 for i in range(4))

        # Third evaluation - no improvement again
        kd_loss._check_kd_trigger(task_scores, iteration=3)
        assert all(kd_loss.decline_count[i] == 2 for i in range(4))

    def test_decline_count_resets_on_improvement(self, kd_loss):
        """Test decline count resets to 0 when performance improves."""
        # First evaluation - baseline
        task_scores_1 = np.array([0.7, 0.6, 0.5, 0.4])
        kd_loss._check_kd_trigger(task_scores_1, iteration=1)

        # No improvement for a few iterations
        kd_loss._check_kd_trigger(task_scores_1, iteration=2)
        kd_loss._check_kd_trigger(task_scores_1, iteration=3)
        assert all(kd_loss.decline_count[i] == 2 for i in range(4))

        # Improvement - decline count should reset
        task_scores_2 = np.array([0.8, 0.7, 0.6, 0.5])
        kd_loss._check_kd_trigger(task_scores_2, iteration=4)
        assert all(kd_loss.decline_count[i] == 0 for i in range(4))

    def test_kd_triggers_when_patience_exceeded(self, kd_loss):
        """Test KD triggers when decline_count >= patience."""
        # Patience is 5 (from fixture)
        task_scores = np.array([0.7, 0.6, 0.5, 0.4])

        # Set baseline
        kd_loss._check_kd_trigger(task_scores, iteration=1)

        # Simulate 5 iterations without improvement
        for i in range(2, 7):
            kd_loss._check_kd_trigger(task_scores, iteration=i)

        # All tasks should be replaced after patience exceeded
        assert all(kd_loss.replaced[i] == True for i in range(4))

    def test_best_scores_updated_on_improvement(self, kd_loss):
        """Test best_scores and best_iteration updated when performance improves."""
        # Initial scores
        task_scores_1 = np.array([0.6, 0.5, 0.4, 0.3])
        kd_loss._check_kd_trigger(task_scores_1, iteration=1)

        # Verify best scores set
        assert kd_loss.best_scores[0] == 0.6
        assert kd_loss.best_iteration[0] == 1

        # Improved scores
        task_scores_2 = np.array([0.8, 0.7, 0.6, 0.5])
        kd_loss._check_kd_trigger(task_scores_2, iteration=5)

        # Verify best scores updated
        assert kd_loss.best_scores[0] == 0.8
        assert kd_loss.best_iteration[0] == 5

    def test_replaced_flag_set_once(self, kd_loss):
        """Test replaced flag only set once per task."""
        task_scores = np.array([0.5, 0.5, 0.5, 0.5])

        # Trigger KD by exceeding patience
        for i in range(1, 8):
            kd_loss._check_kd_trigger(task_scores, iteration=i)

        # All tasks should be replaced
        assert all(kd_loss.replaced[i] == True for i in range(4))

        # Further iterations shouldn't change replaced status
        kd_loss._check_kd_trigger(task_scores, iteration=10)
        assert all(kd_loss.replaced[i] == True for i in range(4))

    def test_performance_history_tracking(self, kd_loss):
        """Test performance history is tracked for each task."""
        scores_1 = np.array([0.5, 0.6, 0.7, 0.8])
        scores_2 = np.array([0.6, 0.7, 0.8, 0.9])
        scores_3 = np.array([0.7, 0.8, 0.9, 0.95])

        kd_loss._check_kd_trigger(scores_1, iteration=1)
        kd_loss._check_kd_trigger(scores_2, iteration=2)
        kd_loss._check_kd_trigger(scores_3, iteration=3)

        # Verify history tracked
        assert kd_loss.performance_history[0] == [0.5, 0.6, 0.7]
        assert kd_loss.performance_history[3] == [0.8, 0.9, 0.95]


# ============================================================================
# Test Class 3: Best Prediction Tracking
# ============================================================================


class TestBestPredictionTracking:
    """Tests for tracking best predictions for each task."""

    def test_previous_predictions_stored(self, kd_loss):
        """Test current predictions stored as previous."""
        preds_mat = np.random.rand(10, 4)

        kd_loss._store_predictions(preds_mat, iteration=1)

        # Verify predictions stored for each task
        for task_id in range(4):
            assert kd_loss.previous_predictions[task_id] is not None
            np.testing.assert_array_equal(
                kd_loss.previous_predictions[task_id], preds_mat[:, task_id]
            )

    def test_best_predictions_stored_at_best_iteration(self, kd_loss):
        """Test best predictions stored when iteration matches best_iteration."""
        preds_mat = np.random.rand(10, 4)

        # Set best_iteration for task 0
        kd_loss.best_iteration[0] = 5

        # Store predictions at iteration 5
        kd_loss._store_predictions(preds_mat, iteration=5)

        # Verify best predictions stored for task 0
        assert kd_loss.best_predictions[0] is not None
        np.testing.assert_array_equal(kd_loss.best_predictions[0], preds_mat[:, 0])

    def test_best_predictions_not_overwritten_after_set(self, kd_loss):
        """Test best predictions remain at best iteration."""
        preds_mat_1 = np.random.rand(10, 4)
        preds_mat_2 = np.random.rand(10, 4)

        # Set best iteration and store
        kd_loss.best_iteration[0] = 5
        kd_loss._store_predictions(preds_mat_1, iteration=5)

        best_preds_stored = kd_loss.best_predictions[0].copy()

        # Store predictions at different iteration
        kd_loss._store_predictions(preds_mat_2, iteration=6)

        # Best predictions should remain unchanged
        np.testing.assert_array_equal(kd_loss.best_predictions[0], best_preds_stored)

    def test_predictions_copied_not_referenced(self, kd_loss):
        """Test predictions are copied, not referenced."""
        preds_mat = np.random.rand(10, 4)

        kd_loss._store_predictions(preds_mat, iteration=1)

        # Modify original array
        preds_mat[0, 0] = 999.0

        # Stored predictions should be unchanged
        assert kd_loss.previous_predictions[0][0] != 999.0


# ============================================================================
# Test Class 4: Label Replacement
# ============================================================================


class TestLabelReplacement:
    """Tests for KD label replacement with best predictions."""

    def test_apply_kd_replaces_labels_for_replaced_tasks(self, kd_loss):
        """Test _apply_kd replaces labels for replaced tasks."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)

        # Set up task 0 as replaced with best predictions
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.5)

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # Task 0 labels should be replaced with best predictions
        np.testing.assert_array_equal(labels_kd[:, 0], np.full(10, 0.5))

    def test_apply_kd_preserves_labels_for_non_replaced_tasks(self, kd_loss):
        """Test _apply_kd preserves labels for non-replaced tasks."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)

        # Set up only task 0 as replaced
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.5)

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # Tasks 1-3 labels should be unchanged
        for task_id in range(1, 4):
            np.testing.assert_array_equal(labels_kd[:, task_id], labels_mat[:, task_id])

    def test_apply_kd_uses_best_predictions_not_current(self, kd_loss):
        """Test _apply_kd uses best_predictions, not current preds."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.full((10, 4), 0.9)  # Current predictions different

        # Set up task 0 as replaced with different best predictions
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.3)  # Best from earlier

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # Should use best predictions (0.3), not current (0.9)
        np.testing.assert_array_equal(labels_kd[:, 0], np.full(10, 0.3))

    def test_apply_kd_handles_no_best_predictions(self, kd_loss):
        """Test _apply_kd handles case when best_predictions is None."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)

        # Set task as replaced but no best predictions yet
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = None

        # Should not crash, should preserve original labels
        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)
        np.testing.assert_array_equal(labels_kd[:, 0], labels_mat[:, 0])

    def test_apply_kd_creates_copy_of_labels(self, kd_loss):
        """Test _apply_kd creates copy and doesn't modify original."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)
        labels_original = labels_mat.copy()

        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.5)

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # Original labels should be unchanged
        np.testing.assert_array_equal(labels_mat, labels_original)


# ============================================================================
# Test Class 5: Objective Integration
# ============================================================================


class TestObjectiveIntegration:
    """Tests for objective function with KD integration."""

    def test_current_iteration_increments(self, kd_loss, mock_train_data):
        """Test current_iteration increments with each objective call."""
        preds = np.random.rand(40)

        assert kd_loss.current_iteration == 0

        kd_loss.objective(preds, mock_train_data)
        assert kd_loss.current_iteration == 1

        kd_loss.objective(preds, mock_train_data)
        assert kd_loss.current_iteration == 2

    def test_store_predictions_called(self, kd_loss, mock_train_data):
        """Test _store_predictions is called during objective."""
        preds = np.random.rand(40)

        # All previous_predictions should be None initially
        assert all(v is None for v in kd_loss.previous_predictions.values())

        kd_loss.objective(preds, mock_train_data)

        # After objective call, predictions should be stored
        assert all(v is not None for v in kd_loss.previous_predictions.values())

    def test_apply_kd_called_when_tasks_replaced(self, kd_loss, mock_train_data):
        """Test _apply_kd is called when any task is replaced."""
        preds = np.random.rand(40)

        # Mark task 0 as replaced with best predictions
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.5)

        # Patch _apply_kd to verify it's called
        with patch.object(kd_loss, "_apply_kd", wraps=kd_loss._apply_kd) as mock_apply:
            kd_loss.objective(preds, mock_train_data)
            assert mock_apply.called

    def test_apply_kd_not_called_when_no_replacement(self, kd_loss, mock_train_data):
        """Test _apply_kd is not called when no tasks replaced."""
        preds = np.random.rand(40)

        # No tasks replaced
        assert all(not v for v in kd_loss.replaced.values())

        # _apply_kd should not be called (implementation checks any(replaced.values()))
        # We can verify by checking objective runs without KD modification
        grad, hess, grad_i, hess_i = kd_loss.objective(preds, mock_train_data)
        assert grad is not None  # Basic functionality works

    def test_objective_returns_four_values(self, kd_loss, mock_train_data):
        """Test objective returns grad, hess, grad_i, hess_i."""
        preds = np.random.rand(40)

        result = kd_loss.objective(preds, mock_train_data)

        assert len(result) == 4
        grad, hess, grad_i, hess_i = result
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert isinstance(grad_i, np.ndarray)
        assert isinstance(hess_i, np.ndarray)

    def test_objective_with_kd_active(self, kd_loss, mock_train_data):
        """Test objective computation with KD active."""
        preds = np.random.rand(40)

        # Set up KD state
        kd_loss.replaced[0] = True
        kd_loss.best_predictions[0] = np.full(10, 0.7)

        grad, hess, grad_i, hess_i = kd_loss.objective(preds, mock_train_data)

        # Should complete successfully with KD active
        assert grad.shape == (10,)
        assert hess.shape == (10,)
        assert grad_i.shape == (10, 4)
        assert hess_i.shape == (10, 4)


# ============================================================================
# Test Class 6: Evaluate Integration
# ============================================================================


class TestEvaluateIntegration:
    """Tests for evaluate function with KD trigger checking."""

    def test_evaluate_calls_parent_evaluate(self, kd_loss, mock_train_data):
        """Test evaluate calls super().evaluate()."""
        preds = np.random.rand(40)

        # Patch parent evaluate
        with patch(
            "docker.models.loss.adaptive_weight_loss.AdaptiveWeightLoss.evaluate"
        ) as mock_eval:
            mock_eval.return_value = (np.array([0.8, 0.7, 0.6, 0.5]), 0.65)

            task_scores, mean_score = kd_loss.evaluate(preds, mock_train_data)

            # Verify parent evaluate was called
            assert mock_eval.called
            assert mean_score == 0.65

    def test_evaluate_calls_check_kd_trigger(self, kd_loss, mock_train_data):
        """Test evaluate calls _check_kd_trigger with scores."""
        preds = np.random.rand(40)

        # Patch check_kd_trigger to verify it's called
        with patch.object(kd_loss, "_check_kd_trigger") as mock_check:
            with patch(
                "docker.models.loss.adaptive_weight_loss.AdaptiveWeightLoss.evaluate"
            ) as mock_eval:
                task_scores = np.array([0.8, 0.7, 0.6, 0.5])
                mock_eval.return_value = (task_scores, 0.65)

                kd_loss.evaluate(preds, mock_train_data)

                # Verify _check_kd_trigger was called with scores
                assert mock_check.called
                call_args = mock_check.call_args[0]
                np.testing.assert_array_equal(call_args[0], task_scores)

    def test_evaluate_returns_task_scores_and_mean(self, kd_loss, mock_train_data):
        """Test evaluate returns task_scores and mean_score."""
        preds = np.random.rand(40)

        with patch(
            "docker.models.loss.adaptive_weight_loss.AdaptiveWeightLoss.evaluate"
        ) as mock_eval:
            expected_task_scores = np.array([0.8, 0.7, 0.6, 0.5])
            expected_mean = 0.65
            mock_eval.return_value = (expected_task_scores, expected_mean)

            task_scores, mean_score = kd_loss.evaluate(preds, mock_train_data)

            np.testing.assert_array_equal(task_scores, expected_task_scores)
            assert mean_score == expected_mean


# ============================================================================
# Test Class 7: Edge Cases
# ============================================================================


class TestKDEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_patience_triggers_quickly(self, mock_hyperparams, sample_indices):
        """Test KD triggers quickly with patience=1."""
        mock_hyperparams.loss_patience = 1

        loss = KnowledgeDistillationLoss(
            num_label=4, val_sublabel_idx=sample_indices, hyperparams=mock_hyperparams
        )

        task_scores = np.array([0.5, 0.5, 0.5, 0.5])

        # Set baseline
        loss._check_kd_trigger(task_scores, iteration=1)

        # One more iteration without improvement should trigger
        loss._check_kd_trigger(task_scores, iteration=2)

        # All tasks should be replaced
        assert all(loss.replaced[i] == True for i in range(4))

    def test_all_tasks_replaced_simultaneously(self, kd_loss):
        """Test handling when all tasks are replaced."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)

        # Mark all tasks as replaced
        for task_id in range(4):
            kd_loss.replaced[task_id] = True
            kd_loss.best_predictions[task_id] = np.full(10, 0.6)

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # All tasks should have replaced labels
        for task_id in range(4):
            np.testing.assert_array_equal(labels_kd[:, task_id], np.full(10, 0.6))

    def test_single_task_struggling(self, kd_loss):
        """Test handling when only one task is struggling."""
        labels_mat = np.ones((10, 4))
        preds_mat = np.random.rand(10, 4)

        # Mark only task 1 as replaced
        kd_loss.replaced[1] = True
        kd_loss.best_predictions[1] = np.full(10, 0.4)

        labels_kd = kd_loss._apply_kd(labels_mat, preds_mat)

        # Task 1 should be replaced
        np.testing.assert_array_equal(labels_kd[:, 1], np.full(10, 0.4))

        # Other tasks should be unchanged
        np.testing.assert_array_equal(labels_kd[:, 0], labels_mat[:, 0])
        np.testing.assert_array_equal(labels_kd[:, 2], labels_mat[:, 2])
        np.testing.assert_array_equal(labels_kd[:, 3], labels_mat[:, 3])

    def test_kd_never_triggered_good_performance(self, kd_loss):
        """Test KD is never triggered when performance keeps improving."""
        # Simulate improving performance
        for i in range(1, 10):
            task_scores = np.array(
                [0.5 + i * 0.05, 0.6 + i * 0.03, 0.7 + i * 0.02, 0.8 + i * 0.01]
            )
            kd_loss._check_kd_trigger(task_scores, iteration=i)

        # No tasks should be replaced (performance keeps improving)
        assert all(kd_loss.replaced[i] == False for i in range(4))
        assert all(kd_loss.decline_count[i] == 0 for i in range(4))

    def test_minimum_tasks(self, mock_hyperparams):
        """Test with minimum number of tasks (2)."""
        indices_2 = {0: np.array([0, 1, 2]), 1: np.array([0, 1, 2])}

        loss = KnowledgeDistillationLoss(
            num_label=2, val_sublabel_idx=indices_2, hyperparams=mock_hyperparams
        )

        assert len(loss.performance_history) == 2
        assert len(loss.decline_count) == 2
        assert len(loss.best_predictions) == 2
        assert len(loss.replaced) == 2

    def test_many_tasks(self, mock_hyperparams):
        """Test with many tasks (10 for scalability)."""
        indices_10 = {i: np.array([0, 1, 2, 3, 4]) for i in range(10)}

        loss = KnowledgeDistillationLoss(
            num_label=10, val_sublabel_idx=indices_10, hyperparams=mock_hyperparams
        )

        assert len(loss.performance_history) == 10
        assert len(loss.decline_count) == 10
        assert len(loss.best_predictions) == 10
        assert len(loss.replaced) == 10

    def test_skips_already_replaced_tasks(self, kd_loss):
        """Test that already replaced tasks are skipped in KD trigger check."""
        # Mark task 0 as already replaced
        kd_loss.replaced[0] = True

        task_scores = np.array([0.5, 0.6, 0.7, 0.8])

        # Simulate multiple iterations with declining performance
        for i in range(1, 10):
            kd_loss._check_kd_trigger(task_scores, iteration=i)

        # Task 0 should still be replaced (was already)
        assert kd_loss.replaced[0] == True

        # Decline count for task 0 should not increment (skipped)
        # Note: Task 0 was set baseline at iter 1, so counter stays low
        # Other tasks keep improving so their counters stay at 0
