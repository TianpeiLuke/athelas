"""
Tests for MtgbmModel implementation.

Tests multi-task gradient boosting model with LightGBM backend.

Following pytest best practices:
- Read source code first (mtgbm_model.py analyzed)
- Test actual implementation behavior
- Strategic mocking (LightGBM components, file I/O)
- Test both happy path and error conditions
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import json

# Import from actual source (following best practice)
from athelas.models.lightgbmmt.implementations.mtgbm_model import MtgbmModel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_loss_function():
    """Create mock loss function with required methods."""
    mock = Mock()
    mock.num_col = 4
    mock.objective = Mock(return_value=(
        np.array([1.0, 2.0]),  # grad
        np.array([1.0, 1.0]),  # hess
        np.array([[1.0], [2.0]]),  # grad_i
        np.array([[1.0], [1.0]])  # hess_i
    ))
    mock.evaluate = Mock(return_value=(
        np.array([0.8, 0.85, 0.9, 0.88]),  # task_scores
        0.8575  # mean_score
    ))
    return mock


@pytest.fixture
def mock_training_state():
    """Create mock training state."""
    mock = Mock()
    mock.current_epoch = 0
    mock.best_epoch = 0
    mock.best_metric = 0.0
    mock.epochs_without_improvement = 0
    mock.to_checkpoint_dict = Mock(return_value={
        "current_epoch": 0,
        "best_epoch": 0,
        "best_metric": 0.0
    })
    mock.from_checkpoint_dict = Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_hyperparams():
    """Create mock hyperparameters with all required attributes."""
    mock = Mock()
    # Data attributes
    mock.full_field_list = ["feature_1", "feature_2", "feature_3"]
    mock.cat_field_list = []
    mock.num_tasks = 4
    
    # LightGBM parameters
    mock.boosting_type = "gbdt"
    mock.num_leaves = 31
    mock.learning_rate = 0.1
    mock.max_depth = -1
    mock.min_data_in_leaf = 20
    mock.feature_fraction = 0.8
    mock.bagging_fraction = 0.8
    mock.bagging_freq = 5
    mock.lambda_l1 = 0.0
    mock.lambda_l2 = 0.0
    mock.min_gain_to_split = 0.0
    mock.seed = 42
    
    # Training parameters
    mock.num_iterations = 100
    mock.early_stopping_rounds = 10
    
    # Loss parameters
    mock.loss_type = "adaptive_kd"
    
    # For model_dump
    mock.model_dump = Mock(return_value={"param": "value"})
    
    return mock


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe with features and task labels."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "task_0": np.random.randint(0, 2, n_samples),
        "task_1": np.random.randint(0, 2, n_samples),
        "task_2": np.random.randint(0, 2, n_samples),
        "task_3": np.random.randint(0, 2, n_samples),
    })
    return df


@pytest.fixture
def mtgbm_model(mock_loss_function, mock_training_state, mock_hyperparams):
    """Create MtgbmModel instance."""
    return MtgbmModel(
        loss_function=mock_loss_function,
        training_state=mock_training_state,
        hyperparams=mock_hyperparams
    )


# ============================================================================
# Test Class 1: Data Preparation
# ============================================================================


class TestMtgbmModelDataPreparation:
    """Tests for data preparation methods."""

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.Dataset')
    def test_prepare_data_creates_datasets(
        self, mock_dataset_class, mtgbm_model, sample_dataframe
    ):
        """Test _prepare_data creates LightGBM datasets."""
        train_df = sample_dataframe.iloc[:60]
        val_df = sample_dataframe.iloc[60:80]
        test_df = sample_dataframe.iloc[80:]
        
        # Mock Dataset creation
        mock_train = Mock()
        mock_val = Mock()
        mock_test = Mock()
        mock_dataset_class.side_effect = [mock_train, mock_val, mock_test]
        
        train_data, val_data, test_data = mtgbm_model._prepare_data(
            train_df, val_df, test_df
        )
        
        # Verify datasets created
        assert train_data == mock_train
        assert val_data == mock_val
        assert test_data == mock_test
        assert mock_dataset_class.call_count == 3

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.Dataset')
    def test_prepare_data_without_test(
        self, mock_dataset_class, mtgbm_model, sample_dataframe
    ):
        """Test _prepare_data handles None test_df."""
        train_df = sample_dataframe.iloc[:70]
        val_df = sample_dataframe.iloc[70:]
        
        mock_train = Mock()
        mock_val = Mock()
        mock_dataset_class.side_effect = [mock_train, mock_val]
        
        train_data, val_data, test_data = mtgbm_model._prepare_data(
            train_df, val_df, None
        )
        
        assert train_data == mock_train
        assert val_data == mock_val
        assert test_data is None
        assert mock_dataset_class.call_count == 2

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.Dataset')
    def test_prepare_data_extracts_features(
        self, mock_dataset_class, mtgbm_model, sample_dataframe
    ):
        """Test _prepare_data extracts correct features."""
        train_df = sample_dataframe
        val_df = sample_dataframe
        
        mtgbm_model._prepare_data(train_df, val_df, None)
        
        # Check first Dataset call (training data)
        call_args = mock_dataset_class.call_args_list[0]
        X_train = call_args[0][0]
        
        # Should have 3 features
        assert X_train.shape[1] == 3

    def test_extract_multi_task_labels_standard_format(
        self, mtgbm_model, sample_dataframe
    ):
        """Test _extract_multi_task_labels with task_0, task_1, ... format."""
        labels = mtgbm_model._extract_multi_task_labels(sample_dataframe)
        
        # Should extract 4 task columns
        assert labels.shape == (100, 4)
        assert labels.dtype in [np.int64, np.int32, np.int_]

    def test_extract_multi_task_labels_alternative_format(self, mtgbm_model):
        """Test _extract_multi_task_labels with alternative column names."""
        df = pd.DataFrame({
            "feature": np.random.randn(50),
            "label_0": np.random.randint(0, 2, 50),
            "label_1": np.random.randint(0, 2, 50),
        })
        
        # Adjust num_tasks
        mtgbm_model.hyperparams.num_tasks = 2
        
        labels = mtgbm_model._extract_multi_task_labels(df)
        
        # Should find label columns
        assert labels.shape[0] == 50
        assert labels.shape[1] >= 2


# ============================================================================
# Test Class 2: Model Initialization
# ============================================================================


class TestMtgbmModelInitialization:
    """Tests for model initialization."""

    def test_initialize_model_sets_params(self, mtgbm_model):
        """Test _initialize_model sets lgb_params correctly."""
        mtgbm_model._initialize_model()
        
        # Verify params set
        assert hasattr(mtgbm_model, "lgb_params")
        assert isinstance(mtgbm_model.lgb_params, dict)
        
        # Check key parameters
        assert mtgbm_model.lgb_params["boosting_type"] == "gbdt"
        assert mtgbm_model.lgb_params["num_leaves"] == 31
        assert mtgbm_model.lgb_params["learning_rate"] == 0.1
        assert mtgbm_model.lgb_params["seed"] == 42

    def test_initialize_model_includes_all_hyperparams(self, mtgbm_model):
        """Test _initialize_model includes all LightGBM hyperparameters."""
        mtgbm_model._initialize_model()
        
        expected_keys = [
            "boosting_type", "num_leaves", "learning_rate", "max_depth",
            "min_data_in_leaf", "feature_fraction", "bagging_fraction",
            "bagging_freq", "lambda_l1", "lambda_l2", "min_gain_to_split",
            "seed", "verbose"
        ]
        
        for key in expected_keys:
            assert key in mtgbm_model.lgb_params

    def test_initialize_model_without_seed(self, mtgbm_model):
        """Test _initialize_model when seed is None."""
        mtgbm_model.hyperparams.seed = None
        mtgbm_model._initialize_model()
        
        # Seed should not be in params
        assert "seed" not in mtgbm_model.lgb_params


# ============================================================================
# Test Class 3: Training
# ============================================================================


class TestMtgbmModelTraining:
    """Tests for model training."""

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.train')
    def test_train_model_calls_lgb_train(self, mock_train, mtgbm_model):
        """Test _train_model calls lgb.train with correct parameters."""
        mock_train_data = Mock()
        mock_val_data = Mock()
        
        # Mock trained model
        mock_model = Mock()
        mock_model.num_trees = Mock(return_value=50)
        mock_model.best_iteration = 45
        mock_model.feature_importance = Mock(return_value=np.array([0.5, 0.3, 0.2]))
        mock_train.return_value = mock_model
        
        # Initialize first
        mtgbm_model._initialize_model()
        
        metrics = mtgbm_model._train_model(mock_train_data, mock_val_data)
        
        # Verify lgb.train called
        assert mock_train.called
        assert metrics["num_iterations"] == 50
        assert metrics["best_iteration"] == 45

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.train')
    def test_train_model_uses_custom_loss(self, mock_train, mtgbm_model):
        """Test _train_model passes custom loss function."""
        mock_train_data = Mock()
        mock_val_data = Mock()
        mock_model = Mock()
        mock_model.num_trees = Mock(return_value=50)
        mock_model.best_iteration = 45
        mock_model.feature_importance = Mock(return_value=np.array([]))
        mock_train.return_value = mock_model
        
        mtgbm_model._initialize_model()
        mtgbm_model._train_model(mock_train_data, mock_val_data)
        
        # Check fobj parameter
        call_kwargs = mock_train.call_args[1]
        assert "fobj" in call_kwargs
        assert call_kwargs["fobj"] == mtgbm_model.loss_function.objective

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.train')
    def test_train_model_uses_custom_eval(self, mock_train, mtgbm_model):
        """Test _train_model uses custom evaluation function."""
        mock_train_data = Mock()
        mock_val_data = Mock()
        mock_model = Mock()
        mock_model.num_trees = Mock(return_value=50)
        mock_model.best_iteration = 45
        mock_model.feature_importance = Mock(return_value=np.array([]))
        mock_train.return_value = mock_model
        
        mtgbm_model._initialize_model()
        mtgbm_model._train_model(mock_train_data, mock_val_data)
        
        # Check feval parameter
        call_kwargs = mock_train.call_args[1]
        assert "feval" in call_kwargs
        assert callable(call_kwargs["feval"])

    def test_create_eval_function_returns_callable(self, mtgbm_model):
        """Test _create_eval_function returns a callable."""
        eval_func = mtgbm_model._create_eval_function()
        
        assert callable(eval_func)

    def test_create_eval_function_wrapper_works(self, mtgbm_model):
        """Test evaluation function wrapper calls loss_function.evaluate."""
        eval_func = mtgbm_model._create_eval_function()
        
        mock_preds = np.array([0.5, 0.6, 0.7])
        mock_data = Mock()
        
        name, score, is_higher_better = eval_func(mock_preds, mock_data)
        
        assert name == "mean_auc"
        assert score == 0.8575  # From mock_loss_function fixture
        assert is_higher_better is True


# ============================================================================
# Test Class 4: Prediction
# ============================================================================


class TestMtgbmModelPrediction:
    """Tests for model prediction."""

    def test_predict_without_trained_model_raises_error(self, mtgbm_model):
        """Test _predict raises ValueError when model not trained."""
        mock_data = Mock()
        mock_data.data = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="Model not trained"):
            mtgbm_model._predict(mock_data)

    def test_predict_with_trained_model(self, mtgbm_model):
        """Test _predict returns predictions with trained model."""
        # Set up trained model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.5, 0.6, 0.7]))
        mtgbm_model.model = mock_model
        
        mock_data = Mock()
        mock_data.data = np.random.randn(3, 3)
        
        predictions = mtgbm_model._predict(mock_data)
        
        assert predictions is not None
        assert len(predictions) == 3
        assert mock_model.predict.called


# ============================================================================
# Test Class 5: Model Persistence
# ============================================================================


class TestMtgbmModelPersistence:
    """Tests for model save/load operations."""

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_creates_directory(
        self, mock_file, mock_path_class, mtgbm_model
    ):
        """Test _save_model creates output directory."""
        mock_dir = Mock()
        # Mock the / operator to return Mock objects
        mock_dir.__truediv__ = Mock(return_value=Mock())
        mock_path_class.return_value = mock_dir
        
        # Set up model
        mtgbm_model.model = Mock()
        mtgbm_model.model.save_model = Mock()
        
        mtgbm_model._save_model("/fake/path")
        
        # Verify directory creation
        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_saves_lightgbm_model(
        self, mock_file, mock_path_class, mtgbm_model
    ):
        """Test _save_model saves LightGBM model file."""
        mock_dir = Mock()
        mock_dir.__truediv__ = Mock(return_value=Mock())
        mock_path_class.return_value = mock_dir
        
        mtgbm_model.model = Mock()
        mtgbm_model.model.save_model = Mock()
        
        mtgbm_model._save_model("/fake/path")
        
        # Verify model saved
        assert mtgbm_model.model.save_model.called

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_saves_hyperparameters(
        self, mock_file, mock_path_class, mtgbm_model
    ):
        """Test _save_model saves hyperparameters."""
        mock_dir = Mock()
        mock_dir.__truediv__ = Mock(return_value=Mock())
        mock_path_class.return_value = mock_dir
        
        mtgbm_model.model = Mock()
        mtgbm_model.model.save_model = Mock()
        
        mtgbm_model._save_model("/fake/path")
        
        # Verify hyperparams model_dump called
        assert mtgbm_model.hyperparams.model_dump.called

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_saves_training_state(
        self, mock_file, mock_path_class, mtgbm_model
    ):
        """Test _save_model saves training state."""
        mock_dir = Mock()
        mock_dir.__truediv__ = Mock(return_value=Mock())
        mock_path_class.return_value = mock_dir
        
        mtgbm_model.model = Mock()
        mtgbm_model.model.save_model = Mock()
        
        mtgbm_model._save_model("/fake/path")
        
        # Verify training state serialized
        assert mtgbm_model.training_state.to_checkpoint_dict.called

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.Booster')
    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_model_loads_lightgbm_model(
        self, mock_file, mock_path_class, mock_booster, mtgbm_model
    ):
        """Test _load_model loads LightGBM model."""
        mock_dir = Mock()
        mock_model_file = Mock()
        mock_state_file = Mock()
        mock_state_file.exists = Mock(return_value=False)  # State file doesn't exist
        mock_dir.__truediv__ = Mock(side_effect=[
            mock_model_file,  # model file
            mock_state_file  # state file
        ])
        mock_path_class.return_value = mock_dir
        
        mock_model = Mock()
        mock_booster.return_value = mock_model
        
        mtgbm_model._load_model("/fake/path")
        
        # Verify model loaded
        assert mtgbm_model.model == mock_model

    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.lgb.Booster')
    @patch('athelas.models.lightgbmmt.implementations.mtgbm_model.Path')
    @patch('builtins.open', new_callable=mock_open, read_data='{"current_epoch": 5}')
    def test_load_model_loads_training_state_if_exists(
        self, mock_file, mock_path_class, mock_booster, mtgbm_model
    ):
        """Test _load_model loads training state when file exists."""
        mock_dir = Mock()
        mock_state_file = Mock()
        mock_state_file.exists = Mock(return_value=True)
        mock_dir.__truediv__ = Mock(side_effect=[
            Mock(),  # model file
            mock_state_file  # state file
        ])
        mock_path_class.return_value = mock_dir
        
        mock_booster.return_value = Mock()
        
        mtgbm_model._load_model("/fake/path")
        
        # Verify training state loaded
        assert mtgbm_model.training_state.from_checkpoint_dict.called


# ============================================================================
# Test Class 6: Integration Tests
# ============================================================================


class TestMtgbmModelIntegration:
    """Integration tests with BaseMultiTaskModel."""

    def test_inherits_from_base_model(self, mtgbm_model):
        """Test MtgbmModel inherits from BaseMultiTaskModel."""
        from athelas.models.lightgbmmt.base.base_model import BaseMultiTaskModel
        
        assert isinstance(mtgbm_model, BaseMultiTaskModel)

    def test_has_all_required_methods(self, mtgbm_model):
        """Test MtgbmModel has all required abstract methods implemented."""
        assert hasattr(mtgbm_model, "_prepare_data")
        assert hasattr(mtgbm_model, "_initialize_model")
        assert hasattr(mtgbm_model, "_train_model")
        assert hasattr(mtgbm_model, "_predict")
        assert hasattr(mtgbm_model, "_save_model")
        assert hasattr(mtgbm_model, "_load_model")

    def test_has_public_interface(self, mtgbm_model):
        """Test MtgbmModel has public interface from BaseMultiTaskModel."""
        assert hasattr(mtgbm_model, "train")
        assert hasattr(mtgbm_model, "save")
        assert hasattr(mtgbm_model, "load")
        
        assert callable(mtgbm_model.train)
        assert callable(mtgbm_model.save)
        assert callable(mtgbm_model.load)

    def test_stores_dependencies(self, mtgbm_model, mock_loss_function, 
                                 mock_training_state, mock_hyperparams):
        """Test MtgbmModel stores all dependencies."""
        assert mtgbm_model.loss_function == mock_loss_function
        assert mtgbm_model.training_state == mock_training_state
        assert mtgbm_model.hyperparams == mock_hyperparams
