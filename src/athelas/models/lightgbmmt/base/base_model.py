"""
Base model for multi-task learning implementations.

Provides template method pattern for training workflow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters
    from ..loss.base_loss_function import BaseLossFunction

from .training_state import TrainingState


class BaseMultiTaskModel(ABC):
    """
    Abstract base class for multi-task learning models.

    Implements template method pattern for training workflow.
    Subclasses implement specific model architectures.
    """

    def __init__(
        self,
        loss_function: Optional["BaseLossFunction"],
        training_state: Optional[TrainingState],
        hyperparams: "LightGBMMtModelHyperparameters",
    ):
        """
        Initialize base multi-task model.

        Parameters
        ----------
        loss_function : BaseLossFunction, optional
            Loss function instance (not required for inference-only mode)
        training_state : TrainingState, optional
            Training state for tracking progress (not required for inference-only mode)
        hyperparams : LightGBMMtModelHyperparameters
            Model hyperparameters
        """
        self.loss_function = loss_function
        self.training_state = training_state or TrainingState()
        self.hyperparams = hyperparams

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Model storage
        self.model = None

        self.logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"loss_type={hyperparams.loss_type}"
        )

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[list] = None,
        task_columns: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Template method for training workflow.

        Orchestrates the training process through defined steps.

        Parameters
        ----------
        train_df : DataFrame
            Training data (should contain only feature_columns + task_columns)
        val_df : DataFrame
            Validation data (should contain only feature_columns + task_columns)
        test_df : DataFrame, optional
            Test data (should contain only feature_columns + task_columns)
        feature_columns : list, optional
            List of feature column names in order
        task_columns : list, optional
            List of task label column names in order

        Returns
        -------
        results : dict
            Training results including metrics and model info
        """
        self.logger.info("Starting training workflow...")

        # Step 1: Prepare data
        self.logger.info("Step 1: Preparing data...")
        train_data, val_data, test_data = self._prepare_data(
            train_df, val_df, test_df, feature_columns, task_columns
        )

        # Step 2: Initialize model
        self.logger.info("Step 2: Initializing model...")
        self._initialize_model()

        # Step 3: Train model (main training loop)
        self.logger.info("Step 3: Training model...")
        train_metrics = self._train_model(train_data, val_data)

        # Step 4: Evaluate model
        self.logger.info("Step 4: Evaluating model...")
        eval_metrics = self._evaluate_model(val_data, test_data)

        # Step 5: Finalize
        self.logger.info("Step 5: Finalizing...")
        results = self._finalize_training(train_metrics, eval_metrics)

        self.logger.info("Training workflow completed successfully")
        return results

    @abstractmethod
    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        feature_columns: Optional[list] = None,
        task_columns: Optional[list] = None,
    ) -> tuple:
        """
        Prepare data for training.

        Subclasses implement specific data preparation logic.
        Training script controls the schema (WHAT columns to use).
        Model just handles format conversion (HOW to convert for ML library).

        Parameters
        ----------
        train_df : DataFrame
            Training data (should contain only feature_columns + task_columns)
        val_df : DataFrame
            Validation data
        test_df : DataFrame, optional
            Test data
        feature_columns : list, optional
            List of feature column names in exact order
        task_columns : list, optional
            List of task label column names in exact order

        Returns
        -------
        train_data, val_data, test_data : tuple
            Prepared datasets in model-specific format
        """
        pass

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize model architecture."""
        pass

    @abstractmethod
    def _train_model(self, train_data: Any, val_data: Any) -> Dict[str, Any]:
        """
        Main training loop.

        Parameters
        ----------
        train_data : Any
            Prepared training data
        val_data : Any
            Prepared validation data

        Returns
        -------
        metrics : dict
            Training metrics
        """
        pass

    def _evaluate_model(
        self, val_data: Any, test_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Default implementation, can be overridden.

        Parameters
        ----------
        val_data : Any
            Validation data (Dataset format)
        test_data : Any, optional
            Test data (Dataset format)

        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        metrics = {}

        # Extract features from Dataset for prediction
        # Model's _predict() expects numpy arrays, not Dataset objects
        val_features = val_data.data
        val_preds = self._predict(val_features)
        val_metrics = self._compute_metrics(val_data, val_preds)
        metrics["validation"] = val_metrics

        # Test metrics if available
        if test_data is not None:
            test_features = test_data.data
            test_preds = self._predict(test_features)
            test_metrics = self._compute_metrics(test_data, test_preds)
            metrics["test"] = test_metrics

        return metrics

    def predict(
        self, df: pd.DataFrame, feature_columns: Optional[list] = None
    ) -> np.ndarray:
        """
        Public prediction method.

        Accepts DataFrame (same format as training), extracts features,
        and generates predictions.

        Parameters
        ----------
        df : DataFrame
            Data to predict on (may contain features + other columns)
        feature_columns : list, optional
            List of feature column names to use
            If not provided, model uses hyperparameters

        Returns
        -------
        predictions : np.ndarray
            Model predictions
        """
        # Prepare data for prediction (extracts features, creates Dataset)
        data = self._prepare_prediction_data(df, feature_columns)

        return self._predict(data)

    @abstractmethod
    def _prepare_prediction_data(
        self, df: pd.DataFrame, feature_columns: Optional[list] = None
    ) -> Any:
        """
        Prepare data for prediction (no labels needed).

        Subclasses implement model-specific prediction data preparation.
        Similar to _prepare_data but without labels.

        Parameters
        ----------
        df : DataFrame
            Data to predict on (may contain features + other columns)
        feature_columns : list, optional
            List of feature column names to use
            If not provided, model uses hyperparameters

        Returns
        -------
        data : Any
            Data in model-specific format
        """
        pass

    @abstractmethod
    def _predict(self, data: Any) -> np.ndarray:
        """
        Generate predictions (internal method).

        Parameters
        ----------
        data : Any
            Data to predict on (model-specific format)

        Returns
        -------
        predictions : np.ndarray
            Model predictions
        """
        pass

    def _compute_metrics(self, data: Any, predictions: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Default implementation uses loss function's evaluate method.

        Parameters
        ----------
        data : Any
            Data with labels
        predictions : np.ndarray
            Model predictions

        Returns
        -------
        metrics : dict
            Computed metrics
        """
        # Use loss function's evaluation if available
        if hasattr(self.loss_function, "evaluate"):
            task_scores, mean_score = self.loss_function.evaluate(predictions, data)
            return {"mean_auc": mean_score, "per_task_auc": task_scores.tolist()}

        return {}

    def _finalize_training(
        self, train_metrics: Dict[str, Any], eval_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Finalize training and prepare results.

        Parameters
        ----------
        train_metrics : dict
            Training metrics
        eval_metrics : dict
            Evaluation metrics

        Returns
        -------
        results : dict
            Complete training results
        """
        results = {
            "training_metrics": train_metrics,
            "evaluation_metrics": eval_metrics,
            "training_state": self.training_state.to_checkpoint_dict(),
            "hyperparameters": self.hyperparams.model_dump(),
            "model_type": self.__class__.__name__,
        }

        return results

    def save(self, output_path: str) -> None:
        """
        Save model artifacts.

        Parameters
        ----------
        output_path : str
            Directory to save artifacts
        """
        self.logger.info(f"Saving model to {output_path}")
        self._save_model(output_path)

    @abstractmethod
    def _save_model(self, output_path: str) -> None:
        """Save model-specific artifacts."""
        pass

    def load(self, model_path: str) -> None:
        """
        Load model artifacts.

        Parameters
        ----------
        model_path : str
            Path to model artifacts
        """
        self.logger.info(f"Loading model from {model_path}")
        self._load_model(model_path)

    @abstractmethod
    def _load_model(self, model_path: str) -> None:
        """Load model-specific artifacts."""
        pass
