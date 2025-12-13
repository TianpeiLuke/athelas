"""
Multi-Task Gradient Boosting Model (MT-GBM) implementation.

Implements shared tree structure for multi-task learning.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.special import expit

from ..lightgbmmt import Dataset, Booster, train
from ..base.base_model import BaseMultiTaskModel


class MtgbmModel(BaseMultiTaskModel):
    """
    Multi-Task Gradient Boosting Model.

    Uses LightGBM with custom multi-task loss function and
    shared tree structures across related tasks.
    """

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        feature_columns: Optional[list] = None,
        task_columns: Optional[list] = None,
    ) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """
        Prepare data for LightGBMT training.

        Converts DataFrames to LightGBMT Dataset format with native 2D label support.
        Training script controls schema (WHAT columns), model handles format conversion (HOW).

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
            If not provided, falls back to tab_field_list + cat_field_list
        task_columns : list, optional
            List of task label column names in exact order
            If not provided, infers from data

        Returns
        -------
        train_data, val_data, test_data : tuple of lgb.Dataset
            Prepared LightGBM datasets
        """
        # Use provided columns or fallback to hyperparams
        if feature_columns is None:
            feature_columns = (
                self.hyperparams.tab_field_list + self.hyperparams.cat_field_list
            )
            self.logger.warning(
                "feature_columns not provided, falling back to hyperparams. "
                "Consider passing explicit columns from training script."
            )

        if task_columns is None:
            # Use explicit task_label_names from hyperparams
            if self.hyperparams.task_label_names:
                task_columns = self.hyperparams.task_label_names
                self.logger.warning(
                    f"task_columns not provided, using hyperparams.task_label_names: {task_columns}. "
                    "Consider passing explicit columns from training script."
                )
            else:
                raise ValueError(
                    "task_columns not provided and hyperparams.task_label_names is empty. "
                    "Must specify task columns either via method parameter or hyperparameters."
                )

        # Extract features and labels
        X_train = train_df[feature_columns].values
        y_train = train_df[task_columns].values

        # Create LightGBMT Dataset with native 2D label support
        # No field hacks needed - Dataset natively supports multi-dimensional labels
        # CRITICAL: Set free_raw_data=False to preserve data for evaluation
        train_data = Dataset(
            X_train,
            label=y_train,  # Pass full 2D array [N_samples, N_tasks]
            feature_name=feature_columns,
            categorical_feature=[
                c for c in feature_columns if c in self.hyperparams.cat_field_list
            ],
            free_raw_data=False,  # Keep raw data for evaluation phase
        )

        # Prepare validation data
        X_val = val_df[feature_columns].values
        y_val = val_df[task_columns].values
        val_data = Dataset(
            X_val,
            label=y_val,  # Full 2D array
            reference=train_data,
            free_raw_data=False,  # Keep raw data for evaluation phase
        )

        # Prepare test data if provided
        test_data = None
        if test_df is not None:
            X_test = test_df[feature_columns].values
            y_test = test_df[task_columns].values
            test_data = Dataset(
                X_test,
                label=y_test,  # Full 2D array
                reference=train_data,
                free_raw_data=False,  # Keep raw data for evaluation phase
            )

        self.logger.info(
            f"Prepared data: train={X_train.shape}, "
            f"val={X_val.shape}, "
            f"test={X_test.shape if test_df is not None else None}"
        )
        self.logger.info(
            f"Features: {len(feature_columns)} columns, Tasks: {len(task_columns)} columns"
        )

        return train_data, val_data, test_data

    def _initialize_model(self) -> None:
        """Initialize LightGBMT model parameters."""
        # Get number of tasks
        num_tasks = (
            len(self.hyperparams.task_label_names)
            if self.hyperparams.task_label_names
            else 1
        )

        # Build LightGBMT parameters from hyperparameters
        self.lgb_params = {
            "objective": "custom",  # Required for multi-task
            "num_labels": num_tasks,  # Number of tasks
            "tree_learner": "serial2",  # Required for multi-task
            "boosting_type": self.hyperparams.boosting_type,
            "num_leaves": self.hyperparams.num_leaves,
            "learning_rate": self.hyperparams.learning_rate,
            "max_depth": self.hyperparams.max_depth,
            "min_data_in_leaf": self.hyperparams.min_data_in_leaf,
            "feature_fraction": self.hyperparams.feature_fraction,
            "bagging_fraction": self.hyperparams.bagging_fraction,
            "bagging_freq": self.hyperparams.bagging_freq,
            "lambda_l1": self.hyperparams.lambda_l1,
            "lambda_l2": self.hyperparams.lambda_l2,
            "min_gain_to_split": self.hyperparams.min_gain_to_split,
            "verbose": -1,
        }

        if self.hyperparams.seed is not None:
            self.lgb_params["seed"] = self.hyperparams.seed

        self.logger.info(f"Initialized multi-task model with {num_tasks} tasks")
        self.logger.info(f"Model params: {self.lgb_params}")

    def _train_model(self, train_data: Dataset, val_data: Dataset) -> Dict[str, Any]:
        """
        Train MT-GBM model with custom multi-task loss function.

        Parameters
        ----------
        train_data : Dataset
            Training data
        val_data : Dataset
            Validation data

        Returns
        -------
        metrics : dict
            Training metrics
        """
        self.logger.info("Starting LightGBMT multi-task training with custom loss...")

        # Get number of tasks
        num_tasks = self.lgb_params["num_labels"]

        # Train with custom train() function (supports epoch passing)
        self.model = train(
            self.lgb_params,
            train_data,
            num_boost_round=self.hyperparams.num_iterations,
            valid_sets=[val_data],
            valid_names=["valid"],
            fobj=self.loss_function.objective,  # Custom loss with multi-task support
            feval=self._create_eval_function(),
            early_stopping_rounds=self.hyperparams.early_stopping_rounds
            if self.hyperparams.early_stopping_rounds
            else None,
            verbose_eval=10,
        )

        # CRITICAL: Set number of labels for multi-task predictions
        self.model.set_num_labels(num_tasks)

        # Extract training metrics
        metrics = {
            "num_iterations": self.model.num_trees(),
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.model.feature_importance().tolist(),
            "num_tasks": num_tasks,
        }

        self.logger.info(
            f"Training completed: {metrics['num_iterations']} trees, {num_tasks} tasks"
        )

        return metrics

    def _create_eval_function(self):
        """Create evaluation function for LightGBM."""

        def eval_func(preds, train_data):
            """Custom evaluation function wrapper."""
            task_scores, mean_score = self.loss_function.evaluate(preds, train_data)
            return "mean_auc", mean_score, True  # (name, value, is_higher_better)

        return eval_func

    def _prepare_prediction_data(
        self, df: pd.DataFrame, feature_columns: Optional[list] = None
    ) -> np.ndarray:
        """
        Prepare data for prediction (no labels needed).

        Following legacy pattern: return numpy array directly instead of Dataset.
        This avoids needing free_raw_data=False and matches LightGBM's standard usage.

        Parameters
        ----------
        df : DataFrame
            Data to predict on (may contain features + other columns)
        feature_columns : list, optional
            List of feature column names to use
            If not provided, falls back to hyperparameters

        Returns
        -------
        X : np.ndarray
            Feature matrix for prediction
        """
        # Use provided feature_columns or fallback to hyperparams
        if feature_columns is None:
            feature_columns = (
                self.hyperparams.tab_field_list + self.hyperparams.cat_field_list
            )
            self.logger.warning(
                "feature_columns not provided for prediction, falling back to hyperparams. "
                "Consider passing explicit columns for consistency with training."
            )

        # Extract features from DataFrame and return as numpy array
        # Legacy pattern: predict directly on arrays, not Datasets
        X = df[feature_columns].values
        
        return X

    def _predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate multi-task predictions.

        Following legacy pattern: accept numpy array directly.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix to predict on

        Returns
        -------
        predictions : np.ndarray
            Multi-task predictions [N_samples, N_tasks]
            Already includes sigmoid transformation from model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Multi-task prediction on raw numpy array (legacy pattern)
        # Returns [N_samples, N_tasks]
        predictions = self.model.predict(data)

        # Apply sigmoid transformation for binary classification
        predictions = expit(predictions)

        self.logger.info(f"Generated predictions with shape: {predictions.shape}")

        return predictions

    def _save_model(self, output_path: str) -> None:
        """
        Save multi-task model artifacts.

        Parameters
        ----------
        output_path : str
            Directory to save artifacts
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LightGBMT model using standard save_model (compatible with Booster load)
        # Note: save_model2() creates incompatible format that Booster() cannot load
        model_file = output_dir / "lightgbmmt_model.txt"
        self.model.save_model(str(model_file))
        self.logger.info(f"Saved multi-task model to {model_file}")

        # Save hyperparameters
        hyperparams_file = output_dir / "hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            json.dump(self.hyperparams.model_dump(), f, indent=2)
        self.logger.info(f"Saved hyperparameters to {hyperparams_file}")

        # Save training state
        state_file = output_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(self.training_state.to_checkpoint_dict(), f, indent=2)
        self.logger.info(f"Saved training state to {state_file}")

    def _load_model(self, model_path: str) -> None:
        """
        Load multi-task model artifacts.

        Parameters
        ----------
        model_path : str
            Path to model artifacts
        """
        model_dir = Path(model_path)

        # Load LightGBMT model using custom Booster
        model_file = model_dir / "lightgbmmt_model.txt"
        self.model = Booster(model_file=str(model_file))
        self.logger.info(f"Loaded multi-task model from {model_file}")

        # Restore num_labels from hyperparameters
        if self.hyperparams.task_label_names:
            num_tasks = len(self.hyperparams.task_label_names)
            self.model.set_num_labels(num_tasks)
            self.logger.info(
                f"Restored num_labels={num_tasks} for multi-task predictions"
            )

        # Load training state if available
        state_file = model_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state_dict = json.load(f)
            self.training_state = self.training_state.from_checkpoint_dict(state_dict)
            self.logger.info(f"Loaded training state from {state_file}")
