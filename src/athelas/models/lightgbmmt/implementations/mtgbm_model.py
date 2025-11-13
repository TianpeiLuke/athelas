"""
Multi-Task Gradient Boosting Model (MT-GBM) implementation.

Implements shared tree structure for multi-task learning.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path

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
    ) -> Tuple[lgb.Dataset, lgb.Dataset, Optional[lgb.Dataset]]:
        """
        Prepare data for LightGBM training.

        Parameters
        ----------
        train_df : DataFrame
            Training data with features and multi-task labels
        val_df : DataFrame
            Validation data
        test_df : DataFrame, optional
            Test data

        Returns
        -------
        train_data, val_data, test_data : tuple of lgb.Dataset
            Prepared LightGBM datasets
        """
        # Extract features and labels
        feature_cols = self.hyperparams.full_field_list

        # Prepare training data
        X_train = train_df[feature_cols].values
        y_train = self._extract_multi_task_labels(train_df)
        train_data = lgb.Dataset(
            X_train,
            label=y_train.flatten(),  # Flatten for LightGBM
            feature_name=feature_cols,
            categorical_feature=self.hyperparams.cat_field_list,
        )

        # Prepare validation data
        X_val = val_df[feature_cols].values
        y_val = self._extract_multi_task_labels(val_df)
        val_data = lgb.Dataset(X_val, label=y_val.flatten(), reference=train_data)

        # Prepare test data if provided
        test_data = None
        if test_df is not None:
            X_test = test_df[feature_cols].values
            y_test = self._extract_multi_task_labels(test_df)
            test_data = lgb.Dataset(
                X_test, label=y_test.flatten(), reference=train_data
            )

        self.logger.info(
            f"Prepared data: train={X_train.shape}, "
            f"val={X_val.shape}, "
            f"test={X_test.shape if test_df is not None else None}"
        )

        return train_data, val_data, test_data

    def _extract_multi_task_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract multi-task labels from dataframe.

        Parameters
        ----------
        df : DataFrame
            Data with task label columns

        Returns
        -------
        labels : np.ndarray
            Multi-task labels [N_samples, N_tasks]
        """
        # Task columns should be specified in hyperparams
        # For now, assume they follow pattern: task_0, task_1, ...
        num_tasks = self.hyperparams.num_tasks or self.loss_function.num_col

        task_cols = [f"task_{i}" for i in range(num_tasks)]
        # Fallback to finding columns
        if not all(col in df.columns for col in task_cols):
            # Try finding task columns by pattern
            task_cols = [col for col in df.columns if col.startswith("task_")]
            if not task_cols:
                # Try other common patterns
                task_cols = [col for col in df.columns if "label" in col.lower()]

        labels = df[task_cols].values
        return labels

    def _initialize_model(self) -> None:
        """Initialize LightGBM model parameters."""
        # Build LightGBM parameters from hyperparameters
        self.lgb_params = {
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

        self.logger.info(f"Initialized model with params: {self.lgb_params}")

    def _train_model(
        self, train_data: lgb.Dataset, val_data: lgb.Dataset
    ) -> Dict[str, Any]:
        """
        Train MT-GBM model with custom loss function.

        Parameters
        ----------
        train_data : lgb.Dataset
            Training data
        val_data : lgb.Dataset
            Validation data

        Returns
        -------
        metrics : dict
            Training metrics
        """
        self.logger.info("Starting LightGBM training with custom loss...")

        # Train with custom loss function
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.hyperparams.num_iterations,
            valid_sets=[val_data],
            valid_names=["valid"],
            fobj=self.loss_function.objective,
            feval=self._create_eval_function(),
            early_stopping_rounds=self.hyperparams.early_stopping_rounds,
            verbose_eval=10,
        )

        # Extract training metrics
        metrics = {
            "num_iterations": self.model.num_trees(),
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.model.feature_importance().tolist(),
        }

        self.logger.info(f"Training completed: {metrics['num_iterations']} trees")

        return metrics

    def _create_eval_function(self):
        """Create evaluation function for LightGBM."""

        def eval_func(preds, train_data):
            """Custom evaluation function wrapper."""
            task_scores, mean_score = self.loss_function.evaluate(preds, train_data)
            return "mean_auc", mean_score, True  # (name, value, is_higher_better)

        return eval_func

    def _predict(self, data: lgb.Dataset) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        data : lgb.Dataset
            Data to predict on

        Returns
        -------
        predictions : np.ndarray
            Raw predictions [N_samples * N_tasks]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get data from Dataset
        X = data.data
        predictions = self.model.predict(X)

        return predictions

    def _save_model(self, output_path: str) -> None:
        """
        Save model artifacts.

        Parameters
        ----------
        output_path : str
            Directory to save artifacts
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_file = output_dir / "lightgbmmt_model.txt"
        self.model.save_model(str(model_file))
        self.logger.info(f"Saved model to {model_file}")

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
        Load model artifacts.

        Parameters
        ----------
        model_path : str
            Path to model artifacts
        """
        model_dir = Path(model_path)

        # Load LightGBM model
        model_file = model_dir / "lightgbmmt_model.txt"
        self.model = lgb.Booster(model_file=str(model_file))
        self.logger.info(f"Loaded model from {model_file}")

        # Load training state if available
        state_file = model_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state_dict = json.load(f)
            self.training_state = self.training_state.from_checkpoint_dict(state_dict)
            self.logger.info(f"Loaded training state from {state_file}")
