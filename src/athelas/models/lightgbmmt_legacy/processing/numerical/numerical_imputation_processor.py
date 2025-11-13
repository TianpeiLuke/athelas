"""
Numerical Imputation Processor - Single Column Architecture

This processor performs imputation on a SINGLE numerical column.
Follows the single-column architecture pattern for real-time inference pipelines.

For batch processing of multiple columns, use one processor per column.
"""

from typing import Any, Union, Optional
import pandas as pd
import numpy as np
import pickle as pkl
import json
import logging
from pathlib import Path

from ..processors import Processor

# Setup logger
logger = logging.getLogger(__name__)


class NumericalVariableImputationProcessor(Processor):
    """
    A processor that performs imputation on a SINGLE numerical variable/column.

    Designed for real-time inference pipelines where each processor
    handles one column and processors can be chained with >> operator.

    For batch processing of multiple columns, use one processor per column.

    Examples:
        >>> # Create processor for single column
        >>> proc = NumericalImputationProcessor(
        ...     column_name='age',
        ...     strategy='mean'
        ... )
        >>>
        >>> # Fit on training data
        >>> proc.fit(train_df['age'])
        >>>
        >>> # Process single value (real-time inference)
        >>> imputed_value = proc.process(None)  # Returns mean value
        >>>
        >>> # Transform Series or DataFrame
        >>> imputed_series = proc.transform(test_df['age'])
    """

    def __init__(
        self,
        column_name: str,
        imputation_value: Optional[Union[int, float]] = None,
        strategy: Optional[str] = None,
    ):
        """
        Initialize numerical imputation processor.

        Args:
            column_name: Name of the column to impute (single column)
            imputation_value: Pre-computed imputation value (for inference)
            strategy: Strategy for fitting ('mean', 'median', 'mode')
                     Required if imputation_value is None

        Raises:
            ValueError: If column_name is empty or invalid
            ValueError: If both imputation_value and strategy are None
        """
        super().__init__()  # Initialize base Processor

        self.processor_name = "numerical_imputation_processor"
        self.function_name_list = ["fit", "process", "transform"]

        # Validate column_name
        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name must be a non-empty string")

        self.column_name = column_name
        self.strategy = strategy

        # Set fitted state based on whether we have an imputation value
        self.is_fitted = imputation_value is not None

        if imputation_value is not None:
            self._validate_imputation_value(imputation_value)
            self.imputation_value = imputation_value
        else:
            self.imputation_value = None
            # If no imputation_value provided, strategy is required for fitting
            if strategy is None:
                raise ValueError("Either imputation_value or strategy must be provided")

    def get_name(self) -> str:
        """Return processor name for base class compatibility."""
        return self.processor_name

    def _validate_imputation_value(self, value: Any) -> None:
        """
        Validate that imputation value is numeric.

        Args:
            value: Value to validate

        Raises:
            ValueError: If value is not numeric
        """
        if not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"Imputation value must be numeric, got {type(value)} for column '{self.column_name}'"
            )
        if pd.isna(value):
            logger.warning(f"Imputation value is NaN for column '{self.column_name}'")

    def fit(
        self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None
    ) -> "NumericalVariableImputationProcessor":
        """
        Fit imputation value on a Series (single column).

        Args:
            X: Series (preferred) or DataFrame with column_name
            y: Ignored (for sklearn compatibility)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If column_name not found in DataFrame
            ValueError: If strategy is unknown
        """
        # Extract Series if DataFrame provided
        if isinstance(X, pd.DataFrame):
            if self.column_name not in X.columns:
                raise ValueError(f"Column '{self.column_name}' not found in DataFrame")
            data = X[self.column_name]
        else:
            data = X

        # Calculate imputation value based on strategy
        if data.isna().all():
            self.imputation_value = 0.0  # Default for all-NaN columns
            logger.warning(
                f"Column '{self.column_name}' has all NaN values, using 0.0 as imputation value"
            )
        elif self.strategy == "mean":
            self.imputation_value = float(data.mean())
        elif self.strategy == "median":
            self.imputation_value = float(data.median())
        elif self.strategy == "mode":
            mode_values = data.mode()
            self.imputation_value = (
                float(mode_values[0]) if len(mode_values) > 0 else 0.0
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True
        logger.info(
            f"Fitted {self.column_name}: strategy={self.strategy}, "
            f"imputation_value={self.imputation_value:.4f}"
        )

        return self

    def process(self, input_value: Union[int, float, Any]) -> Union[int, float]:
        """
        Process a SINGLE numerical value for this column.

        This method is called by __call__ (inherited from base Processor).
        It handles single-value processing for real-time inference.

        Args:
            input_value: Single value to impute if missing

        Returns:
            Imputed value (or original if not missing)

        Raises:
            RuntimeError: If processor not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' must be fitted before processing"
            )

        # Handle missing values
        if pd.isna(input_value):
            return self.imputation_value

        return input_value

    def transform(
        self, X: Union[pd.Series, pd.DataFrame, Any]
    ) -> Union[pd.Series, pd.DataFrame, float]:
        """
        Transform data using the fitted imputation value.

        Args:
            X: Series, DataFrame, or single value

        Returns:
            Imputed data in same format as input

        Raises:
            RuntimeError: If processor not fitted
            ValueError: If column_name not found in DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' must be fitted before transforming"
            )

        # Handle Series
        if isinstance(X, pd.Series):
            return X.fillna(self.imputation_value)

        # Handle DataFrame
        elif isinstance(X, pd.DataFrame):
            if self.column_name not in X.columns:
                raise ValueError(f"Column '{self.column_name}' not found in DataFrame")
            df = X.copy()
            df[self.column_name] = df[self.column_name].fillna(self.imputation_value)
            return df

        # Handle single value (delegate to process)
        else:
            return self.process(X)

    def get_imputation_value(self) -> Union[int, float]:
        """
        Get the fitted imputation value.

        Returns:
            Imputation value for this column

        Raises:
            RuntimeError: If processor not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' has not been fitted"
            )
        return self.imputation_value

    def set_imputation_value(self, value: Union[int, float]) -> None:
        """
        Set imputation value (for pre-fitted processor).

        Args:
            value: Imputation value to use

        Raises:
            ValueError: If value is not numeric
        """
        self._validate_imputation_value(value)
        self.imputation_value = value
        self.is_fitted = True
        logger.info(
            f"Set imputation value for '{self.column_name}': {self.imputation_value:.4f}"
        )

    def get_params(self) -> dict:
        """
        Get processor parameters (DEPRECATED).

        Use get_imputation_value() instead for the fitted value.

        Returns:
            Dictionary with all parameters
        """
        import warnings

        warnings.warn(
            "get_params() is deprecated, use get_imputation_value() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "column_name": self.column_name,
            "imputation_value": self.imputation_value,
            "strategy": self.strategy,
        }

    def save_imputation_value(self, output_dir: Union[Path, str]) -> None:
        """
        Save imputation value to disk.

        Creates two files:
        1. {column_name}_impute_value.pkl (for loading)
        2. {column_name}_impute_value.json (for human readability)

        Args:
            output_dir: Directory to save artifacts to

        Raises:
            RuntimeError: If processor not fitted

        Examples:
            >>> proc = NumericalImputationProcessor('age', strategy='mean')
            >>> proc.fit(train_df['age'])
            >>> proc.save_imputation_value('model_artifacts/')
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Cannot save before fitting processor for column '{self.column_name}'"
            )

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Save pickle (for loading)
        pkl_file = output_dir_path / f"{self.column_name}_impute_value.pkl"
        with open(pkl_file, "wb") as f:
            pkl.dump(self.imputation_value, f)

        # Save JSON (for readability)
        json_file = output_dir_path / f"{self.column_name}_impute_value.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "column_name": self.column_name,
                    "imputation_value": float(self.imputation_value),
                    "strategy": self.strategy,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Saved imputation value for '{self.column_name}' to {output_dir_path}"
        )

    def load_imputation_value(self, filepath: Union[Path, str]) -> None:
        """
        Load imputation value from disk.

        Args:
            filepath: Path to pickle file or directory containing it

        Raises:
            FileNotFoundError: If file not found
            ValueError: If value is invalid

        Examples:
            >>> proc = NumericalImputationProcessor('age', strategy='mean')
            >>> proc.load_imputation_value('model_artifacts/')
            >>> # Or specify exact file
            >>> proc.load_imputation_value('model_artifacts/age_impute_value.pkl')
        """
        filepath_path = Path(filepath)

        # Handle directory path
        if filepath_path.is_dir():
            pkl_file = filepath_path / f"{self.column_name}_impute_value.pkl"
        else:
            pkl_file = filepath_path

        if not pkl_file.exists():
            raise FileNotFoundError(f"Imputation value file not found: {pkl_file}")

        # Load value
        with open(pkl_file, "rb") as f:
            loaded_value = pkl.load(f)

        # Validate
        self._validate_imputation_value(loaded_value)

        self.imputation_value = loaded_value
        self.is_fitted = True

        logger.info(
            f"Loaded imputation value for '{self.column_name}' from {pkl_file}: {self.imputation_value:.4f}"
        )

    @classmethod
    def from_imputation_dict(cls, imputation_dict: dict) -> dict:
        """
        Create processors from script output (impute_dict.pkl).

        This factory method simplifies creating multiple processors from
        the dictionary format used by missing_value_imputation.py script.

        Args:
            imputation_dict: Dictionary mapping column names to imputation values
                            Format: {column_name: imputation_value}

        Returns:
            Dictionary mapping column names to fitted processors

        Raises:
            TypeError: If imputation_dict is not a dictionary
            ValueError: If column name is not a string

        Examples:
            >>> with open("impute_dict.pkl", "rb") as f:
            ...     impute_dict = pkl.load(f)
            >>> processors = NumericalImputationProcessor.from_imputation_dict(impute_dict)
            >>> # Use processors in pipeline
            >>> for col, proc in processors.items():
            ...     dataset.add_pipeline(col, proc)
        """
        if not isinstance(imputation_dict, dict):
            raise TypeError("imputation_dict must be a dictionary")

        processors = {}
        for col, val in imputation_dict.items():
            if not isinstance(col, str):
                raise ValueError(f"Column name must be string, got {type(col)}")

            proc = cls(column_name=col, imputation_value=val)
            processors[col] = proc

        logger.info(f"Created {len(processors)} processors from imputation dict")
        return processors

    @classmethod
    def from_script_artifacts(
        cls, artifacts_dir: Union[Path, str], filename: str = "impute_dict.pkl"
    ) -> dict:
        """
        Load processors from script output directory.

        Looks for impute_dict.pkl in the specified directory and
        creates processors from it.

        Args:
            artifacts_dir: Directory containing impute_dict.pkl
            filename: Name of the imputation dict file (default: impute_dict.pkl)

        Returns:
            Dictionary mapping column names to fitted processors

        Raises:
            FileNotFoundError: If impute_dict.pkl not found

        Examples:
            >>> processors = NumericalImputationProcessor.from_script_artifacts(
            ...     "model_artifacts/"
            ... )
            >>> for col, proc in processors.items():
            ...     dataset.add_pipeline(col, proc)
        """
        artifacts_path = Path(artifacts_dir)
        impute_dict_file = artifacts_path / filename

        if not impute_dict_file.exists():
            raise FileNotFoundError(f"{filename} not found in {artifacts_path}")

        with open(impute_dict_file, "rb") as f:
            impute_dict = pkl.load(f)

        logger.info(f"Loaded imputation dict from {impute_dict_file}")
        return cls.from_imputation_dict(impute_dict)
