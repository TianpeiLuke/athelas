"""
Feature Normalization Processor for Numerical Features

This module provides atomic feature normalization (L1, L2, max normalization).
Extracted from TSA feature normalization requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class FeatureNormalizationProcessor(Processor):
    """
    Normalizes features using L1, L2, or max normalization.

    Extracted from TSA feature normalization requirements.

    Args:
        method: 'l1', 'l2', 'max'
        axis: Axis along which to normalize (0 for columns, 1 for rows)
        columns: Specific columns to normalize
        epsilon: Small value to avoid division by zero
    """

    def __init__(
        self,
        method: str = "l2",
        axis: int = 1,
        columns: Optional[List[str]] = None,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.method = method
        self.axis = axis
        self.columns = columns
        self.epsilon = epsilon
        self.is_fitted = False

        if method not in ["l1", "l2", "max"]:
            raise ValueError(f"method must be one of ['l1', 'l2', 'max'], got {method}")

    def fit(self, data: Any) -> "FeatureNormalizationProcessor":
        """No fitting required for normalization"""
        self.is_fitted = True
        logger.info(f"FeatureNormalizationProcessor fitted with method: {self.method}")
        return self

    def process(
        self, input_data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Apply feature normalization"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, np.ndarray):
            return self._process_numpy_array(input_data)
        elif isinstance(input_data, pd.DataFrame):
            return self._process_dataframe(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        result = input_data.copy().astype(float)

        if self.method == "l1":
            # L1 normalization (Manhattan norm)
            norms = np.sum(np.abs(result), axis=self.axis, keepdims=True)
            norms = np.maximum(norms, self.epsilon)  # Avoid division by zero
            result = result / norms

        elif self.method == "l2":
            # L2 normalization (Euclidean norm)
            norms = np.sqrt(np.sum(result**2, axis=self.axis, keepdims=True))
            norms = np.maximum(norms, self.epsilon)  # Avoid division by zero
            result = result / norms

        elif self.method == "max":
            # Max normalization
            max_vals = np.max(np.abs(result), axis=self.axis, keepdims=True)
            max_vals = np.maximum(max_vals, self.epsilon)  # Avoid division by zero
            result = result / max_vals

        return result

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        result = input_data.copy()

        # Determine columns to normalize
        columns_to_normalize = (
            self.columns or result.select_dtypes(include=[np.number]).columns
        )

        if self.axis == 0:
            # Normalize each column independently
            for col in columns_to_normalize:
                if col in result.columns:
                    col_data = result[col].values.astype(float)

                    if self.method == "l1":
                        norm = np.sum(np.abs(col_data))
                        norm = max(norm, self.epsilon)
                        result[col] = col_data / norm

                    elif self.method == "l2":
                        norm = np.sqrt(np.sum(col_data**2))
                        norm = max(norm, self.epsilon)
                        result[col] = col_data / norm

                    elif self.method == "max":
                        max_val = np.max(np.abs(col_data))
                        max_val = max(max_val, self.epsilon)
                        result[col] = col_data / max_val

        elif self.axis == 1:
            # Normalize each row independently
            numeric_data = result[columns_to_normalize].values.astype(float)

            if self.method == "l1":
                norms = np.sum(np.abs(numeric_data), axis=1, keepdims=True)
                norms = np.maximum(norms, self.epsilon)
                normalized_data = numeric_data / norms

            elif self.method == "l2":
                norms = np.sqrt(np.sum(numeric_data**2, axis=1, keepdims=True))
                norms = np.maximum(norms, self.epsilon)
                normalized_data = numeric_data / norms

            elif self.method == "max":
                max_vals = np.max(np.abs(numeric_data), axis=1, keepdims=True)
                max_vals = np.maximum(max_vals, self.epsilon)
                normalized_data = numeric_data / max_vals

            # Update the DataFrame with normalized values
            result[columns_to_normalize] = normalized_data

        return result

    def get_normalization_info(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get information about the normalization that would be applied"""
        if isinstance(data, np.ndarray):
            if self.method == "l1":
                norms = np.sum(np.abs(data), axis=self.axis)
            elif self.method == "l2":
                norms = np.sqrt(np.sum(data**2, axis=self.axis))
            elif self.method == "max":
                norms = np.max(np.abs(data), axis=self.axis)

            return {
                "method": self.method,
                "axis": self.axis,
                "norms_shape": norms.shape,
                "norms_stats": {
                    "min": float(np.min(norms)),
                    "max": float(np.max(norms)),
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                },
            }

        elif isinstance(data, pd.DataFrame):
            columns_to_normalize = (
                self.columns or data.select_dtypes(include=[np.number]).columns
            )
            numeric_data = data[columns_to_normalize].values.astype(float)

            if self.axis == 0:
                # Column-wise normalization
                norms = {}
                for i, col in enumerate(columns_to_normalize):
                    col_data = numeric_data[:, i]
                    if self.method == "l1":
                        norm = np.sum(np.abs(col_data))
                    elif self.method == "l2":
                        norm = np.sqrt(np.sum(col_data**2))
                    elif self.method == "max":
                        norm = np.max(np.abs(col_data))
                    norms[col] = float(norm)

                return {"method": self.method, "axis": self.axis, "column_norms": norms}

            elif self.axis == 1:
                # Row-wise normalization
                if self.method == "l1":
                    norms = np.sum(np.abs(numeric_data), axis=1)
                elif self.method == "l2":
                    norms = np.sqrt(np.sum(numeric_data**2, axis=1))
                elif self.method == "max":
                    norms = np.max(np.abs(numeric_data), axis=1)

                return {
                    "method": self.method,
                    "axis": self.axis,
                    "row_norms_stats": {
                        "min": float(np.min(norms)),
                        "max": float(np.max(norms)),
                        "mean": float(np.mean(norms)),
                        "std": float(np.std(norms)),
                    },
                }

        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

    def check_normalization(
        self, data: Union[np.ndarray, pd.DataFrame], tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Check if data is already normalized according to the specified method"""
        if isinstance(data, np.ndarray):
            if self.method == "l1":
                norms = np.sum(np.abs(data), axis=self.axis)
                expected_norm = 1.0
            elif self.method == "l2":
                norms = np.sqrt(np.sum(data**2, axis=self.axis))
                expected_norm = 1.0
            elif self.method == "max":
                norms = np.max(np.abs(data), axis=self.axis)
                expected_norm = 1.0

            is_normalized = np.allclose(norms, expected_norm, atol=tolerance)

            return {
                "is_normalized": bool(is_normalized),
                "method": self.method,
                "tolerance": tolerance,
                "norm_deviations": {
                    "max_deviation": float(np.max(np.abs(norms - expected_norm))),
                    "mean_deviation": float(np.mean(np.abs(norms - expected_norm))),
                },
            }

        elif isinstance(data, pd.DataFrame):
            columns_to_check = (
                self.columns or data.select_dtypes(include=[np.number]).columns
            )
            numeric_data = data[columns_to_check].values.astype(float)

            if self.axis == 0:
                # Check column-wise normalization
                results = {}
                for i, col in enumerate(columns_to_check):
                    col_data = numeric_data[:, i]
                    if self.method == "l1":
                        norm = np.sum(np.abs(col_data))
                    elif self.method == "l2":
                        norm = np.sqrt(np.sum(col_data**2))
                    elif self.method == "max":
                        norm = np.max(np.abs(col_data))

                    is_normalized = abs(norm - 1.0) <= tolerance
                    results[col] = {
                        "is_normalized": is_normalized,
                        "norm": float(norm),
                        "deviation": float(abs(norm - 1.0)),
                    }

                overall_normalized = all(r["is_normalized"] for r in results.values())

                return {
                    "is_normalized": overall_normalized,
                    "method": self.method,
                    "axis": self.axis,
                    "tolerance": tolerance,
                    "column_results": results,
                }

            elif self.axis == 1:
                # Check row-wise normalization
                if self.method == "l1":
                    norms = np.sum(np.abs(numeric_data), axis=1)
                elif self.method == "l2":
                    norms = np.sqrt(np.sum(numeric_data**2, axis=1))
                elif self.method == "max":
                    norms = np.max(np.abs(numeric_data), axis=1)

                is_normalized = np.allclose(norms, 1.0, atol=tolerance)

                return {
                    "is_normalized": bool(is_normalized),
                    "method": self.method,
                    "axis": self.axis,
                    "tolerance": tolerance,
                    "norm_deviations": {
                        "max_deviation": float(np.max(np.abs(norms - 1.0))),
                        "mean_deviation": float(np.mean(np.abs(norms - 1.0))),
                    },
                }

        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "method": self.method,
            "axis": self.axis,
            "columns": self.columns,
            "epsilon": self.epsilon,
        }

    def __repr__(self) -> str:
        return (
            f"FeatureNormalizationProcessor(method='{self.method}', "
            f"axis={self.axis}, epsilon={self.epsilon})"
        )
