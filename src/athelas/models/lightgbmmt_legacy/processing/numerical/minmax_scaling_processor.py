"""
MinMax Scaling Processor for Numerical Features

This module provides atomic min-max scaling with learned parameters.
Extracted from TSA preprocessing scaling logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class MinMaxScalingProcessor(Processor):
    """
    Min-max scaling with learned parameters.

    Extracted from TSA preprocessing:
    - seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(seq_num_min_)

    Args:
        feature_range: Target range for scaling
        learned_params: Pre-computed scaling parameters
        columns: Specific columns to scale
        clip_values: Whether to clip to feature_range
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        learned_params: Optional[Dict[str, Dict[str, float]]] = None,
        columns: Optional[List[str]] = None,
        clip_values: bool = True,
    ):
        super().__init__()
        self.feature_range = feature_range
        self.learned_params = learned_params or {}
        self.columns = columns
        self.clip_values = clip_values
        self.scale_params = learned_params or {}
        self.is_fitted = False

        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                f"feature_range[0] must be less than feature_range[1], got {feature_range}"
            )

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "MinMaxScalingProcessor":
        """Learn scaling parameters from data"""
        if not self.scale_params:
            if isinstance(data, np.ndarray):
                self._fit_numpy_array(data)
            elif isinstance(data, pd.DataFrame):
                self._fit_dataframe(data)
            else:
                raise ValueError(f"Unsupported data type for fitting: {type(data)}")

        self.is_fitted = True
        logger.info(
            f"MinMaxScalingProcessor fitted with feature_range: {self.feature_range}"
        )
        return self

    def _fit_numpy_array(self, data: np.ndarray) -> None:
        """Fit scaling parameters for numpy array"""
        # Compute min and max for each column
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min

        # Avoid division by zero
        data_range[data_range == 0] = 1

        # Compute scale and min for transform: X_scaled = X * scale + min
        target_min, target_max = self.feature_range
        scale = (target_max - target_min) / data_range
        min_val = target_min - data_min * scale

        self.scale_params = {
            "scale_": scale,
            "min_": min_val,
            "data_min_": data_min,
            "data_max_": data_max,
            "data_range_": data_range,
        }

    def _fit_dataframe(self, data: pd.DataFrame) -> None:
        """Fit scaling parameters for DataFrame"""
        columns = self.columns or data.select_dtypes(include=[np.number]).columns
        self.scale_params = {}

        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in DataFrame, skipping")
                continue

            col_data = data[col].values
            data_min = np.min(col_data)
            data_max = np.max(col_data)
            data_range = data_max - data_min

            if data_range == 0:
                data_range = 1
                logger.warning(f"Column {col} has zero range, using range=1")

            target_min, target_max = self.feature_range
            scale = (target_max - target_min) / data_range
            min_val = target_min - data_min * scale

            self.scale_params[col] = {
                "scale_": scale,
                "min_": min_val,
                "data_min_": data_min,
                "data_max_": data_max,
                "data_range_": data_range,
            }

    def process(
        self, input_data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Apply min-max scaling"""
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

        # Apply TSA-style scaling: X_scaled = X * scale + min
        if "scale_" in self.scale_params:
            result = result * self.scale_params["scale_"] + self.scale_params["min_"]

        # Apply clipping if requested
        if self.clip_values:
            target_min, target_max = self.feature_range
            result = np.clip(result, target_min, target_max)

        return result

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        result = input_data.copy()

        for col, params in self.scale_params.items():
            if col in result.columns:
                result[col] = result[col] * params["scale_"] + params["min_"]

                # Apply clipping if requested
                if self.clip_values:
                    target_min, target_max = self.feature_range
                    result[col] = result[col].clip(target_min, target_max)

        return result

    def inverse_transform(
        self, scaled_data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform scaled data back to original scale"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before inverse transform")

        if isinstance(scaled_data, np.ndarray):
            return self._inverse_transform_numpy_array(scaled_data)
        elif isinstance(scaled_data, pd.DataFrame):
            return self._inverse_transform_dataframe(scaled_data)
        else:
            raise ValueError(f"Unsupported input type: {type(scaled_data)}")

    def _inverse_transform_numpy_array(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform numpy array"""
        result = scaled_data.copy().astype(float)

        if "scale_" in self.scale_params:
            # Inverse: X_original = (X_scaled - min) / scale
            result = (result - self.scale_params["min_"]) / self.scale_params["scale_"]

        return result

    def _inverse_transform_dataframe(self, scaled_data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform DataFrame"""
        result = scaled_data.copy()

        for col, params in self.scale_params.items():
            if col in result.columns:
                # Inverse: X_original = (X_scaled - min) / scale
                result[col] = (result[col] - params["min_"]) / params["scale_"]

        return result

    def get_data_range(
        self, column: Optional[str] = None
    ) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        """Get the original data range(s)"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before getting data range")

        if column is not None:
            if isinstance(self.scale_params, dict) and column in self.scale_params:
                params = self.scale_params[column]
                return (params["data_min_"], params["data_max_"])
            elif "data_min_" in self.scale_params:
                # Single array case
                return (self.scale_params["data_min_"], self.scale_params["data_max_"])
            else:
                raise KeyError(f"Column {column} not found in scale parameters")
        else:
            # Return all ranges
            if isinstance(self.scale_params, dict) and any(
                isinstance(v, dict) for v in self.scale_params.values()
            ):
                # DataFrame case
                return {
                    col: (params["data_min_"], params["data_max_"])
                    for col, params in self.scale_params.items()
                    if isinstance(params, dict)
                }
            else:
                # Single array case
                return (self.scale_params["data_min_"], self.scale_params["data_max_"])

    def get_scaling_info(self) -> Dict[str, Any]:
        """Get detailed scaling information"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before getting scaling info")

        info = {
            "feature_range": self.feature_range,
            "clip_values": self.clip_values,
            "scale_params": self.scale_params,
        }

        return info

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "feature_range": self.feature_range,
            "learned_params": self.learned_params,
            "columns": self.columns,
            "clip_values": self.clip_values,
            "scale_params": self.scale_params,
        }

    def __repr__(self) -> str:
        return (
            f"MinMaxScalingProcessor(feature_range={self.feature_range}, "
            f"clip_values={self.clip_values}, "
            f"n_features={len(self.scale_params) if isinstance(self.scale_params, dict) else 'unknown'})"
        )
