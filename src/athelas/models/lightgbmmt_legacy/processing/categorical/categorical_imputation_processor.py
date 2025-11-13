"""
Categorical Imputation Processor for Missing Values

This module provides atomic categorical imputation with configurable defaults.
Extracted from TSA default value handling logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from collections import Counter
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class CategoricalImputationProcessor(Processor):
    """
    Handles missing categorical values with configurable defaults.

    Extracted from TSA default value handling logic.

    Args:
        default_values: Dictionary of field -> default value mappings
        missing_indicators: Values that indicate missing data
        strategy: 'default', 'mode', 'constant'
        constant_value: Value to use for constant strategy
    """

    def __init__(
        self,
        default_values: Optional[Dict[str, Any]] = None,
        missing_indicators: Optional[List[Any]] = None,
        strategy: str = "default",
        constant_value: str = "UNKNOWN",
    ):
        super().__init__()

        if missing_indicators is None:
            missing_indicators = ["", "My Text String", None, np.nan]

        self.default_values = default_values or {}
        self.missing_indicators = missing_indicators
        self.strategy = strategy
        self.constant_value = constant_value
        self.learned_defaults = {}
        self.is_fitted = False

        if strategy not in ["default", "mode", "constant"]:
            raise ValueError(
                f"strategy must be one of ['default', 'mode', 'constant'], got {strategy}"
            )

    def fit(self, data: Union[Dict, pd.DataFrame]) -> "CategoricalImputationProcessor":
        """Learn default values from data if needed"""
        if self.strategy == "mode":
            if isinstance(data, pd.DataFrame):
                for col in data.select_dtypes(include=["object"]).columns:
                    # Filter out missing indicators
                    valid_values = data[col][~data[col].isin(self.missing_indicators)]
                    if len(valid_values) > 0:
                        mode_value = valid_values.mode()
                        self.learned_defaults[col] = (
                            mode_value[0]
                            if len(mode_value) > 0
                            else self.constant_value
                        )
                    else:
                        self.learned_defaults[col] = self.constant_value
            elif isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        # Find mode, excluding missing indicators
                        valid_values = [
                            v for v in values if v not in self.missing_indicators
                        ]
                        if valid_values:
                            counter = Counter(valid_values)
                            self.learned_defaults[key] = counter.most_common(1)[0][0]
                        else:
                            self.learned_defaults[key] = self.constant_value

        self.is_fitted = True
        logger.info(
            f"CategoricalImputationProcessor fitted with strategy: {self.strategy}"
        )
        return self

    def process(
        self, input_data: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> Union[Dict, pd.DataFrame, np.ndarray]:
        """Apply categorical imputation"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, pd.DataFrame):
            return self._process_dataframe(input_data)
        elif isinstance(input_data, dict):
            return self._process_dict(input_data)
        elif isinstance(input_data, np.ndarray):
            return self._process_numpy_array(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        result = input_data.copy()
        for col in result.select_dtypes(include=["object"]).columns:
            mask = result[col].isin(self.missing_indicators)
            if mask.any():
                default_val = self._get_default_value(col)
                result.loc[mask, col] = default_val
        return result

    def _process_dict(self, input_data: Dict) -> Dict:
        """Process dictionary input"""
        result = {}
        for key, values in input_data.items():
            if isinstance(values, list):
                default_val = self._get_default_value(key)
                result[key] = [
                    default_val if v in self.missing_indicators else v for v in values
                ]
            else:
                if values in self.missing_indicators:
                    default_val = self._get_default_value(key)
                    result[key] = default_val
                else:
                    result[key] = values
        return result

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        result = input_data.copy()

        # Handle object arrays (string arrays)
        if result.dtype == object:
            # Create mask for missing indicators
            mask = np.isin(result, self.missing_indicators)
            if np.any(mask):
                # Use first available default or constant value
                default_val = (
                    list(self.default_values.values())[0]
                    if self.default_values
                    else list(self.learned_defaults.values())[0]
                    if self.learned_defaults
                    else self.constant_value
                )
                result[mask] = default_val

        return result

    def _get_default_value(self, key: str) -> Any:
        """Get default value for a specific key/column"""
        return (
            self.default_values.get(key)
            or self.learned_defaults.get(key)
            or self.constant_value
        )

    def add_missing_indicator(self, indicator: Any) -> None:
        """Add a new missing indicator"""
        if indicator not in self.missing_indicators:
            self.missing_indicators.append(indicator)
            logger.info(f"Added missing indicator: {indicator}")

    def remove_missing_indicator(self, indicator: Any) -> None:
        """Remove a missing indicator"""
        if indicator in self.missing_indicators:
            self.missing_indicators.remove(indicator)
            logger.info(f"Removed missing indicator: {indicator}")

    def get_missing_statistics(
        self, data: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """Get statistics about missing values in the data"""
        stats = {}

        if isinstance(data, pd.DataFrame):
            for col in data.select_dtypes(include=["object"]).columns:
                missing_count = data[col].isin(self.missing_indicators).sum()
                total_count = len(data[col])
                stats[col] = {
                    "missing_count": missing_count,
                    "total_count": total_count,
                    "missing_percentage": (missing_count / total_count) * 100
                    if total_count > 0
                    else 0,
                }
        elif isinstance(data, dict):
            for key, values in data.items():
                if isinstance(values, list):
                    missing_count = sum(
                        1 for v in values if v in self.missing_indicators
                    )
                    total_count = len(values)
                    stats[key] = {
                        "missing_count": missing_count,
                        "total_count": total_count,
                        "missing_percentage": (missing_count / total_count) * 100
                        if total_count > 0
                        else 0,
                    }
        elif isinstance(data, np.ndarray):
            if data.dtype == object:
                missing_count = np.isin(data, self.missing_indicators).sum()
                total_count = len(data)
                stats["array"] = {
                    "missing_count": missing_count,
                    "total_count": total_count,
                    "missing_percentage": (missing_count / total_count) * 100
                    if total_count > 0
                    else 0,
                }

        return stats

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "default_values": self.default_values,
            "missing_indicators": self.missing_indicators,
            "strategy": self.strategy,
            "constant_value": self.constant_value,
            "learned_defaults": self.learned_defaults,
        }

    def __repr__(self) -> str:
        return (
            f"CategoricalImputationProcessor(strategy='{self.strategy}', "
            f"n_missing_indicators={len(self.missing_indicators)}, "
            f"constant_value='{self.constant_value}')"
        )
