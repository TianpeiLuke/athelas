"""
Dictionary Encoding Processor for Categorical Features

This module provides atomic dictionary-based categorical encoding.
Extracted from TSA CategoricalTransformer functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class DictionaryEncodingProcessor(Processor):
    """
    Pure dictionary-based categorical encoding.

    Extracted from TSA CategoricalTransformer functionality.

    Args:
        categorical_map: Pre-defined category mappings
        unknown_strategy: 'error', 'default', 'ignore'
        default_value: Value for unknown categories
        columns: Specific columns to encode
    """

    def __init__(
        self,
        categorical_map: Optional[Dict[str, Dict[str, int]]] = None,
        unknown_strategy: str = "default",
        default_value: int = 0,
        columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.categorical_map = categorical_map or {}
        self.unknown_strategy = unknown_strategy
        self.default_value = default_value
        self.columns = columns
        self.is_fitted = False

        if unknown_strategy not in ["error", "default", "ignore"]:
            raise ValueError(
                f"unknown_strategy must be one of ['error', 'default', 'ignore'], got {unknown_strategy}"
            )

    def fit(
        self, data: Union[Dict, np.ndarray, pd.DataFrame]
    ) -> "DictionaryEncodingProcessor":
        """Learn categorical mappings from data if not provided"""
        if not self.categorical_map:
            if isinstance(data, pd.DataFrame):
                columns = self.columns or data.select_dtypes(include=["object"]).columns
                for col in columns:
                    unique_values = data[col].unique()
                    # Filter out NaN values
                    unique_values = [v for v in unique_values if pd.notna(v)]
                    self.categorical_map[col] = {
                        val: idx for idx, val in enumerate(unique_values)
                    }
            elif isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list) and all(
                        isinstance(v, str) for v in values if pd.notna(v)
                    ):
                        unique_values = list(set(v for v in values if pd.notna(v)))
                        self.categorical_map[key] = {
                            val: idx for idx, val in enumerate(unique_values)
                        }

        self.is_fitted = True
        logger.info(
            f"DictionaryEncodingProcessor fitted with {len(self.categorical_map)} categorical mappings"
        )
        return self

    def process(
        self, input_data: Union[Dict, np.ndarray, pd.DataFrame]
    ) -> Union[Dict, np.ndarray, pd.DataFrame]:
        """Apply dictionary encoding"""
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
        for col, mapping in self.categorical_map.items():
            if col in result.columns:
                if self.unknown_strategy == "default":
                    result[col] = result[col].map(mapping).fillna(self.default_value)
                elif self.unknown_strategy == "error":
                    unknown_values = set(result[col].dropna().unique()) - set(
                        mapping.keys()
                    )
                    if unknown_values:
                        raise ValueError(
                            f"Unknown categories found in column {col}: {unknown_values}"
                        )
                    result[col] = result[col].map(mapping)
                else:  # ignore
                    result[col] = result[col].map(mapping)
        return result

    def _process_dict(self, input_data: Dict) -> Dict:
        """Process dictionary input"""
        result = {}
        for key, values in input_data.items():
            if key in self.categorical_map:
                mapping = self.categorical_map[key]
                if isinstance(values, list):
                    if self.unknown_strategy == "default":
                        result[key] = [
                            mapping.get(v, self.default_value) for v in values
                        ]
                    elif self.unknown_strategy == "error":
                        unknown_values = set(v for v in values if pd.notna(v)) - set(
                            mapping.keys()
                        )
                        if unknown_values:
                            raise ValueError(
                                f"Unknown categories found in key {key}: {unknown_values}"
                            )
                        result[key] = [mapping.get(v) for v in values]
                    else:  # ignore
                        result[key] = [mapping.get(v) for v in values]
                else:
                    if self.unknown_strategy == "default":
                        result[key] = mapping.get(values, self.default_value)
                    elif self.unknown_strategy == "error":
                        if values not in mapping:
                            raise ValueError(
                                f"Unknown category found in key {key}: {values}"
                            )
                        result[key] = mapping[values]
                    else:  # ignore
                        result[key] = mapping.get(values)
            else:
                result[key] = values
        return result

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        # Handle string arrays
        result = input_data.copy()
        if input_data.dtype.kind in ["U", "S", "O"]:  # Unicode, byte string, or object
            # Apply first available mapping (assumes single column)
            if self.categorical_map:
                mapping = list(self.categorical_map.values())[0]
                if self.unknown_strategy == "default":
                    vectorized_map = np.vectorize(
                        lambda x: mapping.get(x, self.default_value)
                    )
                elif self.unknown_strategy == "error":
                    unknown_values = set(np.unique(input_data)) - set(mapping.keys())
                    if unknown_values:
                        raise ValueError(f"Unknown categories found: {unknown_values}")
                    vectorized_map = np.vectorize(mapping.get)
                else:  # ignore
                    vectorized_map = np.vectorize(mapping.get)
                result = vectorized_map(input_data)
        return result

    def inverse_transform(
        self, encoded_data: Union[Dict, np.ndarray, pd.DataFrame]
    ) -> Union[Dict, np.ndarray, pd.DataFrame]:
        """Inverse transform encoded data back to original categorical values"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before inverse transform")

        # Create inverse mappings
        inverse_maps = {}
        for column_name, mapping in self.categorical_map.items():
            inverse_maps[column_name] = {v: k for k, v in mapping.items()}

        if isinstance(encoded_data, pd.DataFrame):
            result = encoded_data.copy()
            for col, inverse_map in inverse_maps.items():
                if col in result.columns:
                    result[col] = result[col].map(inverse_map)
            return result
        elif isinstance(encoded_data, dict):
            result = {}
            for key, values in encoded_data.items():
                if key in inverse_maps:
                    inverse_map = inverse_maps[key]
                    if isinstance(values, list):
                        result[key] = [
                            inverse_map.get(v, f"UNKNOWN_{v}") for v in values
                        ]
                    else:
                        result[key] = inverse_map.get(values, f"UNKNOWN_{values}")
                else:
                    result[key] = values
            return result
        elif isinstance(encoded_data, np.ndarray):
            if inverse_maps:
                inverse_map = list(inverse_maps.values())[0]
                vectorized_inverse = np.vectorize(
                    lambda x: inverse_map.get(x, f"UNKNOWN_{x}")
                )
                return vectorized_inverse(encoded_data)
            return encoded_data
        else:
            raise ValueError(f"Unsupported input type: {type(encoded_data)}")

    def get_vocab_size(self, column_name: str) -> int:
        """Get vocabulary size for a specific column"""
        if column_name not in self.categorical_map:
            raise KeyError(f"Column {column_name} not found in categorical mappings")
        return (
            max(self.categorical_map[column_name].values()) + 1
            if self.categorical_map[column_name]
            else 1
        )

    def get_all_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all categorical columns"""
        return {col: self.get_vocab_size(col) for col in self.categorical_map.keys()}

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "categorical_map": self.categorical_map,
            "unknown_strategy": self.unknown_strategy,
            "default_value": self.default_value,
            "columns": self.columns,
        }

    def __repr__(self) -> str:
        return (
            f"DictionaryEncodingProcessor(n_mappings={len(self.categorical_map)}, "
            f"unknown_strategy='{self.unknown_strategy}', default_value={self.default_value})"
        )
