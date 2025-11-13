"""
Sequence Ordering Processor for Temporal Self-Attention Model

This module provides atomic sequence ordering for temporal sequences.
Extracted from TSA preprocess_functions.py logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class SequenceOrderingProcessor(Processor):
    """
    Orders sequences by timestamp or other criteria.

    Extracted from TSA preprocess_functions.py sequence validation logic.

    Args:
        sort_field: Field to sort by
        sort_order: 'ascending', 'descending'
        validate_order: Whether to validate ordering consistency
    """

    def __init__(
        self,
        sort_field: str = "orderDate",
        sort_order: str = "ascending",
        validate_order: bool = True,
    ):
        super().__init__()
        self.sort_field = sort_field
        self.sort_order = sort_order
        self.validate_order = validate_order
        self.is_fitted = False

    def fit(self, data: Any) -> "SequenceOrderingProcessor":
        """No fitting required for ordering"""
        self.is_fitted = True
        logger.info(
            f"SequenceOrderingProcessor fitted with sort_field: {self.sort_field}"
        )
        return self

    def process(
        self, input_data: Union[Dict, np.ndarray, pd.DataFrame]
    ) -> Union[Dict, np.ndarray, pd.DataFrame]:
        """Apply sequence ordering"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, np.ndarray):
            return self._process_numpy_array(input_data)
        elif isinstance(input_data, pd.DataFrame):
            return self._process_dataframe(input_data)
        elif isinstance(input_data, dict):
            return self._process_dict(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        # Assume last column contains timestamps for TSA compatibility
        sort_indices = np.argsort(input_data[:, -1])
        if self.sort_order == "descending":
            sort_indices = sort_indices[::-1]

        result = input_data[sort_indices]

        # Validate ordering if requested
        if self.validate_order:
            timestamps = result[:, -1]
            if self.sort_order == "ascending":
                if not np.all(timestamps[:-1] <= timestamps[1:]):
                    logger.warning(
                        "Sequence ordering validation failed for ascending order"
                    )
            else:
                if not np.all(timestamps[:-1] >= timestamps[1:]):
                    logger.warning(
                        "Sequence ordering validation failed for descending order"
                    )

        return result

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        if self.sort_field not in input_data.columns:
            raise ValueError(
                f"Sort field '{self.sort_field}' not found in DataFrame columns"
            )

        ascending = self.sort_order == "ascending"
        result = input_data.sort_values(
            by=self.sort_field, ascending=ascending
        ).reset_index(drop=True)

        # Validate ordering if requested
        if self.validate_order:
            timestamps = result[self.sort_field].values
            if self.sort_order == "ascending":
                if not np.all(timestamps[:-1] <= timestamps[1:]):
                    logger.warning(
                        f"Sequence ordering validation failed for ascending order on field '{self.sort_field}'"
                    )
            else:
                if not np.all(timestamps[:-1] >= timestamps[1:]):
                    logger.warning(
                        f"Sequence ordering validation failed for descending order on field '{self.sort_field}'"
                    )

        return result

    def _process_dict(self, input_data: Dict) -> Dict:
        """Process dictionary input"""
        if self.sort_field not in input_data:
            raise ValueError(
                f"Sort field '{self.sort_field}' not found in input dictionary"
            )

        timestamps = input_data[self.sort_field]
        if not isinstance(timestamps, list):
            # Single value, no sorting needed
            return input_data

        # Create sort indices
        sort_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        if self.sort_order == "descending":
            sort_indices = sort_indices[::-1]

        # Apply sorting to all list values in the dictionary
        result = {}
        for key, values in input_data.items():
            if isinstance(values, list) and len(values) == len(timestamps):
                result[key] = [values[i] for i in sort_indices]
            else:
                result[key] = values

        # Validate ordering if requested
        if self.validate_order:
            sorted_timestamps = result[self.sort_field]
            if self.sort_order == "ascending":
                if not all(
                    sorted_timestamps[i] <= sorted_timestamps[i + 1]
                    for i in range(len(sorted_timestamps) - 1)
                ):
                    logger.warning(
                        f"Sequence ordering validation failed for ascending order on field '{self.sort_field}'"
                    )
            else:
                if not all(
                    sorted_timestamps[i] >= sorted_timestamps[i + 1]
                    for i in range(len(sorted_timestamps) - 1)
                ):
                    logger.warning(
                        f"Sequence ordering validation failed for descending order on field '{self.sort_field}'"
                    )

        return result

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "sort_field": self.sort_field,
            "sort_order": self.sort_order,
            "validate_order": self.validate_order,
        }

    def __repr__(self) -> str:
        return (
            f"SequenceOrderingProcessor(sort_field='{self.sort_field}', "
            f"sort_order='{self.sort_order}', validate_order={self.validate_order})"
        )
