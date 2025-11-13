"""
Time Delta Processor for Temporal Self-Attention Model

This module provides atomic time delta computation for temporal sequences.
Extracted from TSA preprocess_functions.py logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class TimeDeltaProcessor(Processor):
    """
    Computes time deltas relative to a reference point.

    Extracted from TSA preprocess_functions.py:
    - seq_num_mtx[:, -2] = seq_num_mtx[-1, -2] - seq_num_mtx[:, -2]

    Args:
        reference_strategy: 'most_recent', 'first', 'custom'
        reference_field: Field name containing reference timestamp
        output_field: Field name for computed deltas
        time_unit: 'seconds', 'minutes', 'hours', 'days'
        max_delta: Maximum allowed delta (for outlier handling)
    """

    def __init__(
        self,
        reference_strategy: str = "most_recent",
        reference_field: str = "orderDate",
        output_field: str = "time_delta",
        time_unit: str = "seconds",
        max_delta: Optional[float] = 10000000,
    ):
        super().__init__()
        self.reference_strategy = reference_strategy
        self.reference_field = reference_field
        self.output_field = output_field
        self.time_unit = time_unit
        self.max_delta = max_delta
        self.reference_time = None
        self.is_fitted = False

    def fit(self, data: Union[Dict, List, np.ndarray]) -> "TimeDeltaProcessor":
        """Learn reference time from data"""
        if self.reference_strategy == "most_recent":
            if isinstance(data, dict):
                timestamps = data[self.reference_field]
                self.reference_time = (
                    max(timestamps) if isinstance(timestamps, list) else timestamps
                )
            elif isinstance(data, np.ndarray):
                self.reference_time = data[-1, -1]  # Assume last row, last column
            elif isinstance(data, pd.DataFrame):
                self.reference_time = data[self.reference_field].max()
        elif self.reference_strategy == "first":
            if isinstance(data, dict):
                timestamps = data[self.reference_field]
                self.reference_time = (
                    min(timestamps) if isinstance(timestamps, list) else timestamps
                )
            elif isinstance(data, np.ndarray):
                self.reference_time = data[0, -1]  # Assume first row, last column
            elif isinstance(data, pd.DataFrame):
                self.reference_time = data[self.reference_field].min()

        self.is_fitted = True
        logger.info(
            f"TimeDeltaProcessor fitted with reference_time: {self.reference_time}"
        )
        return self

    def process(
        self, input_data: Union[Dict, np.ndarray, pd.DataFrame]
    ) -> Union[Dict, np.ndarray, pd.DataFrame]:
        """Compute time deltas"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, dict):
            timestamps = input_data[self.reference_field]
            if isinstance(timestamps, list):
                deltas = [self.reference_time - t for t in timestamps]
            else:
                deltas = self.reference_time - timestamps

            # Apply max_delta constraint
            if self.max_delta:
                if isinstance(deltas, list):
                    deltas = [min(d, self.max_delta) for d in deltas]
                else:
                    deltas = min(deltas, self.max_delta)

            result = input_data.copy()
            result[self.output_field] = deltas
            return result

        elif isinstance(input_data, np.ndarray):
            # Handle numpy array case (TSA-specific)
            result = input_data.copy()
            result[:, -2] = self.reference_time - result[:, -2]

            # Apply max_delta constraint
            if self.max_delta:
                result[:, -2] = np.minimum(result[:, -2], self.max_delta)

            return result

        elif isinstance(input_data, pd.DataFrame):
            result = input_data.copy()
            result[self.output_field] = (
                self.reference_time - result[self.reference_field]
            )

            # Apply max_delta constraint
            if self.max_delta:
                result[self.output_field] = result[self.output_field].clip(
                    upper=self.max_delta
                )

            return result

        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "reference_strategy": self.reference_strategy,
            "reference_field": self.reference_field,
            "output_field": self.output_field,
            "time_unit": self.time_unit,
            "max_delta": self.max_delta,
            "reference_time": self.reference_time,
        }

    def __repr__(self) -> str:
        return (
            f"TimeDeltaProcessor(reference_strategy='{self.reference_strategy}', "
            f"reference_field='{self.reference_field}', max_delta={self.max_delta})"
        )
