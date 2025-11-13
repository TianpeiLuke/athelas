"""
Sequence Padding Processor for Temporal Self-Attention Model

This module provides atomic sequence padding/truncation for temporal sequences.
Extracted from TSA preprocess_functions.py logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class SequencePaddingProcessor(Processor):
    """
    Pads or truncates sequences to a target length.

    Extracted from TSA preprocess_functions.py:
    - seq_cat_mtx = np.pad(seq_cat_mtx, [(seq_len - 1 - len(seq_cat_vars_mtx), 0), (0, 0)])

    Args:
        target_length: Desired sequence length
        padding_strategy: 'pre', 'post'
        truncation_strategy: 'pre', 'post'
        padding_value: Value to use for padding
        axis: Axis along which to pad/truncate
    """

    def __init__(
        self,
        target_length: int = 51,
        padding_strategy: str = "pre",
        truncation_strategy: str = "post",
        padding_value: Union[int, float] = 0,
        axis: int = 0,
    ):
        super().__init__()
        self.target_length = target_length
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.padding_value = padding_value
        self.axis = axis
        self.is_fitted = False

    def fit(self, data: Any) -> "SequencePaddingProcessor":
        """No fitting required for padding"""
        self.is_fitted = True
        logger.info(
            f"SequencePaddingProcessor fitted with target_length: {self.target_length}"
        )
        return self

    def process(self, input_data: Union[np.ndarray, List]) -> Union[np.ndarray, List]:
        """Apply sequence padding/truncation"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, np.ndarray):
            return self._process_numpy_array(input_data)
        elif isinstance(input_data, list):
            return self._process_list(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        current_length = input_data.shape[self.axis]

        if current_length == self.target_length:
            return input_data
        elif current_length < self.target_length:
            # Padding required
            pad_width = [(0, 0)] * input_data.ndim
            pad_amount = self.target_length - current_length

            if self.padding_strategy == "pre":
                pad_width[self.axis] = (pad_amount, 0)
            else:  # post
                pad_width[self.axis] = (0, pad_amount)

            return np.pad(input_data, pad_width, constant_values=self.padding_value)
        else:
            # Truncation required
            if self.truncation_strategy == "pre":
                # Keep last target_length elements
                slices = [slice(None)] * input_data.ndim
                slices[self.axis] = slice(-self.target_length, None)
                return input_data[tuple(slices)]
            else:  # post
                # Keep first target_length elements
                slices = [slice(None)] * input_data.ndim
                slices[self.axis] = slice(self.target_length)
                return input_data[tuple(slices)]

    def _process_list(self, input_data: List) -> List:
        """Process list input"""
        current_length = len(input_data)

        if current_length == self.target_length:
            return input_data
        elif current_length < self.target_length:
            # Padding required
            pad_amount = self.target_length - current_length
            padding = [self.padding_value] * pad_amount

            if self.padding_strategy == "pre":
                return padding + input_data
            else:  # post
                return input_data + padding
        else:
            # Truncation required
            if self.truncation_strategy == "pre":
                # Keep last target_length elements
                return input_data[-self.target_length :]
            else:  # post
                # Keep first target_length elements
                return input_data[: self.target_length]

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "target_length": self.target_length,
            "padding_strategy": self.padding_strategy,
            "truncation_strategy": self.truncation_strategy,
            "padding_value": self.padding_value,
            "axis": self.axis,
        }

    def __repr__(self) -> str:
        return (
            f"SequencePaddingProcessor(target_length={self.target_length}, "
            f"padding_strategy='{self.padding_strategy}', "
            f"truncation_strategy='{self.truncation_strategy}')"
        )
