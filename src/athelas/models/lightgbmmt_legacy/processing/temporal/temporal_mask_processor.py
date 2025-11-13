"""
Temporal Mask Processor for Temporal Self-Attention Model

This module provides atomic attention mask generation for temporal sequences.
Derived from TSA attention masking requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class TemporalMaskProcessor(Processor):
    """
    Generates attention masks for padded sequences.

    Derived from TSA attention masking requirements.

    Args:
        mask_value: Value indicating valid positions
        padding_value: Value indicating padded positions
        output_format: 'boolean', 'float', 'int'
    """

    def __init__(
        self,
        mask_value: Union[int, float, bool] = True,
        padding_value: Union[int, float] = 0,
        output_format: str = "boolean",
    ):
        super().__init__()
        self.mask_value = mask_value
        self.padding_value = padding_value
        self.output_format = output_format
        self.is_fitted = False

        if output_format not in ["boolean", "float", "int"]:
            raise ValueError(
                f"output_format must be one of ['boolean', 'float', 'int'], got {output_format}"
            )

    def fit(self, data: Any) -> "TemporalMaskProcessor":
        """No fitting required for masking"""
        self.is_fitted = True
        logger.info(
            f"TemporalMaskProcessor fitted with output_format: {self.output_format}"
        )
        return self

    def process(
        self, input_data: Union[np.ndarray, List, pd.DataFrame]
    ) -> Union[np.ndarray, List]:
        """Generate attention mask"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, np.ndarray):
            return self._process_numpy_array(input_data)
        elif isinstance(input_data, list):
            return self._process_list(input_data)
        elif isinstance(input_data, pd.DataFrame):
            return self._process_dataframe(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_numpy_array(self, input_data: np.ndarray) -> np.ndarray:
        """Process numpy array input"""
        # Create mask based on non-padding values
        mask = input_data != self.padding_value

        # Handle multi-dimensional arrays (use any non-padding value in row)
        if mask.ndim > 1:
            mask = np.any(mask, axis=1)

        # Convert to requested format
        return self._convert_mask_format(mask)

    def _process_list(self, input_data: List) -> List:
        """Process list input"""
        # Create mask based on non-padding values
        mask = [item != self.padding_value for item in input_data]

        # Convert to requested format
        if self.output_format == "boolean":
            return mask
        elif self.output_format == "float":
            return [float(m) for m in mask]
        elif self.output_format == "int":
            return [int(m) for m in mask]

    def _process_dataframe(self, input_data: pd.DataFrame) -> np.ndarray:
        """Process DataFrame input"""
        # Create mask based on non-padding values across all columns
        mask = (input_data != self.padding_value).any(axis=1).values

        # Convert to requested format
        return self._convert_mask_format(mask)

    def _convert_mask_format(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to requested format"""
        if self.output_format == "boolean":
            return mask.astype(bool)
        elif self.output_format == "float":
            return mask.astype(float)
        elif self.output_format == "int":
            return mask.astype(int)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def create_causal_mask(self, sequence_length: int) -> np.ndarray:
        """
        Create a causal (lower triangular) attention mask.

        Args:
            sequence_length: Length of the sequence

        Returns:
            Causal attention mask
        """
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before creating causal mask")

        # Create lower triangular matrix
        mask = np.tril(np.ones((sequence_length, sequence_length)))

        # Convert to requested format
        return self._convert_mask_format(mask)

    def create_padding_mask(
        self, sequence_lengths: List[int], max_length: int
    ) -> np.ndarray:
        """
        Create padding masks for batch of sequences with different lengths.

        Args:
            sequence_lengths: List of actual sequence lengths
            max_length: Maximum sequence length (padded length)

        Returns:
            Batch of padding masks
        """
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before creating padding mask")

        batch_size = len(sequence_lengths)
        masks = np.zeros((batch_size, max_length), dtype=bool)

        for i, length in enumerate(sequence_lengths):
            masks[i, :length] = True

        # Convert to requested format
        return self._convert_mask_format(masks)

    def combine_masks(self, *masks: np.ndarray) -> np.ndarray:
        """
        Combine multiple masks using logical AND.

        Args:
            *masks: Variable number of mask arrays

        Returns:
            Combined mask
        """
        if not masks:
            raise ValueError("At least one mask must be provided")

        combined = masks[0].astype(bool)
        for mask in masks[1:]:
            combined = combined & mask.astype(bool)

        # Convert to requested format
        return self._convert_mask_format(combined)

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "mask_value": self.mask_value,
            "padding_value": self.padding_value,
            "output_format": self.output_format,
        }

    def __repr__(self) -> str:
        return (
            f"TemporalMaskProcessor(mask_value={self.mask_value}, "
            f"padding_value={self.padding_value}, output_format='{self.output_format}')"
        )
