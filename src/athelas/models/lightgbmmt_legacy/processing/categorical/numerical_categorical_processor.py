"""
Numerical Categorical Processor for Converting Numbers to Categories

This module provides atomic conversion of numerical values to categorical labels.
Extracted from TSA numerical categorization requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class NumericalCategoricalProcessor(Processor):
    """
    Converts numerical values to categorical labels using binning or thresholds.

    Extracted from TSA str(int(float(cur_var))) conversion patterns.

    Args:
        binning_strategy: 'equal_width', 'equal_frequency', 'custom', 'threshold'
        n_bins: Number of bins for equal_width/equal_frequency
        bin_edges: Custom bin edges for 'custom' strategy
        thresholds: Threshold values for 'threshold' strategy
        labels: Custom labels for categories
        columns: Specific columns to process
    """

    def __init__(
        self,
        binning_strategy: str = "equal_width",
        n_bins: int = 5,
        bin_edges: Optional[List[float]] = None,
        thresholds: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.thresholds = thresholds
        self.labels = labels
        self.columns = columns
        self.learned_bins = {}
        self.is_fitted = False

        if binning_strategy not in [
            "equal_width",
            "equal_frequency",
            "custom",
            "threshold",
        ]:
            raise ValueError(
                f"binning_strategy must be one of ['equal_width', 'equal_frequency', 'custom', 'threshold'], got {binning_strategy}"
            )

    def fit(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> "NumericalCategoricalProcessor":
        """Learn binning parameters from data"""
        if self.binning_strategy in ["equal_width", "equal_frequency"]:
            if isinstance(data, np.ndarray):
                self._fit_numpy_array(data)
            elif isinstance(data, pd.DataFrame):
                self._fit_dataframe(data)
            else:
                raise ValueError(f"Unsupported data type for fitting: {type(data)}")

        self.is_fitted = True
        logger.info(
            f"NumericalCategoricalProcessor fitted with strategy: {self.binning_strategy}"
        )
        return self

    def _fit_numpy_array(self, data: np.ndarray) -> None:
        """Fit binning parameters for numpy array"""
        if self.binning_strategy == "equal_width":
            data_min = np.min(data)
            data_max = np.max(data)
            self.learned_bins["edges"] = np.linspace(
                data_min, data_max, self.n_bins + 1
            )
        elif self.binning_strategy == "equal_frequency":
            self.learned_bins["edges"] = np.percentile(
                data, np.linspace(0, 100, self.n_bins + 1)
            )

    def _fit_dataframe(self, data: pd.DataFrame) -> None:
        """Fit binning parameters for DataFrame"""
        columns = self.columns or data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in DataFrame, skipping")
                continue

            col_data = data[col].dropna().values

            if self.binning_strategy == "equal_width":
                data_min = np.min(col_data)
                data_max = np.max(col_data)
                edges = np.linspace(data_min, data_max, self.n_bins + 1)
            elif self.binning_strategy == "equal_frequency":
                edges = np.percentile(col_data, np.linspace(0, 100, self.n_bins + 1))

            self.learned_bins[col] = {"edges": edges}

    def process(
        self, input_data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Apply numerical to categorical conversion"""
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
        result = input_data.copy()

        if self.binning_strategy == "custom" and self.bin_edges:
            edges = self.bin_edges
        elif self.binning_strategy == "threshold" and self.thresholds:
            # Convert thresholds to bin edges
            edges = [-np.inf] + self.thresholds + [np.inf]
        elif "edges" in self.learned_bins:
            edges = self.learned_bins["edges"]
        else:
            raise ValueError("No binning edges available")

        # Apply binning
        if input_data.ndim == 1:
            result = np.digitize(input_data, edges) - 1
        else:
            for i in range(input_data.shape[1]):
                result[:, i] = np.digitize(input_data[:, i], edges) - 1

        # Apply custom labels if provided
        if self.labels:
            if input_data.ndim == 1:
                result = np.array(
                    [self.labels[min(idx, len(self.labels) - 1)] for idx in result]
                )
            else:
                for i in range(result.shape[1]):
                    result[:, i] = [
                        self.labels[min(int(idx), len(self.labels) - 1)]
                        for idx in result[:, i]
                    ]

        return result

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        result = input_data.copy()

        for col in self.learned_bins.keys():
            if col in result.columns:
                edges = self.learned_bins[col]["edges"]

                # Apply binning
                result[col] = pd.cut(
                    result[col], bins=edges, labels=False, include_lowest=True
                )

                # Apply custom labels if provided
                if self.labels:
                    result[col] = result[col].map(
                        lambda x: self.labels[min(int(x), len(self.labels) - 1)]
                        if pd.notna(x)
                        else x
                    )

        return result

    def get_bin_info(
        self, column: Optional[str] = None
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get binning information"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before getting bin info")

        if column is not None:
            if column in self.learned_bins:
                return self.learned_bins[column]
            else:
                raise KeyError(f"Column {column} not found in learned bins")
        else:
            return self.learned_bins

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "binning_strategy": self.binning_strategy,
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges,
            "thresholds": self.thresholds,
            "labels": self.labels,
            "columns": self.columns,
            "learned_bins": self.learned_bins,
        }

    def __repr__(self) -> str:
        return (
            f"NumericalCategoricalProcessor(strategy='{self.binning_strategy}', "
            f"n_bins={self.n_bins}, n_learned_bins={len(self.learned_bins)})"
        )
