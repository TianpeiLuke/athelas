"""
Field type validation utilities for preprocessing pipelines.

Provides strict validation functions that validate field types before
applying numerical imputation or categorical risk table mapping.
"""

import pandas as pd
from typing import List


def validate_categorical_fields(
    df: pd.DataFrame, cat_fields: List[str], dataset_name: str = "dataset"
) -> None:
    """
    Strictly validate categorical fields before risk table mapping.

    Args:
        df: Input dataframe
        cat_fields: List of categorical field names from config
        dataset_name: Name of dataset for error messages (e.g., "train", "val", "test")

    Raises:
        ValueError: If field not found in dataframe
        TypeError: If field has wrong type with specific field names
    """
    mismatched_fields = []

    for field in cat_fields:
        if field not in df.columns:
            raise ValueError(
                f"Categorical field '{field}' not found in {dataset_name} dataframe"
            )

        dtype = df[field].dtype
        # Allow: object, category, string types
        if dtype not in [
            "object",
            "category",
            "string",
        ] and not pd.api.types.is_string_dtype(df[field]):
            mismatched_fields.append(
                {
                    "field": field,
                    "current_type": str(dtype),
                    "expected_type": "categorical (object/string/category)",
                }
            )

    if mismatched_fields:
        error_msg = f"Categorical field type validation failed for {dataset_name}:\n"
        for info in mismatched_fields:
            error_msg += (
                f"  - Field '{info['field']}': "
                f"expected {info['expected_type']}, "
                f"but got {info['current_type']}\n"
            )
        error_msg += "\nCategorical fields must have object, string, or category dtype before risk table mapping."
        raise TypeError(error_msg)


def validate_numerical_fields(
    df: pd.DataFrame, num_fields: List[str], dataset_name: str = "dataset"
) -> None:
    """
    Strictly validate numerical fields before numerical imputation.

    Args:
        df: Input dataframe
        num_fields: List of numerical field names from config
        dataset_name: Name of dataset for error messages (e.g., "train", "val", "test")

    Raises:
        ValueError: If field not found in dataframe
        TypeError: If field has wrong type with specific field names
    """
    mismatched_fields = []

    for field in num_fields:
        if field not in df.columns:
            raise ValueError(
                f"Numerical field '{field}' not found in {dataset_name} dataframe"
            )

        dtype = df[field].dtype
        # Must be: int or float types
        if not pd.api.types.is_numeric_dtype(df[field]):
            mismatched_fields.append(
                {
                    "field": field,
                    "current_type": str(dtype),
                    "expected_type": "numerical (int/float)",
                }
            )

    if mismatched_fields:
        error_msg = f"Numerical field type validation failed for {dataset_name}:\n"
        for info in mismatched_fields:
            error_msg += (
                f"  - Field '{info['field']}': "
                f"expected {info['expected_type']}, "
                f"but got {info['current_type']}\n"
            )
        error_msg += "\nNumerical fields must have int or float dtype before numerical imputation."
        raise TypeError(error_msg)
