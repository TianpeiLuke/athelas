"""
Categorical Processing Module

This module provides atomic processors for categorical data processing,
including encoding, imputation, validation, and numerical categorization.
"""

from .categorical_label_processor import CategoricalLabelProcessor
from .multiclass_label_processor import MultiClassLabelProcessor
from .dictionary_encoding_processor import DictionaryEncodingProcessor
from .categorical_imputation_processor import CategoricalImputationProcessor
from .numerical_categorical_processor import NumericalCategoricalProcessor
from .categorical_validation_processor import CategoricalValidationProcessor
from .risk_table_processor import RiskTableMappingProcessor


__all__ = [
    "CategoricalLabelProcessor",
    "MultiClassLabelProcessor",
    "DictionaryEncodingProcessor",
    "CategoricalImputationProcessor",
    "NumericalCategoricalProcessor",
    "CategoricalValidationProcessor",
    "RiskTableMappingProcessor",
]
