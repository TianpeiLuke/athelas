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

# Import with optional dependency handling
try:
    from .risk_table_processor import RiskTableProcessor
except ImportError:
    RiskTableProcessor = None

__all__ = [
    "CategoricalLabelProcessor",
    "MultiClassLabelProcessor",
    "DictionaryEncodingProcessor",
    "CategoricalImputationProcessor",
    "NumericalCategoricalProcessor",
    "CategoricalValidationProcessor",
]

# Add optional processors to __all__ if they're available
if RiskTableProcessor is not None:
    __all__.append("RiskTableProcessor")
