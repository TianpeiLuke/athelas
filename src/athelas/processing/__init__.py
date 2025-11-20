"""
Cursus Processing Module

This module provides access to various data processing utilities and processors
that can be used in preprocessing, inference, evaluation, and other ML pipeline steps.

The processors are organized by functionality:
- Base processor classes and composition utilities
- Text processing (tokenization, NLP)
- Numerical processing (imputation, binning)
- Categorical processing (label encoding)
- Domain-specific processors (BSM, risk tables, etc.)
"""

# Import base processor classes
from .processors import Processor, ComposedProcessor, IdentityProcessor


# Export all available processors
__all__ = [
    # Base classes
    "Processor",
    "ComposedProcessor",
    "IdentityProcessor",
]
