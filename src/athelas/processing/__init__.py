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
from .processors import (
    Processor,
    ComposedProcessor,
    IdentityProcessor
)

# Import specific processors
from .categorical_label_processor import CategoricalLabelProcessor
from .multiclass_label_processor import MultiClassLabelProcessor
from .numerical_imputation_processor import NumericalVariableImputationProcessor
from .numerical_binning_processor import NumericalBinningProcessor

# Import text/NLP processors (with optional dependency handling)
try:
    from .bert_tokenize_processor import BertTokenizeProcessor
except ImportError:
    BertTokenizeProcessor = None

try:
    from .gensim_tokenize_processor import GensimTokenizeProcessor
except ImportError:
    GensimTokenizeProcessor = None

# Import domain-specific processors (with optional dependency handling)
try:
    from .bsm_processor import BSMProcessor
except ImportError:
    BSMProcessor = None

try:
    from .cs_processor import CSProcessor
except ImportError:
    CSProcessor = None

try:
    from .risk_table_processor import RiskTableProcessor
except ImportError:
    RiskTableProcessor = None

# Import data loading utilities (with optional dependency handling)
try:
    from .bsm_dataloader import BSMDataLoader
except ImportError:
    BSMDataLoader = None

try:
    from .bsm_datasets import BSMDatasets
except ImportError:
    BSMDatasets = None

# Export all available processors
__all__ = [
    # Base classes
    'Processor',
    'ComposedProcessor', 
    'IdentityProcessor',
    
    # Core processors
    'CategoricalLabelProcessor',
    'MultiClassLabelProcessor',
    'NumericalVariableImputationProcessor',
    'NumericalBinningProcessor',
]

# Add optional processors to __all__ if they're available
_optional_processors = [
    ('BertTokenizeProcessor', BertTokenizeProcessor),
    ('GensimTokenizeProcessor', GensimTokenizeProcessor),
    ('BSMProcessor', BSMProcessor),
    ('CSProcessor', CSProcessor),
    ('RiskTableProcessor', RiskTableProcessor),
    ('BSMDataLoader', BSMDataLoader),
    ('BSMDatasets', BSMDatasets),
]

for name, processor_class in _optional_processors:
    if processor_class is not None:
        __all__.append(name)
