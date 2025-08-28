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
from .base.processors import (
    Processor,
    ComposedProcessor,
    IdentityProcessor
)

# Import specific processors
from .tabular.categorical_label_processor import CategoricalLabelProcessor
from .tabular.multiclass_label_processor import MultiClassLabelProcessor
from .tabular.numerical_imputation_processor import NumericalVariableImputationProcessor
from .tabular.numerical_binning_processor import NumericalBinningProcessor

# Import text/NLP processors (with optional dependency handling)
try:
    from .text.bert_tokenize_processor import TokenizationProcessor as BertTokenizeProcessor
except ImportError:
    BertTokenizeProcessor = None

try:
    from .text.gensim_tokenize_processor import FastTextEmbeddingProcessor as GensimTokenizeProcessor
except ImportError:
    GensimTokenizeProcessor = None

# Import domain-specific processors (with optional dependency handling)
try:
    from .text.bsm_processor import (
        TextNormalizationProcessor,
        DialogueSplitterProcessor,
        DialogueChunkerProcessor,
        EmojiRemoverProcessor,
        HTMLNormalizerProcessor
    )
    BSMProcessor = TextNormalizationProcessor  # Use one as representative
except ImportError:
    BSMProcessor = None

try:
    from .text.cs_processor import CSChatSplitterProcessor as CSProcessor
except ImportError:
    CSProcessor = None

try:
    from .tabular.risk_table_processor import RiskTableMappingProcessor as RiskTableProcessor
except ImportError:
    RiskTableProcessor = None

# Import data loading utilities (with optional dependency handling)
try:
    from .dataset.bsm_dataloader import BSMDataLoader
except ImportError:
    BSMDataLoader = None

try:
    from .dataset.bsm_datasets import BSMDataset as BSMDatasets
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
