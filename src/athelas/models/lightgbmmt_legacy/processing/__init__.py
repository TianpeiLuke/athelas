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

# Import specific processors
from .categorical.categorical_label_processor import CategoricalLabelProcessor
from .categorical.multiclass_label_processor import MultiClassLabelProcessor
from .numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)
from .numerical.numerical_binning_processor import NumericalBinningProcessor

# Import atomic processors
from .temporal.time_delta_processor import TimeDeltaProcessor
from .temporal.sequence_padding_processor import SequencePaddingProcessor
from .temporal.sequence_ordering_processor import SequenceOrderingProcessor
from .temporal.temporal_mask_processor import TemporalMaskProcessor
from .categorical.dictionary_encoding_processor import DictionaryEncodingProcessor
from .categorical.categorical_imputation_processor import CategoricalImputationProcessor
from .categorical.numerical_categorical_processor import NumericalCategoricalProcessor
from .categorical.categorical_validation_processor import CategoricalValidationProcessor
from .numerical.minmax_scaling_processor import MinMaxScalingProcessor
from .numerical.feature_normalization_processor import FeatureNormalizationProcessor

# Import text/NLP processors (with optional dependency handling)
try:
    from .text.bert_tokenize_processor import BertTokenizeProcessor
except ImportError:
    BertTokenizeProcessor = None

try:
    from .text.gensim_tokenize_processor import GensimTokenizeProcessor
except ImportError:
    GensimTokenizeProcessor = None

# Import domain-specific text processors (with optional dependency handling)
try:
    from .text.dialogue_processor import (
        TextNormalizationProcessor,
        TextUpperProcessor,
        DialogueSplitterProcessor,
        DialogueChunkerProcessor,
        EmojiRemoverProcessor,
        HTMLNormalizerProcessor,
    )
except ImportError:
    TextNormalizationProcessor = None
    TextUpperProcessor = None
    DialogueSplitterProcessor = None
    DialogueChunkerProcessor = None
    EmojiRemoverProcessor = None
    HTMLNormalizerProcessor = None

try:
    from .text.cs_format_processor import (
        CSChatSplitterProcessor,
        CSAdapter,
    )
except ImportError:
    CSChatSplitterProcessor = None
    CSAdapter = None

try:
    from .categorical.risk_table_processor import RiskTableMappingProcessor
except ImportError:
    RiskTableMappingProcessor = None

# Import data loading utilities (with optional dependency handling)
try:
    from .dataloaders.bsm_dataloader import build_collate_batch
except ImportError:
    build_collate_batch = None

try:
    from .datasets.bsm_datasets import BSMDataset
except ImportError:
    BSMDataset = None

# Export all available processors
__all__ = [
    # Base classes
    "Processor",
    "ComposedProcessor",
    "IdentityProcessor",
    # Core processors
    "CategoricalLabelProcessor",
    "MultiClassLabelProcessor",
    "NumericalVariableImputationProcessor",
    "NumericalBinningProcessor",
    # Atomic processors
    "TimeDeltaProcessor",
    "SequencePaddingProcessor",
    "SequenceOrderingProcessor",
    "TemporalMaskProcessor",
    "DictionaryEncodingProcessor",
    "CategoricalImputationProcessor",
    "NumericalCategoricalProcessor",
    "CategoricalValidationProcessor",
    "MinMaxScalingProcessor",
    "FeatureNormalizationProcessor",
]

# Add optional processors to __all__ if they're available
_optional_processors = [
    ("BertTokenizeProcessor", BertTokenizeProcessor),
    ("GensimTokenizeProcessor", GensimTokenizeProcessor),
    ("TextNormalizationProcessor", TextNormalizationProcessor),
    ("TextUpperProcessor", TextUpperProcessor),
    ("DialogueSplitterProcessor", DialogueSplitterProcessor),
    ("DialogueChunkerProcessor", DialogueChunkerProcessor),
    ("EmojiRemoverProcessor", EmojiRemoverProcessor),
    ("HTMLNormalizerProcessor", HTMLNormalizerProcessor),
    ("CSChatSplitterProcessor", CSChatSplitterProcessor),
    ("CSAdapter", CSAdapter),
    ("RiskTableMappingProcessor", RiskTableMappingProcessor),
    ("build_collate_batch", build_collate_batch),
    ("BSMDataset", BSMDataset),
]

for name, processor_class in _optional_processors:
    if processor_class is not None:
        __all__.append(name)
