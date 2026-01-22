"""
Processor Registry for Dynamic Pipeline Construction

Maps hyperparameter step names to processor classes for flexible
pipeline composition based on configuration.
"""

from typing import Dict, Type, Optional, Any

# Optional transformers import
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

# Optional gensim import
try:
    from gensim.models import KeyedVectors

    HAS_GENSIM = True
except ImportError:
    KeyedVectors = None
    HAS_GENSIM = False

# Optional custom tokenizer import (HuggingFace tokenizers library)
try:
    from tokenizers import Tokenizer as HFTokenizer

    HAS_CUSTOM_TOKENIZER = True
except ImportError:
    HFTokenizer = None
    HAS_CUSTOM_TOKENIZER = False

from .processors import Processor

# Text processors (always available)
from .text.dialogue_processor import (
    DialogueSplitterProcessor,
    HTMLNormalizerProcessor,
    EmojiRemoverProcessor,
    TextNormalizationProcessor,
    TextUpperProcessor,
)

from .text.cs_format_processor import (
    CSChatSplitterProcessor,
    CSAdapter,
)

# Temporal processors (always available)
from .temporal.sequence_ordering_processor import SequenceOrderingProcessor
from .temporal.sequence_padding_processor import SequencePaddingProcessor
from .temporal.temporal_mask_processor import TemporalMaskProcessor
from .temporal.time_delta_processor import TimeDeltaProcessor

# Optional imports that require transformers
if HAS_TRANSFORMERS:
    from .text.dialogue_processor import DialogueChunkerProcessor
    from .text.bert_tokenize_processor import BertTokenizeProcessor

# Optional imports that require gensim
if HAS_GENSIM:
    from .text.gensim_tokenize_processor import GensimTokenizeProcessor

# Optional imports that require custom tokenizer
if HAS_CUSTOM_TOKENIZER:
    from .text.custom_bpe_tokenize_processor import CustomBPETokenizeProcessor


# Registry mapping hyperparameter step names to processor classes
# Base processors (always available)
PROCESSOR_REGISTRY: Dict[str, Type[Processor]] = {
    # Text processors - dialogue
    "dialogue_splitter": DialogueSplitterProcessor,
    "html_normalizer": HTMLNormalizerProcessor,
    "emoji_remover": EmojiRemoverProcessor,
    "text_normalizer": TextNormalizationProcessor,
    "text_upper": TextUpperProcessor,
    # Text processors - CS format
    "cs_chat_splitter": CSChatSplitterProcessor,
    "cs_adapter": CSAdapter,
    # Temporal processors
    "sequence_ordering": SequenceOrderingProcessor,
    "sequence_padding": SequencePaddingProcessor,
    "temporal_mask": TemporalMaskProcessor,
    "time_delta": TimeDeltaProcessor,
}

# Add transformers-dependent processors if available
if HAS_TRANSFORMERS:
    PROCESSOR_REGISTRY["dialogue_chunker"] = DialogueChunkerProcessor
    PROCESSOR_REGISTRY["tokenizer"] = BertTokenizeProcessor

# Add gensim-dependent processors if available
if HAS_GENSIM:
    PROCESSOR_REGISTRY["fasttext_embedding"] = GensimTokenizeProcessor

# Add custom tokenizer-dependent processors if available
if HAS_CUSTOM_TOKENIZER:
    PROCESSOR_REGISTRY["custom_bpe_tokenizer"] = CustomBPETokenizeProcessor


def build_text_pipeline_from_steps(
    processing_steps: list[str],
    tokenizer: Optional[Any] = None,
    max_sen_len: int = 512,
    chunk_trancate: bool = False,
    max_total_chunks: int = 5,
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
) -> Processor:
    """
    Build a text processing pipeline from a list of step names.

    Args:
        processing_steps: List of processor names from hyperparameters
        tokenizer: HuggingFace tokenizer (optional, required for chunker/tokenizer steps)
        max_sen_len: Maximum sentence length for chunker and tokenizer
        chunk_trancate: Whether to truncate chunks
        max_total_chunks: Maximum number of chunks
        input_ids_key: Key for input IDs (for trimodal support)
        attention_mask_key: Key for attention mask (for trimodal support)

    Returns:
        Chained processor pipeline

    Raises:
        ValueError: If unknown processing step is encountered
        ImportError: If required library not available for specific steps
    """
    pipeline = None

    for step_name in processing_steps:
        # Create processor based on step name
        if step_name == "dialogue_splitter":
            processor = DialogueSplitterProcessor()

        elif step_name == "html_normalizer":
            processor = HTMLNormalizerProcessor()

        elif step_name == "emoji_remover":
            processor = EmojiRemoverProcessor()

        elif step_name == "text_normalizer":
            processor = TextNormalizationProcessor()

        elif step_name == "text_upper":
            processor = TextUpperProcessor()

        elif step_name == "cs_chat_splitter":
            processor = CSChatSplitterProcessor()

        elif step_name == "cs_adapter":
            processor = CSAdapter()

        elif step_name == "dialogue_chunker":
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "dialogue_chunker processor requires transformers library. "
                    "Install with: pip install transformers"
                )
            if tokenizer is None:
                raise ValueError(
                    "dialogue_chunker processor requires a tokenizer argument"
                )
            processor = DialogueChunkerProcessor(
                tokenizer=tokenizer,
                max_tokens=max_sen_len,
                truncate=chunk_trancate,
                max_total_chunks=max_total_chunks,
            )

        elif step_name == "tokenizer":
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "tokenizer processor requires transformers library. "
                    "Install with: pip install transformers"
                )
            if tokenizer is None:
                raise ValueError("tokenizer processor requires a tokenizer argument")
            processor = BertTokenizeProcessor(
                tokenizer,
                add_special_tokens=True,
                max_length=max_sen_len,
                truncation=True,  # Explicitly enable truncation
                padding="max_length",  # CRITICAL FIX: Force padding to max_length instead of "longest"
                input_ids_key=input_ids_key,
                attention_mask_key=attention_mask_key,
            )

        elif step_name == "custom_bpe_tokenizer":
            if not HAS_CUSTOM_TOKENIZER:
                raise ImportError(
                    "custom_bpe_tokenizer processor requires tokenizers library. "
                    "Install with: pip install tokenizers"
                )
            if tokenizer is None:
                raise ValueError(
                    "custom_bpe_tokenizer processor requires a tokenizer argument"
                )
            processor = CustomBPETokenizeProcessor(
                tokenizer=tokenizer,
                add_special_tokens=True,
                max_length=max_sen_len,
                padding=True,  # Enable padding for consistent batch sizes
                input_ids_key=input_ids_key,  # Use parameter from build_text_pipeline_from_steps
                attention_mask_key=attention_mask_key,  # Use parameter from build_text_pipeline_from_steps
                text_length_key="text_length",  # LSTM2Risk expects this field
            )

        elif step_name == "fasttext_embedding":
            if not HAS_GENSIM:
                raise ImportError(
                    "fasttext_embedding processor requires gensim library. "
                    "Install with: pip install gensim"
                )
            raise NotImplementedError(
                "fasttext_embedding processor requires keyed_vectors parameter. "
                "Use direct instantiation instead of build_text_pipeline_from_steps"
            )

        # Temporal processors (note: these typically need separate handling due to fit() requirements)
        elif step_name in [
            "sequence_ordering",
            "sequence_padding",
            "temporal_mask",
            "time_delta",
        ]:
            raise NotImplementedError(
                f"{step_name} processor requires fit() and specific parameters. "
                "Use direct instantiation instead of build_text_pipeline_from_steps"
            )

        else:
            raise ValueError(
                f"Unknown processing step: '{step_name}'. "
                f"Available steps: {list(PROCESSOR_REGISTRY.keys())}"
            )

        # Chain processors using >> operator
        pipeline = processor if pipeline is None else pipeline >> processor

    return pipeline
