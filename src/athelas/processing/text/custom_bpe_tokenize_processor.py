"""Custom BPE Tokenization Processor.

This module provides a tokenization processor compatible with HuggingFace's
tokenizers.Tokenizer library for custom BPE tokenizers trained via the
tokenizer_training step.

Module: processing.text.custom_bpe_tokenize_processor
Purpose: Tokenize text inputs using custom BPE tokenizers for bimodal models
Compatible Models: lstm2risk, transformer2risk

Type Definitions:
    - Input: Union[Dict[str, Any], str, float, int, None]
        * Dict: {"field_name": "text_value"} - extracts first string value
        * str: "pipe|separated|text" - direct text input
        * float/int: Numeric values - converted to string or empty
        * None/NaN: Missing values - returns empty tensors

    - Output: List[Dict[str, Any]]
        * Returns list with single dict containing:
            - input_ids_key: torch.Tensor(dtype=long) - Token IDs
            - attention_mask_key: torch.Tensor(dtype=long) - Mask (1=real, 0=pad)
            - text_length_key: int - Sequence length before padding
        * List format required for pipeline_dataloader compatibility

Key Features:
    - Uses tokenizers.Tokenizer.encode() for BPE tokenization
    - Outputs attention_mask for Transformer2Risk (masks padding tokens)
    - Outputs text_length for LSTM2Risk (enables pack_padded_sequence)
    - Configurable output key names for flexibility
    - Padding applied when max_length is specified
    - Handles missing/NaN values gracefully

Input Handling:
    - Pipe-separated text: "email|name|address|phone" → tokenized
    - Missing values: float('nan'), None → empty tensors
    - Numeric values: 0.5, 123 → converted to string then tokenized
    - Empty/whitespace: "", "   " → empty tensors

Example:
    Basic usage with custom BPE tokenizer:

    >>> from tokenizers import Tokenizer
    >>> from processing.text.custom_bpe_tokenize_processor import CustomBPETokenizeProcessor
    >>>
    >>> # Load custom BPE tokenizer
    >>> tokenizer = Tokenizer.from_file("tokenizer.json")
    >>>
    >>> # Create processor with padding
    >>> processor = CustomBPETokenizeProcessor(
    ...     tokenizer=tokenizer,
    ...     add_special_tokens=True,
    ...     max_length=100,
    ...     padding=True
    ... )
    >>>
    >>> # Process pipe-separated text
    >>> result = processor("alice@email.com|Alice|123 Main St|555-1234")
    >>> result.keys()
    dict_keys(['input_ids', 'attention_mask', 'text_length'])
    >>>
    >>> # Handle missing values
    >>> result_nan = processor(float('nan'))
    >>> result_nan['text_length']
    0

Notes:
    - Sequence length always reflects real tokens (before padding)
    - Special tokens ([CLS], [SEP]) added if add_special_tokens=True
    - Truncation applied if sequence exceeds max_length
    - Padding applied to max_length if padding=True
"""

from typing import List, Optional, Dict, Any, Union
import torch
from tokenizers import Tokenizer  # HuggingFace tokenizers library

from ..processors import Processor


class CustomBPETokenizeProcessor(Processor):
    """
    Tokenization processor for custom BPE tokenizer (HuggingFace tokenizers.Tokenizer).

    Outputs token IDs, attention masks, and sequence lengths for lstm2risk/transformer2risk.

    Args:
        tokenizer (Tokenizer): HuggingFace Tokenizer instance (from tokenizers library)
        add_special_tokens (bool): Whether to add special tokens like [CLS], [SEP].
            Default: True
        max_length (Optional[int]): Maximum sequence length (truncates if longer,
            pads if padding=True).
            ⚠️ CRITICAL: This MUST match the model's max_sen_len/block_size parameter!
            - Transformer2Risk: Set to model's max_sen_len (typically 100)
            - LSTM2Risk: Set to model's block_size (typically 100)
            - If not set, sequences may exceed model's position embedding size
            Default: None (WARNING: Will cause errors if not set!)
        padding (Optional[bool]): Whether to pad sequences to max_length.
            Default: True if max_length is set, otherwise False
        pad_token_id (int): Token ID for padding. Default: 0
        input_ids_key (str): Key name for token IDs in output dict.
            Default: "input_ids"
        attention_mask_key (str): Key name for attention mask in output dict.
            Default: "attention_mask"
        text_length_key (str): Key name for text length in output dict.
            Default: "text_length"

    Returns:
        List[Dict[str, Any]]: List containing single dictionary with token IDs,
            attention mask, and text length (list format required for pipeline_dataloader)

    Example:
        >>> from tokenizers import Tokenizer
        >>> tokenizer = Tokenizer.from_file("tokenizer.json")
        >>> processor = CustomBPETokenizeProcessor(tokenizer, max_length=100)
        >>> data = processor({"text_field": "Alice|alice@email.com"})
        >>> data.keys()
        dict_keys(['text_field', 'text', 'attn_mask', 'text_length'])
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        pad_token_id: int = 0,
        input_ids_key: str = "input_ids",
        attention_mask_key: str = "attention_mask",
        text_length_key: str = "text_length",
    ):
        super().__init__()
        self.processor_name = "custom_bpe_tokenization_processor"
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        # Default: pad if max_length is set
        self.padding = padding if padding is not None else (max_length is not None)
        self.pad_token_id = pad_token_id
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.text_length_key = text_length_key

    def process(
        self, data: Union[Dict[str, Any], str, float, int, None]
    ) -> List[Dict[str, Any]]:
        """
        Tokenize input text using custom BPE tokenizer.

        Args:
            data: Input data - can be:
                - Dictionary containing text field
                - String value directly (e.g., "email|name|...")
                - Numeric value (float, int) - converted to string or empty
                - None/NaN - treated as empty string

        Returns:
            List containing single dictionary with tokenization outputs:
                - input_ids_key: token IDs tensor (L,)
                - attention_mask_key: attention mask (L,) - 1=real, 0=padding
                - text_length_key: sequence length (scalar) - count before padding

            List format is required for pipeline_dataloader compatibility,
            even though names3risk only uses single-sequence (not multi-chunk).

        Note:
            - Returns list with empty tensors for empty/whitespace-only/missing input
            - Applies truncation if max_length is set
            - Applies padding if padding=True and max_length is set
            - Text length always reflects real tokens (before padding)
        """
        # Extract input text from various input formats
        if isinstance(data, dict):
            # Dict input: extract first string value
            input_text = None
            for value in data.values():
                if isinstance(value, str):
                    input_text = value
                    break
        elif isinstance(data, str):
            # String input: use directly
            input_text = data
        else:
            # Handle numeric/missing values (float, int, None, NaN)
            # Convert to string, or empty string if None
            try:
                input_text = str(data) if data is not None else ""
                # Handle NaN case (str(float('nan')) = 'nan')
                if input_text.lower() == "nan":
                    input_text = ""
            except:
                input_text = ""

        # Prepare chunk dictionary
        chunk = {}

        # Handle empty input
        if not input_text or not input_text.strip():
            # Return empty tensors in list format
            chunk[self.input_ids_key] = torch.tensor([], dtype=torch.long)
            chunk[self.attention_mask_key] = torch.tensor([], dtype=torch.long)
            chunk[self.text_length_key] = 0
            return [chunk]  # Return as list for pipeline_dataloader compatibility

        # Encode using custom tokenizer
        # tokenizer.encode() returns Encoding object with .ids attribute
        encoding = self.tokenizer.encode(
            input_text,
            add_special_tokens=self.add_special_tokens,
        )

        # Extract token IDs as list
        ids = encoding.ids

        # Store original length (before truncation/padding)
        text_length = len(ids)

        # Apply truncation if max_length is specified
        # CRITICAL: This prevents IndexError in position embeddings
        # The model's TransformerEncoder/LSTMEncoder expects sequences ≤ max_sen_len
        if self.max_length is not None:
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
                text_length = self.max_length  # Update length after truncation
        else:
            # WARNING: No max_length set - sequences may be too long for model!
            # This will cause IndexError if sequence > model's position embedding size
            pass

        # Apply padding if enabled
        if self.padding and self.max_length:
            # Pad to max_length
            padding_length = self.max_length - len(ids)
            if padding_length > 0:
                ids = ids + [self.pad_token_id] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * text_length
        if self.padding and self.max_length:
            padding_length = self.max_length - text_length
            if padding_length > 0:
                attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors and store in chunk dictionary
        chunk[self.input_ids_key] = torch.tensor(ids, dtype=torch.long)
        chunk[self.attention_mask_key] = torch.tensor(attention_mask, dtype=torch.long)
        chunk[self.text_length_key] = text_length

        # Return as list (single chunk) for pipeline_dataloader compatibility
        return [chunk]
