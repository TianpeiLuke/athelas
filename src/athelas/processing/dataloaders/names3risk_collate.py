"""
Names3Risk DataLoader Collate Functions

Collate functions for batching Names3Risk bimodal (text + tabular) fraud detection samples.

**Core Concept:**
Batching in PyTorch requires handling variable-length sequences and different data modalities.
Names3Risk combines text sequences (customer names, emails) with tabular features (transaction
data), requiring specialized collate functions that handle both modalities correctly.

**Model-Specific Requirements:**

LSTM2Risk:
- Requires sequences sorted by length (descending) for efficient pack_padded_sequence
- Needs explicit length tracking for unpacking
- Variable-length sequences handled via packing/unpacking

Transformer2Risk:
- Requires sequences truncated to fixed block_size (max position embeddings)
- Uses attention masking instead of packing (mask padding tokens)
- No sorting needed (attention mechanism handles variable lengths)

**Architecture:**
Both functions follow factory pattern:
1. Accept configuration parameters (pad_token, block_size)
2. Return closure that captures configuration
3. Closure processes batch and returns standardized dict

**Parameters:**
- pad_token (int): Token ID used for padding shorter sequences
- block_size (int): Maximum sequence length for transformers

**Batch Processing:**
Input batch format (list of dicts):
[
    {"text": tensor([15, 234, ...]), "tabular": tensor([...]), "label": tensor(1)},
    {"text": tensor([89, 45, ...]), "tabular": tensor([...]), "label": tensor(0)},
    ...
]

**Dependencies:**
- torch.nn.utils.rnn.pad_sequence → Pad variable-length sequences
- torch.stack → Stack fixed-size tensors

**Used By:**
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_lstm2risk → LSTM2Risk training
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_transformer2risk → Transformer2Risk training

**Alternative Approaches:**
- torch.utils.data.default_collate → Too simplistic for multi-modal data
- Manual padding loops → Less efficient than pad_sequence
- Single unified collate → Doesn't handle model-specific requirements

**Usage Example:**
```python
from torch.utils.data import DataLoader
from names3risk_pytorch.dockers.processing.dataloaders import (
    build_lstm2risk_collate_fn,
    build_transformer2risk_collate_fn
)

# LSTM2Risk DataLoader
lstm_collate = build_lstm2risk_collate_fn(pad_token=0)
lstm_loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=lstm_collate
)

# Transformer2Risk DataLoader
transformer_collate = build_transformer2risk_collate_fn(
    pad_token=0,
    block_size=100
)
transformer_loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=transformer_collate
)

# Iterate batches
for batch in lstm_loader:
    # batch["text"]: (B, L) - padded, sorted by length
    # batch["text_length"]: (B,) - original lengths
    # batch["tabular"]: (B, F) - tabular features
    # batch["label"]: (B,) - labels
    pass
```

**References:**
- PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html
- pack_padded_sequence: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
"""

from typing import Any, Callable, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def build_lstm2risk_collate_fn(pad_token: int) -> Callable:
    """
    Build collate function for LSTM2Risk model.

    Creates a collate function that handles LSTM-specific requirements:
    - Sorts sequences by length in descending order (required for pack_padded_sequence)
    - Tracks original sequence lengths
    - Pads text sequences to longest sequence in batch
    - Stacks tabular features and labels

    The sorting is critical for LSTM efficiency - pack_padded_sequence requires
    sequences sorted by length to avoid unnecessary computation on padding tokens.

    Args:
        pad_token: Token ID used for padding shorter sequences. Should match
                  tokenizer.pad_token (typically 0 for most tokenizers).

    Returns:
        collate_fn: Function that takes list of samples and returns batched dict.
                   Signature: List[Dict] -> Dict[str, torch.Tensor]

    Batch Output Format:
        {
            "text": (B, L) - Padded text token IDs, sorted by length descending
            "text_length": (B,) - Original sequence lengths before padding
            "tabular": (B, F) - Stacked tabular features
            "label": (B,) - Stacked labels
        }

        Where:
        - B = batch size
        - L = length of longest sequence in batch (after padding)
        - F = number of tabular features

    Example:
        >>> collate = build_lstm2risk_collate_fn(pad_token=0)
        >>> batch = [
        ...     {"text": torch.tensor([1,2,3]), "tabular": torch.randn(10), "label": torch.tensor(1)},
        ...     {"text": torch.tensor([4,5]), "tabular": torch.randn(10), "label": torch.tensor(0)},
        ... ]
        >>> output = collate(batch)
        >>> output["text"].shape  # (2, 3) - padded to longest (3 tokens)
        torch.Size([2, 3])
        >>> output["text_length"]  # [3, 2] - sorted descending
        tensor([3, 2])

    Implementation Notes:
        - Sorting reorders all batch components (text, tabular, labels) consistently
        - sort_idx tracks reordering to maintain sample correspondence
        - pad_sequence handles variable-length padding efficiently
        - Labels and tabular features maintain same order as sorted text
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples into LSTM-compatible format.

        Args:
            batch: List of sample dictionaries with keys: text, tabular, label

        Returns:
            Batched dictionary with sorted, padded sequences and lengths
        """
        # Extract batch components
        texts = [item["text"] for item in batch]
        tabs = [item["tabular"] for item in batch]
        labels = [item["label"] for item in batch]

        # Compute sequence lengths
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

        # Sort by length (descending) - required for LSTM pack_padded_sequence
        # Returns: sorted lengths and indices that would sort the lengths
        lengths, sort_idx = lengths.sort(descending=True)

        # Reorder all components to match sorted order
        texts = [texts[i] for i in sort_idx]
        tabs = [tabs[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]

        # Pad text sequences to longest sequence in batch
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_token)

        # Stack fixed-size tensors
        tabs_stacked = torch.stack(tabs)
        labels_stacked = torch.stack(labels)

        return {
            "text": texts_padded,
            "text_length": lengths,
            "tabular": tabs_stacked,
            "label": labels_stacked,
        }

    return collate_fn


def build_transformer2risk_collate_fn(pad_token: int, block_size: int) -> Callable:
    """
    Build collate function for Transformer2Risk model.

    Creates a collate function that handles Transformer-specific requirements:
    - Truncates sequences to block_size (maximum position embeddings)
    - Creates attention mask (True = attend to token, False = ignore padding)
    - NO sorting needed (attention mechanism handles variable lengths directly)
    - Pads text sequences to longest sequence in batch (up to block_size)
    - Stacks tabular features and labels

    Unlike LSTMs, transformers use position embeddings with fixed maximum length.
    Sequences longer than block_size are truncated. Attention masking allows
    the model to ignore padding tokens without explicit packing/unpacking.

    Args:
        pad_token: Token ID used for padding shorter sequences. Should match
                  tokenizer.pad_token (typically 0 for most tokenizers).
        block_size: Maximum sequence length (determined by position embedding table size).
                   Sequences longer than this are truncated. Typical values: 100-512.

    Returns:
        collate_fn: Function that takes list of samples and returns batched dict.
                   Signature: List[Dict] -> Dict[str, torch.Tensor]

    Batch Output Format:
        {
            "text": (B, L) - Padded text token IDs (no sorting, may be truncated)
            "tabular": (B, F) - Stacked tabular features
            "label": (B,) - Stacked labels
            "attn_mask": (B, L) - Attention mask (True = valid token, False = padding)
        }

        Where:
        - B = batch size
        - L = min(longest_sequence_in_batch, block_size)
        - F = number of tabular features

    Example:
        >>> collate = build_transformer2risk_collate_fn(pad_token=0, block_size=100)
        >>> batch = [
        ...     {"text": torch.tensor([1,2,3]), "tabular": torch.randn(10), "label": torch.tensor(1)},
        ...     {"text": torch.tensor([4,5]), "tabular": torch.randn(10), "label": torch.tensor(0)},
        ... ]
        >>> output = collate(batch)
        >>> output["text"].shape  # (2, 3) - padded to longest (3 tokens)
        torch.Size([2, 3])
        >>> output["attn_mask"]  # [[True, True, True], [True, True, False]]
        tensor([[ True,  True,  True],
                [ True,  True, False]])

    Implementation Notes:
        - Truncation happens before padding ([:block_size] slice)
        - Attention mask computed after padding (True where != pad_token)
        - No sorting preserves original batch order
        - Mask shape matches padded text shape for broadcasting
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples into Transformer-compatible format.

        Args:
            batch: List of sample dictionaries with keys: text, tabular, label

        Returns:
            Batched dictionary with truncated, padded sequences and attention mask
        """
        # Extract batch components and truncate text to block_size
        texts = [item["text"][:block_size] for item in batch]
        tabs = [item["tabular"] for item in batch]
        labels = [item["label"] for item in batch]

        # Pad text sequences to longest sequence in batch (up to block_size)
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_token)

        # Stack fixed-size tensors
        tabs_stacked = torch.stack(tabs)
        labels_stacked = torch.stack(labels)

        # Create attention mask: True where not padding, False where padding
        # This allows transformer to ignore padding tokens in attention computation
        attn_mask = texts_padded != pad_token

        return {
            "text": texts_padded,
            "tabular": tabs_stacked,
            "label": labels_stacked,
            "attn_mask": attn_mask,
        }

    return collate_fn
