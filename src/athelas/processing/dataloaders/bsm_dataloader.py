from collections.abc import Callable, Mapping

import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence


def build_collate_batch(
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
):
    """
    Build a collate function for models with text modalities.

    Handles:
    - Single or multiple text modalities (e.g., chat, shiptrack) with tokenization keys
    - Tabular features
    - Labels

    All text modalities use the same tokenizer output keys since they
    share the same tokenizer.

    Args:
        input_ids_key: Key name for text input_ids (applies to all text modalities)
        attention_mask_key: Key name for text attention_mask (applies to all text modalities)

    Returns:
        Collate function for DataLoader
    """

    def collate_batch(batch):
        if not isinstance(batch[0], dict):
            raise TypeError("Batch must contain dictionaries.")

        output = {}

        def pad_nested(tensors):
            """Pad nested tensors to uniform dimensions."""
            max_chunks = max(t.size(0) for t in tensors)
            max_len = max(t.size(1) for t in tensors)
            padded = []
            for t in tensors:
                pad_chunk = max_chunks - t.size(0)
                pad_len = max_len - t.size(1)
                padded.append(torch.nn.functional.pad(t, (0, pad_len, 0, pad_chunk)))
            return torch.stack(padded)

        def process_text_modality(batch, key, input_ids_key, attention_mask_key):
            """Process text modality by tokenizing and padding sequences."""
            all_input_ids = []
            all_attention_masks = []

            for item in batch:
                input_chunks = [
                    torch.tensor(chunk[input_ids_key], dtype=torch.long)
                    for chunk in item[key]
                ]
                mask_chunks = [
                    torch.tensor(chunk[attention_mask_key], dtype=torch.long)
                    for chunk in item[key]
                ]
                all_input_ids.append(pad_sequence(input_chunks, batch_first=True))
                all_attention_masks.append(pad_sequence(mask_chunks, batch_first=True))

            return pad_nested(all_input_ids), pad_nested(all_attention_masks)

        for key in batch[0]:
            # Check if this is a text field
            if all(
                isinstance(item[key], list)
                and isinstance(item[key][0], dict)
                and input_ids_key in item[key][0]
                for item in batch
            ):
                input_ids, attention_masks = process_text_modality(
                    batch, key, input_ids_key, attention_mask_key
                )
                output[key + "_" + input_ids_key] = input_ids
                output[key + "_" + attention_mask_key] = attention_masks

            # Handle tabular features and labels
            else:
                output[key] = [item[key] for item in batch]

        return output

    return collate_batch


# Alias for backward compatibility - trimodal naming
build_trimodal_collate_batch = build_collate_batch
