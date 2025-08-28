# BERT Tokenization Processor

## Task Summary
The BERT Tokenization Processor converts text into tokenized sequences suitable for BERT and other transformer models. It uses Hugging Face's AutoTokenizer to handle the tokenization process, including special tokens, padding, and truncation.

## Input
- **input_chunks**: List of text strings to tokenize
- **tokenizer**: Hugging Face AutoTokenizer instance
- **add_special_tokens**: Whether to add special tokens like [CLS] and [SEP]
- **max_length**: Maximum sequence length (will truncate if longer)
- **truncation**: Whether to truncate sequences longer than max_length
- **padding**: Padding strategy ('longest', 'max_length', etc.)
- **input_ids_key**: Key name for input IDs in the output dictionary
- **attention_mask_key**: Key name for attention mask in the output dictionary

## Output
- **List[Dict]**: List of dictionaries containing tokenized sequences and attention masks
  - Each dictionary contains:
    - input_ids: List of token IDs
    - attention_mask: List of 1s and 0s indicating which tokens to attend to

## Features
- Handles tokenization for BERT and other transformer models
- Supports customizable maximum sequence length
- Provides options for padding and truncation
- Skips empty or whitespace-only chunks
- Allows customization of output dictionary keys
- Compatible with any tokenizer from Hugging Face's transformers library

## Example Usage
```python
from transformers import AutoTokenizer
from src.processing.bert_tokenize_processor import TokenizationProcessor

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create processor
processor = TokenizationProcessor(
    tokenizer=tokenizer,
    add_special_tokens=True,
    max_length=128,
    truncation=True,
    padding="max_length",
    input_ids_key="input_ids",
    attention_mask_key="attention_mask"
)

# Process text chunks
text_chunks = [
    "This is the first sentence to tokenize.",
    "Here's another sentence with some special characters!",
    "A third, slightly longer sentence that might need truncation depending on the max_length setting."
]

tokenized_output = processor.process(text_chunks)

# Example output structure:
# [
#   {
#     "input_ids": [101, 2023, 2003, 1996, 2034, 6251, 2000, 11401, 1012, 102, ...],
#     "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]
#   },
#   ...
# ]
```

## Use Cases
- Preparing text data for BERT and other transformer models
- Text classification tasks
- Sequence labeling
- Question answering
- Any NLP task requiring tokenized input for transformer models
