# FastText Embedding Processor

## Task Summary
The FastText Embedding Processor converts text into word embeddings using pre-trained FastText vectors. It tokenizes text by splitting on whitespace, maps each token to its corresponding embedding vector, and handles padding and truncation to ensure consistent output dimensions.

## Input
- **input_chunks**: List of text strings to convert to embeddings
- **keyed_vectors**: Gensim KeyedVectors object containing pre-trained FastText embeddings
- **max_length**: Maximum sequence length (will truncate if longer)
- **pad_to_max_length**: Whether to pad sequences shorter than max_length
- **embeddings_key**: Key name for embeddings in the output dictionary
- **attention_mask_key**: Key name for attention mask in the output dictionary

## Output
- **List[Dict]**: List of dictionaries containing embeddings and attention masks
  - Each dictionary contains:
    - embeddings: List of embedding vectors (List[List[float]]) of shape (sequence_length, embedding_dimension)
    - attention_mask: List of 1s and 0s indicating which tokens have valid embeddings

## Features
- Maps words to pre-trained FastText embeddings
- Handles out-of-vocabulary words by using zero vectors
- Supports customizable maximum sequence length
- Provides options for padding to ensure consistent dimensions
- Generates attention masks to identify valid tokens vs. padding
- Allows customization of output dictionary keys

## Example Usage
```python
from gensim.models import KeyedVectors
from src.processing.gensim_tokenize_processor import FastTextEmbeddingProcessor

# Load pre-trained FastText vectors
fasttext_vectors = KeyedVectors.load("path/to/fasttext/vectors.kv")

# Create processor
processor = FastTextEmbeddingProcessor(
    keyed_vectors=fasttext_vectors,
    max_length=50,
    pad_to_max_length=True,
    embeddings_key="embeddings",
    attention_mask_key="attention_mask"
)

# Process text chunks
text_chunks = [
    "This is the first sentence to embed.",
    "Here's another sentence with some special characters!",
    "A third, slightly longer sentence that might need truncation."
]

embedded_output = processor.process(text_chunks)

# Example output structure:
# [
#   {
#     "embeddings": [[0.1, 0.2, ..., 0.3], [0.4, 0.5, ..., 0.6], ...],  # shape: (sequence_length, embedding_dimension)
#     "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...]  # 1s for valid tokens, 0s for padding
#   },
#   ...
# ]
```

## Use Cases
- Feature extraction for NLP tasks
- Text classification using traditional ML models
- Document similarity comparison
- Information retrieval
- Transfer learning with pre-trained word embeddings
