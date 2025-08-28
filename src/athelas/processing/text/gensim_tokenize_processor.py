import numpy as np
from typing import List, Union, Dict, Optional
from gensim.models import KeyedVectors
from .processors import Processor

# --- Processor 7: FastText Embedding Processor ---
class FastTextEmbeddingProcessor(Processor):
    """
    Tokenization processor that maps words to FastText embeddings.
    Accepts a list of text chunks, splits on whitespace, looks up embeddings,
    and pads/truncates to `max_length`. Returns a dict per chunk with:
      - `embeddings_key`: List[List[float]] of shape (L, D)
      - `attention_mask_key`: List[int] of shape (L,)
    """
    def __init__(
        self,
        keyed_vectors: KeyedVectors,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        embeddings_key: str = "embeddings",
        attention_mask_key: str = "attention_mask",
    ):
        super().__init__()
        self.processor_name = "fasttext_embedding_processor"
        self.kv = keyed_vectors
        self.dim = keyed_vectors.vector_size
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.embeddings_key = embeddings_key
        self.attention_mask_key = attention_mask_key

    def process(
        self,
        input_chunks: List[str]
    ) -> List[Dict[str, Union[List[List[float]], List[int]]]]:
        output = []
        for chunk in input_chunks:
            # Split into words
            words = chunk.strip().split()
            # Truncate if needed
            if self.max_length is not None:
                words = words[: self.max_length]

            # Look up embeddings (zeros for unknown)
            embeddings = []
            mask = []
            for w in words:
                if w in self.kv:
                    embeddings.append(self.kv[w].tolist())
                    mask.append(1)
                else:
                    embeddings.append([0.0] * self.dim)
                    mask.append(0)

            # Pad to max_length if configured
            if self.pad_to_max_length and self.max_length is not None:
                pad_len = self.max_length - len(embeddings)
                if pad_len > 0:
                    embeddings.extend([[0.0] * self.dim] * pad_len)
                    mask.extend([0] * pad_len)

            output.append({
                self.embeddings_key: embeddings,
                self.attention_mask_key: mask,
            })
        return output
