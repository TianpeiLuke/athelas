"""
Compression-Tuned BPE Tokenizer

Byte-Pair Encoding tokenizer with automatic vocabulary size tuning to achieve
target compression rate.

**Core Concept:**
For fraud detection on name/email text, optimal vocabulary size balances:
- Too small → Many UNK tokens, loses information
- Too large → Overfitting to training data, poor generalization

Auto-tuning via binary search finds vocabulary size achieving target compression
rate on validation set (e.g., 2.5 chars per token).

**Architecture:**
1. Text normalization (NFKC Unicode normalization)
2. Whitespace pre-tokenization
3. BPE vocabulary learning with configurable min_frequency
4. Binary search on vocab_size to achieve target compression
5. Special tokens: [CLS], [PAD], [UNK], [BOS], [EOS], [MISSING], |

**Parameters:**
- min_frequency (int): Minimum character frequency to include in vocabulary
- target_compression (float): Target compression rate (chars per token)
- max_vocab_size (int): Maximum allowed vocabulary size

**Methods:**
- train(): Train tokenizer on text corpus with compression tuning
- encode(): Tokenize single text to token IDs
- calculate_compression_rate(): Measure compression on text sample

**Dependencies:**
- tokenizers (HuggingFace) → BPE implementation
- unicodedata → Text normalization

**Used By:**
- names3risk_pytorch.dockers.processing.datasets → Tokenize names/emails during preprocessing
- names3risk_pytorch.dockers.lightning_models → Text preprocessing for training/inference

**Alternative Approaches:**
- Character-level tokenization → Simple but loses subword patterns
- Word-level tokenization → Cannot handle OOV words in names/emails
- SentencePiece → Similar but requires separate installation
- Fixed vocab BPE → Requires manual tuning, not data-adaptive

**Usage Example:**
```python
from names3risk_pytorch.dockers.tokenizers import CompressionBPETokenizer

# Train tokenizer
tokenizer = CompressionBPETokenizer(min_frequency=25)
texts = ["john.smith@email.com", "Jane Doe", ...]

tokenizer.train(
    texts,
    target_compression=2.5,  # Aim for 2.5 chars per token
    max_vocab_size=50000
)

# Encode text
tokens = tokenizer.encode("Alice Johnson|alice@email.com")
# Returns: [15, 234, 45, ...]

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"PAD token ID: {tokenizer.pad_token}")
print(f"CLS token ID: {tokenizer.cls_token}")
```

**Training Process:**
1. Split texts into train (80%) and validation (20%) sets
2. Binary search on vocab_size from 1000 to max_vocab_size
3. For each vocab_size:
   a. Train BPE tokenizer on train set
   b. Calculate compression rate on validation set
   c. Track best tokenizer closest to target compression
4. Select tokenizer with compression nearest to target
5. Set up special tokens (PAD, CLS)

**Compression Rate:**
- Measures average characters per token
- Lower rate = more aggressive compression
- Target 2.5 means ~2.5 characters represented per token on average
- Example: "john" (4 chars) → ["jo", "hn"] (2 tokens) = 2.0 compression

**References:**
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - BPE for NLP
- HuggingFace Tokenizers documentation: https://huggingface.co/docs/tokenizers
"""

import unicodedata
import random
from typing import List

from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class CompressionBPETokenizer:
    """
    BPE tokenizer with automatic vocabulary size tuning.

    Trains tokenizer to achieve target compression rate via binary search
    on vocabulary size.

    Attributes:
        min_frequency: Minimum token frequency to include in vocabulary
        pad_token: Token ID for padding ([PAD])
        cls_token: Token ID for classification/start ([CLS])
        vocab_size: Total vocabulary size
    """

    def __init__(self, min_frequency: int = 25):
        """
        Initialize CompressionBPETokenizer.

        Args:
            min_frequency: Minimum token frequency to include in vocabulary.
                          Higher values create smaller vocabularies by filtering
                          rare tokens. Default 25 works well for fraud detection.
        """
        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.min_frequency = min_frequency
        self.pad_token = None
        self.cls_token = None

    def calculate_compression_rate(
        self, texts: List[str], sample_size: int = 10000
    ) -> float:
        """
        Calculate compression rate (chars per token) on text sample.

        Compression rate measures how many characters are represented per token
        on average. Lower values indicate more aggressive compression.

        Args:
            texts: List of texts to measure compression on
            sample_size: Maximum number of texts to sample for calculation.
                        Larger samples are more accurate but slower.

        Returns:
            compression_rate: Average characters per token. For example:
                            - 2.5 means 2.5 characters per token on average
                            - 1.0 means each token represents ~1 character
                            - 4.0 means each token represents ~4 characters

        Example:
            >>> tokenizer.calculate_compression_rate(["john", "jane", "bob"])
            2.35  # Average ~2.35 chars per token
        """
        if len(texts) > sample_size:
            sample_texts = random.sample(texts, sample_size)
        else:
            sample_texts = texts

        total_chars = 0
        total_tokens = 0

        for text in sample_texts:
            normalized_text = unicodedata.normalize("NFKC", text)
            encoding = self._tokenizer.encode(normalized_text)
            total_chars += len(normalized_text)
            total_tokens += len(encoding.ids)

        return total_chars / total_tokens if total_tokens > 0 else 0.0

    def train(
        self,
        texts: List[str],
        target_compression: float = 2.5,
        max_vocab_size: int = 50000,
    ) -> "CompressionBPETokenizer":
        """
        Train tokenizer with automatic vocab size tuning.

        Uses binary search on vocabulary size to achieve target compression rate.
        Splits data into train (80%) and validation (20%) sets to prevent
        overfitting to the training data.

        Args:
            texts: Training texts (customer names, emails, etc.)
            target_compression: Target chars per token (e.g., 2.5).
                              Lower values → more tokens, larger vocab
                              Higher values → fewer tokens, smaller vocab
            max_vocab_size: Maximum vocabulary size. Algorithm searches
                          between 1000 and this value.

        Returns:
            self: Trained tokenizer (enables method chaining)

        Training Process:
            1. Shuffle and split texts (80% train, 20% validation)
            2. Binary search on vocab_size from 1000 to max_vocab_size
            3. For each vocab_size:
               - Train BPE tokenizer on train set
               - Measure compression on validation set
               - Track best tokenizer closest to target
            4. Select tokenizer with compression nearest target
            5. Configure special tokens (PAD, CLS)

        Example:
            >>> tokenizer = CompressionBPETokenizer(min_frequency=25)
            >>> texts = load_customer_names_and_emails()
            >>> tokenizer.train(texts, target_compression=2.5, max_vocab_size=50000)
            Target compression: 250.0%
            Min frequency: 25
            Training on 80000 texts, validating on 20000

            Iteration 1: Testing vocab_size=25500
              Compression: 2.654 (265.4%)
              Actual vocab size: 25342
            ...
            Final tokenizer:
              Min frequency: 25
              Vocab size: 4127
              compression: 2.503 (250.3%)
        """
        # Split data for training and compression validation
        random.shuffle(texts)
        split_idx = int(0.8 * len(texts))
        train_texts = texts[:split_idx]
        validation_texts = texts[split_idx:]

        print(f"Target compression: {target_compression:.1%}")
        print(f"Min frequency: {self.min_frequency}")
        print(
            f"Training on {len(train_texts)} texts, validating on {len(validation_texts)}"
        )

        # Binary search on vocab_size to achieve target compression
        vocab_low = 1000  # Minimum reasonable vocab size
        vocab_high = max_vocab_size
        best_compression = 0.0
        best_tokenizer = None
        best_vocab_size = None

        iteration = 0
        while vocab_low <= vocab_high and iteration < 15:  # Max 15 iterations
            iteration += 1
            current_vocab_size = (vocab_low + vocab_high) // 2

            print(f"\nIteration {iteration}: Testing vocab_size={current_vocab_size}")

            # Create trainer with current vocab size
            trainer = BpeTrainer(
                vocab_size=current_vocab_size,
                special_tokens=[
                    "[CLS]",
                    "[PAD]",
                    "[UNK]",
                    "[BOS]",
                    "[EOS]",
                    "[MISSING]",
                    "|",
                ],
                min_frequency=self.min_frequency,
            )

            # Train tokenizer
            temp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            temp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            temp_tokenizer.train_from_iterator(
                (unicodedata.normalize("NFKC", text) for text in train_texts), trainer
            )

            # Calculate compression on validation set
            self._tokenizer = (
                temp_tokenizer  # Temporarily set for compression calculation
            )
            compression = self.calculate_compression_rate(validation_texts)
            actual_vocab_size = temp_tokenizer.get_vocab_size()

            print(f"  Compression: {compression:.3f} ({compression:.1%})")
            print(f"  Actual vocab size: {actual_vocab_size}")

            # Save best result
            if abs(compression - target_compression) < abs(
                best_compression - target_compression
            ):
                best_compression = compression
                best_tokenizer = temp_tokenizer
                best_vocab_size = actual_vocab_size

            # Adjust search range
            if compression < target_compression:
                # Need higher compression (more chars per token) - increase vocab size
                vocab_low = current_vocab_size + 1
            else:
                # Compression is high enough - can decrease vocab size
                vocab_high = current_vocab_size - 1

            # Early exit if we're close enough
            if abs(compression - target_compression) < 0.005:  # Within 0.5%
                print(f"  ✓ Achieved target compression within tolerance!")
                break

        # Use best tokenizer found
        if best_tokenizer is not None:
            self._tokenizer = best_tokenizer
            print(f"\nFinal tokenizer:")
            print(f"  Min frequency: {self.min_frequency}")
            print(f"  Vocab size: {best_vocab_size}")
            print(f"  compression: {best_compression:.3f} ({best_compression:.1%})")
        else:
            print("\nWarning: No suitable tokenizer found, using last attempt")

        # Set up pad and cls tokens
        pad_tokens = self.encode("[PAD]")
        assert len(pad_tokens) == 1, "PAD token should be single token"
        self.pad_token = pad_tokens[0]

        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1, "CLS token should be single token"
        self.cls_token = cls_tokens[0]

        return self

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Applies NFKC normalization before encoding to handle various Unicode
        representations consistently. For example, "café" can be represented
        as either a single character é or e + combining accent, normalization
        ensures consistent handling.

        Args:
            text: Input text (customer name, email, etc.)

        Returns:
            token_ids: List of token IDs from vocabulary

        Example:
            >>> tokenizer.encode("Alice Johnson")
            [234, 567, 12]  # Three tokens

            >>> tokenizer.encode("alice@email.com")
            [432, 89, 45, 23, 156]  # Five tokens
        """
        normalized_text = unicodedata.normalize("NFKC", text)
        return self.tokenizer.encode(normalized_text).ids

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Get underlying HuggingFace tokenizer.

        Returns:
            Tokenizer: HuggingFace Tokenizer object with trained BPE model
        """
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        """
        Set underlying tokenizer and update special tokens.

        When setting a new tokenizer, automatically updates pad_token and
        cls_token attributes based on the new tokenizer's vocabulary.

        Args:
            value: HuggingFace Tokenizer object to set
        """
        self._tokenizer = value

        pad_tokens = self.encode("[PAD]")
        assert len(pad_tokens) == 1, "PAD token should be single token"
        self.pad_token = pad_tokens[0]

        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1, "CLS token should be single token"
        self.cls_token = cls_tokens[0]

    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.

        Returns:
            int: Total number of tokens in vocabulary, including special tokens
        """
        return self.tokenizer.get_vocab_size()
