import unicodedata
from typing import List, Tuple
import random

from torch.optim.lr_scheduler import OneCycleLR
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class OrderTextTokenizer:

    def __init__(self, min_frequency: int = 25):
        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.min_frequency = min_frequency
        self.pad_token = None
        self.cls_token = None

    def calculate_compression_rate(self, texts, sample_size = 10000):

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

        return total_chars / total_tokens

    def train(
        self,
        texts: List[str],
        target_compression: float = 2.5,
        max_vocab_size: int = 50000,
    ):
        """Train tokenizer with automatic vocab size tuning to achieve target compression."""

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
            self._tokenizer = temp_tokenizer  # Temporarily set for compression calculation
            compression = self.calculate_compression_rate(validation_texts)
            actual_vocab_size = temp_tokenizer.get_vocab_size()

            print(f"  Compression: {compression:.3f} ({compression:.1%})")
            print(f"  Actual vocab size: {actual_vocab_size}")

            # Save best result
            if abs(compression - target_compression) < abs(best_compression - target_compression):
                best_compression = compression
                best_tokenizer = temp_tokenizer
                best_vocab_size = actual_vocab_size

            # Adjust search range
            if compression < target_compression:
                # Need higher compression - increase vocab size
                vocab_low = current_vocab_size + 1
            else:
                # compression is high enough - can decrease vocab size
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
        assert len(pad_tokens) == 1
        self.pad_token = pad_tokens[0]

        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1
        self.cls_token = cls_tokens[0]

        return self

    def encode(self, text):
        normalized_text = unicodedata.normalize("NFKC", text)
        return self.tokenizer.encode(normalized_text).ids

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

        pad_tokens = self.encode("[PAD]")
        assert len(pad_tokens) == 1
        self.pad_token = pad_tokens[0]

        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1
        self.cls_token = cls_tokens[0]

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
