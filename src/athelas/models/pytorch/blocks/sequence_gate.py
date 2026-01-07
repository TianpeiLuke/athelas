#!/usr/bin/env python3
"""
Dual-Sequence Gating Module

Computes learned importance weights for dual-sequence models,
enabling adaptive selection between two sequence sources.

**Common Use Cases:**
- CID (customer) vs CCID (customer-card) transaction sequences
- User history vs. group/cohort history
- Primary vs. auxiliary sequence data

**Key Features:**
- Lightweight attention-based gate computation
- Threshold-based filtering for efficiency
- Padding-aware (handles missing sequences)
- Returns both gate scores and indices for conditional processing

**Mathematical Formulation:**
    Given two sequences (seq1, seq2):

    1. Compute gate embeddings:
       e1 = OrderAttention(seq1)  # [B, 2d]
       e2 = OrderAttention(seq2)  # [B, 2d]

    2. Concatenate and compute scores:
       z = [e1 || e2]             # [B, 4d]
       g = softmax(MLP(z))        # [B, 2]

    3. Apply filtering:
       if seq2_fully_padded:
           g = [1.0, 0.0]         # Force seq1-only

       keep_seq2 = {i : g[i,1] > threshold}

    Returns:
        gate_scores: [B, 2] importance weights (sum to 1.0)
        keep_indices: indices where seq2 should be processed

**Usage Example:**
```python
from pytorch.blocks import DualSequenceGate

gate = DualSequenceGate(config)
gate_scores, keep_idx = gate(
    seq1_cat, seq1_num, time_seq1,
    seq2_cat, seq2_num, time_seq2,
    padding_mask_seq1, padding_mask_seq2
)

# Use gate scores to weight sequence embeddings
ensemble = gate_scores[:, 0] * emb_seq1 + gate_scores[:, 1] * emb_seq2
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional, Tuple

from .order_attention import OrderAttentionModule


class DualSequenceGate(nn.Module):
    """
    Dual-sequence importance weighting module.

    This module implements a learned gating mechanism that dynamically
    determines the importance of two sequences for each sample. It uses
    lightweight attention-based embeddings and an MLP to compute gate scores,
    with threshold-based filtering for computational efficiency.

    Args:
        config: Configuration dictionary with keys:
            - n_cat_features: Number of categorical features
            - n_num_features: Number of numerical features
            - n_embedding: Embedding vocabulary size
            - seq_len: Sequence length (default: 51)
            - gate_embedding_dim: Gate embedding dimension (default: 16)
            - gate_hidden_dim: Gate MLP hidden dimension (default: 256)
            - gate_threshold: Minimum score to process seq2 (default: 0.05)
            - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, config: Dict[str, Union[int, float, str, bool]]):
        super().__init__()

        # Gate-specific configuration
        self.n_cat_features = config["n_cat_features"]
        self.n_num_features = config["n_num_features"]
        self.n_embedding = config["n_embedding"]
        self.seq_len = config.get("seq_len", 51)
        self.gate_embedding_dim = config.get("gate_embedding_dim", 16)
        self.gate_hidden_dim = config.get("gate_hidden_dim", 256)
        self.dropout = config.get("dropout", 0.1)

        # Gate embedding table (smaller than main embedding for efficiency)
        self.embedding_gate = nn.Embedding(
            self.n_embedding + 2, self.gate_embedding_dim, padding_idx=0
        )

        # Gate attention module (simplified configuration)
        gate_config = config.copy()
        gate_config.update(
            {
                "dim_embedding_table": self.gate_embedding_dim,
                "dim_attn_feedforward": 128,
                "num_heads": 1,
                "n_layers_order": 1,
                "use_moe": False,
                "num_experts": 1,
                "use_time_seq": False,
                "return_seq": False,
            }
        )

        self.gate_attention = OrderAttentionModule(gate_config)

        # Override the embedding in gate attention
        self.gate_attention.embedding = self.embedding_gate

        # Gate score computation MLP
        gate_input_dim = 2 * (2 * self.gate_embedding_dim)  # seq1 + seq2 embeddings
        self.gate_score = nn.Sequential(
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        # Gate threshold for seq2 filtering
        self.gate_threshold = config.get("gate_threshold", 0.05)

    def forward(
        self,
        seq1_cat: torch.Tensor,
        seq1_num: torch.Tensor,
        time_seq1: Optional[torch.Tensor],
        seq2_cat: torch.Tensor,
        seq2_num: torch.Tensor,
        time_seq2: Optional[torch.Tensor],
        key_padding_mask_seq1: Optional[torch.Tensor] = None,
        key_padding_mask_seq2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate scores for dual sequences.

        Args:
            seq1_cat: Sequence 1 categorical features [B, L, n_cat_features]
            seq1_num: Sequence 1 numerical features [B, L, n_num_features]
            time_seq1: Sequence 1 time features [B, L, 1]
            seq2_cat: Sequence 2 categorical features [B, L, n_cat_features]
            seq2_num: Sequence 2 numerical features [B, L, n_num_features]
            time_seq2: Sequence 2 time features [B, L, 1]
            key_padding_mask_seq1: Sequence 1 padding mask [B, L]
            key_padding_mask_seq2: Sequence 2 padding mask [B, L]

        Returns:
            gate_scores: Gate scores [B, 2] for (seq1, seq2), sum to 1.0
            seq2_keep_idx: Indices where seq2 should be processed (gate score > threshold)
        """
        # Compute gate embeddings for both sequences
        gate_emb_seq1 = self.gate_attention(
            seq1_cat,
            seq1_num,
            time_seq1,
            attn_mask=None,
            key_padding_mask=key_padding_mask_seq1,
        )

        gate_emb_seq2 = self.gate_attention(
            seq2_cat,
            seq2_num,
            time_seq2,
            attn_mask=None,
            key_padding_mask=key_padding_mask_seq2,
        )

        # Compute raw gate scores
        gate_input = torch.cat([gate_emb_seq1, gate_emb_seq2], dim=-1)
        gate_scores_raw = self.gate_score(gate_input)

        # Apply seq2 filtering based on padding
        gate_scores = gate_scores_raw.clone()

        # Set seq2 gate score to 0 for sequences that are fully padded
        if key_padding_mask_seq2 is not None:
            # Check if seq2 is fully padded (all positions are padded)
            fully_padded_seq2 = key_padding_mask_seq2.sum(dim=1) >= (self.seq_len - 1)
            gate_scores[fully_padded_seq2, 1] = 0.0
            # Renormalize gate scores
            gate_scores = F.softmax(gate_scores, dim=1)

        # Find indices where seq2 should be processed (gate score > threshold)
        seq2_keep_idx = (gate_scores[:, 1] > self.gate_threshold).nonzero().squeeze(-1)

        return gate_scores, seq2_keep_idx
