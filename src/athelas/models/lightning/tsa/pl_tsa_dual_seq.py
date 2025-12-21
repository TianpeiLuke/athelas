#!/usr/bin/env python3
"""
PyTorch Lightning Temporal Self-Attention Dual Sequence Model

Recreates TwoSeqMoEOrderFeatureAttentionClassifier functionality as a Lightning module
while preserving EXACT numerical equivalence with legacy implementation.

Phase 1: Algorithm-Preserving Refactoring
- Recreation using modular Lightning components
- NO optimizations or algorithmic changes
- EXACT legacy behavior preservation
- Goal: rtol ≤ 1e-6 numerical equivalence

Related Documents:
- Design: slipbox/1_design/temporal_self_attention_model_design.md
- Design: slipbox/1_design/pytorch_lightning_temporal_self_attention_design.md
- SOP: slipbox/6_resources/algorithm_preserving_refactoring_sop.md
- Legacy: projects/tsa/scripts/models.py (TwoSeqMoEOrderFeatureAttentionClassifier)

Architecture:
    Dual sequence (CID + CCID) processing with gating:
    1. Order Attention CID: Process customer transaction sequence
    2. Order Attention CCID: Process counterparty transaction sequence
    3. Gate Function: Learn to weight CID vs CCID contributions
    4. Feature Attention: Process current transaction features
    5. Ensemble: Combine gated order embeddings with feature attention
    6. Classifier: Final MLP for prediction

Key Differences from Single Sequence:
    - Two separate order attention modules (CID and CCID)
    - Gating mechanism with separate small embedding table
    - Dynamic masking for empty CCID sequences
    - Weighted ensemble of two order embeddings
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

import lightning.pytorch as pl
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from .pl_order_attention import OrderAttentionModule, OrderAttentionConfig
from .pl_feature_attention import FeatureAttentionModule, FeatureAttentionConfig
from ..utils.dist_utils import all_gather
from ..utils.pl_model_plots import compute_metrics
from ..utils.config_constants import filter_config_for_tensorboard

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


# Configuration class (optional, requires Pydantic V2+)
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict

    class TSADualSeqConfig(BaseModel):
        """
        Configuration schema for TSA Dual Sequence Model.

        Extends TSASingleSeqConfig with dual sequence specific parameters.
        """

        # Pydantic V2 configuration
        model_config = ConfigDict(extra="allow")

        # Core TSA architecture parameters
        n_cat_features: int = Field(ge=1, description="Number of categorical features")
        n_num_features: int = Field(ge=0, description="Number of numerical features")
        n_embedding: int = Field(ge=1, description="Size of embedding table")
        n_engineered_num_features: int = Field(
            default=0, ge=0, description="Number of engineered features"
        )
        seq_len: int = Field(ge=1, description="Maximum sequence length")
        dim_embed: int = Field(ge=1, description="Embedding dimension (must be even)")
        dim_attn_feedforward: int = Field(
            ge=1, description="Feedforward dimension in attention"
        )

        # Attention configuration
        num_heads: int = Field(default=1, ge=1, description="Number of attention heads")
        n_layers_order: int = Field(
            default=6, ge=1, description="Number of order attention layers"
        )
        n_layers_feature: int = Field(
            default=6, ge=1, description="Number of feature attention layers"
        )
        dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")

        # Advanced features
        use_moe: bool = Field(default=True, description="Use Mixture of Experts")
        num_experts: int = Field(
            default=5, ge=1, description="Number of experts in MoE"
        )
        use_time_seq: bool = Field(default=True, description="Use temporal encoding")
        emb_tbl_use_bias: bool = Field(default=True, description="Use embedding bias")

        # Task configuration
        n_classes: int = Field(default=2, ge=2, description="Number of output classes")
        is_binary: bool = Field(default=True, description="Binary classification task")

        # Training configuration
        lr: float = Field(default=2e-5, gt=0, description="Learning rate")
        weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
        adam_epsilon: float = Field(default=1e-8, gt=0, description="Adam epsilon")
        warmup_steps: int = Field(default=0, ge=0, description="Warmup steps")
        run_scheduler: bool = Field(default=True, description="Use LR scheduler")

        @field_validator("dim_embed")
        @classmethod
        def validate_dim_embed(cls, v: int) -> int:
            """Ensure dim_embed is even for splitting between cat and num."""
            if v % 2 != 0:
                raise ValueError(f"dim_embed must be even, got {v}")
            return v

except ImportError:
    TSADualSeqConfig = Dict


class TSADualSeq(pl.LightningModule):
    """
    PyTorch Lightning implementation of TSA Dual Sequence Model.

    Processes two transaction sequences (CID and CCID) using separate order attention
    modules with a learned gating mechanism. Equivalent to legacy TwoSeqMoEOrderFeatureAttentionClassifier.

    Architecture Flow:
        Input CID: x_cat_cid [B, L, D_cat], x_num_cid [B, L, D_num], time_seq_cid [B, L, 1]
        Input CCID: x_cat_ccid [B, L, D_cat], x_num_ccid [B, L, D_num], time_seq_ccid [B, L, 1]
        ↓
        Gate Embeddings: Small embeddings for gate function → [B, 32] each
        Gate Scores: MLP → [B, 2] (softmax weights for CID vs CCID)
        ↓
        Order Attention CID: Full sequence → [B, dim_embed]
        Order Attention CCID: Full sequence → [B, dim_embed]
        Gated Order: gate[0] * CID + gate[1] * CCID → [B, dim_embed]
        ↓
        Feature Attention: Last CID transaction → [B, dim_embed//2]
        ↓
        Ensemble: Concatenate gated order + feature → [B, dim_embed + dim_embed//2]
        ↓
        Classifier MLP: → [B, n_classes]

    Critical Implementation Details:
        - Two separate order attention modules (share main embedding table)
        - Separate small embedding table (dim=16) for gate function
        - Gate function uses simplified attention (no MoE, no temporal encoding)
        - Dynamic masking: CCID gate weight set to 0 if CCID sequence is empty
        - Feature attention uses CID sequence only
        - Layer normalization after gating

    Phase 1 Constraints:
        - NO caching or optimizations
        - NO formula modifications
        - NO architectural changes
        - EXACT legacy computation preserved
    """

    def __init__(
        self,
        config: Union[Dict, TSADualSeqConfig],
    ):
        super().__init__()

        # Convert config to dict if needed
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "dict"):
            config_dict = config.dict()
        else:
            config_dict = config

        self.config = config_dict
        self.model_class = "tsa_dual_seq"

        # === Core configuration ===
        self.id_name = config_dict.get("id_name", None)
        self.label_name = config_dict["label_name"]
        self.is_binary = config_dict.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config_dict.get("n_classes", 2)
        self.metric_choices = config_dict.get(
            "metric_choices", ["accuracy", "f1_score"]
        )

        # Transformed label for multiclass
        if not self.is_binary and self.num_classes > 2:
            self.label_name_transformed = self.label_name + "_processed"
        else:
            self.label_name_transformed = self.label_name

        # Training configuration
        self.model_path = config_dict.get("model_path", "")
        self.lr = config_dict.get("lr", 2e-5)
        self.weight_decay = config_dict.get("weight_decay", 0.0)
        self.adam_epsilon = config_dict.get("adam_epsilon", 1e-8)
        self.warmup_steps = config_dict.get("warmup_steps", 0)
        self.run_scheduler = config_dict.get("run_scheduler", True)

        # For storing predictions and evaluation info
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === TSA Architecture Parameters ===
        self.n_cat_features = config_dict["n_cat_features"]
        self.n_num_features = config_dict["n_num_features"]
        self.n_embedding = config_dict["n_embedding"]
        self.n_engineered_num_features = config_dict.get("n_engineered_num_features", 0)
        self.seq_len = config_dict["seq_len"]
        self.dim_embed = config_dict["dim_embed"]
        self.dim_attn_feedforward = config_dict["dim_attn_feedforward"]

        # Embedding table dimension
        self.dim_embedding_table = self.dim_embed // 2

        # === Create Shared Embedding Table ===
        # CRITICAL: This embedding is shared across all order attention modules
        self.embedding = nn.Embedding(
            self.n_embedding + 2, self.dim_embedding_table, padding_idx=0
        )

        # Separate embedding for engineered features
        self.embedding_engineered = nn.Embedding(
            self.n_engineered_num_features + 1, self.dim_embedding_table, padding_idx=0
        )

        # === Gate Function Embedding ===
        # CRITICAL: Separate smaller embedding for gate (dim=16)
        self.embedding_gate = nn.Embedding(
            self.n_embedding + 2,
            16,  # Smaller dimension for gate
            padding_idx=0,
        )

        # === Order Attention Modules ===
        # CID (customer) order attention
        self.order_attention_cid = OrderAttentionModule(config_dict)
        self.order_attention_cid.embedding = self.embedding

        # CCID (counterparty) order attention
        self.order_attention_ccid = OrderAttentionModule(config_dict)
        self.order_attention_ccid.embedding = self.embedding

        # === Gate Function ===
        # Gate embedding module (simplified order attention)
        # Uses smaller embedding, no MoE, no temporal encoding, single layer
        gate_config = {
            **config_dict,
            "dim_embed": 32,  # Small dimension for gate
            "dim_attn_feedforward": 128,
            "num_heads": 1,
            "n_layers_order": 1,  # Single layer
            "use_moe": False,  # No MoE for gate
            "use_time_seq": False,  # No temporal encoding
        }
        self.gate_emb_cid = OrderAttentionModule(gate_config)
        self.gate_emb_cid.embedding = self.embedding_gate

        self.gate_emb_ccid = OrderAttentionModule(gate_config)
        self.gate_emb_ccid.embedding = self.embedding_gate

        # Gate scoring MLP
        self.gate_score = nn.Sequential(
            nn.Linear(64, 256),  # 32 + 32 = 64
            nn.ReLU(),
            nn.Dropout(config_dict.get("dropout", 0.1)),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

        # === Feature Attention Module ===
        # Uses CID sequence only
        self.feature_attention = FeatureAttentionModule(config_dict)
        self.feature_attention.embedding = self.embedding
        self.feature_attention.embedding_engineered = self.embedding_engineered

        # === Layer Normalization ===
        self.layer_norm = nn.LayerNorm(self.dim_embed)
        self.dropout = nn.Dropout(config_dict.get("dropout", 0.1))

        # === Final Classifier ===
        clf_input_dim = self.dim_embed + self.dim_embedding_table
        self.clf = nn.Sequential(
            nn.Linear(clf_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(config_dict.get("dropout", 0.1)),
            nn.Linear(1024, self.num_classes),
        )

        # === Loss Function ===
        weights = config_dict.get("class_weights", [1.0] * self.num_classes)
        if len(weights) != self.num_classes:
            logger.warning(
                f"class_weights length ({len(weights)}) does not match num_classes ({self.num_classes}). "
                f"Auto-padding with 1.0."
            )
            weights = weights + [1.0] * (self.num_classes - len(weights))

        weights_tensor = torch.tensor(weights[: self.num_classes], dtype=torch.float)
        self.register_buffer("class_weights_tensor", weights_tensor)
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        # Filter config for TensorBoard
        filtered_config = filter_config_for_tensorboard(config_dict)
        self.save_hyperparameters(filtered_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with batch input.

        Args:
            batch: Dictionary containing:
                CID sequence:
                - x_cat_cid: Categorical features [B, L, D_cat]
                - x_num_cid: Numerical features [B, L, D_num]
                - time_seq_cid: Time sequence [B, L, 1]
                - key_padding_mask_cid: Padding mask [B, L] (optional)

                CCID sequence:
                - x_cat_ccid: Categorical features [B, L, D_cat]
                - x_num_ccid: Numerical features [B, L, D_num]
                - time_seq_ccid: Time sequence [B, L, 1]
                - key_padding_mask_ccid: Padding mask [B, L] (optional)

                Shared:
                - x_engineered: Engineered features [B, D_eng] (optional)

        Returns:
            torch.Tensor: Logits [B, n_classes]
        """
        # Extract inputs from batch
        x_cat_cid = batch["x_cat_cid"]
        x_num_cid = batch["x_num_cid"]
        time_seq_cid = batch.get("time_seq_cid", None)
        key_padding_mask_cid = batch.get("key_padding_mask_cid", None)

        x_cat_ccid = batch["x_cat_ccid"]
        x_num_ccid = batch["x_num_ccid"]
        time_seq_ccid = batch.get("time_seq_ccid", None)
        key_padding_mask_ccid = batch.get("key_padding_mask_ccid", None)

        x_engineered = batch.get("x_engineered", None)

        B, L, D = x_cat_cid.shape

        # ===== 1. Gate Function =====
        # Compute gate embeddings for both sequences
        gate_emb_cid = self.gate_emb_cid(
            x_cat=x_cat_cid,
            x_num=x_num_cid,
            time_seq=time_seq_cid,
            key_padding_mask=key_padding_mask_cid,
        )  # [B, 32]

        gate_emb_ccid = self.gate_emb_ccid(
            x_cat=x_cat_ccid,
            x_num=x_num_ccid,
            time_seq=time_seq_ccid,
            key_padding_mask=key_padding_mask_ccid,
        )  # [B, 32]

        # Compute gate scores (softmax weights)
        gate_scores_raw = self.gate_score(
            torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1)
        )  # [B, 2]

        # Clone for modification
        gate_scores = gate_scores_raw.clone()

        # CRITICAL: Mask out CCID contribution if sequence is all padding
        # Check if entire CCID sequence is padded (all True in mask)
        if key_padding_mask_ccid is not None:
            # If all positions are padded (sum == seq_len), set CCID weight to 0
            empty_ccid_mask = torch.sum(key_padding_mask_ccid, dim=1) == L
            gate_scores[empty_ccid_mask, 1] = 0

        # Identify which samples have CCID data (gate score > threshold)
        ccid_keep_idx = (gate_scores[:, 1] > 0.05).nonzero().squeeze(-1)

        # ===== 2. Order Attention =====
        # Always process CID
        x_cid = self.order_attention_cid(
            x_cat=x_cat_cid,
            x_num=x_num_cid,
            time_seq=time_seq_cid,
            key_padding_mask=key_padding_mask_cid,
        )  # [B, dim_embed]

        # Initialize CCID embeddings as zeros
        x_ccid = torch.zeros([B, self.dim_embed], device=x_cid.device)

        # Process CCID only for samples with non-empty CCID
        if len(ccid_keep_idx) > 0:
            x_ccid[ccid_keep_idx, :] = self.order_attention_ccid(
                x_cat=x_cat_ccid[ccid_keep_idx, :, :],
                x_num=x_num_ccid[ccid_keep_idx, :, :],
                time_seq=time_seq_ccid[ccid_keep_idx, :]
                if time_seq_ccid is not None
                else None,
                key_padding_mask=key_padding_mask_ccid[ccid_keep_idx, :]
                if key_padding_mask_ccid is not None
                else None,
            )

        # ===== 3. Gated Ensemble of Order Embeddings =====
        # Weighted combination: gate[0] * CID + gate[1] * CCID
        ensemble_order = torch.einsum(
            "i,ij->ij", gate_scores[:, 0], x_cid
        ) + torch.einsum("i,ij->ij", gate_scores[:, 1], x_ccid)  # [B, dim_embed]

        # Layer normalization
        ensemble_order = self.layer_norm(ensemble_order)

        # ===== 4. Feature Attention =====
        # Process CID features only (current transaction)
        x_feature = self.feature_attention(
            x_cat=x_cat_cid,
            x_num=x_num_cid,
            x_engineered=x_engineered,
        )  # [B, dim_embedding_table]

        # ===== 5. Final Ensemble =====
        ensemble = torch.cat([ensemble_order, x_feature], dim=-1)

        # ===== 6. Classifier =====
        logits = self.clf(ensemble)

        return logits

    def configure_optimizers(self):
        """
        Optimizer + LR scheduler (AdamW + linear warmup).
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.lr, eps=self.adam_epsilon)

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
            )
            if self.run_scheduler
            else get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def run_epoch(self, batch, stage):
        """
        Common epoch logic for train/val/test.

        Args:
            batch: Input batch dictionary
            stage: One of "train", "val", "test", "pred"

        Returns:
            Tuple of (loss, predictions, labels)
        """
        labels = batch.get(self.label_name_transformed) if stage != "pred" else None

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)

            # CrossEntropyLoss expects LongTensor (class indices)
            if self.is_binary:
                labels = labels.long()
            else:
                # Convert one-hot to indices if needed
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1).long()
                else:
                    labels = labels.long()

        # Forward pass
        logits = self(batch)

        # Compute loss
        loss = self.loss_op(logits, labels) if stage != "pred" else None

        # Compute predictions (probabilities)
        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.is_binary else preds

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.run_epoch(batch, "train")
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, "val")
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.pred_lst.extend(preds.detach().cpu().tolist())
        self.label_lst.extend(labels.detach().cpu().tolist())

    def on_validation_epoch_end(self):
        # Sync across GPUs
        device = self.device
        preds = torch.tensor(sum(all_gather(self.pred_lst), []))
        labels = torch.tensor(sum(all_gather(self.label_lst), []))
        metrics = compute_metrics(
            preds.to(device),
            labels.to(device),
            self.metric_choices,
            self.task,
            self.num_classes,
            "val",
        )
        self.log_dict(metrics, prog_bar=True)

    def on_test_epoch_start(self):
        self.id_lst.clear()
        self.pred_lst.clear()
        self.label_lst.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_output_folder = (
            Path(self.model_path) / f"{self.model_class}-{timestamp}"
        )
        self.test_output_folder.mkdir(parents=True, exist_ok=True)

    def test_step(self, batch, batch_idx):
        mode = "test" if self.label_name in batch else "pred"
        self.test_has_label = mode == "test"

        loss, preds, labels = self.run_epoch(batch, mode)
        self.pred_lst.extend(preds.detach().cpu().tolist())
        if labels is not None:
            self.label_lst.extend(labels.detach().cpu().tolist())

        if loss is not None:
            self.log("test_loss", loss, sync_dist=True, prog_bar=True)

        if self.id_name and self.id_name in batch:
            self.id_lst.extend(batch[self.id_name])

    def on_test_epoch_end(self):
        import pandas as pd

        # Save only local results per GPU
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst
        else:
            # Convert multiclass probabilities to JSON strings
            results["prob"] = [json.dumps(p) for p in self.pred_lst]

        if self.test_has_label:
            results[self.label_name] = self.label_lst
        if self.id_name:
            results[self.id_name] = self.id_lst

        df = pd.DataFrame(results)
        test_file = self.test_output_folder / f"test_result_rank{self.global_rank}.tsv"
        df.reset_index(drop=True).to_csv(test_file, sep="\t", index=False)
        logger.info(f"[Rank {self.global_rank}] Saved test results to {test_file}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mode = "test" if self.label_name in batch else "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return (preds, labels) if mode == "test" else preds

    def get_gate_scores(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the gate scores (CID vs CCID weights).

        Useful for interpretability and debugging.

        Args:
            batch: Input batch dictionary

        Returns:
            torch.Tensor: Gate scores [B, 2] (softmax weights)
        """
        x_cat_cid = batch["x_cat_cid"]
        x_num_cid = batch["x_num_cid"]
        time_seq_cid = batch.get("time_seq_cid", None)
        key_padding_mask_cid = batch.get("key_padding_mask_cid", None)

        x_cat_ccid = batch["x_cat_ccid"]
        x_num_ccid = batch["x_num_ccid"]
        time_seq_ccid = batch.get("time_seq_ccid", None)
        key_padding_mask_ccid = batch.get("key_padding_mask_ccid", None)

        # Compute gate embeddings
        gate_emb_cid = self.gate_emb_cid(
            x_cat=x_cat_cid,
            x_num=x_num_cid,
            time_seq=time_seq_cid,
            key_padding_mask=key_padding_mask_cid,
        )

        gate_emb_ccid = self.gate_emb_ccid(
            x_cat=x_cat_ccid,
            x_num=x_num_ccid,
            time_seq=time_seq_ccid,
            key_padding_mask=key_padding_mask_ccid,
        )

        # Compute gate scores
        gate_scores = self.gate_score(torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1))

        # Apply masking
        if key_padding_mask_ccid is not None:
            L = x_cat_ccid.shape[1]
            empty_ccid_mask = torch.sum(key_padding_mask_ccid, dim=1) == L
            gate_scores = gate_scores.clone()
            gate_scores[empty_ccid_mask, 1] = 0

        return gate_scores

    def __repr__(self) -> str:
        return (
            f"TSADualSeq(\n"
            f"  n_cat_features={self.n_cat_features},\n"
            f"  n_num_features={self.n_num_features},\n"
            f"  n_engineered_num_features={self.n_engineered_num_features},\n"
            f"  seq_len={self.seq_len},\n"
            f"  dim_embed={self.dim_embed},\n"
            f"  n_layers_order={self.config.get('n_layers_order', 6)},\n"
            f"  n_layers_feature={self.config.get('n_layers_feature', 6)},\n"
            f"  n_classes={self.num_classes},\n"
            f"  task={self.task},\n"
            f"  has_gating=True\n"
            f")"
        )
