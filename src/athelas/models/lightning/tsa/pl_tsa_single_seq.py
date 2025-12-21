#!/usr/bin/env python3
"""
PyTorch Lightning Temporal Self-Attention Single Sequence Model

Recreates OrderFeatureAttentionClassifier functionality as a Lightning module
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
- Legacy: projects/tsa/scripts/models.py (OrderFeatureAttentionClassifier)

Architecture:
    Single sequence (CID only) processing with:
    1. Order Attention: Process full transaction sequence
    2. Feature Attention: Process current transaction features
    3. Optional MLP: Direct numerical feature processing
    4. Ensemble: Concatenate all embeddings
    5. Classifier: Final MLP for prediction
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

    class TSASingleSeqConfig(BaseModel):
        """
        Configuration schema for TSA Single Sequence Model.

        Combines OrderAttentionConfig and FeatureAttentionConfig with
        training and task-specific parameters.
        """

        # Pydantic V2 configuration
        model_config = ConfigDict(extra="allow")

        # Core TSA architecture parameters (shared between order and feature attention)
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
        use_mlp: bool = Field(default=False, description="Use additional MLP branch")

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
    TSASingleSeqConfig = Dict


class TSASingleSeq(pl.LightningModule):
    """
    PyTorch Lightning implementation of TSA Single Sequence Model.

    Processes a single transaction sequence (CID) using order and feature attention.
    Equivalent to legacy OrderFeatureAttentionClassifier.

    Architecture Flow:
        Input: x_cat [B, L, D_cat], x_num [B, L, D_num], time_seq [B, L, 1]
        ↓
        Order Attention: Full sequence → [B, dim_embed]
        Feature Attention: Last transaction → [B, dim_embed//2]
        Optional MLP: Last transaction features → [B, dim_embed//2]
        ↓
        Ensemble: Concatenate → [B, dim_embed + dim_embed//2 (+ dim_embed//2)]
        ↓
        Classifier MLP: → [B, n_classes]

    Critical Implementation Details:
        - Shared embedding table between order and feature attention
        - Separate engineered features embedding
        - Optional MLP branch for direct feature processing
        - Layer normalization after ensemble
        - Standard cross-entropy loss

    Phase 1 Constraints:
        - NO caching or optimizations
        - NO formula modifications
        - NO architectural changes
        - EXACT legacy computation preserved
    """

    def __init__(
        self,
        config: Union[Dict, TSASingleSeqConfig],
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
        self.model_class = "tsa_single_seq"

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
        self.use_mlp = config_dict.get("use_mlp", False)

        # Embedding table dimension
        self.dim_embedding_table = self.dim_embed // 2

        # === Create Shared Embedding Table ===
        # CRITICAL: This embedding is shared between order and feature attention
        self.embedding = nn.Embedding(
            self.n_embedding + 2, self.dim_embedding_table, padding_idx=0
        )

        # Separate embedding for engineered features
        self.embedding_engineered = nn.Embedding(
            self.n_engineered_num_features + 1, self.dim_embedding_table, padding_idx=0
        )

        # === Order Attention Module ===
        self.order_attention = OrderAttentionModule(config_dict)
        # Share the embedding table
        self.order_attention.embedding = self.embedding

        # === Feature Attention Module ===
        self.feature_attention = FeatureAttentionModule(config_dict)
        # Share the embedding tables
        self.feature_attention.embedding = self.embedding
        self.feature_attention.embedding_engineered = self.embedding_engineered

        # === Optional MLP Branch ===
        if self.use_mlp:
            mlp_input_dim = self.n_num_features + self.n_engineered_num_features
            self.MLP = nn.Sequential(
                nn.Linear(mlp_input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(config_dict.get("dropout", 0.1)),
                nn.Linear(1024, self.dim_embedding_table),
            )
            self.layer_norm_mlp = nn.LayerNorm(self.dim_embedding_table)

        # === Final Classifier ===
        # Input dimension depends on whether MLP is used
        if self.use_mlp:
            clf_input_dim = (
                self.dim_embed + self.dim_embedding_table + self.dim_embedding_table
            )
        else:
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
                - x_cat: Categorical features [B, L, D_cat]
                - x_num: Numerical features [B, L, D_num]
                - time_seq: Time sequence [B, L, 1] (if use_time_seq=True)
                - x_engineered: Engineered features [B, D_eng] (optional)

        Returns:
            torch.Tensor: Logits [B, n_classes]
        """
        # Extract inputs from batch
        x_cat = batch["x_cat"]
        x_num = batch["x_num"]
        time_seq = batch.get("time_seq", None)
        x_engineered = batch.get("x_engineered", None)

        # Order attention: process full sequence
        x_order = self.order_attention(
            x_cat=x_cat,
            x_num=x_num,
            time_seq=time_seq,
        )  # [B, dim_embed]

        # Feature attention: process last transaction
        x_feature = self.feature_attention(
            x_cat=x_cat,
            x_num=x_num,
            x_engineered=x_engineered,
        )  # [B, dim_embedding_table]

        # Optional MLP branch
        if self.use_mlp:
            # Concatenate last numerical features with engineered features
            x_num_last = x_num[:, -1, :]  # [B, D_num]
            if x_engineered is not None:
                mlp_input = torch.cat([x_num_last, x_engineered], dim=-1)
            else:
                mlp_input = x_num_last

            x_mlp = self.MLP(mlp_input)
            x_mlp = self.layer_norm_mlp(x_mlp)  # [B, dim_embedding_table]

            # Ensemble: order + feature + mlp
            ensemble = torch.cat([x_order, x_feature, x_mlp], dim=-1)
        else:
            # Ensemble: order + feature
            ensemble = torch.cat([x_order, x_feature], dim=-1)

        # Final classifier
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

    def get_ensemble_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the ensemble embedding (before final classifier).

        Useful for feature extraction or transfer learning.

        Args:
            batch: Input batch dictionary

        Returns:
            torch.Tensor: Ensemble embedding
        """
        # Extract inputs
        x_cat = batch["x_cat"]
        x_num = batch["x_num"]
        time_seq = batch.get("time_seq", None)
        x_engineered = batch.get("x_engineered", None)

        # Get embeddings
        x_order = self.order_attention(x_cat, x_num, time_seq)
        x_feature = self.feature_attention(x_cat, x_num, x_engineered)

        if self.use_mlp:
            x_num_last = x_num[:, -1, :]
            mlp_input = (
                torch.cat([x_num_last, x_engineered], dim=-1)
                if x_engineered is not None
                else x_num_last
            )
            x_mlp = self.MLP(mlp_input)
            x_mlp = self.layer_norm_mlp(x_mlp)
            ensemble = torch.cat([x_order, x_feature, x_mlp], dim=-1)
        else:
            ensemble = torch.cat([x_order, x_feature], dim=-1)

        return ensemble

    def __repr__(self) -> str:
        return (
            f"TSASingleSeq(\n"
            f"  n_cat_features={self.n_cat_features},\n"
            f"  n_num_features={self.n_num_features},\n"
            f"  n_engineered_num_features={self.n_engineered_num_features},\n"
            f"  seq_len={self.seq_len},\n"
            f"  dim_embed={self.dim_embed},\n"
            f"  n_layers_order={self.config.get('n_layers_order', 6)},\n"
            f"  n_layers_feature={self.config.get('n_layers_feature', 6)},\n"
            f"  use_mlp={self.use_mlp},\n"
            f"  n_classes={self.num_classes},\n"
            f"  task={self.task}\n"
            f")"
        )
