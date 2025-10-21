#!/usr/bin/env python3
"""
Dual Sequence TSA Lightning Module

This module implements the PyTorch Lightning version of the TwoSeqMoEOrderFeatureAttentionClassifier
for dual-sequence temporal self-attention processing with gate function.

Key Features:
- Dual sequence processing (CID and CCID sequences)
- Gate function for dynamic sequence importance weighting
- Lightning integration with full training/validation/test pipeline
- ONNX export and TorchScript support
- Distributed training compatibility
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import lightning.pytorch as pl

import onnx

from .dist_utils import all_gather, get_rank
from .pl_tsa_metrics import compute_tsa_metrics
from .pl_tsa_losses import FocalLoss, CyclicalFocalLoss
from .pl_sequential_attention import SequentialAttentionModule
from .pl_feature_attention import FeatureAttentionModule
from .pl_schedulers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class GateFunction(nn.Module):
    """
    Gate function for dual sequence importance weighting.
    
    This module implements the gate function from TwoSeqMoEOrderFeatureAttentionClassifier
    that dynamically weights the importance of CID vs CCID sequences.
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
        
        # Gate embedding table (smaller than main embedding)
        self.embedding_gate = nn.Embedding(
            self.n_embedding + 2, 
            self.gate_embedding_dim, 
            padding_idx=0
        )
        
        # Gate attention module (simplified configuration)
        gate_config = config.copy()
        gate_config.update({
            "dim_embedding_table": self.gate_embedding_dim,
            "dim_attn_feedforward": 128,
            "num_heads": 1,
            "n_layers_order": 1,
            "use_moe": False,
            "num_experts": 1,
            "use_time_seq": False,
            "return_seq": False
        })
        
        self.gate_attention = SequentialAttentionModule(gate_config)
        
        # Override the embedding in gate attention
        self.gate_attention.embedding = self.embedding_gate
        
        # Gate score computation
        gate_input_dim = 2 * (2 * self.gate_embedding_dim)  # CID + CCID embeddings
        self.gate_score = nn.Sequential(
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Gate threshold for CCID filtering
        self.gate_threshold = config.get("gate_threshold", 0.05)
    
    def forward(
        self,
        x_seq_cat_cid: torch.Tensor,
        x_seq_num_cid: torch.Tensor,
        time_seq_cid: Optional[torch.Tensor],
        x_seq_cat_ccid: torch.Tensor,
        x_seq_num_ccid: torch.Tensor,
        time_seq_ccid: Optional[torch.Tensor],
        key_padding_mask_cid: Optional[torch.Tensor] = None,
        key_padding_mask_ccid: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate scores for CID and CCID sequences.
        
        Args:
            x_seq_cat_cid: CID categorical features [B, L, n_cat_features]
            x_seq_num_cid: CID numerical features [B, L, n_num_features]
            time_seq_cid: CID time sequence [B, L, 1]
            x_seq_cat_ccid: CCID categorical features [B, L, n_cat_features]
            x_seq_num_ccid: CCID numerical features [B, L, n_num_features]
            time_seq_ccid: CCID time sequence [B, L, 1]
            key_padding_mask_cid: CID padding mask [B, L]
            key_padding_mask_ccid: CCID padding mask [B, L]
            
        Returns:
            gate_scores: Gate scores [B, 2] (CID, CCID)
            ccid_keep_idx: Indices where CCID should be processed
        """
        # Compute gate embeddings for both sequences
        gate_emb_cid = self.gate_attention(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, 
            attn_mask=None, key_padding_mask=key_padding_mask_cid
        )
        
        gate_emb_ccid = self.gate_attention(
            x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid,
            attn_mask=None, key_padding_mask=key_padding_mask_ccid
        )
        
        # Compute raw gate scores
        gate_input = torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1)
        gate_scores_raw = self.gate_score(gate_input)
        
        # Apply CCID filtering based on padding
        gate_scores = gate_scores_raw.clone()
        
        # Set CCID gate score to 0 for sequences that are fully padded
        if key_padding_mask_ccid is not None:
            # Check if CCID sequence is fully padded (all positions are padded)
            fully_padded_ccid = (key_padding_mask_ccid.sum(dim=1) >= (self.seq_len - 1))
            gate_scores[fully_padded_ccid, 1] = 0.0
            # Renormalize gate scores
            gate_scores = F.softmax(gate_scores, dim=1)
        
        # Find indices where CCID should be processed (gate score > threshold)
        ccid_keep_idx = (gate_scores[:, 1] > self.gate_threshold).nonzero().squeeze(-1)
        
        return gate_scores, ccid_keep_idx


class DualSequenceTSA(pl.LightningModule):
    """
    Dual Sequence Temporal Self-Attention Lightning Module.
    
    This module implements the Lightning version of TwoSeqMoEOrderFeatureAttentionClassifier
    with dual sequence processing (CID and CCID) and gate function for dynamic weighting.
    """
    
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]],
    ):
        super().__init__()
        self.config = config
        self.model_class = "dual_sequence_tsa"
        self.model_type = "dual_sequence"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config.get("label_name", "label")
        
        # Dual sequence field names
        self.x_cid_cat_key = config.get("x_cid_cat_key", "x_seq_cat_cid")
        self.x_cid_num_key = config.get("x_cid_num_key", "x_seq_num_cid")
        self.x_ccid_cat_key = config.get("x_ccid_cat_key", "x_seq_cat_ccid")
        self.x_ccid_num_key = config.get("x_ccid_num_key", "x_seq_num_ccid")
        self.x_engineered_key = config.get("x_engineered_key", "x_engineered")
        self.time_cid_key = config.get("time_cid_key", "time_seq_cid")
        self.time_ccid_key = config.get("time_ccid_key", "time_seq_ccid")

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score", "auroc", "pr_auc"])

        # ===== transformed label (multiclass case) =======
        if not self.is_binary and self.num_classes > 2:
            self.label_name_transformed = self.label_name + "_processed"
        else:
            self.label_name_transformed = self.label_name

        self.model_path = config.get("model_path", "")
        self.lr = config.get("lr", 1e-5)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.run_scheduler = config.get("run_scheduler", True)

        # For storing predictions and evaluation info
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === Dual Sequence TSA Components ===
        # Gate function for sequence importance weighting
        self.gate_function = GateFunction(config)
        
        # Sequential attention modules for both sequences
        self.sequential_attention_cid = SequentialAttentionModule(config)
        self.sequential_attention_ccid = SequentialAttentionModule(config)
        
        # Feature attention for current transaction processing
        self.feature_attention = FeatureAttentionModule(config)
        
        # Dimensions
        dim_embed = 2 * config["dim_embedding_table"]
        embedding_table_dim = config["dim_embedding_table"]
        
        # Layer normalization for ensemble
        self.layer_norm_ensemble = nn.LayerNorm(dim_embed)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim_embed + embedding_table_dim, 1024),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(1024, self.num_classes),
        )

        # === Loss function ===
        loss_type = config.get("loss", "CrossEntropyLoss")
        if loss_type == "FocalLoss":
            self.loss_op = FocalLoss(
                alpha=config.get("loss_alpha", 0.25),
                gamma=config.get("loss_gamma", 2.0),
                reduction=config.get("loss_reduction", "mean")
            )
        elif loss_type == "CyclicalFocalLoss":
            self.loss_op = CyclicalFocalLoss(
                alpha=config.get("loss_alpha", 0.25),
                gamma_min=config.get("loss_gamma_min", 1.0),
                gamma_max=config.get("loss_gamma_max", 3.0),
                cycle_length=config.get("loss_cycle_length", 1000),
                reduction=config.get("loss_reduction", "mean")
            )
        else:
            # Default CrossEntropyLoss
            weights = config.get("class_weights", [1.0] * self.num_classes)
            if len(weights) != self.num_classes:
                print(
                    f"[Warning] class_weights length ({len(weights)}) does not match num_classes ({self.num_classes}). Auto-padding with 1.0."
                )
                weights = weights + [1.0] * (self.num_classes - len(weights))

            weights_tensor = torch.tensor(weights[: self.num_classes], dtype=torch.float)
            self.register_buffer("class_weights_tensor", weights_tensor)
            self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with batch input.
        Expects dual-sequence TSA-formatted inputs as a dictionary.
        """
        return self._forward_impl(batch)

    def _forward_impl(self, batch) -> torch.Tensor:
        """Forward implementation using dual sequence TSA components."""
        # Extract dual sequence inputs from batch
        x_seq_cat_cid = batch[self.x_cid_cat_key].float()
        x_seq_num_cid = batch[self.x_cid_num_key].float()
        x_seq_cat_ccid = batch[self.x_ccid_cat_key].float()
        x_seq_num_ccid = batch[self.x_ccid_num_key].float()
        x_engineered = batch.get(self.x_engineered_key, torch.zeros(x_seq_cat_cid.size(0), 0, device=x_seq_cat_cid.device)).float()
        
        B = x_seq_cat_cid.size(0)
        
        # Handle time sequences
        if self.time_cid_key in batch:
            time_seq_cid = batch[self.time_cid_key].float()
            if time_seq_cid.dim() == 2:
                time_seq_cid = time_seq_cid.unsqueeze(-1)
        else:
            time_seq_cid = torch.zeros(x_seq_cat_cid.size(0), x_seq_cat_cid.size(1), 1, device=x_seq_cat_cid.device)
            
        if self.time_ccid_key in batch:
            time_seq_ccid = batch[self.time_ccid_key].float()
            if time_seq_ccid.dim() == 2:
                time_seq_ccid = time_seq_ccid.unsqueeze(-1)
        else:
            time_seq_ccid = torch.zeros(x_seq_cat_ccid.size(0), x_seq_cat_ccid.size(1), 1, device=x_seq_cat_ccid.device)

        # Generate attention masks
        attn_mask = None
        key_padding_mask_cid = batch.get("key_padding_mask_cid", None)
        key_padding_mask_ccid = batch.get("key_padding_mask_ccid", None)
        
        if key_padding_mask_cid is None and self.config.get("use_key_padding_mask", True):
            key_padding_mask_cid = (x_seq_cat_cid == 0).all(dim=-1)  # [B, L]
            
        if key_padding_mask_ccid is None and self.config.get("use_key_padding_mask", True):
            key_padding_mask_ccid = (x_seq_cat_ccid == 0).all(dim=-1)  # [B, L]

        # Gate function - compute sequence importance weights
        gate_scores, ccid_keep_idx = self.gate_function(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid,
            x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid,
            key_padding_mask_cid, key_padding_mask_ccid
        )

        # Sequential attention - CID sequence (always processed)
        x_cid = self.sequential_attention_cid(
            x_cat=x_seq_cat_cid,
            x_num=x_seq_num_cid,
            time_seq=time_seq_cid,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_cid
        )  # [B, dim_embed]
        
        # Sequential attention - CCID sequence (conditionally processed)
        x_ccid = torch.zeros([B, x_cid.size(-1)], device=x_seq_cat_cid.device)
        
        if len(ccid_keep_idx) > 0:
            x_ccid[ccid_keep_idx, :] = self.sequential_attention_ccid(
                x_cat=x_seq_cat_ccid[ccid_keep_idx, :, :],
                x_num=x_seq_num_ccid[ccid_keep_idx, :, :],
                time_seq=time_seq_ccid[ccid_keep_idx, :, :] if time_seq_ccid is not None else None,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask_ccid[ccid_keep_idx, :] if key_padding_mask_ccid is not None else None
            )
        
        # Feature attention - current transaction processing (uses CID sequence as reference)
        feature_output = self.feature_attention(
            x_cat=x_seq_cat_cid,
            x_num=x_seq_num_cid,
            x_engineered=x_engineered
        )  # [B, embedding_table_dim]
        
        # Ensemble order embeddings using gate scores
        ensemble_order = torch.einsum("i,ij->ij", gate_scores[:, 0], x_cid) + torch.einsum("i,ij->ij", gate_scores[:, 1], x_ccid)
        ensemble_order = self.layer_norm_ensemble(ensemble_order)
        
        # Combine order and feature outputs
        ensemble = torch.cat([ensemble_order, feature_output], dim=-1)
        
        # Final classification
        scores = self.classifier(ensemble)
        
        return scores

    def configure_optimizers(self):
        """
        Optimizer + LR scheduler (AdamW + linear warmup)
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
        """Run epoch for training/validation/testing."""
        labels = batch.get(self.label_name_transformed) if stage != "pred" else None

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)

            # Important: CrossEntropyLoss always expects LongTensor (class index)
            if self.is_binary:
                labels = labels.long()  # Binary: Expects LongTensor (class indices)
            else:
                # Multiclass: Check if labels are one-hot encoded
                if labels.dim() > 1:  # Assuming one-hot is 2D
                    labels = labels.argmax(dim=1).long()  # Convert one-hot to indices
                else:
                    labels = labels.long()  # Multiclass: Expects LongTensor (class indices)

        logits = self._forward_impl(batch)
        loss = self.loss_op(logits, labels) if stage != "pred" else None

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
        metrics = compute_tsa_metrics(
            preds.to(device),
            labels.to(device),
            self.metric_choices,
            self.task,
            "val",
        )
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        mode = "test" if self.label_name in batch else "pred"
        self.test_has_label = mode == "test"

        loss, preds, labels = self.run_epoch(batch, mode)
        self.pred_lst.extend(preds.detach().cpu().tolist())
        if labels is not None:
            self.label_lst.extend(labels.detach().cpu().tolist())
        if loss is not None:
            self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        if self.id_name:
            self.id_lst.extend(batch[self.id_name])

    def on_test_epoch_start(self):
        self.id_lst.clear()
        self.pred_lst.clear()
        self.label_lst.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_output_folder = (
            Path(self.model_path) / f"{self.model_class}-{timestamp}"
        )
        self.test_output_folder.mkdir(parents=True, exist_ok=True)

    def on_test_epoch_end(self):
        # Save only local results per GPU
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst  # Keep "prob" for binary
        else:
            results["prob"] = [
                json.dumps(p) for p in self.pred_lst
            ]  # convert the [num_class] list into a string

        if self.test_has_label:
            results["label"] = self.label_lst
        if self.id_name:
            results[self.id_name] = self.id_lst

        df = pd.DataFrame(results)
        test_file = self.test_output_folder / f"test_result_rank{self.global_rank}.tsv"
        df.to_csv(test_file, sep="\t", index=False)
        print(f"[Rank {self.global_rank}] Saved test results to {test_file}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mode = "test" if self.label_name in batch else "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return (preds, labels) if mode == "test" else preds

    # === Export ===
    def export_to_onnx(
        self,
        save_path: Union[str, Path],
        sample_batch: Dict[str, Union[torch.Tensor, List]],
    ):
        class DualTSAONNXWrapper(nn.Module):
            def __init__(self, model: DualSequenceTSA):
                super().__init__()
                self.model = model
                self.x_cid_cat_key = model.x_cid_cat_key
                self.x_cid_num_key = model.x_cid_num_key
                self.x_ccid_cat_key = model.x_ccid_cat_key
                self.x_ccid_num_key = model.x_ccid_num_key
                self.x_engineered_key = model.x_engineered_key
                self.time_cid_key = model.time_cid_key
                self.time_ccid_key = model.time_ccid_key

            def forward(
                self,
                x_seq_cat_cid: torch.Tensor,
                x_seq_num_cid: torch.Tensor,
                time_seq_cid: torch.Tensor,
                x_seq_cat_ccid: torch.Tensor,
                x_seq_num_ccid: torch.Tensor,
                time_seq_ccid: torch.Tensor,
                x_engineered: torch.Tensor,
            ):
                batch = {
                    self.x_cid_cat_key: x_seq_cat_cid,
                    self.x_cid_num_key: x_seq_num_cid,
                    self.time_cid_key: time_seq_cid,
                    self.x_ccid_cat_key: x_seq_cat_ccid,
                    self.x_ccid_num_key: x_seq_num_ccid,
                    self.time_ccid_key: time_seq_ccid,
                    self.x_engineered_key: x_engineered,
                }
                # output probability scores instead of logits
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        # Unwrap from FSDP if needed
        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = DualTSAONNXWrapper(model_to_export).to("cpu").eval()

        # === Prepare input tensor list ===
        input_names = [
            self.x_cid_cat_key, self.x_cid_num_key, self.time_cid_key,
            self.x_ccid_cat_key, self.x_ccid_num_key, self.time_ccid_key,
            self.x_engineered_key
        ]
        input_tensors = []

        # Handle dual sequence inputs
        x_seq_cat_cid_tensor = sample_batch.get(self.x_cid_cat_key)
        x_seq_num_cid_tensor = sample_batch.get(self.x_cid_num_key)
        x_seq_cat_ccid_tensor = sample_batch.get(self.x_ccid_cat_key)
        x_seq_num_ccid_tensor = sample_batch.get(self.x_ccid_num_key)
        x_engineered_tensor = sample_batch.get(self.x_engineered_key)
        time_seq_cid_tensor = sample_batch.get(self.time_cid_key)
        time_seq_ccid_tensor = sample_batch.get(self.time_ccid_key)

        if not all(isinstance(t, torch.Tensor) for t in [
            x_seq_cat_cid_tensor, x_seq_num_cid_tensor,
            x_seq_cat_ccid_tensor, x_seq_num_ccid_tensor
        ]):
            raise ValueError("CID and CCID sequence tensors must be torch.Tensor in sample_batch.")

        # Convert to CPU and float
        x_seq_cat_cid_tensor = x_seq_cat_cid_tensor.to("cpu").float()
        x_seq_num_cid_tensor = x_seq_num_cid_tensor.to("cpu").float()
        x_seq_cat_ccid_tensor = x_seq_cat_ccid_tensor.to("cpu").float()
        x_seq_num_ccid_tensor = x_seq_num_ccid_tensor.to("cpu").float()
        
        batch_size = x_seq_cat_cid_tensor.shape[0]
        seq_len = x_seq_cat_cid_tensor.shape[1]

        # Handle engineered features
        if x_engineered_tensor is None:
            x_engineered_tensor = torch.zeros(batch_size, 0).to("cpu").float()
        else:
            x_engineered_tensor = x_engineered_tensor.to("cpu").float()

        # Handle time sequences
        if time_seq_cid_tensor is None:
            time_seq_cid_tensor = torch.zeros(batch_size, seq_len, 1).to("cpu").float()
        else:
            time_seq_cid_tensor = time_seq_cid_tensor.to("cpu").float()
            if time_seq_cid_tensor.dim() == 2:
                time_seq_cid_tensor = time_seq_cid_tensor.unsqueeze(-1)

        if time_seq_ccid_tensor is None:
            time_seq_ccid_tensor = torch.zeros(batch_size, seq_len, 1).to("cpu").float()
        else:
            time_seq_ccid_tensor = time_seq_ccid_tensor.to("cpu").float()
            if time_seq_ccid_tensor.dim() == 2:
                time_seq_ccid_tensor = time_seq_ccid_tensor.unsqueeze(-1)

        input_tensors = [
            x_seq_cat_cid_tensor, x_seq_num_cid_tensor, time_seq_cid_tensor,
            x_seq_cat_ccid_tensor, x_seq_num_ccid_tensor, time_seq_ccid_tensor,
            x_engineered_tensor
        ]

        # Dynamic axes
        dynamic_axes = {}
        for name, tensor in zip(input_names, input_tensors):
            axes = {0: "batch"}
            for i in range(1, tensor.dim()):
                axes[i] = f"dim_{i}"
            dynamic_axes[name] = axes

        try:
            torch.onnx.export(
                wrapper,
                tuple(input_tensors),
                f=save_path,
                input_names=input_names,
                output_names=["probs"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model exported and verified at {save_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    def export_to_torchscript(
        self,
        save_path: Union[str, Path],
        sample_batch: Dict[str, Union[torch.Tensor, List]],
    ):
        self.eval()

        # Clean the sample batch: remove list of strings, convert list of numbers to tensors
        sample_batch_tensorized = {}
        for k, v in sample_batch.items():
            if isinstance(v, list):
                if all(isinstance(x, str) for x in v):
                    continue  # Skip string list
                sample_batch_tensorized[k] = torch.tensor(v).to("cpu")
            elif isinstance(v, torch.Tensor):
                sample_batch_tensorized[k] = v.to("cpu")

        # Unwrap from FSDP if needed
        model_to_export = self
        if isinstance(self, FSDP):
            model_to_export = self.module  # Unwrap the actual LightningModule

        model_to_export = model_to_export.to("cpu")
        model_to_export.eval()

        # Trace the forward method using the cleaned sample batch
        try:
            scripted_model = torch.jit.trace(
                model_to_export, (sample_batch_tensorized,)
            )
        except Exception as e:
            logger.warning(f"Trace failed: {e}. Trying script...")
            scripted_model = torch.jit.script(model_to_export)
        scripted_model.save(str(save_path))
        logger.info(f"TorchScript model saved to: {save_path}")
