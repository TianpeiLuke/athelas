#!/usr/bin/env python3
"""
Dual Sequence TSA Lightning Module

This module implements the PyTorch Lightning version of dual-sequence
temporal self-attention processing with adaptive gate function.

Key Features:
- Dual sequence processing (primary and auxiliary sequences)
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

from ..utils.dist_utils import all_gather, get_rank
from .pl_tsa_metrics import compute_tsa_metrics
from .pl_schedulers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

# Import TSA components from pytorch submodule (Phase 1 & 2 complete)
from ...pytorch.blocks import (
    OrderAttentionModule,
    FeatureAttentionModule,
    DualSequenceGate,
)
from ...pytorch.losses import FocalLoss, CyclicalFocalLoss
from ...pytorch.fusion import WeightedEnsembleFusion

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class DualSequenceTSA(pl.LightningModule):
    """
    Dual Sequence Temporal Self-Attention Lightning Module.

    This module implements dual-sequence temporal self-attention with
    adaptive gate function for dynamic sequence importance weighting.
    Suitable for any application requiring dual-sequence modeling with
    selective processing based on sequence relevance.
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
        self.seq1_cat_key = config.get("seq1_cat_key", "x_seq_cat_cid")
        self.seq1_num_key = config.get("seq1_num_key", "x_seq_num_cid")
        self.seq2_cat_key = config.get("seq2_cat_key", "x_seq_cat_ccid")
        self.seq2_num_key = config.get("seq2_num_key", "x_seq_num_ccid")
        self.engineered_key = config.get("engineered_key", "x_engineered")
        self.seq1_time_key = config.get("seq1_time_key", "time_seq_cid")
        self.seq2_time_key = config.get("seq2_time_key", "time_seq_ccid")

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get(
            "metric_choices", ["accuracy", "f1_score", "auroc", "pr_auc"]
        )

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
        self.gate_function = DualSequenceGate(config)

        # Order attention modules for both sequences (imported from pytorch/blocks/)
        self.sequential_attention_seq1 = OrderAttentionModule(config)
        self.sequential_attention_seq2 = OrderAttentionModule(config)

        # Feature attention for current observation processing
        self.feature_attention = FeatureAttentionModule(config)

        # Dimensions
        dim_embed = 2 * config["dim_embedding_table"]
        embedding_table_dim = config["dim_embedding_table"]

        # Weighted ensemble fusion for dual sequences
        self.ensemble_fusion = WeightedEnsembleFusion(
            embed_dim=dim_embed,
            num_sources=2,
            normalize=True,
        )

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
                reduction=config.get("loss_reduction", "mean"),
            )
        elif loss_type == "CyclicalFocalLoss":
            self.loss_op = CyclicalFocalLoss(
                alpha=config.get("loss_alpha", 0.25),
                gamma_min=config.get("loss_gamma_min", 1.0),
                gamma_max=config.get("loss_gamma_max", 3.0),
                cycle_length=config.get("loss_cycle_length", 1000),
                reduction=config.get("loss_reduction", "mean"),
            )
        else:
            # Default CrossEntropyLoss
            weights = config.get("class_weights", [1.0] * self.num_classes)
            if len(weights) != self.num_classes:
                print(
                    f"[Warning] class_weights length ({len(weights)}) does not match num_classes ({self.num_classes}). Auto-padding with 1.0."
                )
                weights = weights + [1.0] * (self.num_classes - len(weights))

            weights_tensor = torch.tensor(
                weights[: self.num_classes], dtype=torch.float
            )
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
        seq1_cat = batch[self.seq1_cat_key].float()
        seq1_num = batch[self.seq1_num_key].float()
        seq2_cat = batch[self.seq2_cat_key].float()
        seq2_num = batch[self.seq2_num_key].float()
        engineered = batch.get(
            self.engineered_key,
            torch.zeros(seq1_cat.size(0), 0, device=seq1_cat.device),
        ).float()

        B = seq1_cat.size(0)

        # Handle time sequences
        if self.seq1_time_key in batch:
            time_seq1 = batch[self.seq1_time_key].float()
            if time_seq1.dim() == 2:
                time_seq1 = time_seq1.unsqueeze(-1)
        else:
            time_seq1 = torch.zeros(
                seq1_cat.size(0),
                seq1_cat.size(1),
                1,
                device=seq1_cat.device,
            )

        if self.seq2_time_key in batch:
            time_seq2 = batch[self.seq2_time_key].float()
            if time_seq2.dim() == 2:
                time_seq2 = time_seq2.unsqueeze(-1)
        else:
            time_seq2 = torch.zeros(
                seq2_cat.size(0),
                seq2_cat.size(1),
                1,
                device=seq2_cat.device,
            )

        # Generate attention masks
        attn_mask = None
        key_padding_mask_seq1 = batch.get("key_padding_mask_seq1", None)
        key_padding_mask_seq2 = batch.get("key_padding_mask_seq2", None)

        if key_padding_mask_seq1 is None and self.config.get(
            "use_key_padding_mask", True
        ):
            key_padding_mask_seq1 = (seq1_cat == 0).all(dim=-1)  # [B, L]

        if key_padding_mask_seq2 is None and self.config.get(
            "use_key_padding_mask", True
        ):
            key_padding_mask_seq2 = (seq2_cat == 0).all(dim=-1)  # [B, L]

        # Gate function - compute sequence importance weights
        gate_scores, seq2_keep_idx = self.gate_function(
            seq1_cat=seq1_cat,
            seq1_num=seq1_num,
            time_seq1=time_seq1,
            seq2_cat=seq2_cat,
            seq2_num=seq2_num,
            time_seq2=time_seq2,
            key_padding_mask_seq1=key_padding_mask_seq1,
            key_padding_mask_seq2=key_padding_mask_seq2,
        )

        # Sequential attention - Sequence 1 (always processed)
        seq1_embed = self.sequential_attention_seq1(
            x_cat=seq1_cat,
            x_num=seq1_num,
            time_seq=time_seq1,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_seq1,
        )  # [B, dim_embed]

        # Sequential attention - Sequence 2 (conditionally processed)
        seq2_embed = torch.zeros([B, seq1_embed.size(-1)], device=seq1_cat.device)

        if len(seq2_keep_idx) > 0:
            seq2_embed[seq2_keep_idx, :] = self.sequential_attention_seq2(
                x_cat=seq2_cat[seq2_keep_idx, :, :],
                x_num=seq2_num[seq2_keep_idx, :, :],
                time_seq=time_seq2[seq2_keep_idx, :, :]
                if time_seq2 is not None
                else None,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask_seq2[seq2_keep_idx, :]
                if key_padding_mask_seq2 is not None
                else None,
            )

        # Feature attention - current observation processing (uses sequence 1 as reference)
        feature_output = self.feature_attention(
            x_cat=seq1_cat, x_num=seq1_num, x_engineered=engineered
        )  # [B, embedding_table_dim]

        # Ensemble order embeddings using weighted fusion
        ensemble_order = self.ensemble_fusion(
            embeddings=[seq1_embed, seq2_embed], weights=gate_scores
        )

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
                    labels = (
                        labels.long()
                    )  # Multiclass: Expects LongTensor (class indices)

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
                self.seq1_cat_key = model.seq1_cat_key
                self.seq1_num_key = model.seq1_num_key
                self.seq2_cat_key = model.seq2_cat_key
                self.seq2_num_key = model.seq2_num_key
                self.engineered_key = model.engineered_key
                self.seq1_time_key = model.seq1_time_key
                self.seq2_time_key = model.seq2_time_key

            def forward(
                self,
                seq1_cat: torch.Tensor,
                seq1_num: torch.Tensor,
                seq1_time: torch.Tensor,
                seq2_cat: torch.Tensor,
                seq2_num: torch.Tensor,
                seq2_time: torch.Tensor,
                engineered: torch.Tensor,
            ):
                batch = {
                    self.seq1_cat_key: seq1_cat,
                    self.seq1_num_key: seq1_num,
                    self.seq1_time_key: seq1_time,
                    self.seq2_cat_key: seq2_cat,
                    self.seq2_num_key: seq2_num,
                    self.seq2_time_key: seq2_time,
                    self.engineered_key: engineered,
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
            self.seq1_cat_key,
            self.seq1_num_key,
            self.seq1_time_key,
            self.seq2_cat_key,
            self.seq2_num_key,
            self.seq2_time_key,
            self.engineered_key,
        ]
        input_tensors = []

        # Handle dual sequence inputs
        seq1_cat_tensor = sample_batch.get(self.seq1_cat_key)
        seq1_num_tensor = sample_batch.get(self.seq1_num_key)
        seq2_cat_tensor = sample_batch.get(self.seq2_cat_key)
        seq2_num_tensor = sample_batch.get(self.seq2_num_key)
        engineered_tensor = sample_batch.get(self.engineered_key)
        time_seq1_tensor = sample_batch.get(self.seq1_time_key)
        time_seq2_tensor = sample_batch.get(self.seq2_time_key)

        if not all(
            isinstance(t, torch.Tensor)
            for t in [
                seq1_cat_tensor,
                seq1_num_tensor,
                seq2_cat_tensor,
                seq2_num_tensor,
            ]
        ):
            raise ValueError(
                "Sequence 1 and Sequence 2 tensors must be torch.Tensor in sample_batch."
            )

        # Convert to CPU and float
        seq1_cat_tensor = seq1_cat_tensor.to("cpu").float()
        seq1_num_tensor = seq1_num_tensor.to("cpu").float()
        seq2_cat_tensor = seq2_cat_tensor.to("cpu").float()
        seq2_num_tensor = seq2_num_tensor.to("cpu").float()

        batch_size = seq1_cat_tensor.shape[0]
        seq_len = seq1_cat_tensor.shape[1]

        # Handle engineered features
        if engineered_tensor is None:
            engineered_tensor = torch.zeros(batch_size, 0).to("cpu").float()
        else:
            engineered_tensor = engineered_tensor.to("cpu").float()

        # Handle time sequences
        if time_seq1_tensor is None:
            time_seq1_tensor = torch.zeros(batch_size, seq_len, 1).to("cpu").float()
        else:
            time_seq1_tensor = time_seq1_tensor.to("cpu").float()
            if time_seq1_tensor.dim() == 2:
                time_seq1_tensor = time_seq1_tensor.unsqueeze(-1)

        if time_seq2_tensor is None:
            time_seq2_tensor = torch.zeros(batch_size, seq_len, 1).to("cpu").float()
        else:
            time_seq2_tensor = time_seq2_tensor.to("cpu").float()
            if time_seq2_tensor.dim() == 2:
                time_seq2_tensor = time_seq2_tensor.unsqueeze(-1)

        input_tensors = [
            seq1_cat_tensor,
            seq1_num_tensor,
            time_seq1_tensor,
            seq2_cat_tensor,
            seq2_num_tensor,
            time_seq2_tensor,
            engineered_tensor,
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
