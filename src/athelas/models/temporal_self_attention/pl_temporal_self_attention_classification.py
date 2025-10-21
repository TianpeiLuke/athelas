#!/usr/bin/env python3
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


class TemporalSelfAttentionClassification(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]],
    ):
        super().__init__()
        self.config = config
        self.model_class = "temporal_self_attention_classification"
        self.model_type = "single_sequence"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config.get("label_name", "label")
        
        # TSA-specific field names
        self.x_cat_key = config.get("x_cat_key", "x_seq_cat")
        self.x_num_key = config.get("x_num_key", "x_seq_num")
        self.x_engineered_key = config.get("x_engineered_key", "x_engineered")
        self.time_seq_key = config.get("time_seq_key", "time_to_last")

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

        # === TSA Model Components ===
        # Sequential attention for temporal sequence processing
        self.sequential_attention = SequentialAttentionModule(config)
        
        # Feature attention for current transaction processing
        self.feature_attention = FeatureAttentionModule(config)
        
        # Final classifier
        dim_embed = 2 * config["dim_embedding_table"]
        embedding_table_dim = config["dim_embedding_table"]
        use_mlp = config.get("use_mlp", 0)
        
        if use_mlp:
            # MLP for additional numerical features
            self.mlp = nn.Sequential(
                nn.Linear(config["n_num_features"] + config.get("n_engineered_num_features", 0), 1024),
                nn.ReLU(),
                nn.Dropout(config.get("dropout", 0.1)),
                nn.Linear(1024, embedding_table_dim)
            )
            self.layer_norm_mlp = nn.LayerNorm(embedding_table_dim)
            
            # Final classifier with MLP
            self.classifier = nn.Sequential(
                nn.Linear(dim_embed + embedding_table_dim + embedding_table_dim, 1024),
                nn.ReLU(),
                nn.Dropout(config.get("dropout", 0.1)),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.mlp = None
            self.layer_norm_mlp = None
            
            # Final classifier without MLP
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
        Expects TSA-formatted inputs as a dictionary.
        """
        return self._forward_impl(batch)

    def _forward_impl(self, batch) -> torch.Tensor:
        """Forward implementation using refactored TSA components."""
        # Extract TSA inputs from batch
        x_cat = batch[self.x_cat_key].float()
        x_num = batch[self.x_num_key].float()
        x_engineered = batch.get(self.x_engineered_key, torch.zeros(x_cat.size(0), 0, device=x_cat.device)).float()
        
        # Handle time sequence
        if self.time_seq_key in batch:
            time_seq = batch[self.time_seq_key].float()
            if time_seq.dim() == 2:  # Add feature dimension if needed
                time_seq = time_seq.unsqueeze(-1)
        else:
            time_seq = torch.zeros(x_cat.size(0), x_cat.size(1), 1, device=x_cat.device)

        # Generate attention masks
        attn_mask = None
        key_padding_mask = batch.get("key_padding_mask", None)
        if key_padding_mask is None and self.config.get("use_key_padding_mask", True):
            # Generate padding mask: True where all features are 0 (padded)
            key_padding_mask = (x_cat == 0).all(dim=-1)  # [B, L]

        # Sequential attention - temporal sequence processing
        sequential_output = self.sequential_attention(
            x_cat=x_cat,
            x_num=x_num,
            time_seq=time_seq,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )  # [B, dim_embed]
        
        # Feature attention - current transaction processing
        feature_output = self.feature_attention(
            x_cat=x_cat,
            x_num=x_num,
            x_engineered=x_engineered
        )  # [B, embedding_table_dim]
        
        # Optional MLP processing
        if self.mlp is not None:
            # Combine numerical and engineered features for MLP
            mlp_input = torch.cat([x_num[:, -1, :], x_engineered], dim=-1)
            mlp_output = self.mlp(mlp_input)
            mlp_output = self.layer_norm_mlp(mlp_output)
            
            # Combine all outputs
            ensemble = torch.cat([sequential_output, feature_output, mlp_output], dim=-1)
        else:
            # Combine sequential and feature outputs
            ensemble = torch.cat([sequential_output, feature_output], dim=-1)
        
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
        class TSAONNXWrapper(nn.Module):
            def __init__(self, model: TemporalSelfAttentionClassification):
                super().__init__()
                self.model = model
                self.x_cat_key = model.x_cat_key
                self.x_num_key = model.x_num_key
                self.x_engineered_key = model.x_engineered_key
                self.time_seq_key = model.time_seq_key

            def forward(
                self,
                x_cat: torch.Tensor,
                x_num: torch.Tensor,
                x_engineered: torch.Tensor,
                time_seq: torch.Tensor,
            ):
                batch = {
                    self.x_cat_key: x_cat,
                    self.x_num_key: x_num,
                    self.x_engineered_key: x_engineered,
                    self.time_seq_key: time_seq,
                }
                # output probability scores instead of logits
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        # Unwrap from FSDP if needed
        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = TSAONNXWrapper(model_to_export).to("cpu").eval()

        # === Prepare input tensor list ===
        input_names = [self.x_cat_key, self.x_num_key, self.x_engineered_key, self.time_seq_key]
        input_tensors = []

        # Handle TSA inputs
        x_cat_tensor = sample_batch.get(self.x_cat_key)
        x_num_tensor = sample_batch.get(self.x_num_key)
        x_engineered_tensor = sample_batch.get(self.x_engineered_key)
        time_seq_tensor = sample_batch.get(self.time_seq_key)

        if not all(isinstance(t, torch.Tensor) for t in [x_cat_tensor, x_num_tensor]):
            raise ValueError("x_cat and x_num must be torch.Tensor in sample_batch.")

        x_cat_tensor = x_cat_tensor.to("cpu").float()
        x_num_tensor = x_num_tensor.to("cpu").float()
        
        batch_size = x_cat_tensor.shape[0]

        # Handle engineered features
        if x_engineered_tensor is None:
            x_engineered_tensor = torch.zeros(batch_size, 0).to("cpu").float()
        else:
            x_engineered_tensor = x_engineered_tensor.to("cpu").float()

        # Handle time sequence
        if time_seq_tensor is None:
            time_seq_tensor = torch.zeros(batch_size, x_cat_tensor.shape[1], 1).to("cpu").float()
        else:
            time_seq_tensor = time_seq_tensor.to("cpu").float()
            if time_seq_tensor.dim() == 2:
                time_seq_tensor = time_seq_tensor.unsqueeze(-1)

        input_tensors = [x_cat_tensor, x_num_tensor, x_engineered_tensor, time_seq_tensor]

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
