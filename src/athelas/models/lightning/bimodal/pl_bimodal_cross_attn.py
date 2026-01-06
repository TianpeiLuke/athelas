#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import lightning.pytorch as pl

import onnx

from ..utils.dist_utils import all_gather, get_rank
from ..utils.pl_model_plots import compute_metrics
from ..utils.config_constants import filter_config_for_tensorboard

# Import PyTorch components (relative imports)
from ...pytorch.blocks import (
    BertEncoder,
    create_bert_optimizer_groups,
)
from ...pytorch.embeddings import TabularEmbedding, combine_tabular_fields
from ...pytorch.fusion import CrossAttentionFusion
from ...pytorch.schedulers import (
    create_bert_scheduler,
    get_scheduler_config_for_lightning,
)

# ============ Logging Setup ============
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False


class BimodalBertCrossAttn(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]],
    ):
        super().__init__()
        self.config = config
        self.model_class = "bimodal_cross_att"

        # — Core config —
        self.id_name = config.get("id_name")
        self.label_name = config["label_name"]
        # Use unified naming: field_name + "_" + key (no "_processed_" suffix)
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get(
            "text_attention_mask_key", "attention_mask"
        )
        self.text_name = config["text_name"] + "_" + self.text_input_ids_key
        self.text_attention_mask = (
            config["text_name"] + "_" + self.text_attention_mask_key
        )
        self.tab_field_list = config.get("tab_field_list", [])

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        if not self.is_binary and self.num_classes > 2:
            self.label_name_transformed = self.label_name + "_processed"
        else:
            self.label_name_transformed = self.label_name

        self.model_path = config.get("model_path", "")
        self.lr = config.get("lr", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.run_scheduler = config.get("run_scheduler", True)

        # For preds/labels collection
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === BERT Text Encoder using BertEncoder ===
        hidden_dim = config["hidden_common_dim"]
        self.text_encoder = BertEncoder(
            model_name=config.get("tokenizer", "bert-base-cased"),
            output_dim=hidden_dim,
            dropout=0.1,
            reinit_pooler=config.get("reinit_pooler", False),
            reinit_layers=config.get("reinit_layers", 0),
            gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        )
        text_dim = self.text_encoder.output_dim
        
        # === Tabular Encoder using TabularEmbedding ===
        if self.tab_field_list:
            tab_input_dim = len(self.tab_field_list)
            self.tab_encoder = TabularEmbedding(
                input_dim=tab_input_dim,
                hidden_dim=hidden_dim
            )
            tab_dim = self.tab_encoder.hidden_dim
        else:
            self.tab_encoder = None
            tab_dim = 0

        # === Cross-attention fusion using CrossAttentionFusion ===
        num_heads = config.get("num_heads", 4)
        self.cross_att = CrossAttentionFusion(hidden_dim, num_heads)

        # === Final classifier on concat([text,tab]) after fusion ===
        self.final_merge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

        # — Loss function —
        weights = config.get("class_weights", [1.0] * self.num_classes)
        if len(weights) != self.num_classes:
            logger.warning(
                f"class_weights length {len(weights)} != num_classes {self.num_classes}; auto-padding"
            )
            weights = weights + [1.0] * (self.num_classes - len(weights))
        wt = torch.tensor(weights[: self.num_classes], dtype=torch.float)
        self.register_buffer("class_weights_tensor", wt)
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        # Filter config to only save essential hyperparameters to TensorBoard
        # Excludes runtime artifacts (risk_tables, imputation_dict, etc.)
        filtered_config = filter_config_for_tensorboard(config)
        self.save_hyperparameters(filtered_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using PyTorch components.
        """
        # Extract text inputs
        input_ids = batch[self.text_name]
        attention_mask = batch[self.text_attention_mask]
        
        # Encode text with BertEncoder
        text_out = self.text_encoder(input_ids, attention_mask)  # [B, hidden_dim]
        
        # Encode tabular if available
        if self.tab_encoder:
            tab_data = combine_tabular_fields(
                batch, self.tab_field_list, self.device
            )
            tab_out = self.tab_encoder(tab_data)  # [B, hidden_dim]
        else:
            tab_out = torch.zeros((text_out.size(0), 0), device=self.device)
        
        # Unsqueeze to seq-length=1 for attention
        text_seq = text_out.unsqueeze(1)  # [B, 1, H]
        tab_seq = tab_out.unsqueeze(1)  # [B, 1, H]
        
        # Cross-attention fusion using CrossAttentionFusion component
        text_fused, tab_fused = self.cross_att(text_seq, tab_seq)  # Both [B, 1, H]
        
        # Squeeze back to [B, H]
        text_feat = text_fused.squeeze(1)  # [B, H]
        tab_feat = tab_fused.squeeze(1)  # [B, H]
        
        # Concat & classify
        merged = torch.cat([text_feat, tab_feat], dim=1)  # [B, 2H]
        logits = self.final_merge_network(merged)  # [B, num_classes]
        return logits

    def configure_optimizers(self):
        """
        Optimizer + LR scheduler using PyTorch utilities.
        """
        # Use create_bert_optimizer_groups utility
        param_groups = create_bert_optimizer_groups(self, self.weight_decay)
        optimizer = AdamW(param_groups, lr=self.lr, eps=self.adam_epsilon)

        # Use create_bert_scheduler utility
        schedule_type = "linear" if self.run_scheduler else "constant"
        scheduler = create_bert_scheduler(
            optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.warmup_steps,
            schedule_type=schedule_type
        )
        
        # Use get_scheduler_config_for_lightning utility
        scheduler_config = get_scheduler_config_for_lightning(
            scheduler, interval="step"
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def run_epoch(self, batch, stage):
        """Run forward pass and compute loss."""
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

        # Use refactored forward() method
        logits = self(batch)
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
        metrics = compute_metrics(
            preds.to(device),
            labels.to(device),
            self.metric_choices,
            self.task,
            self.num_classes,
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
        import pandas as pd

        # Save only local results per GPU
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst  # Keep "prob" for binary
        else:
            results["prob"] = [
                json.dumps(p) for p in self.pred_lst
            ]  # convert the [num_class] list into a string

        # results = {"prob": self.pred_lst}
        if self.test_has_label:
            results[self.label_name] = self.label_lst
        if self.id_name:
            results[self.id_name] = self.id_lst

        df = pd.DataFrame(results)
        test_file = self.test_output_folder / f"test_result_rank{self.global_rank}.tsv"
        # Fix for pandas 2.x compatibility: reset index before saving
        df.reset_index(drop=True).to_csv(test_file, sep="\t", index=False)
        print(f"[Rank {self.global_rank}] Saved test results to {test_file}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mode = "test" if self.label_name in batch else "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return (preds, labels) if mode == "test" else preds

    def export_to_onnx(
        self,
        save_path: Union[str, Path],
        sample_batch: Dict[str, Union[torch.Tensor, List]],
    ):
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.text_k = model.text_name
                self.mask_k = model.text_attention_mask
                self.tab_keys = model.tab_field_list or []

            def forward(self, input_ids, attention_mask, *tab_tensors):
                batch = {self.text_k: input_ids, self.mask_k: attention_mask}
                for k, t in zip(self.tab_keys, tab_tensors):
                    batch[k] = t
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()
        m = self.module if isinstance(self, FSDP) else self
        wrapper = Wrapper(m.to("cpu")).eval()

        # prepare inputs
        input_names = [self.text_name, self.text_attention_mask]
        input_tensors = [
            sample_batch[self.text_name].to("cpu"),
            sample_batch[self.text_attention_mask].to("cpu"),
        ]
        batch_size = input_tensors[0].shape[0]

        for name in self.tab_field_list:
            input_names.append(name)
            v = sample_batch[name]
            t = (
                v.to("cpu").float()
                if isinstance(v, torch.Tensor)
                else torch.tensor(v, dtype=torch.float32).view(batch_size, -1)
            )
            input_tensors.append(t)

        # dynamic axes for inputs & output
        dynamic_axes = {
            n: {0: "batch", **{i: f"dim_{i}" for i in range(1, t.ndim)}}
            for n, t in zip(input_names, input_tensors)
        }
        dynamic_axes["probs"] = {0: "batch"}

        torch.onnx.export(
            wrapper,
            tuple(input_tensors),
            str(save_path),
            input_names=input_names,
            output_names=["probs"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX export verified at {save_path}")
