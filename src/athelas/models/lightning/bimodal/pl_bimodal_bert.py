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


from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
import onnx

from ..utils.dist_utils import all_gather, get_rank
from ..tabular.pl_tab_ae import TabAE
from ..utils.pl_model_plots import compute_metrics
from ..utils.config_constants import filter_config_for_tensorboard

# Import PyTorch components (relative imports)
from ...pytorch.blocks import (
    BertEncoder,
    create_bert_optimizer_groups,
)
from ...pytorch.embeddings import (
    TabularEmbedding,
    combine_tabular_fields,
)
from ...pytorch.fusion import ConcatenationFusion
from ...pytorch.schedulers import (
    create_bert_scheduler,
    get_scheduler_config_for_lightning,
)

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # <-- THIS LINE IS MISSING

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class BimodalBert(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]],
    ):
        super().__init__()
        self.config = config
        self.model_class = "bimodal_bert"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        # Use configurable key names for text input
        # Use unified naming: field_name + "_" + key (no "_processed_" suffix)
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get(
            "text_attention_mask_key", "attention_mask"
        )
        self.text_name = config["text_name"] + "_" + self.text_input_ids_key
        self.text_attention_mask = (
            config["text_name"] + "_" + self.text_attention_mask_key
        )
        self.tab_field_list = config.get("tab_field_list", None)

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        # ===== transformed label (multiclass case) =======
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

        # For storing predictions and evaluation info
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === Sub-networks using PyTorch components ===
        
        # BERT Text Encoder
        self.text_encoder = BertEncoder(
            model_name=config.get("tokenizer", "bert-base-cased"),
            output_dim=config.get("hidden_common_dim", 256),
            dropout=0.1,
            reinit_pooler=config.get("reinit_pooler", False),
            reinit_layers=config.get("reinit_layers", 0),
            gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        )
        text_dim = self.text_encoder.output_dim
        
        # Tabular Encoder
        if self.tab_field_list:
            # Calculate input dimension from tab fields
            tab_input_dim = len(self.tab_field_list)  # Simplified - adjust if needed
            self.tab_encoder = TabularEmbedding(
                input_dim=tab_input_dim,
                hidden_dim=config.get("hidden_common_dim", 256)
            )
            tab_dim = self.tab_encoder.hidden_dim
        else:
            self.tab_encoder = None
            tab_dim = 0
        
        # === Fusion using ConcatenationFusion ===
        self.fusion = ConcatenationFusion(
            input_dims=[text_dim, tab_dim] if tab_dim > 0 else [text_dim],
            output_dim=self.num_classes,
            use_activation=True  # ReLU before Linear
        )

        # === Loss function ===
        weights = config.get("class_weights", [1.0] * self.num_classes)
        # If weights are shorter than num_classes, pad with 1.0
        if len(weights) != self.num_classes:
            print(
                f"[Warning] class_weights length ({len(weights)}) does not match num_classes ({self.num_classes}). Auto-padding with 1.0."
            )
            weights = weights + [1.0] * (self.num_classes - len(weights))

        weights_tensor = torch.tensor(weights[: self.num_classes], dtype=torch.float)
        self.register_buffer("class_weights_tensor", weights_tensor)
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        # Filter config to only save essential hyperparameters to TensorBoard
        # Excludes runtime artifacts (risk_tables, imputation_dict, etc.)
        filtered_config = filter_config_for_tensorboard(config)
        self.save_hyperparameters(filtered_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using PyTorch components.
        Expects pre-tokenized inputs and tabular data as a dictionary.
        """
        # Extract text inputs
        input_ids = batch[self.text_name]
        attention_mask = batch[self.text_attention_mask]
        
        # Encode text with BertEncoder
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Encode tabular if available
        if self.tab_encoder:
            tab_data = combine_tabular_fields(
                batch, self.tab_field_list, self.device
            )
            tab_features = self.tab_encoder(tab_data)
            # Fuse with ConcatenationFusion
            logits = self.fusion(text_features, tab_features)
        else:
            # Text only
            logits = self.fusion(text_features)
        
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

    # === Export ===
    def export_to_onnx(
        self,
        save_path: Union[str, Path],
        sample_batch: Dict[str, Union[torch.Tensor, List]],
    ):
        class BimodalBertONNXWrapper(nn.Module):
            def __init__(self, model: BimodalBert):
                super().__init__()
                self.model = model
                self.text_key = model.text_name
                self.mask_key = model.text_attention_mask
                self.tab_keys = model.tab_field_list or []

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                *tab_tensors: torch.Tensor,
            ):
                batch = {
                    self.text_key: input_ids,
                    self.mask_key: attention_mask,
                }
                for name, tensor in zip(self.tab_keys, tab_tensors):
                    batch[name] = tensor
                # output probability scores instead of logits
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        # Unwrap from FSDP if needed
        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = BimodalBertONNXWrapper(model_to_export).to("cpu").eval()

        # === Prepare input tensor list ===
        input_names = [self.text_name, self.text_attention_mask]
        input_tensors = []

        # Handle text inputs
        input_ids_tensor = sample_batch.get(self.text_name)
        attention_mask_tensor = sample_batch.get(self.text_attention_mask)

        if not isinstance(input_ids_tensor, torch.Tensor) or not isinstance(
            attention_mask_tensor, torch.Tensor
        ):
            raise ValueError(
                "Both input_ids and attention_mask must be torch.Tensor in sample_batch."
            )

        input_ids_tensor = input_ids_tensor.to("cpu")
        attention_mask_tensor = attention_mask_tensor.to("cpu")

        input_tensors.append(input_ids_tensor)
        input_tensors.append(attention_mask_tensor)

        batch_size = input_ids_tensor.shape[0]

        # Handle tabular inputs
        if self.tab_field_list:
            for field in self.tab_field_list:
                input_names.append(field)
                value = sample_batch.get(field)
                if isinstance(value, torch.Tensor):
                    value = value.to("cpu").float()
                    if value.shape[0] != batch_size:
                        raise ValueError(
                            f"Tensor for field '{field}' has batch size {value.shape[0]} but expected {batch_size}"
                        )
                    input_tensors.append(value)
                elif isinstance(value, list) and all(
                    isinstance(x, (int, float)) for x in value
                ):
                    tensor_val = (
                        torch.tensor(value, dtype=torch.float32)
                        .view(batch_size, -1)
                        .to("cpu")
                    )
                    input_tensors.append(tensor_val)
                else:
                    logger.warning(
                        f"Field '{field}' has unsupported type ({type(value)}); replacing with zeros."
                    )
                    input_tensors.append(
                        torch.zeros((batch_size, 1), dtype=torch.float32).to("cpu")
                    )

        # Final check
        for name, tensor in zip(input_names, input_tensors):
            assert tensor.shape[0] == batch_size, (
                f"Inconsistent batch size for input '{name}': {tensor.shape}"
            )

        dynamic_axes = {}
        for name, tensor in zip(input_names, input_tensors):
            # Assume at least first dimension (batch) is dynamic
            axes = {0: "batch"}
            # Make all further dims dynamic as well
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
