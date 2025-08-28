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

from .dist_utils import all_gather, get_rank
from .pl_tab_ae import TabAE  # Or TabularEmbeddingModule
from .pl_bert import TextBertBase
from .pl_model_plots import compute_metrics
# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # <-- THIS LINE IS MISSING

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class MultimodalBert(pl.LightningModule):
    def __init__(self, config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]]):
        super().__init__()
        self.config = config
        self.model_class = "multimodal_bert"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        # Use configurable key names for text input
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get("text_attention_mask_key", "attention_mask")
        self.text_name = config["text_name"] + "_processed_" + self.text_input_ids_key
        self.text_attention_mask = config["text_name"] + "_processed_" + self.text_attention_mask_key
        self.tab_field_list = config.get("tab_field_list", None)

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])
        
        # ===== transformed label (multiclass case) =======
        if not self.is_binary and self.num_classes > 2:
            self.label_name_transformed = self.label_name + '_processed'
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

        # === Sub-networks ===
        self.tab_subnetwork = TabAE(config) if self.tab_field_list else None  # Or TabularEmbeddingModule
        tab_dim = self.tab_subnetwork.output_tab_dim if self.tab_subnetwork else 0

        self.text_subnetwork = TextBertBase(config)
        text_dim = self.text_subnetwork.output_text_dim

        # === Final classifier ===
        self.final_merge_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(tab_dim + text_dim, self.num_classes),
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

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with batch input.
        Expects pre-tokenized inputs and tabular data as a dictionary.
        """
        tab_data = self.tab_subnetwork.combine_tab_data(batch) if self.tab_subnetwork else None
        return self._forward_impl(batch, tab_data)

    def _forward_impl(self, batch, tab_data) -> torch.Tensor:
        device = next(self.parameters()).device
        # Use configurable key names to access text data]
        text_out = self.text_subnetwork(batch)

        if tab_data is not None:
            tab_data = tab_data.float()  # Ensure tab_data is on the correct device
            tab_out = self.tab_subnetwork(tab_data)  # [B, D]
        else:
            tab_out = torch.zeros((text_out.size(0), 0), device=device)

        combined = torch.cat([text_out, tab_out], dim=1)
        return self.final_merge_network(combined)

    def configure_optimizers(self):
        """
        Optimizer + LR scheduler (AdamW + linear warmup)
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.lr, eps=self.adam_epsilon)

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
            )
            if self.run_scheduler
            else get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def run_epoch(self, batch, stage):
        #labels = batch.get(self.label_name) if stage != "pred" else None
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

        tab_data = self.tab_subnetwork.combine_tab_data(batch) if self.tab_subnetwork else None

        logits = self._forward_impl(batch, tab_data)
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
            preds.to(device), labels.to(device), self.metric_choices, self.task, self.num_classes, "val"
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
        self.test_output_folder = Path(self.model_path) / f"{self.model_class}-{timestamp}"
        self.test_output_folder.mkdir(parents=True, exist_ok=True)


    def on_test_epoch_end(self):
        import pandas as pd

        # Save only local results per GPU
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst  # Keep "prob" for binary
        else:
            results["prob"] = [json.dumps(p) for p in self.pred_lst] # convert the [num_class] list into a string
        
        #results = {"prob": self.pred_lst}
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
    def export_to_onnx(self, save_path: Union[str, Path], sample_batch: Dict[str, Union[torch.Tensor, List]]):
        class MultimodalBertONNXWrapper(nn.Module):
            def __init__(self, model: MultimodalBert):
                super().__init__()
                self.model = model
                self.text_key = model.text_name
                self.mask_key = model.text_attention_mask
                self.tab_keys = model.tab_field_list or []

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *tab_tensors: torch.Tensor):
                batch = {
                    self.text_key: input_ids,
                    self.mask_key: attention_mask,
                }
                for name, tensor in zip(self.tab_keys, tab_tensors):
                    batch[name] = tensor
                #output probability scores instead of logits
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        # Unwrap from FSDP if needed
        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = MultimodalBertONNXWrapper(model_to_export).to("cpu").eval()

        # === Prepare input tensor list ===
        input_names = [self.text_name, self.text_attention_mask]
        input_tensors = []

        # Handle text inputs
        input_ids_tensor = sample_batch.get(self.text_name)
        attention_mask_tensor = sample_batch.get(self.text_attention_mask)

        if not isinstance(input_ids_tensor, torch.Tensor) or not isinstance(attention_mask_tensor, torch.Tensor):
            raise ValueError("Both input_ids and attention_mask must be torch.Tensor in sample_batch.")

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
                        raise ValueError(f"Tensor for field '{field}' has batch size {value.shape[0]} but expected {batch_size}")
                    input_tensors.append(value)
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    tensor_val = torch.tensor(value, dtype=torch.float32).view(batch_size, -1).to("cpu")
                    input_tensors.append(tensor_val)
                else:
                    logger.warning(f"Field '{field}' has unsupported type ({type(value)}); replacing with zeros.")
                    input_tensors.append(torch.zeros((batch_size, 1), dtype=torch.float32).to("cpu"))

        # Final check
        for name, tensor in zip(input_names, input_tensors):
            assert tensor.shape[0] == batch_size, f"Inconsistent batch size for input '{name}': {tensor.shape}"

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



    def export_to_torchscript(self, save_path: Union[str, Path], sample_batch: Dict[str, Union[torch.Tensor, List]]):
        self.eval()

        # Clean the sample batch: remove list of strings, convert list of numbers to tensors
        sample_batch_tensorized = {}
        for k, v in sample_batch.items():
            if isinstance(v, list):
                if all(isinstance(x, str) for x in v):
                    continue  # Skip string list (e.g., dialogue)
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
            scripted_model = torch.jit.trace(model_to_export, (sample_batch_tensorized,))
        except Exception as e:
            logger.warning(f"Trace failed: {e}. Trying script...")
            scripted_model = torch.jit.script(model_to_export)
        scripted_model.save(str(save_path))
        logger.info(f"TorchScript model saved to: {save_path}")
