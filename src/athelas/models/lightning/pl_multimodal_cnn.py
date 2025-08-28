import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import onnx
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .pl_tab_ae import TabAE
from .pl_text_cnn import TextCNN
from .dist_utils import all_gather
from .pl_model_plots import compute_metrics


class MultimodalCNN(pl.LightningModule):
    def __init__(self, config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]], vocab_size: int, word_embeddings: torch.FloatTensor):
        super().__init__()
        self.config = config
        self.model_class = "multimodal_cnn"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_name = config["text_name"] + "_processed_" + self.text_input_ids_key
        self.tab_field_list = config.get("tab_field_list", None)

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        self.model_path = config.get("model_path", ".")
        self.lr = config.get("lr", 0.02)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.run_scheduler = config.get("run_scheduler", True)

        # For storing predictions
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === Subnetworks ===
        self.text_subnetwork = TextCNN(config, vocab_size, word_embeddings)
        self.tab_subnetwork = TabAE(config) if self.tab_field_list else None

        text_dim = self.text_subnetwork.output_text_dim
        tab_dim = self.tab_subnetwork.output_tab_dim if self.tab_subnetwork else 0

        self.final_merge_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(tab_dim + text_dim, self.num_classes),
        )

        # === Loss Function ===
        weights = config.get("class_weights", [1.0] * self.num_classes)
        if len(weights) != self.num_classes:
            weights += [1.0] * (self.num_classes - len(weights))
        self.register_buffer("class_weights_tensor", torch.tensor(weights[:self.num_classes], dtype=torch.float))
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tab_data = self.tab_subnetwork.combine_tab_data(batch) if self.tab_subnetwork else None
        return self._forward_impl(batch, tab_data)

    def _forward_impl(self, batch, tab_data):
        input_ids = batch[self.text_name]
        text_out = self.text_subnetwork(input_ids)
        tab_out = self.tab_subnetwork(tab_data.float()) if tab_data is not None else torch.zeros((text_out.size(0), 0), device=self.device)
        combined = torch.cat([text_out, tab_out], dim=1)
        return self.final_merge_network(combined)

    def run_epoch(self, batch, stage):
        labels = batch.get(self.label_name) if stage != "pred" else None

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
            if self.is_binary:
                labels = labels.long()
            else:
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1).long()
                else:
                    labels = labels.long()

        logits = self.forward(batch)
        loss = self.loss_op(logits, labels) if labels is not None else None
        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.is_binary else preds
        return loss, preds, labels

    def configure_optimizers(self):
        optimizer_type = self.config.get("optimizer_type", "SGD")
        if optimizer_type == "Adam":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, eps=self.adam_epsilon, weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.config.get("momentum", 0.9), weight_decay=self.weight_decay)

        return optimizer

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
        device = self.device
        preds = torch.tensor(sum(all_gather(self.pred_lst), []))
        labels = torch.tensor(sum(all_gather(self.label_lst), []))
        metrics = compute_metrics(preds.to(device), labels.to(device), self.metric_choices, self.task, self.num_classes, "val")
        self.log_dict(metrics, prog_bar=True)

    def on_test_epoch_start(self):
        self.id_lst.clear()
        self.pred_lst.clear()
        self.label_lst.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_output_folder = Path(self.model_path) / f"{self.model_class}-{timestamp}"
        self.test_output_folder.mkdir(parents=True, exist_ok=True)

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

    def on_test_epoch_end(self):
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst
        else:
            results["prob"] = [json.dumps(p) for p in self.pred_lst]

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
        return preds if mode == "pred" else (preds, labels)

    def export_to_onnx(self, save_path: Union[str, Path], sample_batch: Dict[str, Union[torch.Tensor, List]]):
        class MultimodalCNNONNXWrapper(nn.Module):
            def __init__(self, model: MultimodalCNN):
                super().__init__()
                self.model = model
                self.text_key = model.text_name
                self.tab_keys = model.tab_field_list or []

            def forward(self, input_ids: torch.Tensor, *tab_tensors: torch.Tensor):
                batch = {self.text_key: input_ids}
                for name, tensor in zip(self.tab_keys, tab_tensors):
                    batch[name] = tensor
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = MultimodalCNNONNXWrapper(model_to_export).to("cpu").eval()

        input_names = [self.text_name]
        input_tensors = [sample_batch[self.text_name].to("cpu")]

        if self.tab_field_list:
            for field in self.tab_field_list:
                input_names.append(field)
                input_tensors.append(sample_batch[field].to("cpu"))

        dynamic_axes = {name: {0: "batch"} for name in input_names}

        torch.onnx.export(
            wrapper,
            tuple(input_tensors),
            f=save_path,
            input_names=input_names,
            output_names=["probs"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        print(f"ONNX model exported and verified at {save_path}")

