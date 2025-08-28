import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import pandas as pd
import onnx

from .pl_model_plots import compute_metrics
from .dist_utils import all_gather


class TextCNN(pl.LightningModule):
    def __init__(self, config: Dict[str, Union[int, float, str, bool, List[str]]], vocab_size: int, word_embeddings: torch.FloatTensor):
        super().__init__()
        self.config = config
        self.model_class = "text_cnn"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_name = config["text_name"] + "_processed_" + self.text_input_ids_key

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        self.dropout_keep = config.get("dropout_keep", 0.5)
        self.max_sen_len = config.get("max_sen_len", 512)
        self.kernel_size = config.get("kernel_size", [3, 5, 7])
        self.num_layers = config.get("num_layers", 2)
        self.num_channels = config.get("num_channels", [100, 100])
        self.hidden_common_dim = config.get("hidden_common_dim", 100)

        self.model_path = config.get("model_path", ".")
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === Embedding Layer ===
        self.embed_size = word_embeddings.shape[1]
        if vocab_size != word_embeddings.shape[0]:
            raise ValueError("Mismatch between vocab size and embedding matrix")
        self.embeddings = nn.Embedding(vocab_size, self.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=config.get("is_embeddings_trainable", True))

        # === Convolutional Layers ===
        self.conv_output_dims = {k: self._compute_conv_output_dim(k, self.max_sen_len, self.num_layers) for k in self.kernel_size}
        self.conv_input_dims = {k: self._compute_conv_input_dim(self.embed_size, self.num_channels, self.num_layers) for k in self.kernel_size}

        self.convs = nn.ModuleList([
            self._build_conv_layers(k, self.num_layers, self.num_channels, self.conv_input_dims[k], self.conv_output_dims[k])
            for k in self.kernel_size
        ])

        self.output_text_dim = self.hidden_common_dim
        self.network = self._build_text_subnetwork(len(self.kernel_size), self.num_channels, self.output_text_dim)

        # === Loss Function ===
        class_weights = torch.tensor(config.get("class_weights", [1.0] * self.num_classes))
        self.register_buffer("class_weights_tensor", class_weights)
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        self.save_hyperparameters()

    def _compute_conv_output_dim(self, kernel_size, input_dim, num_layers):
        for _ in range(num_layers):
            input_dim = input_dim - kernel_size + 1
            if _ < num_layers - 1:
                input_dim = (input_dim - kernel_size) // kernel_size + 1
        return input_dim

    def _compute_conv_input_dim(self, embed_size, num_channels, num_layers):
        return [embed_size] + num_channels[:-1]

    def _build_conv_layers(self, kernel_size, num_layers, num_channels, input_dims, output_dim):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dims[i], num_channels[i], kernel_size))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.MaxPool1d(kernel_size))
            else:
                layers.append(nn.MaxPool1d(output_dim))
        return nn.Sequential(*layers)

    def _build_text_subnetwork(self, num_kernels, num_channels, output_text_dim):
        return nn.Sequential(
            nn.Dropout(self.dropout_keep),
            nn.Linear(num_channels[-1] * num_kernels, output_text_dim)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = x.permute(0, 2, 1)
        conv_outs = [conv(x).squeeze(2) for conv in self.convs]
        features = torch.cat(conv_outs, dim=1)
        return self.network(features)

    def run_epoch(self, batch, stage):
        input_ids = batch[self.text_name]
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

        logits = self(input_ids)
        loss = self.loss_op(logits, labels) if labels is not None else None

        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.is_binary else preds

        return loss, preds, labels

    def configure_optimizers(self):
        optimizer_type = self.config.get("optimizer_type", "SGD")
        lr = self.config.get("lr", 0.02)
        momentum = self.config.get("momentum", 0.9)

        if optimizer_type == "Adam":
            return optim.Adam(self.parameters(), lr=lr)
        return optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, "train")
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, "val")
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
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
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
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
        class TextCNNONNXWrapper(nn.Module):
            def __init__(self, model: 'TextCNN'):
                super().__init__()
                self.model = model

            def forward(self, input_ids: torch.Tensor):
                logits = self.model(input_ids)
                return nn.functional.softmax(logits, dim=1)

        self.eval()

        model_to_export = self.module if hasattr(self, 'module') else self
        model_to_export = model_to_export.to("cpu")
        wrapper = TextCNNONNXWrapper(model_to_export).to("cpu").eval()

        input_ids_tensor = sample_batch.get(self.text_name)
        if not isinstance(input_ids_tensor, torch.Tensor):
            raise ValueError(f"Sample batch must provide {self.text_name} as a torch.Tensor.")
        input_ids_tensor = input_ids_tensor.to("cpu")

        input_names = [self.text_name]
        output_names = ["probs"]

        dynamic_axes = {
            self.text_name: {0: "batch", 1: "seq_len"},
            "probs": {0: "batch"},
        }

        try:
            torch.onnx.export(
                wrapper,
                (input_ids_tensor,),
                f=save_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported and verified at {save_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
