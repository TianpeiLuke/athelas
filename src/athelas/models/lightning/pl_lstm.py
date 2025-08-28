#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import onnx

# =================== Logging Setup =================================
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

# --- Internal Utilities --------------------------------------------
from .dist_utils import all_gather
from .pl_model_plots import compute_metrics


class TextLSTM(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str]]],
        vocab_size: int,
        word_embeddings: torch.FloatTensor,
    ):
        super().__init__()
        self.config = config
        self.model_class = "text_lstm"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_name = f"{config['text_name']}_processed_{self.text_input_ids_key}"

        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        self.hidden_dimension = config.get("hidden_common_dim", 100)
        self.num_layers = config.get("num_layers", 1)
        self.dropout_keep = config.get("dropout_keep", 0.5)
        self.max_sen_len = config.get("max_sen_len", 512)

        self.model_path = config.get("model_path", ".")
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # === Embedding ===
        self.embed_size = word_embeddings.shape[1]
        if vocab_size != word_embeddings.shape[0]:
            raise ValueError("Mismatch in vocab size and embedding shape")
        self.embeddings = nn.Embedding(vocab_size, self.embed_size)
        self.embeddings.weight = nn.Parameter(
            word_embeddings,
            requires_grad=config.get("is_embeddings_trainable", True),
        )

        # === LSTM + Linear ===
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_dimension,
            num_layers=self.num_layers,
            dropout=self.dropout_keep if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * self.hidden_dimension, self.num_classes)

        # === Loss ===
        class_weights = config.get("class_weights", [1.0] * self.num_classes)
        self.register_buffer(
            "class_weights_tensor",
            torch.tensor(class_weights, dtype=torch.float),
        )
        self.loss_op = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch[self.text_name]
        x = self.embeddings(input_ids)
        lstm_out, _ = self.lstm(x)
        out_fwd = lstm_out[:, -1, : self.hidden_dimension]
        out_rev = lstm_out[:, 0, self.hidden_dimension :]
        out_combined = torch.cat((out_fwd, out_rev), dim=1)
        return self.fc(out_combined)

    def run_epoch(self, batch, stage):
        input_ids = batch[self.text_name]
        labels = batch.get(self.label_name) if stage != "pred" else None

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
            labels = labels.long()

        logits = self(batch)
        loss = self.loss_op(logits, labels) if labels is not None else None

        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.is_binary else preds
        return loss, preds, labels

    def configure_optimizers(self):
        optimizer_type = self.config.get("optimizer", "SGD")
        lr = self.config.get("lr", 0.02)
        momentum = self.config.get("momentum", 0.9)

        if optimizer_type == "Adam":
            return optim.AdamW(self.parameters(), lr=lr)
        return optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.run_epoch(batch, "train")
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
        # sync across GPUs
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
        results = {"prob": self.pred_lst}
        if self.test_has_label:
            results["label"] = self.label_lst
        if self.id_name:
            results[self.id_name] = self.id_lst

        df = pd.DataFrame(results)
        test_file = self.test_output_folder / f"test_result_rank{self.global_rank}.tsv"
        df.to_csv(test_file, sep="\t", index=False)
        logger.info(f"[Rank {self.global_rank}] Saved test results to {test_file}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mode = "test" if self.label_name in batch else "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return (preds, labels) if mode == "test" else preds

    def export_to_onnx(self, save_path: Union[str, Path], sample_batch: Dict[str, Union[torch.Tensor, List]]):
        class TextLSTMONNXWrapper(nn.Module):
            def __init__(self, model: 'TextLSTM'):
                super().__init__()
                self.model = model
                self.text_key = model.text_name

            def forward(self, input_ids: torch.Tensor):
                batch = {self.text_key: input_ids}
                logits = self.model(batch)
                return nn.functional.softmax(logits, dim=1)

        self.eval()
        model_to_export = self.module if hasattr(self, 'module') else self
        model_to_export = model_to_export.to("cpu")
        wrapper = TextLSTMONNXWrapper(model_to_export).to("cpu").eval()

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
            logger.info(f"ONNX model exported and verified at {save_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
