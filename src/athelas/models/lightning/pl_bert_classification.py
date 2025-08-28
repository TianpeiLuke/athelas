#!/usr/bin/env python3
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.optim import AdamW
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from .dist_utils import all_gather
from .pl_model_plots import compute_metrics

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class TextBertClassificationConfig(BaseModel):
    text_name: str
    label_name: str
    tokenizer: str = "bert-base-cased"
    is_binary: bool = True
    num_classes: int = 2
    metric_choices: List[str] = Field(default_factory=lambda: ["accuracy", "f1_score"])
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    lr: float = 2e-5
    run_scheduler: bool = True
    reinit_pooler: bool = False
    reinit_layers: int = 0
    model_path: str
    id_name: Optional[str] = None
    text_input_ids_key: str = "input_ids"
    text_attention_mask_key: str = "attention_mask"

    @field_validator("num_classes")
    @classmethod
    def validate_num_classes(cls, num_classes: int, info: ValidationInfo) -> int:
        """
        In v2, `info.data` holds all the input values.
        """
        is_binary = info.data.get("is_binary", True)
        if is_binary and num_classes != 2:
            raise ValueError("For binary classification, num_classes must be 2")
        if not is_binary and num_classes < 2:
            raise ValueError("For multiclass classification, num_classes must be >= 2")
        return num_classes
    

class TextBertClassification(pl.LightningModule):
    def __init__(self, config: Union[Dict, TextBertClassificationConfig]):
        super().__init__()
        # ensure config object
        if isinstance(config, dict):
            config = TextBertClassificationConfig(**config)
        self.config = config
        # model identifier
        self.model_class = "text_bert_classification"

        # load pretrained BERT for classification
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.config.tokenizer,
            num_labels=self.config.num_classes,
            output_attentions=False,
            return_dict=True,
        )
        self._maybe_reinitialize()

        self.loss_op = None
        self.pred_lst: List = []
        self.label_lst: List = []
        self.id_lst: List = []
        self.test_output_folder: Optional[str] = None
        self.test_has_label: bool = False
        self.save_hyperparameters(logger=False)
        logger.info("Initialized TextBertClassification with model %s", self.model_class)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch[self.config.text_name]
        attention_mask = batch[self.config.text_attention_mask_key]
        labels = batch.get(self.config.label_name)
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _maybe_reinitialize(self):
        if not self.config.reinit_pooler:
            return
        encoder = self.bert.bert
        encoder.pooler.dense.weight.data.normal_(
            mean=0.0, std=encoder.config.initializer_range
        )
        encoder.pooler.dense.bias.data.zero_()
        for p in encoder.pooler.parameters():
            p.requires_grad = True
        if self.config.reinit_layers > 0:
            for layer in encoder.encoder.layer[-self.config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(
                            mean=0.0, std=encoder.config.initializer_range
                        )
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {"params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(grouped_params, lr=self.config.lr, eps=self.config.adam_epsilon)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else None
        scheduler = (get_linear_schedule_with_warmup if self.config.run_scheduler else get_constant_schedule_with_warmup)(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True),
        ]

    def add_loss_op(self, loss_op: Optional[nn.Module] = None):
        if loss_op:
            self.loss_op = loss_op
        else:
            if self.config.is_binary:
                weight = torch.tensor([1.0, self.config.get("pos_weight", 1.0)]).to(self.device)
                self.loss_op = nn.CrossEntropyLoss(weight=weight)
            else:
                self.loss_op = nn.CrossEntropyLoss()

    def run_epoch(self, batch, stage: str):
        input_ids = batch[self.config.text_name]
        attention_mask = batch[self.config.text_attention_mask_key]
        labels = batch.get(self.config.label_name) if stage != "pred" else None
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.config.is_binary else preds
        return loss, preds, labels

    def _shared_step(self, batch, stage: str):
        loss, preds, labels = self.run_epoch(batch, stage)
        if stage != "pred":
            self.pred_lst.extend(preds.detach().cpu().tolist())
            self.label_lst.extend(labels.detach().cpu().tolist())
        return loss, preds, labels

    def training_step(self, batch, batch_idx: int):
        loss, _, _ = self._shared_step(batch, "train")
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        loss, _, _ = self._shared_step(batch, "val")
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()

    def on_validation_epoch_end(self):
        preds = torch.tensor(sum(all_gather(self.pred_lst), []), device=self.device)
        labels = torch.tensor(sum(all_gather(self.label_lst), []), device=self.device)
        metrics = compute_metrics(preds, labels, self.config.metric_choices, self.config.task, self.config.num_classes, "val")
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        self.test_has_label = self.config.label_name in batch
        loss, preds, labels = self._shared_step(batch, "test" if self.test_has_label else "pred")
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        if self.config.id_name and self.config.id_name in batch:
            self.id_lst.extend(batch[self.config.id_name])
        return loss

    def on_test_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()
        self.id_lst.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_output_folder = os.path.join(
            self.config.model_path,
            f"{self.model_class}-{timestamp}",
        )

    @rank_zero_only
    def save_predictions_to_file(self, output_folder, id_name, pred_list, label_list=None):
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame({"prob": pred_list})
        if id_name:
            df[id_name] = sum(all_gather(self.id_lst), [])
        if label_list is not None:
            df["label"] = label_list
        path = os.path.join(output_folder, "test_result.tsv")
        df.to_csv(path, sep="\t", index=False)
        logger.info("Saved test results to %s", path)

    def on_test_epoch_end(self):
        final_preds = sum(all_gather(self.pred_lst), [])
        final_labels = sum(all_gather(self.label_lst), []) if self.test_has_label else None
        if self.test_has_label:
            preds_tensor = torch.tensor(final_preds, device=self.device)
            labels_tensor = torch.tensor(final_labels, device=self.device)
            metrics = compute_metrics(preds_tensor, labels_tensor, self.config.metric_choices, self.config.task, self.config.num_classes, "test")
            self.log_dict(metrics, sync_dist=True, prog_bar=True)
        self.save_predictions_to_file(
            self.test_output_folder,
            self.config.id_name,
            final_preds,
            final_labels if self.test_has_label else None,
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        try:
            _, preds, labels = self.run_epoch(batch, "test")
            return preds, labels
        except KeyError:
            _, preds, _ = self.run_epoch(batch, "pred")
            return preds

    def export_to_onnx(self, save_path, opset_version=11):
        dummy_input = {
            self.config.text_input_ids_key: torch.randint(0, 100, (1, 128)),
            self.config.text_attention_mask_key: torch.ones(1, 128, dtype=torch.long),
        }
        torch.onnx.export(
            self.bert,
            (dummy_input[self.config.text_input_ids_key], dummy_input[self.config.text_attention_mask_key]),
            save_path,
            input_names=[self.config.text_input_ids_key, self.config.text_attention_mask_key],
            output_names=["logits"],
            dynamic_axes={self.config.text_input_ids_key: {0: "batch_size"}, self.config.text_attention_mask_key: {0: "batch_size"}},
            opset_version=opset_version,
        )

    def export_to_torchscript(self, save_path):
        self.bert.eval()
        example_inputs = (
            torch.randint(0, 100, (1, 128)),
            torch.ones(1, 128, dtype=torch.long),
        )
        traced = torch.jit.trace(self.bert, example_inputs)
        traced.save(save_path)

