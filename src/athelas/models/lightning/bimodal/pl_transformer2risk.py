#!/usr/bin/env python3
"""
Transformer2Risk Lightning Module for Bimodal Fraud Detection

PyTorch Lightning module implementing Transformer2Risk architecture for fraud detection
using text (customer names, emails) and tabular features (transaction data).

Architecture matches legacy transformer2risk.py exactly:
- Text encoder: TransformerEncoder with self-attention + feedforward blocks + attention pooling
- Tabular encoder: BatchNorm + 2-layer MLP with LayerNorm
- Fusion classifier: 4x ResidualBlocks + Linear projection

Supports:
- Binary and multiclass classification
- Distributed training (FSDP, DDP)
- ONNX export for inference
- Class weight balancing
- Attention masking for variable-length sequences

Usage:
```python
from names3risk_pytorch.dockers.lightning_models.bimodal import Transformer2Risk
from names3risk_pytorch.dockers.hyperparams import Transformer2RiskHyperparameters

hyperparams = Transformer2RiskHyperparameters(
    n_embed=4000,
    embedding_size=128,
    hidden_size=256,
    n_blocks=8,
    n_heads=8,
    block_size=100,
    dropout_rate=0.2,
    ...
)

model = Transformer2Risk(hyperparams)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader, val_dataloader)
```

References:
- Legacy: projects/names3risk_legacy/transformer2risk.py
- Architecture: slipbox/4_analysis/2026-01-05_names3risk_pytorch_component_correspondence_analysis.md
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import onnx
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from ...pytorch.blocks import TransformerEncoder
from ...pytorch.feedforward import ResidualBlock

# Assuming these utilities exist from other Lightning modules
try:
    from ..utils.dist_utils import all_gather, get_rank
    from ..utils.pl_model_plots import compute_metrics
except ImportError:
    # Fallback for testing without full environment
    def all_gather(data):
        return [data]

    def get_rank():
        return 0

    def compute_metrics(preds, labels, metric_choices, task, num_classes, prefix):
        return {}


# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class Transformer2Risk(pl.LightningModule):
    """
    PyTorch Lightning module for Transformer2Risk bimodal fraud detection.

    Architecture (matches legacy transformer2risk.py exactly):
    1. Text encoder: Transformer with self-attention and pooling
       - Token embedding (vocab_size, embedding_dim)
       - Position embedding (block_size, embedding_dim)
       - 8x Transformer blocks (self-attention + feedforward)
       - Attention pooling (embedding_dim → 1 attention weight)
       - Linear projection (embedding_dim → 2*hidden_dim)
       - Output: (B, 2*hidden_dim)

    2. Tabular encoder: MLP with batch normalization
       - BatchNorm (input_tab_dim)
       - Linear (input_tab_dim → 2*hidden_dim)
       - ReLU + Dropout
       - Linear (2*hidden_dim → 2*hidden_dim)
       - LayerNorm + ReLU + Dropout
       - Output: (B, 2*hidden_dim)

    3. Fusion classifier: Residual blocks + Linear projection
       - 4x ResidualBlock (4*hidden_dim, expansion=1x, post-norm)
       - Each followed by ReLU + Dropout
       - Linear (4*hidden_dim → 1) + Sigmoid
       - Output: (B, 1) binary prediction

    Args:
        hyperparams: Transformer2RiskHyperparameters configuration object

    Batch Input Format:
        - batch["text"]: (B, L) - token IDs (padded, L ≤ block_size)
        - batch["attn_mask"]: (B, L) - attention mask (1=valid, 0=padding)
        - batch["tabular"]: (B, F) - tabular features
        - batch["label"]: (B,) - class labels (LongTensor)

    Training Features:
        - Distributed training support (FSDP, DDP)
        - Mixed precision training (AMP)
        - Class weight balancing
        - Linear warmup + cosine decay scheduler
        - Metrics: accuracy, F1, precision, recall, AUROC, etc.

    ONNX Export:
        - Supports ONNX export for production inference
        - Dynamic batch and sequence dimensions
        - Returns probabilities (not logits)
    """

    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str], torch.FloatTensor]],
    ):
        super().__init__()
        self.config = config
        self.model_class = "transformer2risk"

        # === Field names for batch extraction ===
        self.text_name = config.get("text_name", "text_field")
        self.tab_field_list = config.get("tab_field_list", [])

        # === Task configuration ===
        self.label_name = config.get("label_name", "label")
        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])

        # === Configurable batch key names ===
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get(
            "text_attention_mask_key", "attention_mask"
        )

        # === Training configuration ===
        self.model_path = config.get("model_path", "")
        self.lr = config.get("lr", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.run_scheduler = config.get("run_scheduler", True)

        # === Storage for predictions ===
        self.id_lst, self.pred_lst, self.label_lst = [], [], []
        self.test_output_folder = None
        self.test_has_label = False

        # ============================================================
        # 1. Text Encoder: TransformerEncoder (self-attention + pooling)
        # ============================================================
        n_embed = config.get("n_embed", 4000)
        embedding_size = config.get("embedding_size", 128)
        hidden_size = config.get("hidden_size", 256)
        n_blocks = config.get("n_blocks", 8)
        n_heads = config.get("n_heads", 8)
        max_sen_len = config.get("max_sen_len", 100)
        dropout_rate = config.get("dropout_rate", 0.2)
        # Auto-derive input_tab_dim from tab_field_list if not provided
        input_tab_dim = config.get("input_tab_dim", len(self.tab_field_list))

        self.text_encoder = TransformerEncoder(
            vocab_size=n_embed,
            embedding_dim=embedding_size,
            hidden_dim=hidden_size,
            n_blocks=n_blocks,
            n_heads=n_heads,
            max_seq_len=max_sen_len,
            dropout=dropout_rate,
        )
        # Note: text_encoder already projects to 2*hidden_size internally

        # ============================================================
        # 2. Tabular Encoder: BatchNorm + 2-layer MLP
        # ============================================================
        tab_hidden_dim = 2 * hidden_size
        self.tab_encoder = nn.Sequential(
            nn.BatchNorm1d(input_tab_dim),
            nn.Linear(input_tab_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.LayerNorm(tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ============================================================
        # 3. Fusion Classifier: 4x ResidualBlocks + Linear
        # ============================================================
        fusion_dim = 4 * hidden_size

        # Build classifier with 4 ResidualBlocks (post-norm, 1x expansion)
        classifier_layers = []
        for _ in range(4):
            classifier_layers.extend(
                [
                    ResidualBlock(
                        dim=fusion_dim,
                        expansion_factor=1,  # 1x expansion like legacy
                        dropout=dropout_rate,
                        activation="relu",
                        norm_first=False,  # Post-norm like legacy
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        # Final projection
        classifier_layers.append(nn.Linear(fusion_dim, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # ============================================================
        # 4. Loss Function (CrossEntropyLoss with class weights)
        # ============================================================
        class_weights = config.get("class_weights")
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        else:
            class_weights_tensor = torch.ones(self.num_classes, dtype=torch.float)

        self.register_buffer("class_weights_tensor", class_weights_tensor)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        # Save hyperparameters for Lightning checkpointing
        self.save_hyperparameters(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through Transformer2Risk model.

        Args:
            batch: Dictionary containing:
                - {text_name}_input_ids: (B, 1, L) or (B, L) token IDs (may have chunk dimension)
                - {text_name}_attention_mask: (B, 1, L) or (B, L) attention mask
                - Individual tabular fields as lists (one per field in tab_field_list)
                - label: list of labels

        Returns:
            logits: (B, num_classes) classification logits
        """
        device = next(self.parameters()).device

        # 1. Extract text with field name prefix (from pipeline_dataloader)
        text_input_ids_key = f"{self.text_name}_{self.text_input_ids_key}"
        text_attention_mask_key = f"{self.text_name}_{self.text_attention_mask_key}"

        text_tokens = batch[text_input_ids_key]  # (B, 1, L) or (B, L)
        attn_mask = batch.get(text_attention_mask_key)  # (B, 1, L) or (B, L) or None

        # Squeeze chunk dimension if present (pipeline_dataloader adds it)
        if len(text_tokens.shape) == 3:
            text_tokens = text_tokens.squeeze(1)  # (B, 1, L) → (B, L)
        if attn_mask is not None and len(attn_mask.shape) == 3:
            attn_mask = attn_mask.squeeze(1)  # (B, 1, L) → (B, L)

        # DEBUG: Check token ID ranges before model forward
        if text_tokens.numel() > 0 and logger.isEnabledFor(logging.DEBUG):
            max_token = text_tokens.max().item()
            min_token = text_tokens.min().item()
            n_embed = self.text_encoder.token_embedding.num_embeddings
            logger.debug(
                f"[Transformer2Risk FORWARD] Token ID range: [{min_token}, {max_token}]"
            )
            logger.debug(
                f"[Transformer2Risk FORWARD] Embedding num_embeddings: {n_embed}"
            )
            if max_token >= n_embed:
                logger.error(
                    f"❌ [Transformer2Risk FORWARD] Token {max_token} >= num_embeddings {n_embed}!"
                )

        # 2. Extract and stack individual tabular fields
        for field in self.tab_field_list:
            if field not in batch:
                raise KeyError(f"Missing required tabular field: {field}")

        tab_data = torch.stack(
            [
                torch.tensor(batch[field], device=device, dtype=torch.float32)
                for field in self.tab_field_list
            ],
            dim=1,
        )  # (B, num_features)

        # Text encoding: Transformer + attention pooling + projection
        text_hidden = self.text_encoder(text_tokens, attn_mask)  # (B, 2*H)

        # Tabular encoding: BatchNorm + MLP
        tab_hidden = self.tab_encoder(tab_data)  # (B, 2*H)

        # Fusion: concatenate and classify
        combined = torch.cat([text_hidden, tab_hidden], dim=1)  # (B, 4*H)
        logits = self.classifier(combined)  # (B, num_classes)

        return logits

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Uses AdamW with weight decay + OneCycleLR scheduler (matches legacy).
        Applies weight decay to all parameters except biases and LayerNorm.

        OneCycleLR provides:
        - Linear warmup phase (10% of training by default)
        - Cosine annealing decay phase
        - Automatic per-step scheduling
        - Better convergence than linear warmup alone
        """
        # Separate parameters with/without weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
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

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon
        )

        if self.run_scheduler:
            # OneCycleLR scheduler (matches legacy transformer2risk.py)
            # - max_lr: Peak learning rate (self.lr)
            # - total_steps: Total training steps (automatic via Lightning)
            # - pct_start: Fraction of training for warmup (0.1 = 10%)
            # - anneal_strategy: 'cos' for cosine annealing (default)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup (matches legacy)
                anneal_strategy="cos",
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,  # initial_lr = max_lr / div_factor
                final_div_factor=10000.0,  # min_lr = initial_lr / final_div_factor
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Step per batch (matches legacy)
                },
            }
        else:
            # No scheduler (constant learning rate)
            return optimizer

    def run_epoch(self, batch: Dict[str, torch.Tensor], stage: str):
        """
        Shared logic for train/val/test/predict steps.

        Args:
            batch: Input batch dictionary
            stage: One of "train", "val", "test", "pred"

        Returns:
            loss: Scalar loss (None for pred stage)
            preds: Predictions (full probability distribution for all classes)
            labels: Ground truth labels (None for pred stage)
        """
        # Get labels (if available)
        labels = batch.get(self.label_name) if stage != "pred" else None

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
            labels = labels.long()  # CrossEntropyLoss expects LongTensor

        # Forward pass
        logits = self(batch)  # (B, num_classes)

        # Compute loss
        loss = self.loss_fn(logits, labels) if labels is not None else None

        # Get predictions (probabilities)
        preds = torch.softmax(logits, dim=1)  # (B, num_classes)
        preds = preds[:, 1] if self.is_binary else preds
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _ = self.run_epoch(batch, "train")
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        """Reset accumulators at start of validation."""
        self.pred_lst.clear()
        self.label_lst.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step - accumulate predictions and labels."""
        loss, preds, labels = self.run_epoch(batch, "val")
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        # Accumulate for epoch-level metrics
        self.pred_lst.extend(preds.detach().cpu().tolist())
        self.label_lst.extend(labels.detach().cpu().tolist())

    def on_validation_epoch_end(self):
        """Compute and log validation metrics across all GPUs."""
        # Sync predictions/labels across GPUs
        device = self.device
        preds = torch.tensor(sum(all_gather(self.pred_lst), []))
        labels = torch.tensor(sum(all_gather(self.label_lst), []))

        # Compute metrics
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
        """Setup for test epoch - create output directory."""
        self.id_lst.clear()
        self.pred_lst.clear()
        self.label_lst.clear()

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_output_folder = (
            Path(self.model_path) / f"{self.model_class}-{timestamp}"
        )
        self.test_output_folder.mkdir(parents=True, exist_ok=True)

    def test_step(self, batch, batch_idx):
        """Test step - save predictions per rank."""
        mode = "test" if self.label_name in batch else "pred"
        self.test_has_label = mode == "test"

        loss, preds, labels = self.run_epoch(batch, mode)

        # Accumulate predictions
        self.pred_lst.extend(preds.detach().cpu().tolist())
        if labels is not None:
            self.label_lst.extend(labels.detach().cpu().tolist())

        if loss is not None:
            self.log("test_loss", loss, sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self):
        """Save test results to TSV file (per GPU rank)."""
        import pandas as pd

        # Prepare results
        results = {}
        if self.is_binary:
            results["prob"] = self.pred_lst  # Binary: single probability
        else:
            # Multiclass: serialize probability vectors
            results["prob"] = [json.dumps(p) for p in self.pred_lst]

        if self.test_has_label:
            results[self.label_name] = self.label_lst

        # Save per-rank file
        df = pd.DataFrame(results)
        test_file = self.test_output_folder / f"test_result_rank{self.global_rank}.tsv"
        df.reset_index(drop=True).to_csv(test_file, sep="\t", index=False)
        logger.info(f"[Rank {self.global_rank}] Saved test results to {test_file}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step - return predictions only."""
        mode = "test" if self.label_name in batch else "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return (preds, labels) if mode == "test" else preds

    # ============================================================
    # ONNX Export
    # ============================================================

    def export_to_onnx(
        self,
        save_path: Union[str, Path],
        sample_batch: Dict[str, torch.Tensor],
    ):
        """
        Export model to ONNX format for production inference.

        Args:
            save_path: Path to save ONNX model
            sample_batch: Sample batch for tracing, must contain:
                - "text": (B, L) token IDs
                - "attn_mask": (B, L) attention mask
                - "tabular": (B, F) tabular features

        Example:
            >>> model.export_to_onnx(
            ...     "transformer2risk.onnx",
            ...     {"text": torch.randint(0, 1000, (2, 50)),
            ...      "attn_mask": torch.ones(2, 50, dtype=torch.bool),
            ...      "tabular": torch.randn(2, 100)}
            ... )
        """

        class Transformer2RiskONNXWrapper(nn.Module):
            """
            ONNX-compatible wrapper for Transformer2Risk.

            Returns probabilities (not logits) for production use.
            """

            def __init__(self, model: Transformer2Risk):
                super().__init__()
                self.model = model

            def forward(
                self,
                text_tokens: torch.Tensor,  # (B, L)
                attn_mask: torch.Tensor,  # (B, L)
                tabular: torch.Tensor,  # (B, F)
            ):
                """
                ONNX-compatible forward pass.

                Args:
                    text_tokens: Token IDs (B, L)
                    attn_mask: Attention mask (B, L)
                    tabular: Tabular features (B, F)

                Returns:
                    probs: Class probabilities (B, num_classes)
                """
                # Text encoding (already projects to 2*H internally)
                text_hidden = self.model.text_encoder(text_tokens, attn_mask)

                # Tabular encoding
                tab_hidden = self.model.tab_encoder(tabular.float())

                # Fusion and classification
                combined = torch.cat([text_hidden, tab_hidden], dim=1)
                logits = self.model.classifier(combined)

                # Return probabilities (not logits)
                return torch.softmax(logits, dim=1)

        # Prepare model for export
        self.eval()
        model_to_export = self.module if isinstance(self, FSDP) else self
        model_to_export = model_to_export.to("cpu")
        wrapper = Transformer2RiskONNXWrapper(model_to_export).to("cpu").eval()

        # Construct prefixed keys (same pattern as forward() method)
        text_input_ids_key = f"{self.text_name}_{self.text_input_ids_key}"
        text_attention_mask_key = f"{self.text_name}_{self.text_attention_mask_key}"

        # Prepare inputs using config-driven keys
        text_tokens = sample_batch[text_input_ids_key].to("cpu")

        # Squeeze chunk dimension if present (pipeline_dataloader adds it)
        if len(text_tokens.shape) == 3:
            text_tokens = text_tokens.squeeze(1)  # (B, 1, L) → (B, L)

        attn_mask = sample_batch.get(
            text_attention_mask_key, torch.ones_like(text_tokens).bool()
        ).to("cpu")

        # Squeeze chunk dimension for attention mask if present
        if attn_mask is not None and len(attn_mask.shape) == 3:
            attn_mask = attn_mask.squeeze(1)  # (B, 1, L) → (B, L)

        # Handle tabular data: stack individual fields
        device = torch.device("cpu")
        tabular = torch.stack(
            [
                torch.tensor(sample_batch[field], device=device, dtype=torch.float32)
                for field in self.tab_field_list
            ],
            dim=1,
        )  # (B, num_features)

        # Verify batch consistency
        batch_size = text_tokens.shape[0]
        assert attn_mask.shape[0] == batch_size, "Inconsistent batch size"
        assert tabular.shape[0] == batch_size, "Inconsistent batch size"

        # Define dynamic axes for variable batch/sequence dimensions
        dynamic_axes = {
            "text_tokens": {0: "batch", 1: "sequence"},
            "attn_mask": {0: "batch", 1: "sequence"},
            "tabular": {0: "batch"},
            "probs": {0: "batch"},
        }

        try:
            # Export to ONNX
            torch.onnx.export(
                wrapper,
                (text_tokens, attn_mask, tabular),
                f=save_path,
                input_names=["text_tokens", "attn_mask", "tabular"],
                output_names=["probs"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
            )

            # Verify ONNX model
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"✓ ONNX model exported and verified: {save_path}")
        except Exception as e:
            logger.error(f"✗ ONNX export failed: {e}")
            raise
