# Training and Inference Utilities

## Summary
The `pl_train` module provides utility functions for training and evaluating PyTorch Lightning models. It includes functions for setting up training environments, handling distributed training, performing inference, and exporting models to ONNX format. This module serves as a central hub for training-related functionality across all model types in the MODS_BSM system.

## Key Functions

### model_train
Sets up and executes the training process for a PyTorch Lightning model.

**Parameters:**
- `model`: PyTorch Lightning model to train
- `config`: Configuration dictionary with training parameters
- `train_dataloader`: DataLoader for training data
- `val_dataloader`: DataLoader for validation data
- `device`: Device specification for training
- `model_log_path`: Directory to save logs
- `early_stop_metric`: Metric to monitor for early stopping

**Returns:**
- Trained PyTorch Lightning Trainer object

**Features:**
- Configurable early stopping
- Model checkpointing
- TensorBoard logging
- Learning rate monitoring
- Device statistics monitoring
- Support for distributed training with FSDP or DDP
- Mixed precision training (FP16)
- Gradient clipping

### model_inference
Performs inference using a trained model and returns predictions and labels.

**Parameters:**
- `model`: Trained PyTorch Lightning model
- `dataloader`: DataLoader for inference
- `accelerator`: Accelerator setting
- `device`: Device specification for inference
- `model_log_path`: Directory to save logs
- `return_dataframe`: Whether to return the original dataframe

**Returns:**
- Tuple of (predictions, labels) or (predictions, labels, dataframe)

### extract_preds_and_labels
Extracts predictions and labels from a DataFrame.

**Parameters:**
- `df`: DataFrame containing predictions and labels
- `is_binary`: Whether the task is binary classification

**Returns:**
- Tuple of (predictions, labels) as PyTorch tensors

### is_fsdp_available
Checks if Fully Sharded Data Parallel (FSDP) training is available.

**Returns:**
- Boolean indicating FSDP availability

### my_auto_wrap_policy
Custom FSDP auto-wrap policy for multimodal models.

**Parameters:**
- `module`: Module to inspect
- `recurse`: Whether FSDP is recursing
- `unwrapped_params`: Number of unwrapped parameters
- `min_num_params`: Minimum number of parameters to wrap

**Returns:**
- Boolean indicating whether to wrap the module

## Model Imports
The module imports all model classes from the MODS_BSM system:
- [TextBertBase](pl_bert.md): BERT-based text encoder
- [TextBertClassification](pl_bert_classification.md): BERT-based text classifier
- [TextLSTM](pl_lstm.md): LSTM-based text classifier
- [TextCNN](pl_text_cnn.md): CNN-based text classifier
- [TabAE](pl_tab_ae.md): Tabular autoencoder
- [MultimodalBert](pl_multimodal_bert.md): Multimodal BERT model
- [MultimodalCNN](pl_multimodal_cnn.md): Multimodal CNN model
- [MultimodalBertGateFusion](pl_multimodal_gate_fusion.md): Multimodal BERT with gate fusion
- [MultimodalBertMoE](pl_multimodal_moe.md): Multimodal BERT with mixture of experts
- [MultimodalBertCrossAttn](pl_multimodal_cross_attn.md): Multimodal BERT with cross-attention

## Example Usage
```python
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from pl_train import model_train, model_inference
from pl_multimodal_bert import MultimodalBert

# Set random seed for reproducibility
seed_everything(42)

# Create model and dataloaders
config = {
    "text_name": "text",
    "label_name": "label",
    "tab_field_list": ["feature1", "feature2", "feature3"],
    "tokenizer": "bert-base-uncased",
    "is_binary": True,
    "num_classes": 2,
    "hidden_common_dim": 128,
    "max_epochs": 5,
    "early_stop_patience": 3,
    "model_class": "multimodal_bert",
    "fp16": True
}

model = MultimodalBert(config)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# Train model
trainer = model_train(
    model=model,
    config=config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device="auto",
    model_log_path="./logs",
    early_stop_metric="val/f1_score"
)

# Run inference
preds, labels = model_inference(
    model=model,
    dataloader=test_dataloader,
    device="auto"
)

# Compute metrics
from pl_model_plots import compute_metrics
metrics = compute_metrics(preds, labels, ["accuracy", "f1_score", "auroc"], "binary", 2)
print(f"Test metrics: {metrics}")
```
