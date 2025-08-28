# Model Plots and Metrics Utilities

## Summary
The `pl_model_plots` module provides utilities for computing classification metrics and generating performance plots for PyTorch Lightning models. It supports a wide range of metrics for both binary and multiclass classification tasks and includes functions for visualizing ROC curves and other performance metrics.

## Key Functions

### compute_metrics
Calculates various classification metrics based on model predictions and ground truth labels.

**Parameters:**
- `preds`: Model predictions (probabilities or logits)
- `target`: Ground truth labels
- `metric_choices`: List of metrics to compute
- `task`: Classification task type ('binary' or 'multiclass')
- `num_classes`: Number of classes for multiclass classification
- `stage`: Optional stage name for prefixing metric keys (e.g., 'val', 'test')

**Supported Metrics:**
- `accuracy`: Classification accuracy
- `f1_score`: F1 score (harmonic mean of precision and recall)
- `auroc`: Area Under the Receiver Operating Characteristic curve
- `average_precision`: Average precision score
- `precision`: Precision score
- `recall`: Recall score
- `specificity`: Specificity score
- `kl_divergence`: Kullback-Leibler divergence
- `binary_recall_at_fixed_precision`: Recall at fixed precision for binary classification
- `multiclass_recall_at_fixed_precision`: Recall at fixed precision for multiclass classification

### plot_to_tensorboard
Converts a matplotlib figure to a TensorBoard image and logs it.

**Parameters:**
- `writer`: TensorBoard SummaryWriter instance
- `tag`: Tag for the image in TensorBoard
- `figure`: Matplotlib figure to log
- `global_step`: Global step for TensorBoard logging

### roc_metric_plot
Generates and saves ROC curve plots for binary or multiclass classification.

**Parameters:**
- `y_pred`: Model predictions on test data
- `y_true`: Ground truth labels for test data
- `y_val_pred`: Model predictions on validation data
- `y_val_true`: Ground truth labels for validation data
- `path`: Directory path to save the plot
- `task`: Classification task type ('binary' or 'multiclass')
- `num_classes`: Number of classes for multiclass classification
- `writer`: Optional TensorBoard SummaryWriter for logging
- `global_step`: Global step for TensorBoard logging

## Usage in Models
This utility module is used by various model classes in the MODS_BSM system to:

1. Compute and log performance metrics during training, validation, and testing
2. Generate ROC curves and other performance plots
3. Log metrics and visualizations to TensorBoard

## Example Usage
```python
import torch
from pl_model_plots import compute_metrics, roc_metric_plot

# Compute metrics
preds = torch.tensor([0.1, 0.9, 0.8, 0.3])
labels = torch.tensor([0, 1, 1, 0])
metrics = compute_metrics(
    preds=preds,
    target=labels,
    metric_choices=["accuracy", "f1_score", "auroc"],
    task="binary",
    num_classes=2,
    stage="val"
)
print(metrics)  # {'val/accuracy': tensor(1.), 'val/f1_score': tensor(1.), 'val/auroc': tensor(1.)}

# Generate ROC plot
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
roc_metric_plot(
    y_pred=preds,
    y_true=labels,
    y_val_pred=preds,
    y_val_true=labels,
    path="./output",
    task="binary",
    writer=writer,
    global_step=10
)
```
