# PyTorch Lightning Models

This directory contains documentation for the various PyTorch Lightning models in the MODS_BSM system. Each markdown file provides a detailed description of a specific model, including its architecture, input/output formats, and usage examples.

## Text Models

- [TextBertBase](pl_bert.md): A BERT-based text encoder that serves as a foundation for text processing in multimodal models.
- [TextBertClassification](pl_bert_classification.md): A complete BERT-based text classification model for binary or multiclass classification.
- [TextLSTM](pl_lstm.md): A bidirectional LSTM-based text classification model that uses pre-trained word embeddings.
- [TextCNN](pl_text_cnn.md): A convolutional neural network for text classification that captures n-gram features at various scales.

## Tabular Models

- [TabAE](pl_tab_ae.md): A tabular autoencoder that normalizes and transforms tabular features into a dense representation.

## Multimodal Models

- [MultimodalBert](pl_multimodal_bert.md): A model that combines text and tabular data by concatenating their representations.
- [MultimodalCNN](pl_multimodal_cnn.md): A model that combines text (processed with CNN) and tabular data for classification.
- [MultimodalBertCrossAttn](pl_multimodal_cross_attn.md): A model that uses cross-attention mechanisms to fuse text and tabular representations.
- [MultimodalBertGateFusion](pl_multimodal_gate_fusion.md): A model that uses a gated fusion mechanism to control the contribution of each modality.
- [MultimodalBertMoE](pl_multimodal_moe.md): A model that uses a mixture of experts approach to dynamically weight the contributions of text and tabular data.

## Utility Modules

- [Model Plots and Metrics](pl_model_plots.md): Utilities for computing classification metrics and generating performance plots.
- [Distributed Training Utilities](dist_utils.md): Utilities for distributed training and GPU memory monitoring.
- [Training and Inference Utilities](pl_train.md): Functions for training, evaluating, and performing inference with PyTorch Lightning models.

## Model Architecture Overview

The models in this directory follow a modular design pattern:

1. **Text Encoders**: Process text data using BERT, LSTM, or CNN architectures
2. **Tabular Encoders**: Process structured tabular data using neural networks
3. **Fusion Modules**: Combine text and tabular representations using various techniques:
   - Simple concatenation (MultimodalBert)
   - Cross-attention (MultimodalBertCrossAttn)
   - Gated fusion (MultimodalBertGateFusion)
   - Mixture of experts (MultimodalBertMoE)
4. **Classification Heads**: Transform fused representations into class predictions

## Common Usage Pattern

```python
# 1. Define configuration
config = {
    "text_name": "text",
    "label_name": "label",
    "tab_field_list": ["feature1", "feature2", "feature3"],
    "tokenizer": "bert-base-uncased",
    "is_binary": True,
    "num_classes": 2,
    "hidden_common_dim": 128
}

# 2. Initialize model
model = MultimodalBert(config)

# 3. Train model
from pl_train import model_train
trainer = model_train(
    model=model,
    config=config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader
)

# 4. Perform inference
from pl_train import model_inference
preds, labels = model_inference(
    model=model,
    dataloader=test_dataloader
)

# 5. Compute metrics
from pl_model_plots import compute_metrics
metrics = compute_metrics(preds, labels, ["accuracy", "f1_score"], "binary", 2)
```

## Model Selection Guide

- **Text-only data**: Use TextBertClassification, TextLSTM, or TextCNN
- **Tabular-only data**: Use a standard classifier with TabAE as a feature extractor
- **Text + Tabular data with simple fusion**: Use MultimodalBert or MultimodalCNN
- **Text + Tabular data with advanced fusion**: Use MultimodalBertCrossAttn, MultimodalBertGateFusion, or MultimodalBertMoE
- **Resource-constrained environments**: Use TextLSTM, TextCNN, or MultimodalCNN
- **Maximum accuracy**: Use MultimodalBertCrossAttn or MultimodalBertMoE
