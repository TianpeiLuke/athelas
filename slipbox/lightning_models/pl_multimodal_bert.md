# MultimodalBert Model

## Model Summary
MultimodalBert is a PyTorch Lightning module that combines text and tabular data for classification tasks. It uses a BERT-based text encoder and a tabular data encoder, concatenates their outputs, and passes the combined representation through a classifier. This model is designed for tasks where both text and structured tabular data are available and can contribute to the prediction.

## Architecture

### Components
1. **Text Subnetwork**: A [TextBertBase](pl_bert.md) model that encodes text data
2. **Tabular Subnetwork**: A [TabAE](pl_tab_ae.md) model that encodes tabular data (optional)
3. **Final Merge Network**: A sequential module consisting of:
   - ReLU activation
   - Linear layer that maps the concatenated representations to class logits
4. **Loss Function**: Cross-entropy loss with optional class weighting

### Data Flow
1. Text data is processed through the text subnetwork to produce text representations
2. Tabular data (if available) is processed through the tabular subnetwork to produce tabular representations
3. The text and tabular representations are concatenated
4. The concatenated representation is passed through the final merge network to produce logits
5. During training, the loss is calculated using the logits and the true labels
6. During inference, the logits are converted to probabilities using softmax

## Benefits and Use Cases
- Leverages both textual and structured data for improved predictions
- Combines the strengths of BERT for text understanding and neural networks for tabular data
- Suitable for tasks where both modalities provide complementary information
- Can be used for:
  - Customer support ticket classification with user metadata
  - Product categorization with both descriptions and attributes
  - Fraud detection using transaction text and numerical features
  - Medical diagnosis using clinical notes and patient data

## Input Format

### Expected Input
- A dictionary containing both text and tabular data:
  - `{text_name}_processed_input_ids`: Tensor of token IDs [batch_size, num_chunks, seq_length]
  - `{text_name}_processed_attention_mask`: Tensor of attention masks [batch_size, num_chunks, seq_length]
  - Tabular features: Individual tensors for each tabular feature
  - `{label_name}`: Tensor of labels [batch_size] (optional during inference)

### Configuration Parameters
- `text_name`: Base name for text inputs
- `label_name`: Name of the label field
- `tab_field_list`: List of tabular field names (optional)
- `tokenizer`: Name of the pre-trained BERT model to use (default: "bert-base-cased")
- `is_binary`: Whether the task is binary classification (default: True)
- `num_classes`: Number of classes for classification
- `metric_choices`: List of metrics to compute (default: ["accuracy", "f1_score"])
- `hidden_common_dim`: Output dimension of the text and tabular encoders
- `lr`: Learning rate (default: 2e-5)
- `weight_decay`: Weight decay for AdamW optimizer (default: 0.0)
- `warmup_steps`: Number of warmup steps for learning rate scheduler (default: 0)
- `adam_epsilon`: Epsilon for AdamW optimizer (default: 1e-8)
- `run_scheduler`: Whether to use a learning rate scheduler (default: True)
- `class_weights`: List of class weights for the loss function (default: [1.0] * num_classes)
- `id_name`: Name of the ID field (optional)
- `model_path`: Path to save model outputs
- `text_input_ids_key`: Key suffix for input IDs (default: "input_ids")
- `text_attention_mask_key`: Key suffix for attention mask (default: "attention_mask")

## Output Format
- During training/validation: Loss, metrics, and predictions
- During inference: Probabilities for each class
  - For binary classification: Tensor of shape [batch_size] with probabilities for the positive class
  - For multiclass classification: Tensor of shape [batch_size, num_classes] with probabilities for each class

## Example Usage
```python
config = {
    "text_name": "text",
    "label_name": "label",
    "tab_field_list": ["feature1", "feature2", "feature3"],
    "tokenizer": "bert-base-uncased",
    "is_binary": True,
    "num_classes": 2,
    "hidden_common_dim": 128,
    "metric_choices": ["accuracy", "f1_score", "auroc"],
    "lr": 3e-5,
    "class_weights": [1.0, 2.0],
    "model_path": "./output"
}

model = MultimodalBert(config)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_dataloader, val_dataloader)
```
