# TextBertClassification Model

## Model Summary
TextBertClassification is a PyTorch Lightning module that implements a complete BERT-based text classification model. It uses a pre-trained BERT model with a classification head to perform binary or multiclass classification on text data. The model is designed for end-to-end text classification tasks and includes training, validation, and testing functionality.

## Architecture

### Components
1. **BERT Model**: A pre-trained BERT model from Hugging Face's transformers library with a classification head
2. **Classification Head**: Automatically added by the AutoModelForSequenceClassification class
3. **Loss Function**: Cross-entropy loss with optional class weighting

### Data Flow
1. Input text is tokenized and passed to the BERT model
2. BERT processes the tokens and produces logits for each class
3. During training, the loss is calculated using the logits and the true labels
4. During inference, the logits are converted to probabilities using softmax

### Initialization Options
- **Pooler Reinitialization**: Option to reinitialize the BERT pooler layer
- **Layer Reinitialization**: Option to reinitialize a specified number of BERT layers from the end

## Benefits and Use Cases
- Complete solution for text classification tasks
- Supports both binary and multiclass classification
- Includes metrics calculation and evaluation
- Can handle imbalanced datasets through class weighting
- Suitable for a wide range of text classification applications:
  - Sentiment analysis
  - Topic classification
  - Intent recognition
  - Content categorization

## Input Format

### Expected Input
- A dictionary containing tokenized text data with keys:
  - `{text_name}`: Tensor of token IDs [batch_size, seq_length]
  - `{text_attention_mask}`: Tensor of attention masks [batch_size, seq_length]
  - `{label_name}`: Tensor of labels [batch_size] (optional during inference)

### Configuration Parameters
- `text_name`: Name of the input text field
- `label_name`: Name of the label field
- `tokenizer`: Name of the pre-trained BERT model to use (default: "bert-base-cased")
- `is_binary`: Whether the task is binary classification (default: True)
- `num_classes`: Number of classes for classification
- `metric_choices`: List of metrics to compute (default: ["accuracy", "f1_score"])
- `lr`: Learning rate (default: 2e-5)
- `weight_decay`: Weight decay for AdamW optimizer (default: 0.0)
- `warmup_steps`: Number of warmup steps for learning rate scheduler (default: 0)
- `adam_epsilon`: Epsilon for AdamW optimizer (default: 1e-8)
- `run_scheduler`: Whether to use a learning rate scheduler (default: True)
- `reinit_pooler`: Whether to reinitialize the BERT pooler layer (default: False)
- `reinit_layers`: Number of BERT layers to reinitialize from the end (default: 0)
- `model_path`: Path to save model outputs
- `id_name`: Name of the ID field (optional)
- `text_input_ids_key`: Key for input IDs (default: "input_ids")
- `text_attention_mask_key`: Key for attention mask (default: "attention_mask")

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
    "tokenizer": "bert-base-uncased",
    "is_binary": True,
    "num_classes": 2,
    "metric_choices": ["accuracy", "f1_score", "auroc"],
    "lr": 3e-5,
    "model_path": "./output"
}

model = TextBertClassification(config)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_dataloader, val_dataloader)
```
