# TextLSTM Model

## Model Summary
TextLSTM is a PyTorch Lightning module that implements a bidirectional LSTM-based text classification model. It uses pre-trained word embeddings to represent text tokens and processes them through a bidirectional LSTM network to capture sequential information. The model is designed for text classification tasks and provides a lightweight alternative to transformer-based models like BERT.

## Architecture

### Components
1. **Word Embeddings**: Pre-trained word embeddings loaded from an external source
2. **Bidirectional LSTM**: A multi-layer bidirectional LSTM network
3. **Linear Classifier**: A linear layer that maps the LSTM output to class logits
4. **Loss Function**: Cross-entropy loss with optional class weighting

### Data Flow
1. Input tokens are mapped to embeddings using the embedding layer
2. The embeddings are processed through the bidirectional LSTM
3. The final hidden states from both directions are concatenated
4. The concatenated representation is passed through the linear classifier to produce logits
5. During training, the loss is calculated using the logits and the true labels
6. During inference, the logits are converted to probabilities using softmax

## Benefits and Use Cases
- More computationally efficient than transformer models
- Captures sequential dependencies in text
- Suitable for resource-constrained environments
- Effective for tasks where word order and context are important
- Can be used for:
  - Sentiment analysis
  - Text categorization
  - Intent classification
  - Simple question answering

## Input Format

### Expected Input
- A dictionary containing tokenized text data with keys:
  - `{text_name}_processed_input_ids`: Tensor of token IDs [batch_size, seq_length]
  - `{label_name}`: Tensor of labels [batch_size] (optional during inference)

### Required Initialization Parameters
- `config`: Dictionary containing model configuration
- `vocab_size`: Size of the vocabulary
- `word_embeddings`: Pre-trained word embeddings tensor of shape [vocab_size, embed_size]

### Configuration Parameters
- `text_name`: Name of the input text field
- `label_name`: Name of the label field
- `is_binary`: Whether the task is binary classification (default: True)
- `num_classes`: Number of classes for classification (default: 2)
- `metric_choices`: List of metrics to compute (default: ["accuracy", "f1_score"])
- `hidden_common_dim`: Hidden dimension of the LSTM (default: 100)
- `num_layers`: Number of LSTM layers (default: 1)
- `dropout_keep`: Dropout rate (default: 0.5)
- `max_sen_len`: Maximum sentence length (default: 512)
- `is_embeddings_trainable`: Whether to fine-tune the embeddings (default: True)
- `class_weights`: List of class weights for the loss function (default: [1.0] * num_classes)
- `id_name`: Name of the ID field (optional)
- `model_path`: Path to save model outputs (default: ".")
- `text_input_ids_key`: Key suffix for input IDs (default: "input_ids")

## Output Format
- During training/validation: Loss, metrics, and predictions
- During inference: Probabilities for each class
  - For binary classification: Tensor of shape [batch_size] with probabilities for the positive class
  - For multiclass classification: Tensor of shape [batch_size, num_classes] with probabilities for each class

## Example Usage
```python
import torch
import numpy as np
from gensim.models import KeyedVectors

# Load pre-trained word embeddings
word_vectors = KeyedVectors.load("word_vectors.kv")
vocab_size = len(word_vectors.key_to_index)
embedding_dim = word_vectors.vector_size
embeddings = np.zeros((vocab_size, embedding_dim))
for word, idx in word_vectors.key_to_index.items():
    embeddings[idx] = word_vectors[word]
embeddings_tensor = torch.FloatTensor(embeddings)

config = {
    "text_name": "text",
    "label_name": "label",
    "is_binary": True,
    "num_classes": 2,
    "hidden_common_dim": 128,
    "num_layers": 2,
    "dropout_keep": 0.7,
    "is_embeddings_trainable": False
}

model = TextLSTM(config, vocab_size, embeddings_tensor)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)
```
