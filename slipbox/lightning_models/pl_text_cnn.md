# TextCNN Model

## Model Summary
TextCNN is a PyTorch Lightning module that implements a Convolutional Neural Network (CNN) for text classification. It uses pre-trained word embeddings and multiple convolutional layers with different kernel sizes to capture n-gram features at various scales. The model is designed for text classification tasks and provides a computationally efficient alternative to recurrent or transformer-based models.

## Architecture

### Components
1. **Word Embeddings**: Pre-trained word embeddings loaded from an external source
2. **Convolutional Layers**: Multiple parallel convolutional networks with different kernel sizes
3. **Pooling Layers**: Max pooling layers after each convolution
4. **Dropout**: Regularization to prevent overfitting
5. **Linear Classifier**: A linear layer that maps the concatenated CNN outputs to the output dimension
6. **Loss Function**: Cross-entropy loss with optional class weighting

### Data Flow
1. Input tokens are mapped to embeddings using the embedding layer
2. The embeddings are processed through multiple parallel CNN networks with different kernel sizes
3. Each CNN network consists of multiple convolutional and pooling layers
4. The outputs from all CNN networks are concatenated
5. The concatenated features are passed through dropout and a linear layer
6. During training, the loss is calculated using the output and the true labels
7. During inference, the output is converted to probabilities using softmax

## Benefits and Use Cases
- Captures local patterns and n-gram features in text
- More computationally efficient than RNNs or transformers
- Parallelizable computation (no sequential dependencies)
- Effective for tasks where local patterns are important
- Can be used for:
  - Text classification
  - Sentiment analysis
  - Topic categorization
  - Intent recognition
  - Short text classification

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
- `dropout_keep`: Dropout rate (default: 0.5)
- `max_sen_len`: Maximum sentence length (default: 512)
- `kernel_size`: List of kernel sizes for the convolutional layers (default: [3, 5, 7])
- `num_layers`: Number of convolutional layers (default: 2)
- `num_channels`: List of channel sizes for each layer (default: [100, 100])
- `hidden_common_dim`: Output dimension of the network (default: 100)
- `is_embeddings_trainable`: Whether to fine-tune the embeddings (default: True)
- `class_weights`: List of class weights for the loss function (default: [1.0] * num_classes)
- `optimizer_type`: Type of optimizer to use (default: "SGD")
- `lr`: Learning rate (default: 0.02)
- `momentum`: Momentum for SGD optimizer (default: 0.9)
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
    "kernel_size": [2, 3, 4],
    "num_layers": 2,
    "num_channels": [128, 128],
    "hidden_common_dim": 100,
    "dropout_keep": 0.5,
    "optimizer_type": "Adam",
    "lr": 0.001
}

model = TextCNN(config, vocab_size, embeddings_tensor)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)
```
