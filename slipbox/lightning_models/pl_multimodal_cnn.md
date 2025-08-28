# MultimodalCNN Model

## Model Summary
MultimodalCNN is a PyTorch Lightning module that combines text and tabular data for classification tasks using a CNN-based architecture. It uses a TextCNN model for processing text data and a TabAE model for processing tabular data, concatenates their outputs, and passes the combined representation through a classifier. This model is designed for tasks where both text and structured tabular data are available and can contribute to the prediction.

## Architecture

### Components
1. **Text Subnetwork**: A [TextCNN](pl_text_cnn.md) model that processes text data using convolutional networks
2. **Tabular Subnetwork**: A [TabAE](pl_tab_ae.md) model that encodes tabular data (optional)
3. **Final Merge Network**: A sequential module consisting of:
   - ReLU activation
   - Linear layer that maps the concatenated representations to class logits
4. **Loss Function**: Cross-entropy loss with optional class weighting

### Data Flow
1. Text data is processed through the TextCNN subnetwork to produce text representations
2. Tabular data (if available) is processed through the TabAE subnetwork to produce tabular representations
3. The text and tabular representations are concatenated
4. The concatenated representation is passed through the final merge network to produce logits
5. During training, the loss is calculated using the logits and the true labels
6. During inference, the logits are converted to probabilities using softmax

## Benefits and Use Cases
- Leverages both textual and structured data for improved predictions
- Uses CNN-based text processing which is computationally efficient
- Combines the strengths of CNNs for text understanding and neural networks for tabular data
- Suitable for tasks where both modalities provide complementary information
- Can be used for:
  - Customer support ticket classification with user metadata
  - Product categorization with both descriptions and attributes
  - Fraud detection using transaction text and numerical features
  - Content moderation with user and post metadata

## Input Format

### Expected Input
- A dictionary containing both text and tabular data:
  - `{text_name}_processed_input_ids`: Tensor of token IDs [batch_size, seq_length]
  - Tabular features: Individual tensors for each tabular feature
  - `{label_name}`: Tensor of labels [batch_size] (optional during inference)

### Required Initialization Parameters
- `config`: Dictionary containing model configuration
- `vocab_size`: Size of the vocabulary
- `word_embeddings`: Pre-trained word embeddings tensor of shape [vocab_size, embed_size]

### Configuration Parameters
- `text_name`: Base name for text inputs
- `label_name`: Name of the label field
- `tab_field_list`: List of tabular field names (optional)
- `is_binary`: Whether the task is binary classification (default: True)
- `num_classes`: Number of classes for classification (default: 2)
- `metric_choices`: List of metrics to compute (default: ["accuracy", "f1_score"])
- `hidden_common_dim`: Output dimension of the text and tabular encoders
- `lr`: Learning rate (default: 0.02)
- `weight_decay`: Weight decay for optimizer (default: 0.0)
- `adam_epsilon`: Epsilon for Adam optimizer (default: 1e-8)
- `warmup_steps`: Number of warmup steps for learning rate scheduler (default: 0)
- `run_scheduler`: Whether to use a learning rate scheduler (default: True)
- `class_weights`: List of class weights for the loss function (default: [1.0] * num_classes)
- `id_name`: Name of the ID field (optional)
- `model_path`: Path to save model outputs (default: ".")
- `text_input_ids_key`: Key suffix for input IDs (default: "input_ids")
- Additional parameters for TextCNN (see [TextCNN documentation](pl_text_cnn.md))

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
    "tab_field_list": ["feature1", "feature2", "feature3"],
    "is_binary": True,
    "num_classes": 2,
    "hidden_common_dim": 128,
    "kernel_size": [3, 4, 5],
    "num_layers": 2,
    "num_channels": [100, 100],
    "dropout_keep": 0.5,
    "lr": 0.01,
    "class_weights": [1.0, 2.0],
    "model_path": "./output"
}

model = MultimodalCNN(config, vocab_size, embeddings_tensor)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)
```
