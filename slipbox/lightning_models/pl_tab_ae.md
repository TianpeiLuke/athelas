# TabAE (Tabular Autoencoder)

## Model Summary
TabAE is a PyTorch Lightning module that encodes tabular data into a fixed-dimensional representation. It serves as a component in multimodal models, providing a way to incorporate structured tabular features alongside other modalities like text. The model normalizes and transforms tabular features into a dense representation that can be used for downstream tasks.

## Architecture

### Components
1. **Embedding Layer**: A sequential module consisting of:
   - LayerNorm: Normalizes the input features
   - Linear layer: Maps the input features to the hidden dimension
   - ReLU activation: Adds non-linearity to the representation

### Data Flow
1. Tabular features are combined into a single tensor
2. The combined features are normalized using LayerNorm
3. The normalized features are transformed using a linear layer
4. The transformed features are passed through a ReLU activation

## Benefits and Use Cases
- Provides a simple yet effective way to encode tabular data
- Normalizes features to improve training stability
- Creates fixed-dimensional representations that can be combined with other modalities
- Serves as a building block for multimodal models like [MultimodalBert](pl_multimodal_bert.md)
- Can be used for:
  - Feature extraction from structured data
  - Dimensionality reduction
  - Creating embeddings for tabular data

## Input Format

### Expected Input
- A dictionary containing tabular features with keys matching the `tab_field_list`
- Each feature should be a tensor of shape [batch_size, 1] or [batch_size]

### Configuration Parameters
- `tab_field_list`: List of tabular field names
- `hidden_common_dim`: Output dimension of the embedding layer
- `is_binary`: Whether the task is binary classification (not used directly by TabAE)
- `num_classes`: Number of classes for classification (not used directly by TabAE)

## Output Format
- A tensor of shape [batch_size, hidden_common_dim] containing the tabular representations
- These representations can be used directly for classification or combined with other modalities

## Example Usage
```python
config = {
    "tab_field_list": ["age", "income", "education_level", "num_purchases"],
    "hidden_common_dim": 64
}

tabular_encoder = TabAE(config)
tabular_representation = tabular_encoder(batch)
```

## Notes
- TabAE is a simplified version of TabularEmbeddingModule with the same functionality
- The model automatically computes `input_tab_dim` based on the length of `tab_field_list`
- The `combine_tab_data` method handles converting different input formats to a consistent tensor
