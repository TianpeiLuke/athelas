# TextBertBase Model

## Model Summary
TextBertBase is a PyTorch Lightning module that implements a BERT-based text encoder. It serves as a foundation for text processing in multimodal models, providing a way to extract meaningful representations from text data. The model uses a pre-trained BERT model to encode text and adds a customizable head layer to transform the BERT output to a specified dimension.

## Architecture

### Components
1. **BERT Encoder**: A pre-trained BERT model from Hugging Face's transformers library
2. **Head Layer**: A sequential module consisting of:
   - Dropout layer (0.1 dropout rate)
   - Linear layer that transforms BERT's output dimension to a configurable hidden dimension

### Data Flow
1. Input text is tokenized and passed to the BERT model
2. BERT processes the tokens and produces a pooled output representation
3. For batches with multiple chunks per sample, the model:
   - Reshapes the input to process all chunks
   - Averages the embeddings across chunks
4. The pooled output is passed through the head layer to produce the final representation

### Initialization Options
- **Pooler Reinitialization**: Option to reinitialize the BERT pooler layer
- **Layer Reinitialization**: Option to reinitialize a specified number of BERT layers from the end

## Benefits and Use Cases
- Provides high-quality text representations for downstream tasks
- Serves as a building block for multimodal models like [MultimodalBert](pl_multimodal_bert.md)
- Flexible configuration allows adaptation to different text processing needs
- Can be used for various NLP tasks when combined with appropriate classification heads
- Supports processing of multiple text chunks per sample

## Input Format

### Expected Input
- A dictionary containing tokenized text data with keys:
  - `{text_name}_processed_input_ids`: Tensor of token IDs [batch_size, num_chunks, seq_length]
  - `{text_name}_processed_attention_mask`: Tensor of attention masks [batch_size, num_chunks, seq_length]

### Configuration Parameters
- `text_name`: Base name for text inputs
- `tokenizer`: Name of the pre-trained BERT model to use (default: "bert-base-cased")
- `hidden_common_dim`: Output dimension of the head layer
- `reinit_pooler`: Whether to reinitialize the BERT pooler layer (default: False)
- `reinit_layers`: Number of BERT layers to reinitialize from the end (default: 0)
- `text_input_ids_key`: Key suffix for input IDs (default: "input_ids")
- `text_attention_mask_key`: Key suffix for attention mask (default: "attention_mask")

## Output Format
- A tensor of shape [batch_size, hidden_common_dim] containing the text representations
- These representations can be used directly for classification or combined with other modalities

## Example Usage
```python
config = {
    "text_name": "text",
    "hidden_common_dim": 128,
    "tokenizer": "bert-base-uncased",
    "reinit_pooler": True,
    "reinit_layers": 2
}

text_encoder = TextBertBase(config)
text_representation = text_encoder(batch)
```
