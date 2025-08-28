# MultiClass Label Processor

## Task Summary
The MultiClass Label Processor converts categorical labels into numeric tensors suitable for deep learning models, particularly for multimodal BERT models. It supports both standard integer encoding and one-hot encoding of labels.

## Input
- **labels**: A single label or list of labels (strings, integers, or floats)
- **label_list**: Optional predefined list of unique label strings
- **one_hot**: Boolean flag indicating whether to output one-hot encoded labels
- **strict**: Boolean flag indicating whether to raise an error for unknown labels

## Output
- **torch.Tensor**: Encoded labels as a LongTensor (standard encoding) or FloatTensor (one-hot encoding)

## Features
- Converts categorical labels to numeric tensors
- Supports both single labels and batches of labels
- Can be initialized with a predefined set of labels
- Dynamically adds new labels to the mapping when encountered (unless strict=True)
- Provides one-hot encoding option for classification tasks
- Maintains bidirectional mapping between labels and IDs

## Example Usage
```python
# Initialize with predefined labels
processor = MultiClassLabelProcessor(
    label_list=["positive", "negative", "neutral"],
    one_hot=True,
    strict=True
)

# Process a single label
encoded_label = processor.process("positive")
# Returns tensor([[1., 0., 0.]]) (one-hot encoded)

# Process a batch of labels
encoded_batch = processor.process(["positive", "negative", "neutral", "positive"])
# Returns tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

# Initialize with dynamic label addition
dynamic_processor = MultiClassLabelProcessor(one_hot=False)

# Process labels dynamically
encoded1 = dynamic_processor.process(["apple", "banana", "cherry"])
# Returns tensor([0, 1, 2])

encoded2 = dynamic_processor.process(["banana", "apple", "date"])
# Returns tensor([1, 0, 3]) (adds "date" as a new label)
```

## Use Cases
- Text classification tasks
- Sentiment analysis
- Intent recognition
- Any multiclass classification problem with categorical labels
- Preparing labels for deep learning models, especially transformer-based models
