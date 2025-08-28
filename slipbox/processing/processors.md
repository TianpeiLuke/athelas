# Processor Base Classes

## Processor

### Task Summary
The Processor is an abstract base class that defines the interface for all text processing components in the system. It provides a common structure and composition capabilities for building processing pipelines.

### Key Features
- **Abstract Base Class**: Enforces implementation of the `process` method in subclasses
- **Callable Interface**: Allows processors to be used as functions via `__call__`
- **Composition Support**: Enables chaining processors using the `>>` operator
- **Naming Convention**: Each processor has a name for identification and debugging
- **Function List**: Tracks available functions for potential introspection

### Methods
- **get_name()**: Returns the processor's name
- **process()**: Abstract method that must be implemented by subclasses
- **__call__()**: Makes the processor callable, delegating to process()
- **__rshift__()**: Implements the `>>` operator for processor composition

## ComposedProcessor

### Task Summary
The ComposedProcessor combines multiple processors into a single processing pipeline, applying them sequentially to input data.

### Input
- **processors**: List of Processor instances to be applied in sequence

### Output
- The result of applying each processor in sequence to the input

### Key Features
- **Sequential Processing**: Applies processors in the order they were added
- **Unified Interface**: Presents multiple processors as a single processor
- **Name Tracking**: Maintains a list of component processor names
- **Composition Support**: Can be further composed with other processors

## IdentityProcessor

### Task Summary
The IdentityProcessor is a simple processor that returns its input unchanged. It serves as a placeholder or default processor in situations where processing is optional.

### Input
- Any input data

### Output
- The same input data, unchanged

### Key Features
- **Pass-through Behavior**: Returns input without modification
- **Utility Function**: Useful as a default or placeholder in processing pipelines

## Example Usage
```python
from src.processing.processors import Processor, ComposedProcessor, IdentityProcessor
from src.processing.bsm_processor import TextNormalizationProcessor, EmojiRemoverProcessor

# Create individual processors
normalizer = TextNormalizationProcessor()
emoji_remover = EmojiRemoverProcessor()
identity = IdentityProcessor()

# Method 1: Compose using the >> operator
pipeline = normalizer >> emoji_remover

# Method 2: Create a ComposedProcessor directly
pipeline = ComposedProcessor([normalizer, emoji_remover])

# Process text through the pipeline
input_text = "Hello WORLD! 😊"
processed_text = pipeline(input_text)  # "hello world!"

# Using IdentityProcessor as a placeholder
optional_pipeline = identity if skip_processing else pipeline
result = optional_pipeline(input_text)
```

## Design Pattern
The processor system implements the Composite design pattern, allowing individual processors and compositions of processors to be treated uniformly. It also follows the Chain of Responsibility pattern, where each processor handles a specific aspect of text processing and passes the result to the next processor in the chain.
