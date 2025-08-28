# Processing Components

This directory contains documentation for the various processing components used in the MODS_BSM system. Each markdown file provides a detailed description of a specific processor, including its purpose, inputs, outputs, and usage examples.

## Base Components

- [Processor Base Classes](processors.md): Abstract base classes and composition utilities for building processing pipelines

## Text Processing

- [BSM Processors](bsm_processor.md): Text normalization, dialogue splitting, chunking, and HTML processing components
- [BERT Tokenization Processor](bert_tokenize_processor.md): Converts text into tokenized sequences for transformer models
- [FastText Embedding Processor](gensim_tokenize_processor.md): Maps words to pre-trained FastText embeddings
- [Customer Service Processors](cs_processor.md): Parses and formats customer service chat transcripts

## Categorical Data Processing

- [Categorical Label Processor](categorical_label_processor.md): Converts categorical text values into numeric labels
- [MultiClass Label Processor](multiclass_label_processor.md): Encodes categorical labels for deep learning models
- [Risk Table Mapping Processor](risk_table_processor.md): Maps categorical variables to risk scores based on target correlation
- [DataFrame Category Risk Calculator](df_category_risk.md): Computes risk ratios for categorical variables

## Numerical Data Processing

- [Numerical Binning Processor](numerical_binning_processor.md): Transforms continuous numerical values into categorical bins
- [Numerical Variable Imputation Processor](numerical_imputation_processor.md): Handles missing values in numerical data

## Processing Pipeline Architecture

The processing components follow a consistent design pattern:

1. **Base Processor Interface**: All processors inherit from the abstract `Processor` class
2. **Composition Support**: Processors can be chained using the `>>` operator
3. **Callable Interface**: Processors can be used as functions via `__call__`
4. **Stateful Processing**: Many processors have fit/transform methods for stateful processing

## Common Usage Pattern

```python
# Create processors
processor1 = ProcessorType1(param1=value1)
processor2 = ProcessorType2(param2=value2)

# Option 1: Compose using the >> operator
pipeline = processor1 >> processor2

# Option 2: Create a ComposedProcessor directly
from src.processing.processors import ComposedProcessor
pipeline = ComposedProcessor([processor1, processor2])

# For stateful processors, fit first
processor1.fit(training_data)

# Process data
processed_data = pipeline(input_data)
```

## Key Features Across Processors

- **Configurable Parameters**: Most processors accept parameters to customize their behavior
- **Error Handling**: Processors include validation and error handling for edge cases
- **Serialization Support**: Many processors can save/load their state for model persistence
- **Flexible Input/Output**: Processors handle various input types (strings, lists, DataFrames)
- **Efficient Implementation**: Processors are designed for performance with large datasets
