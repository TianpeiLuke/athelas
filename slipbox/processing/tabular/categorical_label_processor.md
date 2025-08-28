# Categorical Label Processor

## Task Summary
The Categorical Label Processor converts categorical text values into numeric labels by assigning a unique integer to each category. It can either use a predefined mapping or build the mapping dynamically as new categories are encountered.

## Input
- **input_text**: A string representing a categorical value
- **initial_categories**: Optional list of categories to initialize the mapping
- **update_on_new**: Boolean flag indicating whether to add new categories to the mapping
- **unknown_label**: Label to assign if update_on_new is False and a new category is encountered

## Output
- **Integer**: A numeric label corresponding to the input category

## Features
- Converts categorical text values to numeric labels
- Can be initialized with a predefined set of categories
- Supports dynamic addition of new categories
- Provides configurable handling for unknown categories
- Simple and efficient implementation

## Example Usage
```python
# Initialize with predefined categories
processor = CategoricalLabelProcessor(
    initial_categories=["low", "medium", "high"],
    update_on_new=False,
    unknown_label=-1
)

# Process known categories
low_label = processor.process("low")     # Returns 0
medium_label = processor.process("medium")  # Returns 1
high_label = processor.process("high")   # Returns 2

# Process unknown category
unknown_label = processor.process("very_high")  # Returns -1 (unknown_label)

# Initialize with dynamic category addition
dynamic_processor = CategoricalLabelProcessor(update_on_new=True)

# Process categories dynamically
label1 = dynamic_processor.process("apple")   # Returns 0
label2 = dynamic_processor.process("banana")  # Returns 1
label3 = dynamic_processor.process("apple")   # Returns 0 (existing category)
label4 = dynamic_processor.process("cherry")  # Returns 2 (new category)
```

## Use Cases
- Text classification tasks
- Feature engineering for machine learning models
- Converting categorical variables for algorithms that require numeric inputs
- Creating one-hot encoding mappings
