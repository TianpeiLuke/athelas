# Numerical Binning Processor

## Task Summary
The Numerical Binning Processor transforms continuous numerical values into categorical bins. It supports both equal-width and quantile-based binning strategies, with customizable bin labels and handling for missing or out-of-range values.

## Input
- **DataFrame or Series**: Contains numerical values to be binned
- **column_name**: Name of the column to bin
- **n_bins**: Number of bins to create (default: 5)
- **strategy**: Binning strategy ('quantile' or 'equal-width')
- **bin_labels**: Custom labels for bins (list of strings, boolean, or None)
- **output_column_name**: Name for the output column (default: "{column_name}_binned")
- **handle_missing_value**: How to handle missing values ('as_is' or custom string)
- **handle_out_of_range**: How to handle values outside the fitted range ('boundary_bins' or custom string)

## Output
- **DataFrame or Series**: Original data with an additional column containing binned values as categorical labels

## Key Methods

### fit()
- **Input**: DataFrame containing the column to be binned
- **Output**: Fitted processor with bin edges and labels determined
- **Task**: Analyzes the data distribution and creates bin edges according to the specified strategy

### process()
- **Input**: Single numerical value
- **Output**: Bin label as a string
- **Task**: Assigns a single value to its appropriate bin

### transform()
- **Input**: DataFrame or Series containing values to bin
- **Output**: DataFrame with added column of binned values or Series of binned values
- **Task**: Applies binning to all values in the input data

### save_params() / load_params()
- **Input/Output**: File path or dictionary for parameter storage/retrieval
- **Task**: Serializes/deserializes the processor's fitted parameters for later use

## Features
- Supports both quantile-based and equal-width binning strategies
- Handles edge cases like single-value columns or columns with few unique values
- Provides options for custom bin labels or interval notation
- Offers flexible handling of missing values and out-of-range values
- Includes parameter serialization for model persistence

## Example Usage
```python
# Create and fit the processor
processor = NumericalBinningProcessor(
    column_name="age",
    n_bins=5,
    strategy="quantile",
    bin_labels=["Very Young", "Young", "Middle-aged", "Senior", "Elderly"],
    handle_missing_value="Unknown",
    handle_out_of_range="boundary_bins"
)
processor.fit(training_data)

# Transform new data
transformed_data = processor.transform(new_data)

# Save the fitted parameters
processor.save_params("./model_artifacts/")

# Load the processor later
loaded_processor = NumericalBinningProcessor.load_params("./model_artifacts/numerical_binning_processor_age_params.json")
```
