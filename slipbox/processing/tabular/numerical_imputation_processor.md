# Numerical Variable Imputation Processor

## Task Summary
The Numerical Variable Imputation Processor handles missing values in numerical data by replacing them with statistically derived values. It supports multiple imputation strategies and can be applied to specific variables or all numerical variables in a dataset.

## Input
- **DataFrame or Series**: Contains numerical values with potential missing values
- **variables**: List of column names to apply imputation to (optional, defaults to all numerical columns)
- **imputation_dict**: Pre-defined dictionary mapping column names to imputation values (optional)
- **strategy**: Imputation strategy to use ('mean', 'median', or 'mode')

## Output
- **DataFrame or Series**: Original data with missing values replaced by imputed values

## Key Methods

### fit()
- **Input**: DataFrame containing the columns to be imputed
- **Output**: Fitted processor with imputation values calculated for each column
- **Task**: Calculates imputation values for each specified column based on the chosen strategy

### process()
- **Input**: Dictionary of variable names and values
- **Output**: Dictionary with missing values imputed
- **Task**: Replaces missing values in the input dictionary with their corresponding imputation values

### transform()
- **Input**: DataFrame or Series containing values to impute
- **Output**: DataFrame or Series with missing values imputed
- **Task**: Applies imputation to all missing values in the input data

## Features
- Supports multiple imputation strategies: mean, median, and mode
- Can be initialized with pre-calculated imputation values
- Works with both DataFrame and Series inputs
- Can be composed with other processors using the `>>` operator
- Handles edge cases like columns with all missing values

## Example Usage
```python
# Create and fit the processor
processor = NumericalVariableImputationProcessor(
    variables=["age", "income", "years_experience"],
    strategy="median"
)
processor.fit(training_data)

# Transform new data
transformed_data = processor.transform(new_data)

# Use with pre-defined imputation values
imputation_dict = {"age": 35.0, "income": 50000.0, "years_experience": 5.0}
processor = NumericalVariableImputationProcessor(imputation_dict=imputation_dict)
transformed_data = processor.transform(new_data)
```
