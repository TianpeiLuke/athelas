# Risk Table Mapping Processor

## Task Summary
The Risk Table Mapping Processor converts categorical variables into numerical risk scores based on their correlation with a binary target variable. It calculates the proportion of positive outcomes for each category and applies optional smoothing to handle rare categories.

## Input
- **DataFrame**: Contains categorical variables and a binary target variable
- **column_name**: Name of the categorical column to be mapped to risk scores
- **label_name**: Name of the binary target variable (expected to be 0 or 1)
- **smooth_factor**: Smoothing factor for risk calculation (0 to 1)
- **count_threshold**: Minimum count for considering a category's calculated risk
- **risk_tables**: Optional pre-computed risk tables

## Output
- **DataFrame or Series**: Original data with categorical values replaced by their corresponding risk scores
- **Risk Tables**: Dictionary containing mapping from categorical values to risk scores and a default risk value

## Key Methods

### fit()
- **Input**: DataFrame containing the categorical column and target variable
- **Output**: Fitted processor with risk tables calculated
- **Task**: Calculates risk scores for each category based on its correlation with the target variable

### process()
- **Input**: Single categorical value
- **Output**: Corresponding risk score as a float
- **Task**: Maps a single categorical value to its risk score

### transform()
- **Input**: DataFrame, Series, or single value
- **Output**: Data with categorical values replaced by risk scores
- **Task**: Applies risk mapping to all values in the input data

### save_risk_tables() / load_risk_tables()
- **Input/Output**: File path for risk table storage/retrieval
- **Task**: Serializes/deserializes the risk tables for later use

## Features
- Calculates risk scores based on the proportion of positive outcomes for each category
- Applies smoothing to handle rare categories and prevent overfitting
- Supports minimum count thresholds to ignore unreliable categories
- Handles missing values and categories not seen during training
- Saves risk tables in both pickle and JSON formats for portability

## Risk Calculation
The risk score for each category is calculated as:
```
smooth_risk = (category_count * raw_risk + smooth_samples * default_risk) / (category_count + smooth_samples)
```
Where:
- **raw_risk**: Proportion of positive outcomes for the category
- **default_risk**: Overall proportion of positive outcomes in the dataset
- **smooth_samples**: Number of virtual samples used for smoothing (derived from smooth_factor)

## Example Usage
```python
# Create and fit the processor
processor = RiskTableMappingProcessor(
    column_name="country_code",
    label_name="fraud_flag",
    smooth_factor=0.1,
    count_threshold=10
)
processor.fit(training_data)

# Transform new data
transformed_data = processor.transform(new_data)

# Save the risk tables
processor.save_risk_tables("./model_artifacts/")

# Load the processor later
new_processor = RiskTableMappingProcessor(column_name="country_code", label_name="fraud_flag")
new_processor.load_risk_tables("./model_artifacts/risk_table_mapping_processor_country_code_risk_tables.pkl")
```
