# DataFrame Category Risk Calculator

## Task Summary
The DataFrame Category Risk Calculator computes risk ratios for categorical variables based on their correlation with a binary target variable. It calculates the proportion of positive outcomes for each category, applies smoothing, and handles categories with insufficient data.

## Input
- **df**: Pandas DataFrame containing the categorical columns
- **col_names**: List of column names to compute risk values for
- **tags**: Binary target variable (array of 0/1 or -1/1 values)
- **default_risk**: Default risk value for categories with insufficient data (default: 0.001)
- **cnt_threshold**: Minimum count threshold for reliable risk calculation (default: 5)
- **smoothing_factor**: Factor for Laplace smoothing (default: 1)
- **tag_range**: Specifies the range of tag values (1 for {0,1}, 2 for {-1,1}) (default: 1)

## Output
- **ratio_dict_list**: List of dictionaries mapping categories to risk ratios for each column
- **count_dict_list**: List of dictionaries mapping categories to positive count for each column

## Risk Calculation Process
1. Determines the binary set of labels ({0,1} or {-1,1}) based on tag_range
2. Calculates the global positive ratio as a prior distribution P(y)
3. For each specified column:
   - Groups data by column value and target
   - Calculates the total count for each category
   - Calculates the positive count for each category
   - Applies smoothing to handle rare categories
   - Filters out categories with counts below the threshold
   - Assigns default risk or global positive ratio to categories with no positive examples

## Smoothing Formula
The risk ratio for each category is calculated as:
```
risk_ratio = (positive_count + smoothing_factor * global_pos_ratio) / (total_count + smoothing_factor)
```

## Example Usage
```python
import pandas as pd
import numpy as np
from src.processing.df_category_risk import df_category_risk

# Create sample data
df = pd.DataFrame({
    'country': ['US', 'UK', 'CA', 'US', 'UK', 'FR', 'DE', 'US', 'CA', 'UK'],
    'device': ['mobile', 'desktop', 'mobile', 'tablet', 'mobile', 'desktop', 'mobile', 'desktop', 'tablet', 'mobile']
})
tags = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])

# Calculate risk ratios
ratio_dict_list, count_dict_list = df_category_risk(
    df=df,
    col_names=['country', 'device'],
    tags=tags,
    default_risk=0.01,
    cnt_threshold=2,
    smoothing_factor=0.5
)

# Example output:
# ratio_dict_list = [
#   {'US': 0.75, 'UK': 0.67, 'CA': 0.0, 'FR': 0.0, 'DE': 0.0},  # country risks
#   {'mobile': 0.5, 'desktop': 0.67, 'tablet': 0.33}  # device risks
# ]
# count_dict_list = [
#   {'US': 3, 'UK': 2, 'CA': 0, 'FR': 0, 'DE': 0},  # country positive counts
#   {'mobile': 2, 'desktop': 2, 'tablet': 1}  # device positive counts
# ]
```

## Use Cases
- Feature engineering for risk-based models
- Creating risk scores for categorical variables
- Weight of Evidence (WoE) calculation
- Handling high-cardinality categorical features
- Preprocessing for fraud detection or risk assessment models
