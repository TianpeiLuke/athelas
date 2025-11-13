---
tags:
  - design
  - step_builders
  - label_generation
  - multi_label
  - multi_task_learning
  - rule_based_classification
keywords:
  - label ruleset
  - multilabel extension
  - rule-based multilabel
  - sparse label generation
  - category-conditional rules
topics:
  - label ruleset multilabel extension
  - rule-based multilabel generation
  - multi-task rule evaluation
language: python
date of note: 2025-11-11
---

# Label Ruleset Multilabel Extension Design

## Overview

This document defines the design for extending the Label Ruleset Generation and Execution system to support multi-label output for multi-task learning scenarios. The extension enables category-conditional rule evaluation where rules can generate sparse multi-label columns based on categorical features (e.g., payment methods), combining the power of rule-based classification with multi-task learning architectures.

## Related Documents

### Current Ruleset System
- **[Label Ruleset Generation Contract](../../src/cursus/steps/contracts/label_ruleset_generation_contract.py)** - Current generation contract
- **[Label Ruleset Execution Contract](../../src/cursus/steps/contracts/label_ruleset_execution_contract.py)** - Current execution contract
- **[Label Ruleset Generation Script](../../src/cursus/steps/scripts/label_ruleset_generation.py)** - Generation implementation
- **[Label Ruleset Execution Script](../../src/cursus/steps/scripts/label_ruleset_execution.py)** - Execution implementation

### Related Design Documents
- **[Multilabel Preprocessing Step Design](./multilabel_preprocessing_step_design.md)** - Simple category-based multilabel
- **[LightGBM Multi-Task Training Step Design](./lightgbm_multi_task_training_step_design.md)** - Multi-task training consumer
- **[MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)** - Comprehensive MTGBM architecture

## Motivation

### Limitations of Current System

The current Label Ruleset system supports **single-label** output only:
- One rule evaluation per row → one output label
- Cannot generate category-specific labels
- No support for sparse multi-label matrices
- Limited to single-task classification

### Limitations of Simple Multilabel Preprocessing

The proposed simple multilabel preprocessing:
- Only copies existing labels to category-specific columns
- Cannot encode domain knowledge or business logic
- Limited to exact category matching
- No sophisticated label generation rules

### Value Proposition of Multilabel Ruleset Extension

Combining rule-based systems with multilabel output enables:
- **Category-conditional business logic**: Different rules per payment method
- **Domain knowledge encoding**: Expert-defined fraud patterns per category
- **Sparse representation**: Efficient multi-label matrices
- **Auditability**: Track which rules fire for which categories
- **Validation**: Automatic rule conflict detection
- **Optimization**: Rule reordering for performance

## Design Principles

### Extension Principles
- **Backward Compatibility**: Single-label mode remains default
- **Minimal Changes**: Extend, don't rewrite existing system
- **Clear Semantics**: Explicit multi-label mode configuration
- **Performance**: Efficient sparse evaluation
- **Validation**: Comprehensive multi-label rule validation

### Multi-Label Philosophy
- **Sparse by Default**: NaN for non-applicable category labels
- **Category-Conditional**: Rules target specific categories
- **Priority-Based**: First-match wins per label column
- **Fail-Safe**: Graceful handling of missing categories
- **Auditable**: Track rule matches per category

## Architecture Design

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Label Ruleset Generation                     │
│  (Validates & Optimizes Multi-Label Rules)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ validated_ruleset.json
                     │ (with multi-label config)
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Label Ruleset Execution                         │
│  (Applies Rules → Generates Sparse Multi-Labels)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ DataFrames with sparse label columns
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Task Training                             │
│  (Trains on Category-Specific Labels)                      │
└─────────────────────────────────────────────────────────────┘
```

### Component Changes

#### 1. Label Configuration Extension

**Current (Single-Label):**
```json
{
  "output_label_name": "is_fraud",
  "output_label_type": "binary",
  "label_values": [0, 1],
  "default_label": 0,
  "evaluation_mode": "priority"
}
```

**Extended (Multi-Label):**
```json
{
  "label_mode": "multi_label",
  "output_label_columns": [
    "is_fraud_CC",
    "is_fraud_DC", 
    "is_fraud_ACH"
  ],
  "output_label_type": "binary",
  "label_values": [0, 1],
  "default_label": 0,
  "evaluation_mode": "priority",
  "category_column": "payment_method",
  "categories": ["CC", "DC", "ACH"],
  "sparse_representation": true
}
```

**New Fields:**
- `label_mode`: "single_label" (default) or "multi_label"
- `output_label_columns`: List of label columns to generate
- `category_column`: Column defining categories for sparse representation
- `categories`: List of valid category values
- `sparse_representation`: Use NaN for non-matching categories (default: true)

#### 2. Rule Definition Extension

**Current (Single-Label):**
```json
{
  "rule_id": "R001",
  "name": "High value transaction",
  "priority": 1,
  "enabled": true,
  "conditions": {
    "all_of": [
      {"field": "amount", "operator": ">", "value": 1000}
    ]
  },
  "output_label": 1
}
```

**Extended (Multi-Label Option 1 - Target Column):**
```json
{
  "rule_id": "R001_CC",
  "name": "High value CC transaction",
  "priority": 1,
  "enabled": true,
  "conditions": {
    "all_of": [
      {"field": "payment_method", "operator": "equals", "value": "CC"},
      {"field": "amount", "operator": ">", "value": 1000}
    ]
  },
  "output_labels": {
    "is_fraud_CC": 1
  }
}
```

**Extended (Multi-Label Option 2 - Auto-Category):**
```json
{
  "rule_id": "R001_auto",
  "name": "High value transaction (any method)",
  "priority": 1,
  "enabled": true,
  "conditions": {
    "field": "amount",
    "operator": ">",
    "value": 1000
  },
  "output_labels": "auto",
  "output_value": 1,
  "category_conditional": true
}
```

**New Fields:**
- `output_labels`: Dict mapping column names to values, or "auto" for category-based
- `output_value`: Value to assign (when using "auto" mode)
- `category_conditional`: Enable automatic category detection

## Detailed Design

### 1. Label Ruleset Generation Changes

#### Configuration Schema Extension

```python
# In label_config.json
{
  "label_mode": "multi_label",  # NEW: single_label | multi_label
  
  # Single-label fields (backward compatible)
  "output_label_name": "is_fraud",  # Used if label_mode == "single_label"
  
  # Multi-label fields (new)
  "output_label_columns": [  # Used if label_mode == "multi_label"
    "is_fraud_CC",
    "is_fraud_DC",
    "is_fraud_ACH"
  ],
  
  # Common fields
  "output_label_type": "binary",
  "label_values": [0, 1],
  "default_label": 0,
  "evaluation_mode": "priority",
  
  # Multi-label specific
  "category_column": "payment_method",  # Column for category filtering
  "categories": ["CC", "DC", "ACH"],  # Valid category values
  "sparse_representation": true,  # Use NaN for non-matching (default: true)
  "allow_multiple_matches": false  # Allow multiple rules per column (default: false)
}
```

#### Rule Definition Extension

**Option 1: Explicit Target Columns**
```json
{
  "rule_id": "R001_CC",
  "conditions": {...},
  "output_labels": {
    "is_fraud_CC": 1  # Explicitly target this column
  }
}
```

**Option 2: Category-Conditional Auto**
```json
{
  "rule_id": "R001_auto",
  "conditions": {...},
  "output_labels": "auto",  # Auto-determine column from row's category
  "output_value": 1,
  "category_conditional": true
}
```

#### Validation Extensions

**New Validators:**

1. **MultiLabelConfigValidator**
```python
class MultiLabelConfigValidator:
    """Validates multi-label configuration."""
    
    def validate(self, label_config: dict) -> ValidationResult:
        """
        Validates:
        - label_mode is valid
        - output_label_columns exist if multi_label mode
        - category_column specified if multi_label
        - categories list not empty
        - No duplicate column names
        """
        result = ValidationResult()
        
        label_mode = label_config.get("label_mode", "single_label")
        
        if label_mode not in ["single_label", "multi_label"]:
            result.valid = False
            result.errors.append(f"Invalid label_mode: {label_mode}")
        
        if label_mode == "multi_label":
            # Check required fields
            if "output_label_columns" not in label_config:
                result.valid = False
                result.errors.append("output_label_columns required for multi_label mode")
            
            output_columns = label_config.get("output_label_columns", [])
            if not output_columns:
                result.valid = False
                result.errors.append("output_label_columns cannot be empty")
            
            # Check for duplicates
            if len(output_columns) != len(set(output_columns)):
                result.valid = False
                result.errors.append("Duplicate columns in output_label_columns")
            
            # Check category configuration
            if "category_column" not in label_config:
                result.warnings.append("category_column not specified, sparse representation may not work correctly")
            
            categories = label_config.get("categories", [])
            if not categories:
                result.warnings.append("categories list is empty")
        
        return result
```

2. **MultiLabelRuleValidator**
```python
class MultiLabelRuleValidator:
    """Validates multi-label rule definitions."""
    
    def validate_rule(self, rule: dict, label_config: dict) -> ValidationResult:
        """
        Validates:
        - output_labels format (dict or "auto")
        - Target columns exist in configuration
        - Category conditions match label columns
        - No conflicting category assignments
        """
        result = ValidationResult()
        
        output_labels = rule.get("output_labels")
        label_mode = label_config.get("label_mode", "single_label")
        
        if label_mode == "single_label":
            # Backward compatibility: check output_label field
            if "output_label" not in rule:
                result.valid = False
                result.errors.append(f"Rule {rule.get('rule_id')}: missing output_label")
        
        elif label_mode == "multi_label":
            if output_labels == "auto":
                # Validate auto mode
                if "output_value" not in rule:
                    result.valid = False
                    result.errors.append(f"Rule {rule.get('rule_id')}: output_value required for auto mode")
                
                if not rule.get("category_conditional", False):
                    result.warnings.append(f"Rule {rule.get('rule_id')}: auto mode without category_conditional")
            
            elif isinstance(output_labels, dict):
                # Validate explicit target columns
                valid_columns = set(label_config.get("output_label_columns", []))
                
                for col in output_labels.keys():
                    if col not in valid_columns:
                        result.valid = False
                        result.errors.append(
                            f"Rule {rule.get('rule_id')}: target column '{col}' not in output_label_columns"
                        )
            
            else:
                result.valid = False
                result.errors.append(
                    f"Rule {rule.get('rule_id')}: output_labels must be dict or 'auto'"
                )
        
        return result
```

3. **CategoryConsistencyValidator**
```python
class CategoryConsistencyValidator:
    """Validates category-label column consistency."""
    
    def validate(self, label_config: dict, rules: List[dict]) -> ValidationResult:
        """
        Validates:
        - Label column names match category convention
        - Each category has at least one rule
        - No orphan label columns
        """
        result = ValidationResult()
        
        if label_config.get("label_mode") != "multi_label":
            return result
        
        categories = set(label_config.get("categories", []))
        output_columns = label_config.get("output_label_columns", [])
        category_column = label_config.get("category_column")
        
        # Check naming convention
        base_name = label_config.get("output_label_name", "")
        if base_name:
            expected_columns = {f"{base_name}_{cat}" for cat in categories}
            actual_columns = set(output_columns)
            
            if expected_columns != actual_columns:
                result.warnings.append(
                    f"Column names don't follow convention: expected {expected_columns}, got {actual_columns}"
                )
        
        # Check rule coverage
        covered_columns = set()
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            
            output_labels = rule.get("output_labels")
            if isinstance(output_labels, dict):
                covered_columns.update(output_labels.keys())
        
        uncovered = set(output_columns) - covered_columns
        if uncovered:
            result.warnings.append(
                f"Label columns without rules: {uncovered}"
            )
        
        return result
```

#### Optimization Extensions

**Multi-Label Rule Optimization:**
```python
def optimize_multilabel_ruleset(
    ruleset: dict,
    enable_category_grouping: bool = True,
    log: Callable[[str], None] = print
) -> dict:
    """
    Optimize multi-label ruleset.
    
    Strategies:
    1. Group rules by target category
    2. Order by category frequency
    3. Optimize per-category complexity
    """
    label_mode = ruleset.get("label_config", {}).get("label_mode", "single_label")
    
    if label_mode == "single_label":
        # Use standard optimization
        return optimize_ruleset(ruleset, log=log)
    
    rules = copy.deepcopy(ruleset.get("ruleset", []))
    
    if enable_category_grouping:
        log("[INFO] Grouping rules by target category...")
        
        # Group rules by target column
        column_rules = {}
        for rule in rules:
            output_labels = rule.get("output_labels")
            if isinstance(output_labels, dict):
                for col in output_labels.keys():
                    column_rules.setdefault(col, []).append(rule)
        
        # Sort each group by complexity
        for col, col_rules in column_rules.items():
            for rule in col_rules:
                rule["complexity_score"] = calculate_complexity(rule.get("conditions", {}))
            col_rules.sort(key=lambda r: r["complexity_score"])
            log(f"  {col}: {len(col_rules)} rules optimized")
        
        # Interleave rules from different categories
        optimized_rules = []
        max_rules_per_col = max(len(rules) for rules in column_rules.values())
        
        for i in range(max_rules_per_col):
            for col_rules in column_rules.values():
                if i < len(col_rules):
                    optimized_rules.append(col_rules[i])
        
        rules = optimized_rules
    
    # Assign final priorities
    for i, rule in enumerate(rules, start=1):
        rule["priority"] = i
    
    return {
        **ruleset,
        "ruleset": rules,
        "optimization_metadata": {
            "multi_label_optimization": True,
            "category_grouping_enabled": enable_category_grouping
        }
    }
```

### 2. Label Ruleset Execution Changes

#### RuleEngine Extension

**Modified `__init__`:**
```python
class RuleEngine:
    def __init__(self, validated_ruleset: dict):
        """Initialize rule engine with multi-label support."""
        self.label_config = validated_ruleset["label_config"]
        self.field_config = validated_ruleset["field_config"]
        self.ruleset = validated_ruleset["ruleset"]
        self.metadata = validated_ruleset.get("metadata", {})
        
        # Filter to enabled rules only
        self.active_rules = [r for r in self.ruleset if r.get("enabled", True)]
        
        # Determine label mode
        self.label_mode = self.label_config.get("label_mode", "single_label")
        
        if self.label_mode == "single_label":
            # Single-label configuration (backward compatible)
            self.output_label_name = self.label_config["output_label_name"]
            self.output_columns = [self.output_label_name]
        
        elif self.label_mode == "multi_label":
            # Multi-label configuration
            self.output_columns = self.label_config["output_label_columns"]
            self.category_column = self.label_config.get("category_column")
            self.categories = self.label_config.get("categories", [])
            self.sparse_representation = self.label_config.get("sparse_representation", True)
        
        # Common configuration
        self.default_label = self.label_config["default_label"]
        self.evaluation_mode = self.label_config.get("evaluation_mode", "priority")
        
        # Statistics tracking (per column)
        self.rule_match_counts = {
            col: {r["rule_id"]: 0 for r in self.active_rules}
            for col in self.output_columns
        }
        self.default_label_counts = {col: 0 for col in self.output_columns}
        self.total_evaluated = 0
```

**Modified `evaluate_row` for Multi-Label:**
```python
def evaluate_row(self, row: pd.Series) -> Dict[str, Any]:
    """
    Evaluate rules against a single row.
    
    Returns:
        - Single-label mode: int (label value)
        - Multi-label mode: Dict[str, Any] (column → value mapping)
    """
    self.total_evaluated += 1
    
    if self.label_mode == "single_label":
        return self._evaluate_row_single_label(row)
    else:
        return self._evaluate_row_multi_label(row)

def _evaluate_row_single_label(self, row: pd.Series) -> int:
    """Evaluate rules for single-label mode (backward compatible)."""
    for rule in self.active_rules:
        try:
            if self._evaluate_conditions(rule["conditions"], row):
                rule_id = rule["rule_id"]
                self.rule_match_counts[self.output_label_name][rule_id] += 1
                return rule["output_label"]
        except Exception as e:
            logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
            continue
    
    self.default_label_counts[self.output_label_name] += 1
    return self.default_label

def _evaluate_row_multi_label(self, row: pd.Series) -> Dict[str, Any]:
    """Evaluate rules for multi-label mode with sparse representation."""
    import numpy as np
    
    # Initialize all columns with NaN (sparse) or default (dense)
    if self.sparse_representation:
        result = {col: np.nan for col in self.output_columns}
    else:
        result = {col: self.default_label for col in self.output_columns}
    
    # Get row's category if category_column specified
    row_category = None
    if self.category_column and self.category_column in row.index:
        row_category = row[self.category_column]
    
    # Evaluate rules in priority order
    for rule in self.active_rules:
        try:
            if not self._evaluate_conditions(rule["conditions"], row):
                continue
            
            # Rule matched - determine output
            output_labels = rule.get("output_labels")
            
            if output_labels == "auto":
                # Auto mode: use category to determine column
                if row_category and self.sparse_representation:
                    # Find matching column for this category
                    target_col = self._get_column_for_category(row_category)
                    if target_col and target_col in result:
                        # Only set if not already set (priority order)
                        if pd.isna(result[target_col]) or result[target_col] == self.default_label:
                            output_value = rule.get("output_value", 1)
                            result[target_col] = output_value
                            self.rule_match_counts[target_col][rule["rule_id"]] += 1
            
            elif isinstance(output_labels, dict):
                # Explicit target columns
                for col, value in output_labels.items():
                    if col not in result:
                        continue
                    
                    # Check if this column matches row's category (for sparse mode)
                    if self.sparse_representation and self.category_column:
                        expected_category = self._get_category_for_column(col)
                        if expected_category and expected_category != row_category:
                            # This column doesn't match row's category, keep as NaN
                            continue
                    
                    # Only set if not already set (priority order)
                    if pd.isna(result[col]) or result[col] == self.default_label:
                        result[col] = value
                        self.rule_match_counts[col][rule["rule_id"]] += 1
        
        except Exception as e:
            logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
            continue
    
    # Fill remaining NaN/default values with default_label (for non-sparse or unmatched)
    for col in result:
        if pd.isna(result[col]):
            self.default_label_counts[col] += 1
            if not self.sparse_representation:
                result[col] = self.default_label
    
    return result

def _get_column_for_category(self, category: str) -> Optional[str]:
    """Map category to output column name."""
    # Try common naming patterns
    for col in self.output_columns:
        if col.endswith(f"_{category}"):
            return col
        if category.lower() in col.lower():
            return col
    return None

def _get_category_for_column(self, column: str) -> Optional[str]:
    """Map output column name to category."""
    # Extract category from column name
    for category in self.categories:
        if column.endswith(f"_{category}"):
            return category
        if category.lower() in column.lower():
            return category
    return None
```

**Modified `evaluate_batch`:**
```python
def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate rules for entire DataFrame.
    
    Returns:
        DataFrame with label column(s) added
    """
    if self.label_mode == "single_label":
        # Single column result (backward compatible)
        df[self.output_label_name] = df.apply(self.evaluate_row, axis=1)
        return df
    
    else:
        # Multi-column result
        results = df.apply(self.evaluate_row, axis=1, result_type='expand')
        
        # Add all label columns to original dataframe
        for col in self.output_columns:
            df[col] = results[col]
        
        return df
```

**Modified Statistics:**
```python
def get_statistics(self) -> Dict[str, Any]:
    """Get execution statistics with multi-label support."""
    if self.label_mode == "single_label":
        # Single-label statistics (backward compatible)
        return {
            "total_evaluated": self.total_evaluated,
            "rule_match_counts": self.rule_match_counts[self.output_label_name],
            "default_label_count": self.default_label_counts[self.output_label_name],
            "rule_match_percentages": {
                rule_id: (count / self.total_evaluated * 100) if self.total_evaluated > 0 else 0
                for rule_id, count in self.rule_match_counts[self.output_label_name].items()
            },
            "default_label_percentage": (
                self.default_label_counts[self.output_label_name] / self.total_evaluated * 100
                if self.total_evaluated > 0 else 0
            )
        }
    
    else:
        # Multi-label statistics (per column)
        stats = {
            "label_mode": "multi_label",
            "total_evaluated": self.total_evaluated,
            "per_column_statistics": {}
        }
        
        for col in self.output_columns:
            col_stats = {
                "rule_match_counts": self.rule_match_counts[col],
                "default_label_count": self.default_label_counts[col],
                "rule_match_percentages": {
                    rule_id: (count / self.total_evaluated * 100) if self.total_evaluated > 0 else 0
                    for rule_id, count in self.rule_match_counts[col].items()
                },
                "default_label_percentage": (
                    self.default_label_counts[col] / self.total_evaluated * 100
                    if self.total_evaluated > 0 else 0
                )
            }
            stats["per_column_statistics"][col] = col_stats
        
        return stats
```

## Usage Examples

### Example 1: Credit Card Fraud Detection (Category-Specific Rules)

**Label Configuration:**
```json
{
  "label_mode": "multi_label",
  "output_label_columns": [
    "is_fraud_CC",
    "is_fraud_DC",
    "is_fraud_ACH"
  ],
  "output_label_type": "binary",
  "label_values": [0, 1],
  "default_label": 0,
  "evaluation_mode": "priority",
  "category_column": "payment_method",
  "categories": ["CC", "DC", "ACH"],
  "sparse_representation": true
}
```

**Rules:**
```json
[
  {
    "rule_id": "R001_CC_overseas",
    "name": "Credit card overseas high value",
    "priority": 1,
    "conditions": {
      "all_of": [
        {"field": "payment_method", "operator": "equals", "value": "CC"},
        {"field": "amount", "operator": ">", "value": 5000},
        {"field": "country", "operator": "not_in", "value": ["US", "CA"]}
      ]
    },
    "output_labels": {
      "is_fraud_CC": 1
    }
  },
  {
    "rule_id": "R002_CC_new_customer",
    "name": "Credit card new customer high value",
    "priority": 2,
    "conditions": {
      "all_of": [
        {"field": "payment_method", "operator": "equals", "value": "CC"},
        {"field": "amount", "operator": ">", "value": 2000},
        {"field": "customer_age_days", "operator": "<", "value": 30}
      ]
    },
    "output_labels": {
      "is_fraud_CC": 1
    }
  },
  {
    "rule_id": "R003_DC_unusual_pattern",
    "name": "Debit card unusual transaction pattern",
    "priority": 3,
    "conditions": {
      "all_of": [
        {"field": "payment_method", "operator": "equals", "value": "DC"},
        {"field": "transactions_last_hour", "operator": ">", "value": 5},
        {"field": "amount", "operator": ">", "value": 1000}
      ]
    },
    "output_labels": {
      "is_fraud_DC": 1
    }
  },
  {
    "rule_id": "R004_ACH_large_transfer",
    "name": "ACH large transfer",
    "priority": 4,
    "conditions": {
      "all_of": [
        {"field": "payment_method", "operator": "equals", "value": "ACH"},
        {"field": "amount", "operator": ">", "value": 10000}
      ]
    },
    "output_labels": {
      "is_fraud_ACH": 1
    }
  }
]
```

**Input Data:**
```
| txn_id | payment_method | amount | country | customer_age_days | transactions_last_hour |
|--------|----------------|--------|---------|-------------------|------------------------|
| 1      | CC             | 6000   | UK      | 45                | 1                      |
| 2      | CC             | 3000   | US      | 15                | 2                      |
| 3      | DC             | 1500   | US      | 100               | 8                      |
| 4      | ACH            | 12000  | US      | 200               | 1                      |
```

**Output Data:**
```
| txn_id | payment_method | ... | is_fraud_CC | is_fraud_DC | is_fraud_ACH |
|--------|----------------|-----|-------------|-------------|--------------|
| 1      | CC             | ... | 1           | NaN         | NaN          |  # R001 matched
| 2      | CC             | ... | 1           | NaN         | NaN          |  # R002 matched
| 3      | DC             | ... | NaN         | 1           | NaN          |  # R003 matched
| 4      | ACH            | ... | NaN         | NaN         | 1            |  # R004 matched
```

### Example 2: Auto Mode (Simplified Rules)

**Rules with Auto Mode:**
```json
[
  {
    "rule_id": "R001_auto_high_value",
    "name": "High value transaction (auto-category)",
    "priority": 1,
    "conditions": {
      "field": "amount",
      "operator": ">",
      "value": 5000
    },
    "output_labels": "auto",
    "output_value": 1,
    "category_conditional": true
  }
]
```

**Behavior**: Rule automatically targets the column matching the row's payment method.
- CC transaction → sets `is_fraud_CC`
- DC transaction → sets `is_fraud_DC`  
- ACH transaction → sets `is_fraud_ACH`

## Comparison: Simple vs Ruleset-Based Multilabel

### Simple Category-Based Multilabel Preprocessing

**Strengths:**
- ✅ Simple to implement and understand
- ✅ Fast execution (just category filtering)
- ✅ Good for basic category decomposition
- ✅ No rule definition required

**Limitations:**
- ❌ Only copies existing labels
- ❌ No business logic possible
- ❌ Cannot encode domain knowledge
- ❌ Limited to exact category matching
- ❌ No sophisticated label generation

**Use Cases:**
- Simple multi-task learning setups
- When labels are already correct
- Quick prototyping
- No domain expertise needed

### Ruleset-Based Multilabel Extension

**Strengths:**
- ✅ Complex category-specific business logic
- ✅ Domain knowledge encoding per category
- ✅ Validation and optimization built-in
- ✅ Transparent and auditable
- ✅ Maintainable by business users
- ✅ Category-specific fraud patterns
- ✅ All benefits of rule-based systems

**Limitations:**
- ❌ More complex initial setup
- ❌ Requires rule definition effort
- ❌ Overkill for simple cases
- ❌ Performance overhead for complex rules

**Use Cases:**
- Sophisticated fraud detection
- Category-specific risk scoring
- Domain expert-driven label generation
- Need for auditability and transparency
- Regulatory compliance requirements

## Implementation Checklist

### Phase 1: Extend Generation Script
- [ ] Add `label_mode` field to label_config schema
- [ ] Add multi-label specific fields (output_label_columns, categories, etc.)
- [ ] Implement MultiLabelConfigValidator
- [ ] Implement MultiLabelRuleValidator
- [ ] Implement CategoryConsistencyValidator
- [ ] Extend rule validation for output_labels format
- [ ] Implement multi-label rule optimization
- [ ] Update validated_ruleset.json format
- [ ] Add multi-label validation tests

### Phase 2: Extend Execution Script
- [ ] Extend RuleEngine.__init__ for multi-label mode
- [ ] Implement _evaluate_row_multi_label method
- [ ] Implement _get_column_for_category helper
- [ ] Implement _get_category_for_column helper
- [ ] Update evaluate_batch for multi-column output
- [ ] Update get_statistics for per-column tracking
- [ ] Handle sparse representation (NaN) correctly
- [ ] Add auto mode support
- [ ] Implement per-column rule match tracking
- [ ] Add multi-label execution tests

### Phase 3: Contract Updates
- [ ] Update generation contract documentation
- [ ] Update execution contract documentation
- [ ] Add multi-label mode examples
- [ ] Document new environment variables
- [ ] Update expected output formats

### Phase 4: Configuration Integration
- [ ] Extend LabelRulesetGenerationConfig
- [ ] Extend LabelRulesetExecutionConfig  
- [ ] Add multi-label specific fields to configs
- [ ] Update config validators
- [ ] Add backward compatibility tests

### Phase 5: Testing & Documentation
- [ ] Unit tests for multi-label validation
- [ ] Unit tests for multi-label execution
- [ ] Integration tests with multi-task training
- [ ] Performance benchmarks
- [ ] Update user documentation
- [ ] Create usage examples
- [ ] Add troubleshooting guide

## Performance Considerations

### Computational Efficiency

**Sparse Representation:**
- Memory: ~1.2x single-label (sparse columns compress well)
- Execution: ~1.5-2x single-label (multiple column evaluation)
- Optimization: Rule grouping by category reduces redundant evaluation

**Dense Representation:**
- Memory: ~3x single-label for 3 categories (all columns populated)
- Execution: ~2-3x single-label
- Not recommended for >5 categories

### Scalability

**Scales well with:**
- Number of rows (linear complexity)
- Number of rules per category (<20 optimal)
- Sparse representation (NaN for non-applicable)

**Performance degrades with:**
- Too many categories (>10)
- Too many rules per category (>50)
- Dense representation with many categories
- Complex nested conditions

### Optimization Strategies

1. **Category Grouping**: Group rules by target category
2. **Complexity Ordering**: Simple rules first within each category
3. **Sparse Evaluation**: Skip non-matching categories early
4. **Vectorization**: Batch processing where possible
5. **Caching**: Cache category-column mappings

## Migration Path

### From Single-Label to Multi-Label

**Step 1: Keep Single-Label Mode (Default)**
```json
{
  "label_mode": "single_label",
  "output_label_name": "is_fraud",
  ...
}
```

**Step 2: Add Category-Specific Rules**
```json
{
  "label_mode": "multi_label",
  "output_label_columns": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
  "category_column": "payment_method",
  "categories": ["CC", "DC", "ACH"],
  ...
}
```

**Step 3: Migrate Rules**
- Convert single-label rules to multi-label format
- Add category conditions where appropriate
- Test with small dataset first
- Validate output matches expectations

### Backward Compatibility

**Guaranteed:**
- Existing single-label configurations work unchanged
- Default `label_mode` is "single_label"
- Single-label rule format unchanged
- Existing validation logic preserved

**Migration Support:**
- Utility script to convert single→multi-label config
- Validation warnings for potential issues
- Gradual migration per rule possible

## Future Enhancements

### Planned Improvements

1. **Hierarchical Categories**
   - Support category hierarchies (e.g., payment_type → payment_method)
   - Enable multi-level multi-labeling

2. **Dynamic Category Detection**
   - Auto-detect categories from data
   - Suggest column names automatically

3. **Rule Templates**
   - Pre-defined rule templates per category
   - Easier rule creation for common patterns

4. **Advanced Optimization**
   - ML-based rule ordering
   - Sample-based optimization
   - Adaptive priority adjustment

5. **Multi-Label Metrics**
   - Per-category precision/recall
   - Rule effectiveness tracking
   - A/B testing support

6. **Integration Enhancements**
   - Direct integration with MTGBM training
   - Automatic hyperparameter suggestions
   - Pipeline optimization

## Best Practices

### 1. Rule Design

**Category-Specific Rules:**
```python
# Good: Explicit category check in conditions
{
  "conditions": {
    "all_of": [
      {"field": "payment_method", "operator": "equals", "value": "CC"},
      {"field": "amount", "operator": ">", "value": 1000}
    ]
  },
  "output_labels": {"is_fraud_CC": 1}
}

# Also Good: Use auto mode with category_conditional
{
  "conditions": {
    "field": "amount",
    "operator": ">",
    "value": 1000
  },
  "output_labels": "auto",
  "output_value": 1,
  "category_conditional": true
}
```

### 2. Validation

**Always validate:**
- Label column naming matches convention
- Each category has at least one rule
- No conflicting rules per category
- Category values match data

### 3. Testing

**Test with:**
- Representative sample from each category
- Edge cases (missing categories, null values)
- Performance benchmarks
- Validation reports

### 4. Documentation

**Document:**
- Business logic behind each rule
- Category-specific patterns
- Expected label distributions
- Rule modification history

## Troubleshooting Guide

### Common Issues

#### Issue 1: Rules Not Firing for Category

**Symptoms**: All labels remain NaN or default value for a category

**Solutions**:
- Check category value in data matches rule conditions exactly
- Verify category_column specified correctly
- Check sparse_representation setting
- Review rule priorities

#### Issue 2: Multiple Labels Set Per Row

**Symptoms**: Row has multiple non-NaN label values

**Solutions**:
- Verify sparse_representation=true
- Check category_column configuration
- Review _get_category_for_column logic
- Ensure rules have correct category conditions

#### Issue 3: Performance Issues

**Symptoms**: Slow execution, high memory usage

**Solutions**:
- Enable category grouping optimization
- Reduce number of rules per category
- Use sparse representation
- Simplify complex nested conditions
- Consider caching strategies

#### Issue 4: Validation Failures

**Symptoms**: Ruleset generation fails validation

**Solutions**:
- Check output_label_columns list is complete
- Verify categories list matches data
- Ensure no duplicate column names
- Review rule target columns

## Conclusion

The Label Ruleset Multilabel Extension provides a powerful, systematic approach to multi-label generation for multi-task learning. Key benefits include:

- **Business Logic Integration**: Encode domain knowledge per category
- **Validation & Optimization**: Automatic rule checking and optimization
- **Sparse Efficiency**: Efficient sparse representation with NaN
- **Backward Compatibility**: Seamless integration with existing system
- **Auditability**: Track which rules fire for which categories
- **Flexibility**: Support both explicit and auto category modes

This extension enables sophisticated multi-label classification scenarios while maintaining the transparency, maintainability, and validation benefits of rule-based systems.

## References

### Related Design Documents
- [Multilabel Preprocessing Step Design](./multilabel_preprocessing_step_design.md)
- [LightGBM Multi-Task Training Step Design](./lightgbm_multi_task_training_step_design.md)
- [MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)

### Implementation References
- [Label Ruleset Generation Script](../../src/cursus/steps/scripts/label_ruleset_generation.py)
- [Label Ruleset Execution Script](../../src/cursus/steps/scripts/label_ruleset_execution.py)
- [Label Ruleset Generation Contract](../../src/cursus/steps/contracts/label_ruleset_generation_contract.py)
- [Label Ruleset Execution Contract](../../src/cursus/steps/contracts/label_ruleset_execution_contract.py)

---

*This design document provides comprehensive specification for extending the Label Ruleset system to support multi-label output for multi-task learning, enabling sophisticated category-conditional rule-based classification with domain knowledge integration.*
