---
tags:
  - design
  - step_builders
  - ruleset_execution
  - rule_engine
  - patterns
  - sagemaker
  - data_preprocessing
  - label_generation
keywords:
  - ruleset execution
  - rule-based label mapping
  - rule engine
  - label generation
  - batch processing
topics:
  - step builder patterns
  - ruleset execution implementation
  - rule engine design
  - SageMaker data processing
language: python
date of note: 2025-11-09
updated: 2025-11-09
---

# Label Ruleset Execution Step Builder Patterns

## Overview

This document defines the design patterns for the **RulesetExecutor** step builder implementation in the cursus framework. The RulesetExecutor step creates **ProcessingStep** instances that apply pre-validated rulesets to data, transforming multi-source input data (LLM outputs, human annotations, tabular features) into numerical classification labels. This step focuses on efficient execution of validated rules at scale.

## Purpose and Motivation

### Problem Statement

After rules are validated and structured (by RulesetGenerator), we need to:

1. **Apply rules efficiently** to large datasets
2. **Handle train/val/test splits** appropriately
3. **Generate labeled data** for model training
4. **Track execution statistics** for monitoring
5. **Provide debugging information** when needed

### Separation of Concerns

By separating execution from generation:
- **Generation**: Validates once (expensive, comprehensive)
- **Execution**: Applies many times (optimized, fast)
- **Benefits**: Better performance, clearer code, easier debugging

## Integration with Data Processing Ecosystem

The RulesetExecutor integrates as follows:

```
Validated Ruleset + Processed Data → RulesetExecutor → Labeled Data
                                                            ↓
                                                    Training Step
```

**Full Pipeline Integration:**
```
Raw Data → Tabular Preprocessing → Processed Data (train/val/test)
                                         ↓
Bedrock Processing → LLM Outputs
                         ↓
User Rules → RulesetGenerator → Validated Ruleset
                                       ↓
        Validated Ruleset + Processed Data + LLM Outputs
                                       ↓
                              RulesetExecutor
                                       ↓
                              Labeled Data (train/val/test)
                                       ↓
                              Training Step
```

## Input Requirements

### 1. Validated Ruleset (from RulesetGenerator)

```json
{
  "version": "1.0",
  "generated_timestamp": "2025-11-09T11:30:00Z",
  "label_config": {
    "output_label_name": "final_reversal_flag",
    "output_label_type": "binary",
    "label_values": [0, 1],
    "label_mapping": {
      "0": "No_Reversal",
      "1": "Reversal"
    },
    "default_label": 1,
    "evaluation_mode": "priority"
  },
  "field_config": {
    "required_fields": ["category", "confidence_score", "reversal_flag", "conc_si"],
    "field_types": {
      "category": "string",
      "confidence_score": "float",
      "reversal_flag": "int",
      "conc_si": "float"
    }
  },
  "ruleset": [
    {
      "rule_id": "rule_001",
      "name": "High confidence TrueDNR",
      "priority": 1,
      "enabled": true,
      "conditions": {
        "all_of": [
          {
            "field": "category",
            "operator": "equals",
            "value": "TrueDNR"
          },
          {
            "field": "confidence_score",
            "operator": ">=",
            "value": 0.8
          }
        ]
      },
      "output_label": 0,
      "description": "High confidence TrueDNR cases indicate no reversal",
      "complexity_score": 2
    }
  ],
  "metadata": {
    "total_rules": 5,
    "enabled_rules": 4,
    "disabled_rules": 1,
    "field_usage": {
      "category": 3,
      "confidence_score": 5,
      "reversal_flag": 2,
      "conc_si": 1
    },
    "validation_summary": {
      "field_validation": "passed_at_config_level",
      "label_validation": "passed",
      "logic_validation": "passed",
      "warnings": []
    }
  }
}
```

**Key Points:**
- `field_config` contains only `required_fields` and `field_types` (no `optional_fields`)
- All fields used in rules are marked as required
- Rules include `complexity_score` from optimization
- Metadata includes field usage statistics

### 2. Processed Data

```
INPUT_PROCESSED_DATA/
├── train/
│   └── train_processed_data.csv
├── val/
│   └── val_processed_data.csv
└── test/
    └── test_processed_data.csv
```

Each CSV contains columns referenced by the ruleset (e.g., category, confidence_score, reversal_flag, etc.)

## Output Structure

### Processed Data with Labels

To support stacked preprocessing steps, the output uses the same logical name as input (`processed_data`), allowing seamless chaining:

```
OUTPUT_PROCESSED_DATA/
├── train/
│   └── train_processed_data.csv (original + label column)
├── val/
│   └── val_processed_data.csv (original + label column)
├── test/
│   └── test_processed_data.csv (original + label column)
├── execution_report.json
└── rule_match_statistics.json (optional)
```

**Key Design Decision:** Using `processed_data` for both input and output enables preprocessing pipeline composition:
- **Risk Table Mapping**: `processed_data` → `processed_data` (adds risk columns)
- **Missing Value Imputation**: `processed_data` → `processed_data` (imputes nulls)
- **Stratified Sampling**: `processed_data` → `processed_data` (balances classes)
- **Ruleset Executor**: `processed_data` → `processed_data` (adds labels)

Each step reads `processed_data`, transforms it, and outputs `processed_data` for the next step.

## Field Availability Validation (Execution-Time)

Before executing rules, the execution step validates that all required fields exist in the actual data:

```python
class RulesetFieldValidator:
    """Validates field availability in actual data at execution time."""
    
    def validate_fields(
        self, 
        ruleset: dict,
        data_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validates all field references exist in actual data.
        
        This is an EXECUTION-TIME validator that checks:
        - All required fields exist in DataFrame
        - All fields used in rules exist in DataFrame  
        - Field null percentages (data quality check)
        
        This validator does NOT check:
        - Field type declarations (that's generation-time)
        - Rule logic errors (that's generation-time)
        
        Args:
            ruleset: Validated ruleset configuration
            data_df: Actual DataFrame to check
            
        Returns:
            Dictionary with validation results:
            - valid: bool
            - missing_fields: List[str] - fields referenced but not in data
            - warnings: List[str] - high null percentages, etc.
        """
        result = {
            "valid": True,
            "missing_fields": [],
            "warnings": []
        }
        
        field_config = ruleset.get("field_config", {})
        required_fields = set(field_config.get("required_fields", []))
        optional_fields = set(field_config.get("optional_fields", []))
        
        rules = ruleset.get("rules", [])
        
        # Extract all field references from rules
        used_fields = set()
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            fields = self._extract_fields_from_conditions(rule.get("conditions", {}))
            used_fields.update(fields)
        
        # Check field availability in data
        available_fields = set(data_df.columns)
        
        # Check required fields exist
        missing_required = required_fields - available_fields
        if missing_required:
            result["valid"] = False
            result["missing_fields"].extend(list(missing_required))
            logger.error(f"Required fields missing in data: {missing_required}")
        
        # Check used fields exist
        missing_used = used_fields - available_fields
        if missing_used:
            result["valid"] = False
            result["missing_fields"].extend(list(missing_used))
            logger.error(f"Fields used in rules but not in data: {missing_used}")
        
        # Check for high null percentages
        for field in used_fields & available_fields:
            null_pct = data_df[field].isnull().sum() / len(data_df)
            if null_pct > 0.5:
                result["warnings"].append(
                    f"Field '{field}' has {null_pct:.1%} null values"
                )
                logger.warning(f"Field '{field}' has {null_pct:.1%} null values")
        
        return result
    
    def _extract_fields_from_conditions(self, condition: dict) -> List[str]:
        """Recursively extract all field names from a condition."""
        fields = []
        
        if "all_of" in condition:
            for subcond in condition["all_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "any_of" in condition:
            for subcond in condition["any_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "none_of" in condition:
            for subcond in condition["none_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "field" in condition:
            fields.append(condition["field"])
        
        return fields
```

**Key Points:**
- This validation happens at execution time, not generation time
- Checks actual DataFrame columns, not just schema declarations
- Fails fast if required fields are missing
- Provides warnings for data quality issues (high null percentages)

## Rule Evaluation Engine

### Core Engine Implementation

```python
class RuleEngine:
    """
    Evaluates validated rules against data rows to produce labels.
    
    Optimized for:
    - Batch processing (vectorized where possible)
    - Priority-based evaluation (first match wins)
    - Efficient condition checking
    - Minimal memory footprint
    """
    
    def __init__(self, validated_ruleset: dict):
        """
        Initialize rule engine with validated ruleset.
        
        Args:
            validated_ruleset: Pre-validated ruleset from RulesetGenerator
        """
        # Extract configuration
        self.label_config = validated_ruleset['label_config']
        self.field_config = validated_ruleset['field_config']
        self.ruleset = validated_ruleset['ruleset']
        self.metadata = validated_ruleset.get('metadata', {})
        
        # Filter to enabled rules only (already sorted by priority)
        self.active_rules = [r for r in self.ruleset if r.get('enabled', True)]
        
        # Configuration
        self.output_label_name = self.label_config['output_label_name']
        self.default_label = self.label_config['default_label']
        self.evaluation_mode = self.label_config.get('evaluation_mode', 'priority')
        
        # Statistics tracking
        self.rule_match_counts = {r['rule_id']: 0 for r in self.active_rules}
        self.default_label_count = 0
        self.total_evaluated = 0
        
    def evaluate_row(self, row: pd.Series) -> int:
        """
        Evaluate rules against a single DataFrame row.
        
        Args:
            row: DataFrame row as Series
            
        Returns:
            Numerical label value (int)
        """
        self.total_evaluated += 1
        
        # Evaluate rules in priority order
        for rule in self.active_rules:
            try:
                if self._evaluate_conditions(rule['conditions'], row):
                    # Match found
                    self.rule_match_counts[rule['rule_id']] += 1
                    return rule['output_label']
            except Exception as e:
                # Log but continue (fail-safe approach)
                logger.warning(
                    f"Error evaluating rule {rule['rule_id']}: {e}"
                )
                continue
        
        # No rule matched, use default
        self.default_label_count += 1
        return self.default_label
    
    def evaluate_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate rules for entire DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series of label values, one per row
        """
        return df.apply(self.evaluate_row, axis=1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_evaluated": self.total_evaluated,
            "rule_match_counts": self.rule_match_counts,
            "default_label_count": self.default_label_count,
            "rule_match_percentages": {
                rule_id: (count / self.total_evaluated * 100) if self.total_evaluated > 0 else 0
                for rule_id, count in self.rule_match_counts.items()
            },
            "default_label_percentage": (
                (self.default_label_count / self.total_evaluated * 100) 
                if self.total_evaluated > 0 else 0
            )
        }
```

### Condition Evaluation

```python
def _evaluate_conditions(self, conditions: dict, row: pd.Series) -> bool:
    """
    Recursively evaluate nested conditions.
    
    Args:
        conditions: Condition dictionary with logical operators
        row: DataFrame row as Series
        
    Returns:
        Boolean indicating whether conditions are satisfied
    """
    # Handle logical operators
    if 'all_of' in conditions:
        return all(
            self._evaluate_conditions(cond, row) 
            for cond in conditions['all_of']
        )
    
    elif 'any_of' in conditions:
        return any(
            self._evaluate_conditions(cond, row) 
            for cond in conditions['any_of']
        )
    
    elif 'none_of' in conditions:
        return not any(
            self._evaluate_conditions(cond, row) 
            for cond in conditions['none_of']
        )
    
    # Handle leaf condition (field comparison)
    else:
        return self._evaluate_leaf_condition(conditions, row)

def _evaluate_leaf_condition(self, condition: dict, row: pd.Series) -> bool:
    """
    Evaluate a single leaf condition (field comparison).
    
    Args:
        condition: Single condition with field, operator, value
        row: DataFrame row as Series
        
    Returns:
        Boolean indicating whether condition is satisfied
    """
    field = condition['field']
    operator = condition['operator']
    expected_value = condition['value']
    
    # Get actual value from row
    if field not in row.index:
        return False
    
    actual_value = row[field]
    
    # Handle null values
    if pd.isna(actual_value):
        if operator == 'is_null':
            return True
        elif operator == 'is_not_null':
            return False
        else:
            return False  # Null doesn't match comparisons
    
    # Apply operator (see supported operators below)
    return self._apply_operator(operator, actual_value, expected_value)
```

### Supported Operators

```python
def _apply_operator(
    self, 
    operator: str, 
    actual: Any, 
    expected: Any
) -> bool:
    """Apply comparison operator."""
    
    # Comparison operators
    if operator == 'equals':
        return actual == expected
    elif operator == 'not_equals':
        return actual != expected
    elif operator == '>':
        return float(actual) > float(expected)
    elif operator == '>=':
        return float(actual) >= float(expected)
    elif operator == '<':
        return float(actual) < float(expected)
    elif operator == '<=':
        return float(actual) <= float(expected)
    
    # Collection operators
    elif operator == 'in':
        return actual in expected
    elif operator == 'not_in':
        return actual not in expected
    
    # String operators
    elif operator == 'contains':
        return str(expected) in str(actual)
    elif operator == 'not_contains':
        return str(expected) not in str(actual)
    elif operator == 'starts_with':
        return str(actual).startswith(str(expected))
    elif operator == 'ends_with':
        return str(actual).endswith(str(expected))
    elif operator == 'regex_match':
        import re
        return bool(re.search(expected, str(actual)))
    
    # Null operators
    elif operator == 'is_null':
        return False  # Already handled null case
    elif operator == 'is_not_null':
        return True  # Already handled null case
    
    else:
        raise ValueError(f"Unsupported operator: {operator}")
```

## Job Type Variants

The RulesetExecutor step supports multiple job type variants to handle different execution scenarios:

### Supported Job Types

Aligned with tabular preprocessing step's job types:

| Job Type | Description | Splits Processed | Use Case |
|----------|-------------|-----------------|----------|
| `training` | Full pipeline execution | train, val, test | Complete model training pipeline |
| `validation` | Validation split only | val | Validation set labeling |
| `testing` | Test split only | test | Test set labeling |
| `calibration` | Calibration split only | calibration | Calibration set labeling |

### Job Type Behavior

```python
# In main() function
if job_args.job_type == "training":
    # Process all splits for complete training pipeline
    # Training mode creates three splits: train, val, test
    splits = ["train", "val", "test"]
else:
    # Process single specified split using job_type as directory name
    # validation → validation/
    # testing → testing/
    # calibration → calibration/
    splits = [job_args.job_type]
```

**Important:** For non-training job types, the directory name matches the job_type exactly:
- `validation` job type → `validation/` directory (not `val/`)
- `testing` job type → `testing/` directory (not `test/`)
- `calibration` job type → `calibration/` directory

This aligns with tabular preprocessing output structure where:
- Training mode: creates `train/`, `val/`, `test/` directories
- Other modes: create directory matching job_type name (e.g., `validation/`, `testing/`, `calibration/`)

### Configuration Examples

**Training Pipeline (all splits):**
```python
ruleset_executor_config = RulesetExecutorConfig(
    job_type="training",
    # ... other config
)
```

**Single Split Processing:**
```python
# Process validation split only
ruleset_executor_config = RulesetExecutorConfig(
    job_type="validation",
    # ... other config
)

# Process test split only
ruleset_executor_config = RulesetExecutorConfig(
    job_type="testing",
    # ... other config
)

# Process calibration split only
ruleset_executor_config = RulesetExecutorConfig(
    job_type="calibration",
    # ... other config
)
```

### Command Line Arguments

The script accepts job_type as a command line argument, matching tabular preprocessing conventions:

```bash
# Training mode (all splits: train, val, test)
python ruleset_execution.py --job-type training

# Single splits
python ruleset_execution.py --job-type validation  # Processes val/ directory
python ruleset_execution.py --job-type testing     # Processes test/ directory
python ruleset_execution.py --job-type calibration # Processes calibration/ directory
```

## Script Implementation

### Main Processing Logic

```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for ruleset execution.
    
    Args:
        input_paths: Dictionary with keys:
            - "validated_ruleset": Path to validated ruleset JSON
            - "processed_data": Directory with train/val/test splits
        output_paths: Dictionary with keys:
            - "labeled_data": Directory for output with labels
            - "execution_report": Path for execution statistics
            - "rule_match_statistics": Optional path for detailed statistics
        environ_vars: Environment variables
        job_args: Command line arguments (job_type)
        logger: Optional logger function
        
    Returns:
        Dictionary of processed DataFrames by split name
    """
    log = logger or print
    
    # 1. Load validated ruleset
    ruleset_path = input_paths["validated_ruleset"]
    with open(ruleset_path, 'r') as f:
        validated_ruleset = json.load(f)
    
    log(f"[INFO] Loaded validated ruleset v{validated_ruleset.get('version')}")
    log(f"[INFO] Rules: {validated_ruleset['metadata']['enabled_rules']} enabled")
    
    # 2. Initialize field validator
    field_validator = RulesetFieldValidator()
    log(f"[INFO] Initialized field validator")
    
    # 3. Initialize rule engine
    rule_engine = RuleEngine(validated_ruleset)
    log(f"[INFO] Initialized rule engine")
    
    # 4. Determine splits to process
    input_dir = Path(input_paths["processed_data"])
    output_dir = Path(output_paths["labeled_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if job_args.job_type == "training":
        splits = ["train", "val", "test"]
    else:
        splits = [job_args.job_type]
    
    # 4. Process each split
    processed_splits = {}
    split_statistics = {}
    
    for split_name in splits:
        log(f"[INFO] Processing {split_name} split...")
        
        # Load data
        split_dir = input_dir / split_name
        if not split_dir.exists():
            log(f"[WARNING] Split directory not found: {split_dir}")
            continue
        
        data_file = split_dir / f"{split_name}_processed_data.csv"
        if not data_file.exists():
            log(f"[WARNING] Data file not found: {data_file}")
            continue
        
        df = pd.read_csv(data_file)
        log(f"[INFO] Loaded {split_name}: {df.shape}")
        
        # Validate field availability in data
        validation_result = field_validator.validate_fields(validated_ruleset, df)
        if not validation_result["valid"]:
            error_msg = f"Field validation failed for {split_name}: {validation_result['missing_fields']}"
            log(f"[ERROR] {error_msg}")
            if environ_vars.get("FAIL_ON_MISSING_FIELDS", "true").lower() == "true":
                raise ValueError(error_msg)
            else:
                log(f"[WARNING] Skipping {split_name} due to validation failure")
                continue
        
        # Log warnings if any
        for warning in validation_result.get("warnings", []):
            log(f"[WARNING] {warning}")
        
        log(f"[INFO] Field validation passed for {split_name}")
        
        # Apply rules to generate labels
        output_label_name = rule_engine.output_label_name
        df[output_label_name] = rule_engine.evaluate_batch(df)
        
        # Compute label distribution
        label_dist = df[output_label_name].value_counts().to_dict()
        log(f"[INFO] {split_name} label distribution: {label_dist}")
        
        # Save statistics
        split_statistics[split_name] = {
            "total_rows": len(df),
            "label_distribution": label_dist,
            "execution_stats": rule_engine.get_statistics()
        }
        
        # Reset engine statistics for next split
        rule_engine.rule_match_counts = {
            r['rule_id']: 0 for r in rule_engine.active_rules
        }
        rule_engine.default_label_count = 0
        rule_engine.total_evaluated = 0
        
        # Save labeled data
        output_split_dir = output_dir / split_name
        output_split_dir.mkdir(exist_ok=True)
        output_file = output_split_dir / f"{split_name}_processed_data.csv"
        df.to_csv(output_file, index=False)
        log(f"[INFO] Saved {output_file}")
        
        processed_splits[split_name] = df
    
    # 5. Save execution report
    execution_report = {
        "ruleset_version": validated_ruleset.get('version'),
        "ruleset_timestamp": validated_ruleset.get('generated_timestamp'),
        "execution_timestamp": datetime.now().isoformat(),
        "label_config": validated_ruleset['label_config'],
        "split_statistics": split_statistics,
        "total_rules_evaluated": validated_ruleset['metadata']['enabled_rules']
    }
    
    report_path = Path(output_paths["execution_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(execution_report, f, indent=2)
    log(f"[INFO] Saved execution report: {report_path}")
    
    # 6. Save detailed rule match statistics (optional)
    if "rule_match_statistics" in output_paths:
        stats_path = Path(output_paths["rule_match_statistics"])
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(split_statistics, f, indent=2)
        log(f"[INFO] Saved rule match statistics: {stats_path}")
    
    log("[INFO] Ruleset execution complete")
    return processed_splits
```

## Configuration Class

```python
from pydantic import Field
from typing import Optional
from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase


class RulesetExecutorConfig(ProcessingStepConfigBase):
    """
    Configuration for RulesetExecutor step.
    
    Inherits from ProcessingStepConfigBase which provides:
    - Base pipeline configuration
    - Processing instance settings
    - Script configuration
    
    Adds ruleset execution-specific configuration.
    """
    
    # ===== System Inputs with Defaults (Tier 2) =====
    
    # Execution Options
    enable_rule_match_tracking: bool = Field(
        default=True,
        description="Track which rules match for statistics"
    )
    
    save_rule_match_details: bool = Field(
        default=False,
        description="Save detailed per-row rule match information (can be large)"
    )
    
    # Performance Options
    batch_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=100000,
        description="Process data in batches for memory efficiency (None = all at once)"
    )
    
    enable_progress_logging: bool = Field(
        default=True,
        description="Log progress during batch processing"
    )
    
    # Error Handling
    fail_on_missing_fields: bool = Field(
        default=True,
        description="Fail if expected fields are missing from data"
    )
    
    skip_rows_with_errors: bool = Field(
        default=False,
        description="Skip rows that cause evaluation errors (use default label)"
    )
    
    # Output Configuration
    include_execution_statistics: bool = Field(
        default=True,
        description="Include execution statistics in output"
    )
    
    save_label_distribution: bool = Field(
        default=True,
        description="Save label distribution per split"
    )
    
    # Override default processing entry point
    processing_entry_point: Optional[str] = Field(
        default="ruleset_execution.py",
        description="Entry point script for ruleset execution"
    )
```

## Environment Variables

```bash
# Execution Options
ENABLE_RULE_MATCH_TRACKING="true"
SAVE_RULE_MATCH_DETAILS="false"

# Performance Options
BATCH_SIZE=""  # Empty = process all at once
ENABLE_PROGRESS_LOGGING="true"

# Error Handling
FAIL_ON_MISSING_FIELDS="true"
SKIP_ROWS_WITH_ERRORS="false"

# Output Configuration
INCLUDE_EXECUTION_STATISTICS="true"
SAVE_LABEL_DISTRIBUTION="true"
```

## Input/Output Structure

### Input Structure

```
INPUT_VALIDATED_RULESET/
└── validated_ruleset.json

INPUT_PROCESSED_DATA/
├── train/
│   └── train_processed_data.csv
├── val/
│   └── val_processed_data.csv
└── test/
    └── test_processed_data.csv
```

### Output Structure

```
OUTPUT_LABELED_DATA/
├── train/
│   └── train_processed_data.csv (with label column)
├── val/
│   └── val_processed_data.csv (with label column)
├── test/
│   └── test_processed_data.csv (with label column)
├── execution_report.json
└── rule_match_statistics.json (optional)
```

### Execution Report Format

```json
{
  "ruleset_version": "1.0",
  "ruleset_timestamp": "2025-11-09T11:30:00Z",
  "execution_timestamp": "2025-11-09T12:00:00Z",
  "label_config": { /* label configuration */ },
  "split_statistics": {
    "train": {
      "total_rows": 10000,
      "label_distribution": {"0": 6234, "1": 3766},
      "execution_stats": {
        "total_evaluated": 10000,
        "rule_match_counts": {
          "rule_001": 4521,
          "rule_002": 1713,
          "rule_003": 3012
        },
        "default_label_count": 754,
        "rule_match_percentages": {
          "rule_001": 45.21,
          "rule_002": 17.13,
          "rule_003": 30.12
        },
        "default_label_percentage": 7.54
      }
    }
  },
  "total_rules_evaluated": 9
}
```

## Pipeline Integration

### Step Dependencies and Input/Output Mapping

The RulesetExecutor step depends on TWO upstream steps:

1. **TabularPreprocessingStep**: Provides processed data (train/val/test splits)
   - Output: `processed_data` 
   
2. **RulesetGeneratorStep**: Provides validated ruleset
   - Output: `validated_ruleset`

### Step Creation Example (Simple Pipeline)

```python
# Step 1: Tabular preprocessing - produces processed data splits
tabular_preprocessing_step = TabularPreprocessingStepBuilder(
    config=preprocessing_config
).create_step(
    dependencies=[data_load_step]
)

# Step 2: Generate validated ruleset
ruleset_generator_step = RulesetGeneratorStepBuilder(
    config=generator_config
).create_step(
    inputs={
        'ruleset_configs': 's3://bucket/rules/configs/',  # Auto-generated from config
    },
    outputs={
        'validated_ruleset': 's3://bucket/rulesets/validated_ruleset.json',
        'validation_report': 's3://bucket/reports/validation_report.json'
    }
)

# Step 3: Execute ruleset on processed data (DEPENDS ON BOTH STEPS)
# Output: processed_data (with labels) - ready for training or further preprocessing
ruleset_executor_step = RulesetExecutorStepBuilder(
    config=executor_config
).create_step(
    inputs={
        'validated_ruleset': ruleset_generator_step.properties
                             .ProcessingOutputConfig
                             .Outputs['validated_ruleset']
                             .S3Output.S3Uri,
        'processed_data': tabular_preprocessing_step.properties
                          .ProcessingOutputConfig
                          .Outputs['processed_data']
                          .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/processed-data-with-labels/',
        'execution_report': 's3://bucket/reports/execution_report.json'
    },
    dependencies=[ruleset_generator_step, tabular_preprocessing_step]
)

# Step 4: Training step (consumes processed_data with labels)
training_step = TrainingStepBuilder(
    config=training_config
).create_step(
    inputs={
        'training_data': ruleset_executor_step.properties
                         .ProcessingOutputConfig
                         .Outputs['processed_data']
                         .S3Output.S3Uri
    },
    dependencies=[ruleset_executor_step]
)
```

### Stacked Preprocessing Pipeline Example

```python
# Demonstrate how processed_data enables preprocessing composition

# Step 1: Base tabular preprocessing
base_preprocessing_step = TabularPreprocessingStepBuilder(
    config=base_config
).create_step(
    dependencies=[data_load_step]
)

# Step 2: Risk table mapping (adds risk columns)
risk_mapping_step = RiskTableMappingStepBuilder(
    config=risk_config
).create_step(
    inputs={
        'processed_data': base_preprocessing_step.properties
                          .ProcessingOutputConfig
                          .Outputs['processed_data']
                          .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/processed-data-with-risk/'
    },
    dependencies=[base_preprocessing_step]
)

# Step 3: Missing value imputation
imputation_step = MissingValueImputationStepBuilder(
    config=imputation_config
).create_step(
    inputs={
        'processed_data': risk_mapping_step.properties
                          .ProcessingOutputConfig
                          .Outputs['processed_data']
                          .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/processed-data-imputed/'
    },
    dependencies=[risk_mapping_step]
)

# Step 4: Ruleset execution (adds labels)
ruleset_executor_step = RulesetExecutorStepBuilder(
    config=executor_config
).create_step(
    inputs={
        'validated_ruleset': ruleset_generator_step.properties
                             .ProcessingOutputConfig
                             .Outputs['validated_ruleset']
                             .S3Output.S3Uri,
        'processed_data': imputation_step.properties
                          .ProcessingOutputConfig
                          .Outputs['processed_data']
                          .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/processed-data-labeled/'
    },
    dependencies=[ruleset_generator_step, imputation_step]
)

# Step 5: Stratified sampling (balances classes)
sampling_step = StratifiedSamplingStepBuilder(
    config=sampling_config
).create_step(
    inputs={
        'processed_data': ruleset_executor_step.properties
                          .ProcessingOutputConfig
                          .Outputs['processed_data']
                          .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/processed-data-final/'
    },
    dependencies=[ruleset_executor_step]
)

# Step 6: Training (consumes final processed_data)
training_step = TrainingStepBuilder(
    config=training_config
).create_step(
    inputs={
        'training_data': sampling_step.properties
                         .ProcessingOutputConfig
                         .Outputs['processed_data']
                         .S3Output.S3Uri
    },
    dependencies=[sampling_step]
)
```

**Benefits of Stacked Preprocessing:**
1. **Composability**: Each step reads and writes `processed_data`
2. **Flexibility**: Easy to add/remove/reorder preprocessing steps
3. **Consistency**: Same logical name throughout pipeline
4. **Clarity**: Data flow is clear and predictable

### Input Argument Mapping

The `RulesetExecutorStepBuilder` defines the following input arguments:

| Argument | Description | Required | Source Step | Output Name |
|----------|-------------|----------|-------------|-------------|
| validated_ruleset | Validated ruleset JSON | Yes | RulesetGeneratorStep | validated_ruleset |
| processed_data | Processed data splits | Yes | TabularPreprocessingStep | processed_data |

### Output Property Mapping

The `RulesetExecutorStepBuilder` provides the following output properties:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| labeled_data | Labeled data with train/val/test splits | `step.properties.ProcessingOutputConfig.Outputs["labeled_data"].S3Output.S3Uri` |
| execution_report | Execution statistics and metrics | `step.properties.ProcessingOutputConfig.Outputs["execution_report"].S3Output.S3Uri` |

### Pipeline DAG Structure

```
┌─────────────────┐
│  Data Load Step │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Tabular Preprocessing   │
│ Output: processed_data  │
└─────────┬───────────────┘
          │
          │
          │    ┌──────────────────────┐
          │    │ Ruleset Generator    │
          │    │ (Independent)        │
          │    │ Output: validated_   │
          │    │         ruleset      │
          │    └──────────┬───────────┘
          │               │
          │               │
          └───────┐   ┐───┘
                  │   │
                  ▼   ▼
        ┌────────────────────────────┐
        │ Ruleset Executor           │
        │ Inputs:                    │
        │   - validated_ruleset      │
        │   - processed_data         │
        │ Output: labeled_data       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────┐
        │  Training Step     │
        └────────────────────┘
```

**Key Points:**
1. **RulesetGenerator is INDEPENDENT** - validates rules without needing actual data
2. **TabularPreprocessing is INDEPENDENT** - processes data without needing rules
3. **RulesetExecutor depends on BOTH** - needs validated rules AND processed data
4. Generator and Preprocessing can run in parallel
5. Both must complete before RulesetExecutor can run

## Performance Optimizations

### 1. Batch Processing

For large datasets, process in batches to manage memory:

```python
def evaluate_in_batches(
    self, 
    df: pd.DataFrame, 
    batch_size: int = 10000
) -> pd.Series:
    """Process DataFrame in batches."""
    results = []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        batch_labels = self.evaluate_batch(batch)
        results.append(batch_labels)
    return pd.concat(results)
```

### 2. Early Termination

Stop at first matching rule (priority mode):
- Most common rules should have higher priority
- Reduces average conditions evaluated per row

### 3. Field Access Caching

Cache frequently accessed fields within row evaluation:
- Reduces Series indexing overhead
- Particularly beneficial for complex nested conditions

## Error Handling Strategies

### 1. Fail-Safe Evaluation

```python
try:
    if self._evaluate_conditions(rule['conditions'], row):
        return rule['output_label']
except Exception as e:
    logger.warning(f"Rule {rule['rule_id']} failed: {e}")
    continue  # Try next rule
```

### 2. Missing Field Handling

```python
if field not in row.index:
    if self.fail_on_missing_fields:
        raise KeyError(f"Required field '{field}' not found")
    else:
        return False  # Condition fails gracefully
```

### 3. Type Coercion Safety

```python
try:
    return float(actual) > float(expected)
except (ValueError, TypeError):
    logger.warning(f"Type coercion failed: {actual} vs {expected}")
    return False
```

## Key Design Principles

1. **Performance First**: Optimized for large-scale batch processing
2. **Fail-Safe**: Graceful error handling, continue processing
3. **Observable**: Detailed statistics and tracking
4. **Separation of Concerns**: Only execution, no validation
5. **Framework Compliance**: Follows ProcessingStepConfigBase patterns

## Expected Benefits

1. **Fast Execution**: Optimized for performance
2. **Scalability**: Handles large datasets efficiently
3. **Monitoring**: Detailed execution statistics
4. **Reliability**: Robust error handling
5. **Reusability**: Apply same ruleset to different data

## Summary

The RulesetExecutor step provides:

1. **Efficient execution** of pre-validated rulesets
2. **Batch processing** for large-scale data
3. **Comprehensive statistics** for monitoring
4. **Robust error handling** for production use
5. **Framework integration** following cursus patterns

This design ensures fast, reliable label generation at scale while maintaining full compatibility with the cursus framework patterns and SageMaker pipeline infrastructure.
