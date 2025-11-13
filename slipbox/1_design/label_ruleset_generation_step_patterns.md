---
tags:
  - design
  - step_builders
  - ruleset_generation
  - rule_validation
  - patterns
  - sagemaker
  - data_preprocessing
keywords:
  - ruleset generation
  - rule validation
  - JSON schema validation
  - declarative rules
  - rule structure
topics:
  - step builder patterns
  - ruleset generation implementation
  - rule validation design
  - SageMaker data processing
language: python
date of note: 2025-11-09
updated: 2025-11-09
---

# Label Ruleset Generation Step Builder Patterns

## Overview

This document defines the design patterns for the **RulesetGenerator** step builder implementation in the cursus framework. The RulesetGenerator step creates **ProcessingStep** instances that validate and transform user-defined rules into a structured, validated ruleset format that can be consumed by downstream execution steps. This step ensures rule quality, consistency, and correctness before execution.

## Purpose and Motivation

### Problem Statement

Users need to define classification rules in a human-readable format, but these rules must be validated and structured before execution to ensure:

1. **Field references exist** in the target data
2. **Label values are valid** and consistent
3. **Rule logic is sound** (no contradictions, unreachable rules)
4. **Format is correct** for efficient execution

### Separation of Concerns

By separating ruleset generation from execution:
- **Generation**: Validates, structures, and optimizes rules once
- **Execution**: Efficiently applies pre-validated rules to data
- **Benefits**: Faster execution, clearer error messages, better debugging

## Integration with Data Processing Ecosystem

The RulesetGenerator integrates as follows:

```
User-Defined Rules (JSON/YAML) → RulesetGenerator → Validated Ruleset
                                                           ↓
                                              Ruleset Execution Step
                                                           ↓
                                                    Labeled Data
```

**Full Pipeline Integration:**
```
Raw Data → Tabular Preprocessing → Processed Data
                                         ↓
Bedrock Processing → LLM Outputs
                         ↓
User Rules → RulesetGenerator → Validated Ruleset
                                       ↓
              Processed Data + LLM Outputs + Validated Ruleset
                                       ↓
                              Ruleset Execution Step
                                       ↓
                                  Labeled Data
```

## Configuration via Sub-Configs (Pydantic Models)

Following the cursus framework pattern (see `BedrockPromptTemplateGenerationConfig`), users define rules through typed Pydantic models rather than raw JSON files.

### Sub-Config Classes

#### 1. LabelConfig (Label Configuration)

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Literal

class LabelConfig(BaseModel):
    """
    Configuration for label generation.
    
    Defines output label structure, values, and evaluation mode.
    """
    output_label_name: str = Field(
        ...,
        description="Name of the output label column"
    )
    
    output_label_type: Literal["binary", "multiclass"] = Field(
        ...,
        description="Type of classification (binary or multiclass)"
    )
    
    label_values: List[int] = Field(
        ...,
        min_length=2,
        description="Valid label values (e.g., [0, 1] for binary)"
    )
    
    label_mapping: Dict[str, str] = Field(
        ...,
        description="Mapping of label values to human-readable names"
    )
    
    default_label: int = Field(
        ...,
        description="Default label when no rules match"
    )
    
    evaluation_mode: str = Field(
        default="priority",
        description="Rule evaluation mode (priority, best_match, confidence_weighted)"
    )
    
    model_config = {"extra": "forbid", "validate_assignment": True}
```

#### 2. RuleCondition (Condition Expression)

**Note**: FieldConfig is automatically inferred from rule definitions and is not exposed as a user-facing configuration.

```python
from enum import Enum

class ComparisonOperator(str, Enum):
    """
    Supported comparison operators for rule conditions.
    
    Categories:
    - Comparison: equals, not_equals, gt, gte, lt, lte
    - Collection: in_collection, not_in_collection
    - String: contains, not_contains, starts_with, ends_with, regex_match
    - Null: is_null, is_not_null
    """
    # Comparison operators
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    
    # Collection operators
    IN = "in"
    NOT_IN = "not_in"
    
    # String operators
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    
    # Null operators
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class RuleCondition(BaseModel):
    """
    Single condition in a rule.
    
    Supports nested logical operators (all_of, any_of, none_of) and
    leaf conditions with field comparisons using validated operators.
    """
    # Leaf condition fields
    field: Optional[str] = Field(
        default=None,
        description="Field name for leaf condition"
    )
    
    operator: Optional[ComparisonOperator] = Field(
        default=None,
        description="Comparison operator from ComparisonOperator enum"
    )
    
    value: Optional[Any] = Field(
        default=None,
        description="Expected value for comparison"
    )
    
    # Logical operators (for nested conditions)
    all_of: Optional[List['RuleCondition']] = Field(
        default=None,
        description="All conditions must be true (AND logic)"
    )
    
    any_of: Optional[List['RuleCondition']] = Field(
        default=None,
        description="At least one condition must be true (OR logic)"
    )
    
    none_of: Optional[List['RuleCondition']] = Field(
        default=None,
        description="All conditions must be false (NOT logic)"
    )
    
    model_config = {"extra": "forbid", "validate_assignment": True}
```

### RuleCondition Examples

Below are comprehensive examples showing how to use each operator category:

#### Comparison Operators

```python
# Example 1: Simple equality check
condition_equals = RuleCondition(
    field="category",
    operator=ComparisonOperator.EQUALS,
    value="TrueDNR"
)

# Example 2: Not equals
condition_not_equals = RuleCondition(
    field="status",
    operator=ComparisonOperator.NOT_EQUALS,
    value="cancelled"
)

# Example 3: Greater than
condition_gt = RuleCondition(
    field="confidence_score",
    operator=ComparisonOperator.GT,
    value=0.8
)

# Example 4: Greater than or equal
condition_gte = RuleCondition(
    field="amount",
    operator=ComparisonOperator.GTE,
    value=100.0
)

# Example 5: Less than
condition_lt = RuleCondition(
    field="risk_score",
    operator=ComparisonOperator.LT,
    value=0.3
)

# Example 6: Less than or equal
condition_lte = RuleCondition(
    field="processing_time",
    operator=ComparisonOperator.LTE,
    value=5000
)
```

#### Collection Operators

```python
# Example 7: Value in collection
condition_in = RuleCondition(
    field="country_code",
    operator=ComparisonOperator.IN,
    value=["US", "CA", "GB", "AU"]
)

# Example 8: Value not in collection
condition_not_in = RuleCondition(
    field="payment_method",
    operator=ComparisonOperator.NOT_IN,
    value=["cash", "check"]
)
```

#### String Operators

```python
# Example 9: Contains substring
condition_contains = RuleCondition(
    field="description",
    operator=ComparisonOperator.CONTAINS,
    value="urgent"
)

# Example 10: Does not contain substring
condition_not_contains = RuleCondition(
    field="notes",
    operator=ComparisonOperator.NOT_CONTAINS,
    value="spam"
)

# Example 11: Starts with prefix
condition_starts_with = RuleCondition(
    field="transaction_id",
    operator=ComparisonOperator.STARTS_WITH,
    value="TXN-"
)

# Example 12: Ends with suffix
condition_ends_with = RuleCondition(
    field="email",
    operator=ComparisonOperator.ENDS_WITH,
    value="@amazon.com"
)

# Example 13: Regex match
condition_regex = RuleCondition(
    field="order_id",
    operator=ComparisonOperator.REGEX_MATCH,
    value=r"^ORD-\d{8}$"  # Matches ORD-12345678
)
```

#### Null Operators

```python
# Example 14: Is null (missing or NaN)
condition_is_null = RuleCondition(
    field="optional_comment",
    operator=ComparisonOperator.IS_NULL,
    value=None  # value parameter ignored for null checks
)

# Example 15: Is not null (has value)
condition_is_not_null = RuleCondition(
    field="customer_id",
    operator=ComparisonOperator.IS_NOT_NULL,
    value=None  # value parameter ignored for null checks
)
```

#### Nested Logical Operators

```python
# Example 16: AND logic (all_of) - All conditions must be true
condition_and = RuleCondition(
    all_of=[
        RuleCondition(field="category", operator=ComparisonOperator.EQUALS, value="Premium"),
        RuleCondition(field="confidence_score", operator=ComparisonOperator.GTE, value=0.9),
        RuleCondition(field="amount", operator=ComparisonOperator.GT, value=1000)
    ]
)

# Example 17: OR logic (any_of) - At least one condition must be true
condition_or = RuleCondition(
    any_of=[
        RuleCondition(field="priority", operator=ComparisonOperator.EQUALS, value="high"),
        RuleCondition(field="urgent_flag", operator=ComparisonOperator.EQUALS, value=1),
        RuleCondition(field="sla_hours", operator=ComparisonOperator.LT, value=4)
    ]
)

# Example 18: NOT logic (none_of) - All conditions must be false
condition_not = RuleCondition(
    none_of=[
        RuleCondition(field="status", operator=ComparisonOperator.EQUALS, value="cancelled"),
        RuleCondition(field="status", operator=ComparisonOperator.EQUALS, value="failed"),
        RuleCondition(field="error_code", operator=ComparisonOperator.IS_NOT_NULL, value=None)
    ]
)

# Example 19: Complex nested logic - (A AND B) OR C
condition_complex = RuleCondition(
    any_of=[
        RuleCondition(
            all_of=[
                RuleCondition(field="category", operator=ComparisonOperator.EQUALS, value="TrueDNR"),
                RuleCondition(field="confidence_score", operator=ComparisonOperator.GTE, value=0.8)
            ]
        ),
        RuleCondition(field="manual_review_flag", operator=ComparisonOperator.EQUALS, value=1)
    ]
)

# Example 20: Very complex nested logic - (A AND B) OR (C AND NOT D)
condition_very_complex = RuleCondition(
    any_of=[
        # First branch: High confidence TrueDNR
        RuleCondition(
            all_of=[
                RuleCondition(field="category", operator=ComparisonOperator.EQUALS, value="TrueDNR"),
                RuleCondition(field="confidence_score", operator=ComparisonOperator.GTE, value=0.9)
            ]
        ),
        # Second branch: Low risk and not flagged
        RuleCondition(
            all_of=[
                RuleCondition(field="risk_score", operator=ComparisonOperator.LT, value=0.3),
                RuleCondition(
                    none_of=[
                        RuleCondition(field="fraud_flag", operator=ComparisonOperator.EQUALS, value=1),
                        RuleCondition(field="dispute_flag", operator=ComparisonOperator.EQUALS, value=1)
                    ]
                )
            ]
        )
    ]
)
```

#### Practical Full Rule Examples

```python
# Example 21: Complete rule for high-confidence reversals
rule_high_confidence_reversal = RuleDefinition(
    name="High Confidence Reversal",
    priority=1,
    conditions=RuleCondition(
        all_of=[
            RuleCondition(field="category", operator=ComparisonOperator.EQUALS, value="Reversal"),
            RuleCondition(field="confidence_score", operator=ComparisonOperator.GTE, value=0.85),
            RuleCondition(field="amount", operator=ComparisonOperator.GT, value=0)
        ]
    ),
    output_label=1,
    description="High confidence reversal with positive amount"
)

# Example 22: Complete rule for suspected fraud
rule_fraud_detection = RuleDefinition(
    name="Suspected Fraud",
    priority=2,
    conditions=RuleCondition(
        all_of=[
            # Multiple transactions in short time
            RuleCondition(field="transaction_count_1h", operator=ComparisonOperator.GT, value=10),
            # From risky countries
            RuleCondition(
                field="country_code",
                operator=ComparisonOperator.IN,
                value=["XX", "YY", "ZZ"]
            ),
            # High amount
            RuleCondition(field="total_amount", operator=ComparisonOperator.GT, value=5000),
            # Not verified customer
            RuleCondition(
                none_of=[
                    RuleCondition(field="verification_status", operator=ComparisonOperator.EQUALS, value="verified"),
                    RuleCondition(field="trusted_customer", operator=ComparisonOperator.EQUALS, value=True)
                ]
            )
        ]
    ),
    output_label=1,
    description="Multiple fraud indicators detected"
)

# Example 23: Complete rule with string pattern matching
rule_email_pattern = RuleDefinition(
    name="Corporate Email Pattern",
    priority=3,
    conditions=RuleCondition(
        any_of=[
            RuleCondition(field="email", operator=ComparisonOperator.ENDS_WITH, value="@amazon.com"),
            RuleCondition(field="email", operator=ComparisonOperator.ENDS_WITH, value="@aws.amazon.com"),
            RuleCondition(field="email", operator=ComparisonOperator.REGEX_MATCH, value=r"^[a-z]+@(corp|internal)\.company\.com$")
        ]
    ),
    output_label=0,
    description="Trusted corporate email domains"
)

# Example 24: Complete rule with null handling
rule_incomplete_data = RuleDefinition(
    name="Incomplete Data",
    priority=10,
    conditions=RuleCondition(
        any_of=[
            RuleCondition(field="customer_id", operator=ComparisonOperator.IS_NULL, value=None),
            RuleCondition(field="transaction_date", operator=ComparisonOperator.IS_NULL, value=None),
            RuleCondition(field="amount", operator=ComparisonOperator.IS_NULL, value=None)
        ]
    ),
    output_label=1,
    description="Missing required fields - needs manual review"
)
```

#### Best Practices for Rule Conditions

1. **Prefer Simple Conditions**: Start with simple leaf conditions before adding complexity
2. **Use Appropriate Operators**: Choose operators that match your data types (string operators for strings, comparison for numbers)
3. **Avoid Deep Nesting**: Keep nesting to 2-3 levels maximum for maintainability
4. **Test Boundary Cases**: Ensure operators work correctly at edge values (0, empty strings, etc.)
5. **Handle Nulls Explicitly**: Use `is_null`/`is_not_null` operators to handle missing data
6. **Order Matters in `all_of`**: Place most selective conditions first for efficiency
7. **Use `any_of` Sparingly**: Too many OR conditions can be hard to debug
8. **Document Complex Logic**: Add clear descriptions for rules with nested conditions

#### 4. RuleDefinition (Individual Rule)

```python
import uuid
from pydantic import PrivateAttr

class RuleDefinition(BaseModel):
    """
    Definition of a single classification rule.
    
    Maps conditions to output labels with priority ordering.
    
    Tier 1: User-facing fields (name, priority, conditions, output_label)
    Tier 3: Auto-generated fields (rule_id - private, derived)
    """
    
    # ===== Tier 1: User Inputs (Required/Optional) =====
    
    name: str = Field(
        ...,
        description="Human-readable rule name"
    )
    
    priority: int = Field(
        ...,
        ge=1,
        description="Priority for evaluation (lower = higher priority)"
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether rule is active"
    )
    
    conditions: RuleCondition = Field(
        ...,
        description="Condition expression that must be satisfied"
    )
    
    output_label: int = Field(
        ...,
        description="Label value to output when rule matches"
    )
    
    description: str = Field(
        default="",
        description="Description of what this rule identifies"
    )
    
    # ===== Tier 3: Derived Fields (Private, Auto-Generated) =====
    
    _rule_id: str = PrivateAttr(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    
    @property
    def rule_id(self) -> str:
        """Get auto-generated unique rule identifier."""
        return self._rule_id
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include rule_id in output."""
        data = super().model_dump(**kwargs)
        data["rule_id"] = self.rule_id
        return data
    
    model_config = {"extra": "forbid", "validate_assignment": True}
```

#### 5. RuleDefinitionList (Rule Collection)

```python
class RuleDefinitionList(BaseModel):
    """
    Collection of rule definitions with validation.
    
    Ensures rule IDs are unique and provides utility methods.
    """
    ruleset: List[RuleDefinition] = Field(
        ...,
        min_length=1,
        description="List of rule definitions (at least one required)"
    )
    
    @field_validator("ruleset")
    @classmethod
    def ruleset_must_have_unique_ids(cls, v: List[RuleDefinition]) -> List[RuleDefinition]:
        """Validate all rule IDs are unique."""
        if not v:
            raise ValueError("At least one rule is required")
        
        rule_ids = set()
        for i, rule in enumerate(v):
            if rule.rule_id in rule_ids:
                raise ValueError(f"Duplicate rule_id: '{rule.rule_id}' at index {i}")
            rule_ids.add(rule.rule_id)
        
        return v
    
    def sort_by_priority(self) -> "RuleDefinitionList":
        """Return new list sorted by priority."""
        sorted_rules = sorted(self.ruleset, key=lambda x: x.priority)
        return RuleDefinitionList(ruleset=sorted_rules)
    
    def to_script_format(self) -> List[Dict[str, Any]]:
        """Convert to format expected by script."""
        return [rule.model_dump() for rule in self.ruleset]
    
    model_config = {"extra": "forbid", "validate_assignment": True}
```

### Example User Configuration

```python
# Define label configuration
label_config = LabelConfig(
    output_label_name="final_reversal_flag",
    output_label_type="binary",
    label_values=[0, 1],
    label_mapping={
        "0": "No_Reversal",
        "1": "Reversal"
    },
    default_label=1,
    evaluation_mode="priority"
)

# Define field configuration
field_config = FieldConfig(
    required_fields=["category", "confidence_score"],
    optional_fields=["reversal_flag", "conc_si", "ttm_conc_amt"],
    field_types={
        "category": "string",
        "confidence_score": "float",
        "reversal_flag": "int",
        "conc_si": "float",
        "ttm_conc_amt": "float"
    }
)

# Define rules
rule_001 = RuleDefinition(
    rule_id="rule_001",
    name="High confidence TrueDNR",
    priority=1,
    enabled=True,
    conditions=RuleCondition(
        all_of=[
            RuleCondition(field="category", operator="equals", value="TrueDNR"),
            RuleCondition(field="confidence_score", operator=">=", value=0.8)
        ]
    ),
    output_label=0,
    description="High confidence TrueDNR cases indicate no reversal"
)

# Create configuration
config = RulesetGeneratorConfig(
    # Base pipeline config fields
    author="data-scientist",
    bucket="my-bucket",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    region="NA",
    service_name="classification",
    pipeline_version="1.0",
    project_root_folder="my_project",
    
    # Ruleset-specific config
    label_settings=label_config,
    field_settings=field_config,
    rule_definitions=[rule_001, rule_002, rule_003]
)
```

## Output Format: Validated Ruleset

The validated ruleset includes:
- Original rules with validation status
- Optimization metadata
- Field usage statistics
- Rule priority ordering
- Validation report

```json
{
  "version": "1.0",
  "generated_timestamp": "2025-11-09T11:30:00Z",
  "label_config": { /* same as input */ },
  "field_config": { /* enriched with validation */ },
  "rules": [ /* sorted by priority, validated */ ],
  "metadata": {
    "total_rules": 10,
    "enabled_rules": 9,
    "disabled_rules": 1,
    "field_usage": {
      "category": 8,
      "confidence_score": 7,
      "reversal_flag": 3
    },
    "validation_summary": {
      "field_validation": "passed",
      "label_validation": "passed",
      "logic_validation": "passed_with_warnings",
      "warnings": ["rule_007 unreachable"]
    }
  }
}
```

## Two-Tier Validation Architecture

**Note**: Field schema validation has been moved to the configuration layer. The config automatically infers `field_config` from rule definitions and validates consistency at initialization time, ensuring field references are correct before the script runs.

### 1. Label Value Validation

```python
class RulesetLabelValidator:
    """Validates output labels match configuration."""
    
    def validate_labels(self, ruleset: dict) -> ValidationResult:
        """
        Validates all output_label values in rules.
        
        Checks:
        - All output_label values are in label_values
        - For binary: only 0 and 1 are used
        - For multiclass: check class balance potential
        - Default label is valid
        - No conflicting rules (same priority, different outputs)
        
        Returns:
        - valid: bool
        - invalid_labels: List[Tuple[rule_id, invalid_label]]
        - uncovered_classes: List[int]
        - conflicting_rules: List[Tuple[rule_id1, rule_id2]]
        - warnings: List[str]
        """
```

**Validation Logic:**
1. Extract all `output_label` values from rules
2. Verify each is in `label_values`
3. Check binary constraints (only 0, 1)
4. Identify unreachable label values
5. Detect potential rule conflicts

### 2. Rule Logic Validation

```python
class RulesetLogicValidator:
    """Validates rule logic for errors."""
    
    def validate_logic(self, ruleset: dict) -> ValidationResult:
        """
        Validates rule logic for common errors.
        
        Checks:
        - No tautologies (always true)
        - No contradictions (always false)
        - No unreachable rules (shadowed by higher priority)
        - Numeric ranges are valid (min <= max)
        - Operator compatibility with field types
        
        Returns:
        - valid: bool
        - tautologies: List[str]
        - contradictions: List[str]
        - unreachable_rules: List[str]
        - type_mismatches: List[str]
        - warnings: List[str]
        """
```

**Validation Logic:**
1. Check condition logic for each rule
2. Detect always-true conditions (tautologies)
3. Detect always-false conditions (contradictions)
4. Analyze rule priority for shadowing
5. Verify operator-type compatibility

## Rule Optimization

### Priority Ordering

```python
def optimize_rule_priority(ruleset: dict) -> dict:
    """
    Optimize rule priority ordering for execution efficiency.
    
    Strategies:
    1. Place high-frequency match rules first
    2. Group rules by field usage for cache efficiency
    3. Order by condition complexity (simple first)
    
    Returns:
        Optimized ruleset with adjusted priorities
    """
```

### Field Usage Analysis

```python
def analyze_field_usage(ruleset: dict) -> Dict[str, int]:
    """
    Analyze which fields are used most frequently.
    
    Returns:
        Dictionary mapping field names to usage count
        
    Use cases:
    - Identify critical fields for validation
    - Optimize data loading (only load needed fields)
    - Guide rule optimization
    """
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
) -> Dict[str, Any]:
    """
    Main logic for ruleset generation and validation.
    
    Reads JSON files auto-generated from Pydantic sub-configs by RulesetGeneratorConfig.
    
    Args:
        input_paths: Dictionary with keys:
            - "ruleset_configs": Directory with auto-generated JSON files:
                * label_config.json (required)
                * field_config.json (optional)
                * rules.json (required)
            - "sample_data": Optional sample data for field validation
        output_paths: Dictionary with keys:
            - "validated_ruleset": Path for validated ruleset output
            - "validation_report": Path for detailed validation report
        environ_vars: Environment variables
        job_args: Command line arguments
        logger: Optional logger function
        
    Returns:
        Dictionary with processing results
    """
    log = logger or print
    
    # 1. Load auto-generated JSON configuration files
    configs_dir = Path(input_paths["ruleset_configs"])
    
    # Load label_config.json (required)
    label_config_file = configs_dir / "label_config.json"
    with open(label_config_file, 'r') as f:
        label_config = json.load(f)
    log(f"[INFO] Loaded label config from {label_config_file}")
    
    # Load field_config.json (optional - may be inferred from sample data)
    field_config_file = configs_dir / "field_config.json"
    if field_config_file.exists():
        with open(field_config_file, 'r') as f:
            field_config = json.load(f)
        log(f"[INFO] Loaded field config from {field_config_file}")
    else:
        log("[INFO] No field_config.json found - will infer from sample data")
        field_config = {
            "required_fields": [],
            "optional_fields": [],
            "field_types": {}
        }
    
    # Load rules.json (required)
    rules_file = configs_dir / "rules.json"
    with open(rules_file, 'r') as f:
        rules = json.load(f)
    log(f"[INFO] Loaded {len(rules)} rules from {rules_file}")
    
    # Assemble user ruleset
    user_ruleset = {
        "label_config": label_config,
        "field_config": field_config,
        "rules": rules
    }
    
    # 2. Load sample data if provided (for field validation)
    sample_df = None
    if "sample_data" in input_paths:
        sample_data_path = input_paths["sample_data"]
        if os.path.exists(sample_data_path):
            # Load sample (number of rows from environment variable)
            sample_size = int(environ_vars.get("SAMPLE_SIZE", "100"))
            sample_df = pd.read_csv(sample_data_path, nrows=sample_size)
            log(f"[INFO] Loaded sample data: {sample_df.shape}")
            
            # If field_config was not provided, infer from sample data
            if not field_config_file.exists():
                log("[INFO] Inferring field config from sample data...")
                inferred_fields = list(sample_df.columns)
                inferred_types = {
                    col: str(sample_df[col].dtype) 
                    for col in sample_df.columns
                }
                field_config["optional_fields"] = inferred_fields
                field_config["field_types"] = inferred_types
                user_ruleset["field_config"] = field_config
                log(f"[INFO] Inferred {len(inferred_fields)} fields from sample data")
    
    # 3. Initialize validators
    field_validator = RulesetFieldValidator()
    label_validator = RulesetLabelValidator()
    logic_validator = RulesetLogicValidator()
    
    # 4. Run validation (based on environment variables)
    log("[INFO] Running validation...")
    
    enable_field = environ_vars.get("ENABLE_FIELD_VALIDATION", "true").lower() == "true"
    enable_label = environ_vars.get("ENABLE_LABEL_VALIDATION", "true").lower() == "true"
    enable_logic = environ_vars.get("ENABLE_LOGIC_VALIDATION", "true").lower() == "true"
    
    field_validation = field_validator.validate_fields(user_ruleset, sample_df) if enable_field else None
    label_validation = label_validator.validate_labels(user_ruleset) if enable_label else None
    logic_validation = logic_validator.validate_logic(user_ruleset) if enable_logic else None
    
    # 5. Check validation results
    all_valid = (
        (field_validation.valid if field_validation else True) and 
        (label_validation.valid if label_validation else True) and 
        (logic_validation.valid if logic_validation else True)
    )
    
    if not all_valid:
        log("[ERROR] Validation failed!")
        
        # Save detailed validation report
        validation_report = {
            "validation_status": "failed",
            "field_validation": field_validation.__dict__ if field_validation else None,
            "label_validation": label_validation.__dict__ if label_validation else None,
            "logic_validation": logic_validation.__dict__ if logic_validation else None
        }
        
        report_path = output_paths.get("validation_report")
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
        
        raise RuntimeError("Ruleset validation failed. See validation report for details.")
    
    log("[INFO] Validation passed")
    
    # 6. Optimize ruleset (if enabled)
    enable_optimization = environ_vars.get("ENABLE_RULE_OPTIMIZATION", "true").lower() == "true"
    if enable_optimization:
        log("[INFO] Optimizing ruleset...")
        optimized_ruleset = optimize_ruleset(user_ruleset)
    else:
        log("[INFO] Skipping optimization")
        optimized_ruleset = user_ruleset
    
    # 7. Generate validated ruleset with metadata
    validated_ruleset = {
        "version": "1.0",
        "generated_timestamp": datetime.now().isoformat(),
        "label_config": optimized_ruleset["label_config"],
        "field_config": optimized_ruleset["field_config"],
        "rules": optimized_ruleset["rules"],
        "metadata": {
            "total_rules": len(optimized_ruleset["rules"]),
            "enabled_rules": sum(1 for r in optimized_ruleset["rules"] if r.get("enabled", True)),
            "disabled_rules": sum(1 for r in optimized_ruleset["rules"] if not r.get("enabled", True)),
            "field_usage": analyze_field_usage(optimized_ruleset),
            "validation_summary": {
                "field_validation": "passed" if not field_validation or field_validation.valid else "failed",
                "label_validation": "passed" if not label_validation or label_validation.valid else "failed",
                "logic_validation": (
                    "passed" if not logic_validation or 
                    (logic_validation.valid and not logic_validation.warnings) else 
                    "passed_with_warnings" if logic_validation.valid else "failed"
                ),
                "warnings": logic_validation.warnings if logic_validation else []
            }
        }
    }
    
    # 8. Save validated ruleset
    validated_ruleset_path = output_paths["validated_ruleset"]
    os.makedirs(os.path.dirname(validated_ruleset_path), exist_ok=True)
    with open(validated_ruleset_path, 'w') as f:
        json.dump(validated_ruleset, f, indent=2)
    
    log(f"[INFO] Saved validated ruleset to {validated_ruleset_path}")
    
    # 9. Save validation report
    validation_report = {
        "validation_status": "passed",
        "field_validation": field_validation.__dict__ if field_validation else {"skipped": True},
        "label_validation": label_validation.__dict__ if label_validation else {"skipped": True},
        "logic_validation": logic_validation.__dict__ if logic_validation else {"skipped": True},
        "optimization_applied": enable_optimization,
        "metadata": validated_ruleset["metadata"]
    }
    
    report_path = output_paths.get("validation_report")
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        log(f"[INFO] Saved validation report to {report_path}")
    
    log("[INFO] Ruleset generation complete")
    
    return {
        "validated_ruleset": validated_ruleset,
        "validation_report": validation_report
    }
```

## Configuration Class

```python
from pydantic import Field, PrivateAttr, model_validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase


class RulesetGeneratorConfig(ProcessingStepConfigBase):
    """
    Configuration for RulesetGenerator step using sub-config pattern.
    
    Inherits from ProcessingStepConfigBase which provides:
    - Base pipeline configuration
    - Processing instance settings
    - Script configuration
    
    Adds ruleset generation-specific configuration with typed sub-configs.
    """
    
    # ===== Tier 1: Essential User Inputs (Required) =====
    
    # Sub-configuration objects (typed Pydantic models)
    label_settings: LabelConfig = Field(
        ...,
        description="Label configuration with validation"
    )
    
    field_settings: Optional[FieldConfig] = Field(
        default=None,
        description="Field configuration (optional, can be inferred from sample data)"
    )
    
    rule_definitions: List[RuleDefinition] = Field(
        ...,
        min_length=1,
        description="List of rule definitions (at least one required)"
    )
    
    # ===== Tier 2: System Inputs with Defaults (Optional) =====
    
    # Configuration path - where JSON files will be generated
    ruleset_configs_path: str = Field(
        default="ruleset_configs",
        description="Subdirectory for generated ruleset configuration files (label_config.json, field_config.json, rules.json)"
    )
    
    # Validation Configuration
    enable_field_validation: bool = Field(
        default=True,
        description="Enable field existence validation"
    )
    
    enable_label_validation: bool = Field(
        default=True,
        description="Enable label value validation"
    )
    
    enable_logic_validation: bool = Field(
        default=True,
        description="Enable rule logic validation"
    )
    
    fail_on_validation_warning: bool = Field(
        default=False,
        description="Fail if validation produces warnings (strict mode)"
    )
    
    # Optimization Configuration
    enable_rule_optimization: bool = Field(
        default=True,
        description="Enable rule priority optimization"
    )
    
    optimize_by_frequency: bool = Field(
        default=False,
        description="Optimize rules based on expected match frequency (requires sample data)"
    )
    
    # Sample Data Configuration
    use_sample_data_validation: bool = Field(
        default=True,
        description="Use sample data for field validation if available"
    )
    
    sample_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of rows to use for sample data validation"
    )
    
    # Output Configuration
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in validated ruleset"
    )
    
    include_optimization_report: bool = Field(
        default=True,
        description="Include optimization details in validation report"
    )
    
    # Override default processing entry point
    processing_entry_point: Optional[str] = Field(
        default="ruleset_generation.py",
        description="Entry point script for ruleset generation"
    )
    
    # ===== Tier 3: Derived Fields (Private with Property Access) =====
    
    _effective_label_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _effective_field_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _effective_rules: Optional[RuleDefinitionList] = PrivateAttr(default=None)
    _resolved_ruleset_configs_path: Optional[str] = PrivateAttr(default=None)
    
    # Public properties for derived fields
    
    @property
    def effective_label_config(self) -> Dict[str, Any]:
        """Get label configuration from typed settings."""
        if self._effective_label_config is None:
            self._effective_label_config = self.label_settings.model_dump()
        return self._effective_label_config
    
    @property
    def effective_field_config(self) -> Dict[str, Any]:
        """Get field configuration from typed settings or empty dict."""
        if self._effective_field_config is None:
            if self.field_settings is not None:
                self._effective_field_config = self.field_settings.model_dump()
            else:
                # Empty field config - will be inferred from sample data
                self._effective_field_config = {
                    "required_fields": [],
                    "optional_fields": [],
                    "field_types": {}
                }
        return self._effective_field_config
    
    @property
    def effective_rules(self) -> RuleDefinitionList:
        """Get effective rule definitions with validation."""
        if self._effective_rules is None:
            self._effective_rules = RuleDefinitionList(ruleset=self.rule_definitions)
        return self._effective_rules
    
    @property
    def resolved_ruleset_configs_path(self) -> Optional[str]:
        """
        Get resolved absolute path for ruleset configurations.
        
        Uses effective_source_dir from base class for consistency.
        
        Returns:
            Absolute path to ruleset configs directory
        """
        if self._resolved_ruleset_configs_path is None:
            resolved_source_dir = self.effective_source_dir
            if resolved_source_dir is None:
                raise ValueError(
                    "Cannot resolve ruleset_configs_path: no processing source directory configured"
                )
            
            self._resolved_ruleset_configs_path = str(
                Path(resolved_source_dir) / self.ruleset_configs_path
            )
        
        return self._resolved_ruleset_configs_path
    
    def generate_ruleset_config_bundle(self) -> None:
        """
        Generate complete ruleset configuration bundle as JSON files.
        
        Creates JSON files in the configured ruleset_configs_path:
        - label_config.json (always generated)
        - field_config.json (only if field_settings provided)
        - rules.json (always generated)
        
        The script will read these JSON files for validation and optimization.
        """
        output_dir = Path(self.resolved_ruleset_configs_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate label_config.json (always required)
        label_config_file = output_dir / "label_config.json"
        with open(label_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.effective_label_config, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated label config: {label_config_file}")
        generated_files.append("label_config.json")
        
        # Generate field_config.json (only if field_settings provided)
        if self.field_settings is not None:
            field_config_file = output_dir / "field_config.json"
            with open(field_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.effective_field_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Generated field config: {field_config_file}")
            generated_files.append("field_config.json")
        else:
            logger.info(
                "Skipping field_config.json generation (field_settings is None - will infer from sample data)"
            )
        
        # Generate rules.json (always required)
        rules_file = output_dir / "rules.json"
        # Sort by priority before saving
        sorted_rules = self.effective_rules.sort_by_priority()
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_rules.to_script_format(), f, indent=2, ensure_ascii=False)
        logger.info(f"Generated rules config: {rules_file}")
        generated_files.append("rules.json")
        
        logger.info(f"Generated ruleset configuration bundle in: {output_dir}")
        logger.info(
            f"Bundle contains {len(generated_files)} JSON files: {', '.join(generated_files)}"
        )
    
    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "RulesetGeneratorConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        
        # Initialize ruleset-specific derived fields
        _ = self.effective_label_config
        _ = self.effective_field_config
        _ = self.effective_rules
        
        # Auto-generate ruleset config bundle
        try:
            self.generate_ruleset_config_bundle()
            logger.info(
                f"Auto-generated ruleset configuration bundle at: {self.resolved_ruleset_configs_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to auto-generate ruleset config bundle: {e}")
            logger.info(
                "You can manually call generate_ruleset_config_bundle() after providing missing settings"
            )
        
        return self
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        
        # Add derived properties
        data["effective_label_config"] = self.effective_label_config
        data["effective_field_config"] = self.effective_field_config
        data["effective_rules"] = self.effective_rules.model_dump()
        
        if self.ruleset_configs_path is not None:
            data["resolved_ruleset_configs_path"] = self.resolved_ruleset_configs_path
        
        return data
```

## Input/Output Structure

### Input Structure (Auto-Generated from Config)

The RulesetGeneratorConfig automatically generates JSON files from Pydantic sub-configs:

```
INPUT_RULESET_CONFIGS/ (auto-generated by config)
├── label_config.json       (from label_settings)
├── field_config.json       (from field_settings, optional)
└── rules.json              (from rule_definitions)

INPUT_SAMPLE_DATA/ (optional)
└── sample_data.csv  (sample for field validation)
```

**Key Pattern**: Users never write JSON files directly. Instead:
1. Users define typed Pydantic models (`LabelConfig`, `FieldConfig`, `RuleDefinition`)
2. RulesetGeneratorConfig validates and auto-generates JSON files
3. Script reads validated JSON files for processing

### Output Structure

```
OUTPUT_VALIDATED_RULESET/
├── validated_ruleset.json  (validated, optimized ruleset)
└── validation_report.json  (detailed validation report)
```

### Auto-Generation Flow

```python
# 1. User defines configuration with sub-configs
config = RulesetGeneratorConfig(
    label_settings=label_config,
    field_settings=field_config,
    rule_definitions=[rule_001, rule_002],
    # ... other config fields
)

# 2. Config initialization auto-generates JSON files
# Happens in initialize_derived_fields() via generate_ruleset_config_bundle()
# Creates: label_config.json, field_config.json (if provided), rules.json

# 3. Step builder uses config to create ProcessingStep
# Script reads auto-generated JSON files as inputs
```

## Environment Variables

```bash
# Validation Configuration
ENABLE_FIELD_VALIDATION="true"
ENABLE_LABEL_VALIDATION="true"
ENABLE_LOGIC_VALIDATION="true"
FAIL_ON_VALIDATION_WARNING="false"

# Optimization Configuration
ENABLE_RULE_OPTIMIZATION="true"
OPTIMIZE_BY_FREQUENCY="false"

# Sample Data Configuration
USE_SAMPLE_DATA_VALIDATION="true"
SAMPLE_SIZE="100"

# Output Configuration
INCLUDE_METADATA="true"
INCLUDE_OPTIMIZATION_REPORT="true"
```

## Key Design Principles

1. **Validation First**: Catch all errors before execution
2. **Optimization**: Structure rules for efficient execution
3. **Clear Reporting**: Detailed validation feedback
4. **Separation of Concerns**: Generate once, execute many times
5. **Framework Compliance**: Follows ProcessingStepConfigBase patterns

## Expected Benefits

1. **Early Error Detection**: Catch rule errors before execution
2. **Better Performance**: Optimized rulesets execute faster
3. **Clear Debugging**: Validation reports pinpoint issues
4. **Reusability**: Validated rulesets can be reused across executions
5. **Maintainability**: Easier to understand and modify rules

## Summary

The RulesetGenerator step provides:

1. **Comprehensive validation** of user-defined rules
2. **Optimization** for efficient execution
3. **Clear error reporting** for debugging
4. **Structured output** for downstream execution
5. **Framework integration** following cursus patterns

This design ensures rules are validated once and executed many times efficiently.
