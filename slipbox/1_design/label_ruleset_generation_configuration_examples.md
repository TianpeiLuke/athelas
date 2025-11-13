---
tags:
  - design
  - implementation
  - label_ruleset
  - configuration_examples
  - documentation
  - user_guide
keywords:
  - label ruleset configuration
  - binary classification
  - multiclass classification
  - multilabel classification
  - rule-based classification
  - fraud detection
topics:
  - configuration examples
  - real-world use cases
  - classification scenarios
language: json, python
date of note: 2025-11-11
---

# Label Ruleset Generation Configuration Examples

This document provides complete, working examples showing how to configure the **Label Ruleset Generation** system for different classification scenarios: binary, multiclass, and multilabel.

## Overview

This example demonstrates three classification patterns:
- **Binary Classification**: Single-label with two values (0/1, Yes/No)
- **Multiclass Classification**: Single-label with multiple values (3+ categories)
- **Multilabel Classification**: Multiple label columns for multi-task learning

## Example Use Case: E-Commerce Fraud Detection

We'll use fraud detection as our running example, showing how to configure rules for:
- **Binary**: Is this transaction fraudulent? (Yes/No)
- **Multiclass**: What type of fraud is this? (No Fraud, Account Takeover, Card Fraud, Refund Abuse, None)
- **Multilabel**: Which payment methods show fraud? (Credit Card, Debit Card, ACH)

---

# Part 1: Binary Classification - Simple Fraud Detection

## Use Case
Detect whether a transaction is fraudulent using simple rule-based logic.

## Configuration Files

### 1.1 Label Configuration (`label_config.json`)

```json
{
  "output_label_name": "is_fraud",
  "output_label_type": "binary",
  "label_values": [0, 1],
  "label_mapping": {
    "0": "Legitimate",
    "1": "Fraudulent"
  },
  "default_label": 0,
  "evaluation_mode": "priority"
}
```

**Key Points:**
- `output_label_name`: Single string for single-label mode
- `output_label_type`: "binary" for two-class classification
- `label_values`: List of two values (0 and 1)
- `label_mapping`: Human-readable descriptions
- `default_label`: Used when no rules match (legitimate by default)

### 1.2 Rule Definitions (`ruleset.json`)

```json
[
  {
    "rule_id": "rule_001",
    "name": "High value transaction from new account",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "transaction_amount",
          "operator": ">",
          "value": 5000
        },
        {
          "field": "account_age_days",
          "operator": "<",
          "value": 7
        }
      ]
    },
    "output_label": 1,
    "description": "New accounts with high-value transactions are suspicious"
  },
  {
    "rule_id": "rule_002",
    "name": "Velocity anomaly",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "velocity_score",
          "operator": ">",
          "value": 0.9
        },
        {
          "field": "device_risk",
          "operator": "equals",
          "value": "high"
        }
      ]
    },
    "output_label": 1,
    "description": "High velocity score with risky device indicates fraud"
  },
  {
    "rule_id": "rule_003",
    "name": "Trusted customer pattern",
    "priority": 3,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "customer_tier",
          "operator": "in",
          "value": ["gold", "platinum"]
        },
        {
          "field": "transaction_amount",
          "operator": "<",
          "value": 1000
        }
      ]
    },
    "output_label": 0,
    "description": "Trusted customers with normal amounts are legitimate"
  }
]
```

**Key Points:**
- `output_label`: Simple integer (0 or 1) for binary mode
- `priority`: Lower number = higher priority (evaluated first)
- `conditions`: Nested boolean logic (all_of, any_of, none_of)
- Rules evaluated top-to-bottom until first match

### 1.3 Python Configuration (Using DAGConfigFactory)

**Recommended Approach**: Use `DAGConfigFactory` for pipeline integration

```python
from cursus.api.factory.dag_config_factory import DAGConfigFactory
from cursus.api.dag.base_dag import PipelineDAG
from cursus.steps.configs.config_label_ruleset_generation_step import (
    LabelConfig,
    RuleDefinition,
    RulesetDefinitionList,
    RuleCondition,
    ComparisonOperator
)

# Step 1: Create pipeline DAG
dag = PipelineDAG()
dag.add_node("LabelRulesetGeneration")
dag.add_node("LabelRulesetExecution")
dag.add_edge("LabelRulesetGeneration", "LabelRulesetExecution")

# Step 2: Initialize DAGConfigFactory
factory = DAGConfigFactory(dag)

# Step 3: Set base pipeline configuration
factory.set_base_config(
    bucket="my-fraud-detection-bucket",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    region="NA",
    aws_region="us-east-1",
    author="data-science-team",
    service_name="FraudDetection",
    pipeline_version="1.0.0",
    model_class="binary_classifier",
    framework_version="2.1.0",
    py_version="py310",
    source_dir="docker",
    project_root_folder="fraud_detection"
)

# Step 4: Set base processing configuration
factory.set_base_processing_config(
    processing_source_dir="docker/scripts",
    processing_instance_type_large="ml.m5.12xlarge",
    processing_instance_type_small="ml.m5.4xlarge"
)

# Step 5: Define label configuration
label_config = LabelConfig(
    output_label_name="is_fraud",
    output_label_type="binary",
    label_values=[0, 1],
    label_mapping={"0": "Legitimate", "1": "Fraudulent"},
    default_label=0,
    evaluation_mode="priority"
)

# Step 6: Define rules
rules = [
    RuleDefinition(
        name="High value transaction from new account",
        priority=1,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="transaction_amount", operator=ComparisonOperator.GT, value=5000),
                RuleCondition(field="account_age_days", operator=ComparisonOperator.LT, value=7)
            ]
        ),
        output_label=1,
        description="New accounts with high-value transactions are suspicious"
    ),
    RuleDefinition(
        name="Velocity anomaly",
        priority=2,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="velocity_score", operator=ComparisonOperator.GT, value=0.9),
                RuleCondition(field="device_risk", operator=ComparisonOperator.EQUALS, value="high")
            ]
        ),
        output_label=1,
        description="High velocity score with risky device indicates fraud"
    ),
    RuleDefinition(
        name="Trusted customer pattern",
        priority=3,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="customer_tier", operator=ComparisonOperator.IN, value=["gold", "platinum"]),
                RuleCondition(field="transaction_amount", operator=ComparisonOperator.LT, value=1000)
            ]
        ),
        output_label=0,
        description="Trusted customers with normal amounts are legitimate"
    )
]

# Wrap rules in RulesetDefinitionList
ruleset_definitions = RulesetDefinitionList(rules=rules)

# Step 7: Configure the label ruleset generation step
factory.set_step_config(
    "LabelRulesetGeneration",
    # Pydantic models for configuration
    label_config=label_config,
    rule_definitions=ruleset_definitions,
    # Optional settings (with defaults)
    enable_field_validation=True,
    enable_label_validation=True,
    enable_logic_validation=True,
    enable_rule_optimization=True,
    ruleset_configs_path="ruleset_configs",
    processing_entry_point="label_ruleset_generation.py"
)

# Step 8: Configure the label ruleset execution step
factory.set_step_config(
    "LabelRulesetExecution",
    job_type="training",
    fail_on_missing_fields=True,
    enable_rule_match_tracking=True,
    enable_progress_logging=True,
    processing_entry_point="label_ruleset_execution.py"
)

# Step 9: Generate all configurations
configs = factory.generate_all_configs()

# Step 10: Save to JSON
from cursus.steps.configs.utils import merge_and_save_configs

config_path = "config/fraud_detection/config.json"
merged_config = merge_and_save_configs(configs, config_path)

print(f"✅ Binary classification pipeline configured and saved to {config_path}")
```

**Key Benefits of DAGConfigFactory**:
- ✅ Base configuration set once, inherited by all steps
- ✅ Step-by-step guidance with validation
- ✅ Automatic config class mapping
- ✅ Unified JSON output for pipeline compilation
- ✅ Clear separation of concerns

### 1.4 Expected Output

When executed, this produces a labeled dataset:

| transaction_id | transaction_amount | account_age_days | velocity_score | device_risk | customer_tier | **is_fraud** |
|----------------|-------------------|------------------|----------------|-------------|---------------|-------------|
| txn_001        | 6000              | 3                | 0.5            | medium      | silver        | **1**       |
| txn_002        | 200               | 365              | 0.2            | low         | gold          | **0**       |
| txn_003        | 1500              | 90               | 0.95           | high        | bronze        | **1**       |
| txn_004        | 500               | 730              | 0.1            | low         | platinum      | **0**       |

---

# Part 2: Multiclass Classification - Fraud Type Detection

## Use Case
Classify transactions into multiple fraud categories to understand fraud patterns.

## Configuration Files

### 2.1 Label Configuration (`label_config.json`)

```json
{
  "output_label_name": "fraud_type",
  "output_label_type": "multiclass",
  "label_values": [0, 1, 2, 3],
  "label_mapping": {
    "0": "No_Fraud",
    "1": "Account_Takeover",
    "2": "Card_Fraud",
    "3": "Refund_Abuse"
  },
  "default_label": 0,
  "evaluation_mode": "priority"
}
```

**Key Points:**
- `output_label_type`: "multiclass" for 3+ categories
- `label_values`: List of all possible integer values
- `label_mapping`: Descriptive name for each value
- Still single-label mode - one output column

### 2.2 Rule Definitions (`ruleset.json`)

```json
[
  {
    "rule_id": "rule_ato_001",
    "name": "Account takeover pattern",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "login_from_new_device",
          "operator": "equals",
          "value": true
        },
        {
          "field": "password_changed_recently",
          "operator": "equals",
          "value": true
        },
        {
          "field": "shipping_address_changed",
          "operator": "equals",
          "value": true
        }
      ]
    },
    "output_label": 1,
    "description": "Multiple account changes indicate account takeover"
  },
  {
    "rule_id": "rule_card_001",
    "name": "Card testing pattern",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "small_transactions_count_1h",
          "operator": ">",
          "value": 10
        },
        {
          "field": "card_bin_risk_score",
          "operator": ">",
          "value": 0.8
        }
      ]
    },
    "output_label": 2,
    "description": "Multiple small transactions with risky card indicates testing"
  },
  {
    "rule_id": "rule_refund_001",
    "name": "Refund abuse pattern",
    "priority": 3,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "refund_rate_30d",
          "operator": ">",
          "value": 0.5
        },
        {
          "field": "return_without_item_count",
          "operator": ">",
          "value": 3
        }
      ]
    },
    "output_label": 3,
    "description": "High refund rate with item retention indicates abuse"
  },
  {
    "rule_id": "rule_legitimate_001",
    "name": "Verified customer",
    "priority": 4,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "identity_verified",
          "operator": "equals",
          "value": true
        },
        {
          "field": "transaction_amount",
          "operator": "<",
          "value": 2000
        }
      ]
    },
    "output_label": 0,
    "description": "Verified customers with normal amounts are legitimate"
  }
]
```

**Key Points:**
- Each rule maps to a specific fraud type (1, 2, 3) or legitimate (0)
- Priority determines which category wins if multiple rules match
- Rules should be ordered from specific (fraud types) to general (legitimate)

### 2.3 Python Configuration (Using DAGConfigFactory)

**Recommended Approach**: Use `DAGConfigFactory` for pipeline integration

```python
from cursus.api.factory.dag_config_factory import DAGConfigFactory
from cursus.api.dag.base_dag import PipelineDAG
from cursus.steps.configs.config_label_ruleset_generation_step import (
    LabelConfig,
    RuleDefinition,
    RulesetDefinitionList,
    RuleCondition,
    ComparisonOperator
)

# Step 1: Create pipeline DAG
dag = PipelineDAG()
dag.add_node("LabelRulesetGeneration")
dag.add_node("LabelRulesetExecution")
dag.add_edge("LabelRulesetGeneration", "LabelRulesetExecution")

# Step 2: Initialize factory (assumes base configs already set)
factory = DAGConfigFactory(dag)
factory.set_base_config(
    bucket="my-fraud-detection-bucket",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    region="NA",
    aws_region="us-east-1",
    author="data-science-team",
    service_name="FraudTypeDetection",
    pipeline_version="1.0.0"
)

factory.set_base_processing_config(
    processing_source_dir="docker/scripts",
    processing_instance_type_large="ml.m5.12xlarge"
)

# Step 3: Define multiclass label configuration
label_config = LabelConfig(
    output_label_name="fraud_type",
    output_label_type="multiclass",
    label_values=[0, 1, 2, 3],
    label_mapping={
        "0": "No_Fraud",
        "1": "Account_Takeover",
        "2": "Card_Fraud",
        "3": "Refund_Abuse"
    },
    default_label=0,
    evaluation_mode="priority"
)

# Step 4: Define rules for each fraud category
rules = [
    RuleDefinition(
        name="Account takeover pattern",
        priority=1,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="login_from_new_device", operator=ComparisonOperator.EQUALS, value=True),
                RuleCondition(field="password_changed_recently", operator=ComparisonOperator.EQUALS, value=True),
                RuleCondition(field="shipping_address_changed", operator=ComparisonOperator.EQUALS, value=True)
            ]
        ),
        output_label=1,
        description="Multiple account changes indicate account takeover"
    ),
    RuleDefinition(
        name="Card testing pattern",
        priority=2,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="small_transactions_count_1h", operator=ComparisonOperator.GT, value=10),
                RuleCondition(field="card_bin_risk_score", operator=ComparisonOperator.GT, value=0.8)
            ]
        ),
        output_label=2,
        description="Multiple small transactions with risky card indicates testing"
    ),
    RuleDefinition(
        name="Refund abuse pattern",
        priority=3,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="refund_rate_30d", operator=ComparisonOperator.GT, value=0.5),
                RuleCondition(field="return_without_item_count", operator=ComparisonOperator.GT, value=3)
            ]
        ),
        output_label=3,
        description="High refund rate with item retention indicates abuse"
    ),
    RuleDefinition(
        name="Verified customer",
        priority=4,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="identity_verified", operator=ComparisonOperator.EQUALS, value=True),
                RuleCondition(field="transaction_amount", operator=ComparisonOperator.LT, value=2000)
            ]
        ),
        output_label=0,
        description="Verified customers with normal amounts are legitimate"
    )
]

ruleset_definitions = RulesetDefinitionList(rules=rules)

# Step 5: Configure label ruleset generation step
factory.set_step_config(
    "LabelRulesetGeneration",
    label_config=label_config,
    rule_definitions=ruleset_definitions,
    enable_field_validation=True,
    enable_label_validation=True,
    enable_logic_validation=True,
    enable_rule_optimization=True,
    ruleset_configs_path="ruleset_configs",
    processing_entry_point="label_ruleset_generation.py"
)

# Step 6: Configure label ruleset execution step
factory.set_step_config(
    "LabelRulesetExecution",
    job_type="training",
    fail_on_missing_fields=True,
    enable_rule_match_tracking=True,
    enable_progress_logging=True,
    processing_entry_point="label_ruleset_execution.py"
)

# Step 7: Generate and save configurations
configs = factory.generate_all_configs()

from cursus.steps.configs.utils import merge_and_save_configs
config_path = "config/fraud_detection/multiclass/config.json"
merge_and_save_configs(configs, config_path)

print(f"✅ Multiclass classification pipeline configured and saved to {config_path}")
```

**Key Benefits**:
- ✅ Consistent pattern across binary and multiclass configurations
- ✅ Easy to add additional pipeline steps (preprocessing, training, etc.)
- ✅ Unified configuration output for pipeline compilation

### 2.4 Expected Output

| transaction_id | login_new_device | password_changed | shipping_changed | small_txn_count | **fraud_type** | **fraud_type_name** |
|----------------|------------------|------------------|------------------|-----------------|----------------|---------------------|
| txn_001        | true             | true             | true             | 2               | **1**          | **Account_Takeover** |
| txn_002        | false            | false            | false            | 15              | **2**          | **Card_Fraud**       |
| txn_003        | false            | false            | false            | 1               | **0**          | **No_Fraud**         |
| txn_004        | false            | true             | false            | 3               | **3**          | **Refund_Abuse**     |

---

# Part 3: Multilabel Classification - Payment Method Fraud

## Use Case
Detect fraud patterns specific to each payment method simultaneously. A single transaction might show fraud signals for multiple payment types.

## Configuration Files

### 3.1 Label Configuration (`label_config.json`)

**Simple Global Configuration:**

```json
{
  "output_label_name": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
  "output_label_type": "multilabel",
  "label_values": [0, 1],
  "label_mapping": {
    "0": "No_Fraud",
    "1": "Fraud"
  },
  "default_label": 0,
  "evaluation_mode": "priority",
  "sparse_representation": true
}
```

**Advanced Per-Column Configuration:**

```json
{
  "output_label_name": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
  "output_label_type": "multilabel",
  "label_values": {
    "is_fraud_CC": [0, 1],
    "is_fraud_DC": [0, 1],
    "is_fraud_ACH": [0, 1, 2]
  },
  "label_mapping": {
    "is_fraud_CC": {"0": "No_Fraud", "1": "Fraud"},
    "is_fraud_DC": {"0": "No_Fraud", "1": "Fraud"},
    "is_fraud_ACH": {"0": "No_Fraud", "1": "Low_Risk", "2": "High_Risk"}
  },
  "default_label": {
    "is_fraud_CC": 0,
    "is_fraud_DC": 0,
    "is_fraud_ACH": 0
  },
  "evaluation_mode": "priority",
  "sparse_representation": true
}
```

**Key Points:**
- `output_label_name`: **List** of column names (not string)
- `output_label_type`: "multilabel" for multi-task learning
- `label_values`: Can be global (list) or per-column (dict)
- `label_mapping`: Can be global (dict) or per-column (dict of dicts)
- `default_label`: Can be global (int/str) or per-column (dict)
- `sparse_representation`: Use NaN for non-matching columns (memory efficient)

### 3.2 Rule Definitions (`ruleset.json`)

**Single-Column Targeting:**

```json
[
  {
    "rule_id": "rule_cc_001",
    "name": "High value CC transaction",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "payment_method",
          "operator": "equals",
          "value": "CC"
        },
        {
          "field": "transaction_amount",
          "operator": ">",
          "value": 1000
        }
      ]
    },
    "output_label": {"is_fraud_CC": 1},
    "description": "High value credit card transaction flagged as fraud"
  },
  {
    "rule_id": "rule_dc_001",
    "name": "International DC transaction",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "payment_method",
          "operator": "equals",
          "value": "DC"
        },
        {
          "field": "is_international",
          "operator": "equals",
          "value": true
        }
      ]
    },
    "output_label": {"is_fraud_DC": 1},
    "description": "International debit card transaction flagged as fraud"
  },
  {
    "rule_id": "rule_ach_001",
    "name": "Large ACH transfer",
    "priority": 3,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "payment_method",
          "operator": "equals",
          "value": "ACH"
        },
        {
          "field": "transaction_amount",
          "operator": ">",
          "value": 10000
        }
      ]
    },
    "output_label": {"is_fraud_ACH": 2},
    "description": "Large ACH transfer marked as high risk"
  }
]
```

**Multi-Column Targeting:**

```json
[
  {
    "rule_id": "rule_multi_001",
    "name": "Suspicious pattern across all methods",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "velocity_score",
          "operator": ">",
          "value": 0.9
        },
        {
          "field": "device_risk",
          "operator": "equals",
          "value": "high"
        }
      ]
    },
    "output_label": {
      "is_fraud_CC": 1,
      "is_fraud_DC": 1,
      "is_fraud_ACH": 1
    },
    "description": "High-risk pattern applies to all payment methods"
  },
  {
    "rule_id": "rule_cc_dc_001",
    "name": "Card fraud pattern",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "field": "card_bin_risk_score",
      "operator": ">",
      "value": 0.85
    },
    "output_label": {
      "is_fraud_CC": 1,
      "is_fraud_DC": 1
    },
    "description": "Risky card BIN affects both credit and debit cards"
  }
]
```

**Key Points:**
- `output_label`: **Dictionary** mapping column names to values
- Can target single column: `{"is_fraud_CC": 1}`
- Can target multiple columns: `{"is_fraud_CC": 1, "is_fraud_DC": 1}`
- Priority-based evaluation: first match wins for each column

### 3.3 Python Configuration (Using DAGConfigFactory)

**Simple Global Configuration (Recommended):**

```python
from cursus.api.factory.dag_config_factory import DAGConfigFactory
from cursus.api.dag.base_dag import PipelineDAG
from cursus.steps.configs.config_label_ruleset_generation_step import (
    LabelConfig,
    RuleDefinition,
    RulesetDefinitionList,
    RuleCondition,
    ComparisonOperator
)

# Step 1: Create pipeline DAG
dag = PipelineDAG()
dag.add_node("LabelRulesetGeneration")
dag.add_node("LabelRulesetExecution")
dag.add_edge("LabelRulesetGeneration", "LabelRulesetExecution")

# Step 2: Initialize factory
factory = DAGConfigFactory(dag)
factory.set_base_config(
    bucket="my-fraud-detection-bucket",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    region="NA",
    aws_region="us-east-1",
    author="data-science-team",
    service_name="PaymentMethodFraudDetection",
    pipeline_version="1.0.0"
)

factory.set_base_processing_config(
    processing_source_dir="docker/scripts",
    processing_instance_type_large="ml.m5.12xlarge"
)

# Step 3: Define multilabel configuration (simple global config)
label_config = LabelConfig(
    output_label_name=["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
    output_label_type="multilabel",
    label_values=[0, 1],  # Global: same values for all columns
    label_mapping={"0": "No_Fraud", "1": "Fraud"},  # Global: same mapping for all
    default_label=0,  # Global: same default for all columns
    evaluation_mode="priority",
    sparse_representation=True
)

# Step 4: Define rules with single and multi-column targeting
rules = [
    RuleDefinition(
        name="High value CC transaction",
        priority=1,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="payment_method", operator=ComparisonOperator.EQUALS, value="CC"),
                RuleCondition(field="transaction_amount", operator=ComparisonOperator.GT, value=1000)
            ]
        ),
        output_label={"is_fraud_CC": 1},  # Single column
        description="High value credit card transaction flagged as fraud"
    ),
    RuleDefinition(
        name="International DC transaction",
        priority=2,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="payment_method", operator=ComparisonOperator.EQUALS, value="DC"),
                RuleCondition(field="is_international", operator=ComparisonOperator.EQUALS, value=True)
            ]
        ),
        output_label={"is_fraud_DC": 1},  # Single column
        description="International debit card transaction flagged as fraud"
    ),
    RuleDefinition(
        name="Suspicious pattern across all methods",
        priority=3,
        enabled=True,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="velocity_score", operator=ComparisonOperator.GT, value=0.9),
                RuleCondition(field="device_risk", operator=ComparisonOperator.EQUALS, value="high")
            ]
        ),
        output_label={  # Multiple columns
            "is_fraud_CC": 1,
            "is_fraud_DC": 1,
            "is_fraud_ACH": 1
        },
        description="High-risk pattern applies to all payment methods"
    )
]

ruleset_definitions = RulesetDefinitionList(rules=rules)

# Step 5: Configure label ruleset generation step
factory.set_step_config(
    "LabelRulesetGeneration",
    label_config=label_config,
    rule_definitions=ruleset_definitions,
    enable_field_validation=True,
    enable_label_validation=True,
    enable_logic_validation=True,
    enable_rule_optimization=True,
    ruleset_configs_path="ruleset_configs",
    processing_entry_point="label_ruleset_generation.py"
)

# Step 6: Configure label ruleset execution step
factory.set_step_config(
    "LabelRulesetExecution",
    job_type="training",
    fail_on_missing_fields=True,
    enable_rule_match_tracking=True,
    enable_progress_logging=True,
    processing_entry_point="label_ruleset_execution.py"
)

# Step 7: Generate and save configurations
configs = factory.generate_all_configs()

from cursus.steps.configs.utils import merge_and_save_configs
config_path = "config/fraud_detection/multilabel/config.json"
merge_and_save_configs(configs, config_path)

print(f"✅ Multilabel classification pipeline configured and saved to {config_path}")
```

**Advanced Per-Column Configuration:**

```python
# Create multilabel configuration with per-column settings
label_config = LabelConfig(
    output_label_name=["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
    output_label_type="multilabel",
    label_values={
        "is_fraud_CC": [0, 1],
        "is_fraud_DC": [0, 1],
        "is_fraud_ACH": [0, 1, 2]  # ACH has 3 levels
    },
    label_mapping={
        "is_fraud_CC": {"0": "No_Fraud", "1": "Fraud"},
        "is_fraud_DC": {"0": "No_Fraud", "1": "Fraud"},
        "is_fraud_ACH": {"0": "No_Fraud", "1": "Low_Risk", "2": "High_Risk"}
    },
    default_label={
        "is_fraud_CC": 0,
        "is_fraud_DC": 0,
        "is_fraud_ACH": 0
    },
    evaluation_mode="priority",
    sparse_representation=True
)

# Rules can now use different values for ACH
rules = [
    RuleDefinition(
        name="Large ACH transfer",
        priority=1,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="payment_method", operator="equals", value="ACH"),
                RuleCondition(field="transaction_amount", operator=">", value=10000)
            ]
        ),
        output_label={"is_fraud_ACH": 2},  # High risk
        description="Large ACH transfer marked as high risk"
    ),
    RuleDefinition(
        name="Medium ACH transfer",
        priority=2,
        conditions=RuleCondition(
            all_of=[
                RuleCondition(field="payment_method", operator="equals", value="ACH"),
                RuleCondition(field="transaction_amount", operator=">", value=5000)
            ]
        ),
        output_label={"is_fraud_ACH": 1},  # Low risk
        description="Medium ACH transfer marked as low risk"
    )
]
```

### 3.4 Expected Output

**Sparse Representation (default):**

| transaction_id | payment_method | amount | velocity | **is_fraud_CC** | **is_fraud_DC** | **is_fraud_ACH** |
|----------------|----------------|--------|----------|-----------------|-----------------|------------------|
| txn_001        | CC             | 6000   | 0.5      | **1**           | NaN             | NaN              |
| txn_002        | DC             | 200    | 0.2      | NaN             | **0**           | NaN              |
| txn_003        | ACH            | 15000  | 0.1      | NaN             | NaN             | **2**            |
| txn_004        | CC             | 1500   | 0.95     | **1**           | NaN             | NaN              |
| txn_005        | All            | 500    | 0.99     | **1**           | **1**           | **1**            |

**Dense Representation:**

| transaction_id | payment_method | amount | velocity | **is_fraud_CC** | **is_fraud_DC** | **is_fraud_ACH** |
|----------------|----------------|--------|----------|-----------------|-----------------|------------------|
| txn_001        | CC             | 6000   | 0.5      | **1**           | **0**           | **0**            |
| txn_002        | DC             | 200    | 0.2      | **0**           | **0**           | **0**            |
| txn_003        | ACH            | 15000  | 0.1      | **0**           | **0**           | **2**            |
| txn_004        | CC             | 1500   | 0.95     | **1**           | **0**           | **0**            |
| txn_005        | All            | 500    | 0.99     | **1**           | **1**           | **1**            |

---

# Part 4: Comparison and Best Practices

## When to Use Each Type

### Binary Classification
**Use When:**
- Simple yes/no decision needed
- Two distinct outcomes
- Balanced class distribution acceptable

**Examples:**
- Fraud vs Legitimate
- Approve vs Deny
- Pass vs Fail

**Benefits:**
- Simplest to configure and understand
- Easiest to validate and monitor
- Clear decision boundary

### Multiclass Classification
**Use When:**
- Multiple mutually exclusive categories
- Need to understand type/reason for classification
- Single aspect has multiple values

**Examples:**
- Fraud type (Account Takeover, Card Fraud, Refund Abuse)
- Risk level (Low, Medium, High, Critical)
- Product category (Electronics, Clothing, Food)

**Benefits:**
- Rich categorical information
- One evaluation per transaction
- Natural for exclusive categories

### Multilabel Classification
**Use When:**
- Multiple independent classification tasks
- Categories not mutually exclusive
- Multi-task learning required

**Examples:**
- Fraud detection per payment method
- Risk assessment per product category
- Compliance checks per regulation

**Benefits:**
- Sparse representation (memory efficient)
- Independent evaluation per label
- Supports multi-task ML models
- Category-conditional business logic

## Configuration Patterns

### 1. Rule Priority Design

**Binary/Multiclass:**
```
Priority 1: Most specific fraud patterns (high confidence)
Priority 2: Moderate fraud patterns
Priority 3: Low confidence fraud patterns
Priority 4: Legitimate patterns
Priority 5: Default (catch-all)
```

**Multilabel:**
```
Priority 1: Cross-cutting patterns (affect all columns)
Priority 2: Two-column patterns (affect subset)
Priority 3: Single-column patterns (specific to one)
```

### 2. Condition Complexity

**Simple Conditions (Recommended):**
```json
{
  "field": "transaction_amount",
  "operator": ">",
  "value": 1000
}
```

**Nested Conditions:**
```json
{
  "all_of": [
    {"field": "amount", "operator": ">", "value": 1000},
    {
      "any_of": [
        {"field": "country", "operator": "equals", "value": "US"},
        {"field": "country", "operator": "equals", "value": "CA"}
      ]
    }
  ]
}
```

### 3. Default Label Strategy

**Conservative (Recommended):**
```json
{
  "default_label": 0  // For binary/multiclass: flag as legitimate/safe
}
```
- Use when false negatives are acceptable
- Safer for business operations
- Example: Fraud detection (default to legitimate)

**Aggressive:**
```json
{
  "default_label": 1  // For binary/multiclass: flag as fraud/suspicious
}
```
- Use when false positives are acceptable
- More cautious approach
- Example: High-risk transactions (default to review)

**Multilabel Considerations:**
```json
{
  "default_label": {
    "is_fraud_CC": 0,
    "is_fraud_DC": 0,
    "is_fraud_ACH": 1  // More conservative for ACH transfers
  }
}
```
- Per-column defaults enable risk-based strategies
- Different payment methods can have different default behaviors
- Balance business risk with operational efficiency

---

# Part 5: Advanced Topics

## Operator Reference

### Comparison Operators
- `equals` / `not_equals`: Exact match
- `>`, `>=`, `<`, `<=`: Numeric comparison
- `in`, `not_in`: Collection membership

### String Operators
- `contains` / `not_contains`: Substring search
- `starts_with` / `ends_with`: Prefix/suffix match
- `regex_match`: Pattern matching

### Null Operators
- `is_null` / `is_not_null`: Null checking

## Evaluation Modes

### Priority Mode (Default)
```json
{
  "evaluation_mode": "priority"
}
```
- First matching rule wins
- Rules evaluated in priority order (1, 2, 3...)
- Deterministic and predictable
- **Recommended for most use cases**

### Confidence Mode (Future)
```json
{
  "evaluation_mode": "confidence"
}
```
- Highest confidence rule wins
- Requires confidence scores in rules
- More flexible but complex
- Reserved for advanced scenarios

## Validation Best Practices

### Field Validation
```python
factory.set_step_config(
    "LabelRulesetGeneration",
    enable_field_validation=True,  # Validates fields exist in data
    # ...
)
```
- Catches typos in field names early
- Ensures rules reference valid data fields
- Fails fast before execution

### Label Validation
```python
factory.set_step_config(
    "LabelRulesetGeneration",
    enable_label_validation=True,  # Validates output labels are valid
    # ...
)
```
- Ensures output labels match configuration
- Prevents invalid label values
- Critical for multilabel correctness

### Logic Validation
```python
factory.set_step_config(
    "LabelRulesetGeneration",
    enable_logic_validation=True,  # Validates rule logic
    # ...
)
```
- Detects tautologies (always true)
- Detects contradictions (never true)
- Identifies unreachable rules

## Performance Optimization

### Rule Optimization
```python
factory.set_step_config(
    "LabelRulesetGeneration",
    enable_rule_optimization=True,  # Reorders rules for efficiency
    # ...
)
```
- Reorders rules by complexity
- Simpler rules evaluated first
- Reduces average evaluation time

### Sparse Representation
```python
label_config = LabelConfig(
    output_label_type="multilabel",
    sparse_representation=True,  # Use NaN for non-matching columns
    # ...
)
```
- **Memory efficient**: Only stores matching columns
- **Training friendly**: Many ML frameworks handle NaN
- **Interpretable**: Clear which columns were evaluated

## Troubleshooting

### Common Issues

**Issue 1: No rules matching**
```
Problem: All records get default_label
Solution: 
- Review rule conditions
- Check field names for typos
- Verify data values match expectations
- Add debug logging
```

**Issue 2: Wrong label values**
```
Problem: Labels don't match configuration
Solution:
- Enable label validation
- Check output_label in rules
- Verify label_values includes all outputs
- Review label_mapping
```

**Issue 3: Multilabel columns missing**
```
Problem: Some label columns not created
Solution:
- Check output_label_name is a list
- Verify all columns in output_label
- Enable sparse_representation for NaN support
```

**Issue 4: Field not found errors**
```
Problem: Field referenced in rules doesn't exist
Solution:
- Enable field validation
- Check field names match data exactly
- Review required_fields in field config
```

## Testing Strategy

### Unit Testing Rules
1. **Test individual rules** with known inputs
2. **Test rule priority** with multiple matching rules
3. **Test default behavior** with no matches
4. **Test edge cases** (nulls, empty strings, extremes)

### Integration Testing
1. **End-to-end pipeline** with realistic data
2. **Label distribution** matches expectations
3. **Performance** on large datasets
4. **Rule coverage** - all rules fire at least once

### Validation Testing
1. **Schema validation** catches configuration errors
2. **Field validation** catches missing fields
3. **Label validation** catches invalid outputs
4. **Logic validation** catches unreachable rules

---

# Summary

This document provided complete, working examples for configuring label ruleset generation across three classification types:

## Key Takeaways

### Binary Classification
- ✅ Simplest configuration
- ✅ Two output values (0, 1)
- ✅ Single label column
- ✅ Use for yes/no decisions

### Multiclass Classification
- ✅ Multiple exclusive categories
- ✅ Single label column with 3+ values
- ✅ Rich categorical information
- ✅ Use for type/reason classification

### Multilabel Classification
- ✅ Multiple label columns
- ✅ Independent task per column
- ✅ Sparse or dense representation
- ✅ Use for multi-task learning

## Configuration Pattern

All three types follow the same DAGConfigFactory workflow:
1. Create pipeline DAG
2. Initialize factory
3. Set base configs
4. Define label config and rules
5. Configure generation step
6. Configure execution step
7. Generate and save unified config

## Best Practices

- ✅ **Use DAGConfigFactory** for pipeline integration
- ✅ **Enable all validations** during development
- ✅ **Start with simple rules** and iterate
- ✅ **Test with realistic data** early
- ✅ **Monitor rule coverage** in production
- ✅ **Document rule intent** in descriptions
- ✅ **Version control** rule configurations

## Next Steps

1. **Choose your classification type** (binary, multiclass, or multilabel)
2. **Copy the relevant example** from this document
3. **Customize** for your use case
4. **Test** with your data
5. **Deploy** to your pipeline
6. **Monitor** and iterate

## Related Documentation

### Design Documents
- **[Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)** - Complete design specification for multilabel support
- **[Implementation Plan](../2_project_planning/2025-11-11_label_ruleset_multilabel_extension_implementation_plan.md)** - Detailed implementation phases and technical approach

### Source Code Reference
- **Configuration Models**: `src/cursus/steps/configs/config_label_ruleset_generation_step.py`
  - `LabelConfig` - Label configuration with multilabel support
  - `RuleDefinition` - Rule definition with Pydantic validation
  - `RuleCondition` - Nested condition structures
  - `RulesetDefinitionList` - Validated rule collections
  - `LabelRulesetGenerationConfig` - Complete step configuration

- **Processing Scripts**:
  - `src/cursus/steps/scripts/label_ruleset_generation.py` - Generation and validation logic
  - `src/cursus/steps/scripts/label_ruleset_execution.py` - Rule execution engine

- **Contracts**:
  - `src/cursus/steps/contracts/label_ruleset_generation_contract.py` - Generation step contract
  - `src/cursus/steps/contracts/label_ruleset_execution_contract.py` - Execution step contract

### Factory Pattern Integration
- **DAGConfigFactory**: `src/cursus/api/factory/dag_config_factory.py`
- **Example Notebook**: `projects/rnr_pytorch_bedrock/demo_config_widget_dummy_bedrock.ipynb`

### Label Ruleset Design Documentation
- **[Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)** - Complete design specification for multilabel support
- **[Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md)** - Generation step design patterns
- **[Label Ruleset Execution Step Patterns](../1_design/label_ruleset_execution_step_patterns.md)** - Execution step design patterns
- **[Label Ruleset Optimization Patterns](../1_design/label_ruleset_optimization_patterns.md)** - Rule optimization strategies
- **[LLM Category to Binary Label Ruleset Example](../1_design/llm_category_to_binary_label_ruleset_example.md)** - Real-world example workflow

### Configuration System Design
- **[DAG Config Factory Design](../1_design/dag_config_factory_design.md)** - Factory pattern for pipeline configuration
- **[Config Tiered Design](../1_design/config_tiered_design.md)** - Three-tier configuration architecture
- **[Config Manager Three Tier Implementation](../1_design/config_manager_three_tier_implementation.md)** - Implementation details

### Implementation Plans
- **[Label Ruleset Multilabel Extension Plan](../2_project_planning/2025-11-11_label_ruleset_multilabel_extension_implementation_plan.md)** - Detailed implementation phases
- **[DAG Config Factory Implementation Plan](../2_project_planning/2025-10-15_dag_config_factory_implementation_plan.md)** - Factory pattern implementation
- **[Bedrock Prompt Template Generation Plan](../2_project_planning/2025-11-02_bedrock_prompt_template_generation_step_implementation_plan.md)** - Related step implementation example

### Analysis and Comparisons
- **[Label Generation Workflow Comparison](../4_analysis/2025-11-11_label_generation_workflow_comparison.md)** - Comparison of label generation approaches

### Entry Points
- **[Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)** - Central index for step documentation
- **[Processing Steps Index](../00_entry_points/processing_steps_index.md)** - Overview of all processing steps
