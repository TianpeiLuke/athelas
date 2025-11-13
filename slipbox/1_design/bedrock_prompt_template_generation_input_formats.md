---
tags:
  - design
  - implementation
  - bedrock_steps
  - input_formats
  - documentation
  - user_guide
keywords:
  - category definitions format
  - task requirements format
  - output schema template format
  - input specifications
topics:
  - input file formats
  - data structure requirements
  - template generation inputs
language: json, csv
date of note: 2025-11-02
---

# Bedrock Prompt Template Generation - Input Format Guide

## Overview

This document provides comprehensive instructions on the required formats for input files used by the Bedrock Prompt Template Generation script. The script accepts three types of input files to generate high-quality prompt templates.

## Input File Locations

The script expects input files in the following container directories:

- **Category Definitions**: `/opt/ml/processing/input/categories` (Required)
- **Output Schema Template**: `/opt/ml/processing/input/schema` (Optional - for advanced customization only)

**Note**: Task Requirements are not needed as input since the script has comprehensive defaults and auto-configuration.

## 1. Category Definitions (Required)

### Purpose
Category definitions are the core input that defines the classification categories, their conditions, exceptions, and metadata.

### Supported Formats
- **JSON** (preferred): `.json` files
- **CSV** (alternative): `.csv` files

### JSON Format Structure

#### Single Category File
```json
{
  "name": "Positive",
  "description": "Positive sentiment or favorable opinion",
  "conditions": [
    "Contains positive language or expressions",
    "Expresses satisfaction or approval",
    "Shows enthusiasm or excitement"
  ],
  "exceptions": [
    "Sarcastic statements with positive words",
    "Backhanded compliments",
    "Ironic positive expressions"
  ],
  "key_indicators": [
    "good", "excellent", "satisfied", "happy", "pleased",
    "wonderful", "amazing", "fantastic", "love", "great"
  ],
  "examples": [
    "This product is amazing!",
    "I'm very satisfied with the service",
    "Excellent work, thank you!"
  ],
  "priority": 1,
  "validation_rules": [
    "Must contain at least one positive indicator",
    "Cannot contain negative qualifiers"
  ],
  "aliases": ["positive_sentiment", "favorable", "good"]
}
```

#### Multiple Categories File
```json
[
  {
    "name": "Positive",
    "description": "Positive sentiment or favorable opinion",
    "conditions": [
      "Contains positive language or expressions",
      "Expresses satisfaction or approval"
    ],
    "exceptions": [
      "Sarcastic statements with positive words"
    ],
    "key_indicators": [
      "good", "excellent", "satisfied", "happy"
    ],
    "priority": 1
  },
  {
    "name": "Negative",
    "description": "Negative sentiment or unfavorable opinion",
    "conditions": [
      "Contains negative language or expressions",
      "Expresses dissatisfaction or disapproval"
    ],
    "exceptions": [
      "Constructive criticism with positive intent"
    ],
    "key_indicators": [
      "bad", "terrible", "disappointed", "angry"
    ],
    "priority": 2
  },
  {
    "name": "Neutral",
    "description": "Neutral sentiment or factual statements",
    "conditions": [
      "Factual information without emotional language",
      "Balanced statements with both positive and negative aspects"
    ],
    "exceptions": [
      "Mixed sentiment with clear positive or negative lean"
    ],
    "key_indicators": [
      "okay", "average", "standard", "normal"
    ],
    "priority": 3
  }
]
```

### CSV Format Structure

#### CSV File Format
```csv
name,description,conditions,exceptions,key_indicators,priority,examples,validation_rules,aliases
Positive,"Positive sentiment or favorable opinion","Contains positive language;Expresses satisfaction","Sarcastic statements;Backhanded compliments","good;excellent;satisfied;happy",1,"This is great!;Love this product","Must contain positive indicator","positive_sentiment;favorable"
Negative,"Negative sentiment or unfavorable opinion","Contains negative language;Expresses dissatisfaction","Constructive criticism","bad;terrible;disappointed;angry",2,"This is awful;Very disappointed","Must contain negative indicator","negative_sentiment;unfavorable"
Neutral,"Neutral sentiment or factual statements","Factual information;Balanced statements","Mixed sentiment with clear lean","okay;average;standard;normal",3,"It's okay;Standard quality","Must be factual","neutral_sentiment;balanced"
```

### Required Fields
- **name** (string): Unique category name
- **description** (string): Clear description of the category
- **conditions** (array/semicolon-separated): Conditions that must be met for this category
- **key_indicators** (array/semicolon-separated): Key words or phrases that indicate this category

### Optional Fields
- **exceptions** (array/semicolon-separated): Conditions that exclude this category
- **examples** (array/semicolon-separated): Example texts for this category
- **priority** (integer): Priority order (lower numbers = higher priority)
- **validation_rules** (array/semicolon-separated): Additional validation rules
- **aliases** (array/semicolon-separated): Alternative names for this category

### Validation Rules
1. Each category must have a unique `name`
2. `conditions` and `key_indicators` cannot be empty
3. All text fields should be non-empty strings
4. Priority should be a positive integer (defaults to 999 if not provided)

## 2. Output Schema Template (Optional)

### Purpose
Output schema template allows customization of the generated output format and validation schema.

### Format
JSON Schema format

### Structure
```json
{
  "title": "Classification Output Schema",
  "description": "Schema for classification results",
  "properties": {
    "category": {
      "description": "The classified category name",
      "enum": ["Positive", "Negative", "Neutral"]
    },
    "confidence": {
      "description": "Confidence score between 0.0 and 1.0",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "key_evidence": {
      "description": "Specific evidence supporting the classification",
      "minLength": 10,
      "maxLength": 500
    },
    "reasoning": {
      "description": "Detailed explanation of the classification decision",
      "minLength": 20,
      "maxLength": 1000
    },
    "metadata": {
      "description": "Additional metadata about the classification",
      "properties": {
        "processing_time": {
          "description": "Time taken for classification in milliseconds"
        },
        "model_version": {
          "description": "Version of the classification model used"
        },
        "flags": {
          "description": "Any special flags or warnings"
        }
      }
    }
  },
  "required": ["category", "confidence", "key_evidence", "reasoning"]
}
```

### Customization Options

#### Field Definitions
- **properties**: Define custom fields and their validation rules
- **required**: Specify which fields are mandatory
- **additionalProperties**: Control whether extra fields are allowed

#### Validation Rules
- **type**: Data type for each field
- **minimum/maximum**: Numeric constraints
- **minLength/maxLength**: String length constraints
- **enum**: Allowed values for categorical fields
- **pattern**: Regular expression patterns for string validation

#### Advanced Features
- **nested objects**: Support for complex nested structures
- **arrays**: Support for list-type fields
- **conditional validation**: Rules that depend on other field values

## File Organization Examples

### Example 1: Simple Setup (Minimal Required)
```
/opt/ml/processing/input/
└── categories/
    └── sentiment_categories.json
```

### Example 2: Multiple Category Files
```
/opt/ml/processing/input/
├── categories/
│   ├── primary_categories.json
│   ├── secondary_categories.json
│   └── edge_cases.csv
└── schema/
    └── validation_schema.json
```

### Example 3: CSV-Based Setup
```
/opt/ml/processing/input/
├── categories/
│   └── all_categories.csv
└── schema/
    └── output_format.json
```

## Configuration Parameters

### INPUT_PLACEHOLDERS
**Purpose**: Defines all input fields that will be included in the generated prompt template.

**Format**: JSON array of strings
**Default**: `["input_data"]`

**Examples**:
```json
// Simple input (default)
INPUT_PLACEHOLDERS='["input_data"]'

// Multiple input fields
INPUT_PLACEHOLDERS='["input_data", "product_category", "customer_tier", "purchase_date"]'

// Complex context
INPUT_PLACEHOLDERS='["customer_comment", "product_category", "customer_tier", "purchase_date", "support_channel"]'
```

**Generated Template Output**:
```
Analysis Instructions:

Please analyze:
Input_data: {input_data}
Product_category: {product_category}
Customer_tier: {customer_tier}
Purchase_date: {purchase_date}
```

**Usage in Downstream Processing**:
```python
# Format template with all placeholder values
formatted_prompt = template['user_prompt_template'].format(
    input_data="Customer review text here...",
    product_category="Electronics",
    customer_tier="Premium",
    purchase_date="2025-10-15"
)
```

## Best Practices

### Category Definitions
1. **Be Specific**: Write clear, unambiguous conditions
2. **Include Exceptions**: Define what should NOT be classified in each category
3. **Use Examples**: Provide concrete examples for each category
4. **Set Priorities**: Use priority ordering for overlapping categories
5. **Test Coverage**: Ensure categories cover all expected input cases

### Input Placeholders
1. **Start Simple**: Begin with basic `["input_data"]` and add fields as needed
2. **Use Descriptive Names**: Choose clear, meaningful field names
3. **Consider Context**: Include fields that provide relevant classification context
4. **Avoid Redundancy**: Don't duplicate information across multiple fields
5. **Test Integration**: Ensure downstream processing can provide all placeholder values

### Output Schema
1. **Start Simple**: Begin with basic schema and add complexity as needed
2. **Validate Constraints**: Set appropriate validation rules for your use case
3. **Document Fields**: Provide clear descriptions for all fields
4. **Test Schema**: Validate that your schema works with expected outputs

## Common Issues and Solutions

### Issue: Categories Not Loading
**Cause**: Invalid JSON format or missing required fields
**Solution**: Validate JSON syntax and ensure all required fields are present

### Issue: CSV Parsing Errors
**Cause**: Incorrect delimiter usage or special characters
**Solution**: Use semicolons for array separators, escape special characters

### Issue: Schema Validation Failures
**Cause**: Incompatible schema with default output format
**Solution**: Ensure schema includes all required default fields

### Issue: Empty Output
**Cause**: No valid category files found
**Solution**: Check file locations and naming conventions

## Validation Checklist

Before running the script, verify:

- [ ] Category definition files are in correct location
- [ ] JSON files have valid syntax
- [ ] CSV files use correct delimiter (semicolon for arrays)
- [ ] All required fields are present in categories
- [ ] Category names are unique
- [ ] File permissions allow reading
- [ ] Optional files (requirements, schema) are valid JSON

This comprehensive input format guide ensures successful prompt template generation with high-quality, customized outputs tailored to your specific classification needs.
