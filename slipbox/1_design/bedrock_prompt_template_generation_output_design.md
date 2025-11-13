---
tags:
  - design
  - implementation
  - bedrock_steps
  - output_formats
  - template_generation
keywords:
  - bedrock prompt output
  - template generation output
  - prompt template files
  - validation schema
  - metadata format
topics:
  - output file structure
  - template format design
  - validation and quality metrics
  - integration specifications
language: json
date of note: 2025-11-02
---

# Bedrock Prompt Template Generation - Output Design

## Overview

This document defines the comprehensive output design for the Bedrock Prompt Template Generation step, including file structures, formats, and integration specifications. The output design ensures seamless integration with downstream Bedrock processing steps while providing comprehensive validation and quality assurance.

## Output Architecture

### Container Output Directories

The step generates outputs in three primary directories:

```
/opt/ml/processing/output/
├── templates/          # Generated prompt templates (primary output)
├── metadata/           # Template metadata and validation results
└── schema/             # Validation schemas for output format
```

### Output File Types

1. **Primary Template Files**: Ready-to-use prompt templates
2. **Metadata Files**: Generation statistics and validation results
3. **Schema Files**: JSON schemas for output validation
4. **Quality Reports**: Template quality metrics and recommendations

## Primary Output Files

### 1. Prompt Templates (`/opt/ml/processing/output/templates/`)

#### Main Template File: `prompts.json`

**Purpose**: Primary output containing the generated prompt templates ready for Bedrock processing.

**Structure**:
```json
{
  "system_prompt": "You are an expert analyst with extensive knowledge in data analysis, classification, pattern recognition. Your task is to analyze data accurately, classify content systematically, provide clear reasoning. Always be precise, be objective, be thorough, be consistent in your analysis.",
  "user_prompt_template": "Categories and their criteria:\n\n1. Positive\n    - Positive sentiment or favorable opinion\n    - Key elements:\n        * good\n        * excellent\n        * satisfied\n        * happy\n    - Conditions:\n        * Contains positive language or expressions\n        * Expresses satisfaction or approval\n    - Must NOT include:\n        * Sarcastic statements with positive words\n\nAnalysis Instructions:\n\nPlease analyze:\nInput_data: {input_data}\nCustomer_id: {customer_id}\nTransaction_amount: {transaction_amount}\n\nProvide your analysis in the following structured format:\n\n1. Carefully review all provided data\n2. Identify key patterns and indicators\n3. Match against category criteria\n4. Select the most appropriate category\n5. Validate evidence against conditions and exceptions\n6. Provide confidence assessment and reasoning\n\nDecision Criteria:\n- Base decisions on explicit evidence in the data\n- Consider all category conditions and exceptions\n- Choose the category with the strongest evidence match\n- Provide clear reasoning for your classification\n\nKey Evidence Validation:\n- Evidence MUST align with at least one condition for the selected category\n- Evidence MUST NOT match any exceptions listed for the selected category\n- Evidence should reference specific content from the input data\n- Multiple pieces of supporting evidence strengthen the classification\n\n## Required Output Format\n\n**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**\n\n```json\n{\n    \"category\": \"One of: Positive, Negative, Neutral\",\n    \"confidence\": \"Number between 0.0 and 1.0\",\n    \"key_evidence\": \"Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.\",\n    \"reasoning\": \"Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected\"\n}\n```\n\nField Descriptions:\n- **category** (string): The classified category name (must be exactly one of the defined categories) (must be one of: Positive, Negative, Neutral)\n- **confidence** (number): Confidence score between 0.0 and 1.0 indicating certainty of classification (minimum: 0.0, maximum: 1.0)\n- **key_evidence** (string): Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.\n- **reasoning** (string): Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected\n\n**Category Validation:**\n- The category field must exactly match one of: Positive, Negative, Neutral\n- Category names are case-sensitive and must match exactly\n\nDo not include any text before or after the JSON object. Only return valid JSON.",
  "input_placeholders": ["input_data", "customer_id", "transaction_amount"]
}
```

**Key Features**:
- **system_prompt**: Role definition and behavioral guidelines
- **user_prompt_template**: Complete 5-component template with embedded placeholders
- **input_placeholders**: Array of placeholder names for direct access by Processing steps
- **Integration Ready**: Direct compatibility with Bedrock processing steps
- **Zero Configuration Processing**: Placeholders automatically mapped without regex extraction
- **Dynamic Content Injection**: Supports `{variable_name}` syntax within the template text

#### Template Components Breakdown

**Component 1: System Prompt**
- Expert role assignment
- Expertise areas definition
- Behavioral guidelines
- Professional tone establishment

**Component 2: Category Definitions**
- Structured category descriptions
- Conditions and exceptions
- Key indicators and examples
- Priority-based ordering

**Component 3: Input Placeholders**
- Dynamic variable placeholders from `INPUT_PLACEHOLDERS` environment variable
- Default: `["input_data"]` but configurable for multiple fields
- Context field support for additional data columns
- Flexible data injection points via `{placeholder_name}` syntax
- Example: `Input_data: {input_data}`, `Customer_id: {customer_id}`, `Amount: {transaction_amount}`

**Component 4: Instructions and Rules**
- Step-by-step analysis process
- Decision criteria guidelines
- Evidence validation requirements
- Quality assurance rules

**Component 5: Output Format Schema**
- Structured JSON specification
- Field descriptions and constraints
- Validation requirements
- Evidence alignment rules

### 2. Template Metadata (`/opt/ml/processing/output/metadata/`)

#### Metadata File: `template_metadata_{timestamp}.json`

**Purpose**: Comprehensive metadata about template generation, validation results, and quality metrics.

**Structure**:
```json
{
  "template_version": "1.0",
  "generation_timestamp": "2025-11-02T16:20:00.123456",
  "task_type": "classification",
  "template_style": "structured",
  "category_count": 3,
  "category_names": ["Positive", "Negative", "Neutral"],
  "output_format": "structured_json",
  "validation_level": "standard",
  "includes_examples": true,
  "generator_config": {
    "system_prompt_config": {
      "role_definition": "expert analyst",
      "expertise_areas": ["data analysis", "classification", "pattern recognition"],
      "behavioral_guidelines": ["be precise", "be objective", "be thorough", "be consistent"]
    },
    "output_format_config": {
      "format_type": "structured_json",
      "required_fields": ["category", "confidence", "key_evidence", "reasoning"],
      "evidence_validation_rules": [
        "Evidence MUST align with at least one condition for the selected category",
        "Evidence MUST NOT match any exceptions listed for the selected category"
      ]
    }
  },
  "validation_results": {
    "is_valid": true,
    "quality_score": 0.95,
    "validation_details": [
      {
        "component": "system_prompt",
        "is_valid": true,
        "score": 1.0,
        "issues": []
      },
      {
        "component": "user_prompt_template",
        "is_valid": true,
        "score": 0.9,
        "issues": []
      },
      {
        "component": "metadata",
        "is_valid": true,
        "score": 1.0,
        "issues": []
      }
    ],
    "recommendations": []
  },
  "generation_config": {
    "task_type": "classification",
    "template_style": "structured",
    "validation_level": "standard",
    "category_count": 3
  }
}
```

**Key Sections**:

**Template Information**:
- Version and timestamp tracking
- Task type and style classification
- Category statistics and names

**Generation Configuration**:
- System prompt settings
- Output format specifications
- Validation level and rules

**Quality Validation**:
- Overall validity status
- Component-specific scores
- Quality recommendations
- Issue identification

**Usage Statistics**:
- Processing metrics
- Performance indicators
- Resource utilization

### 3. Validation Schema (`/opt/ml/processing/output/schema/`)

#### Schema File: `validation_schema_{timestamp}.json`

**Purpose**: Enhanced JSON schema for validating outputs generated BY Bedrock when using the prompt template, with additional processing metadata for Bedrock Processing step integration. This schema is NOT used during template generation but is provided as an output for downstream validation and processing configuration.

**Structure**:
```json
{
  "title": "Bedrock Response Validation Schema",
  "description": "Schema for validating Bedrock LLM responses with processing metadata",
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["Positive", "Negative", "Neutral"],
      "description": "The classified category name (must be exactly one of the defined categories)"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence score between 0.0 and 1.0 indicating certainty of classification"
    },
    "key_evidence": {
      "type": "string",
      "description": "Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision."
    },
    "reasoning": {
      "type": "string",
      "description": "Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected"
    }
  },
  "required": ["category", "confidence", "key_evidence", "reasoning"],
  "additionalProperties": false,
  
  "processing_config": {
    "format_type": "structured_json",
    "response_model_name": "ClassificationResponse",
    "validation_level": "standard"
  },
  
  "template_metadata": {
    "template_version": "1.0",
    "generation_timestamp": "20251102_162000",
    "output_format_source": "OUTPUT_FORMAT_CONFIG",
    "task_type": "classification",
    "template_style": "structured"
  }
}
```

**Enhanced Validation Features**:
- **Standard JSON Schema**: Core validation for Bedrock responses
- **Category Constraints**: Enum validation against defined categories
- **Confidence Bounds**: Numeric range validation (0.0-1.0)
- **Field Requirements**: All 4 fields mandatory
- **Type Safety**: Strict type checking
- **Additional Properties**: Blocked to ensure clean output

**Processing Configuration** (New):
- **Format Type**: Response format specification for processing steps
- **Response Model Name**: Dynamic Pydantic model name generation
- **Validation Level**: Processing validation strictness

**Template Integration Metadata** (New):
- **Template Version**: Version tracking for template evolution
- **Generation Timestamp**: Creation time for audit trails
- **Category Statistics**: Count and names of defined categories
- **Output Format Source**: Configuration source tracking
- **Task and Style**: Template type and style classification

## Output Format Specifications

### JSON Structure Standards

#### Field Definitions

**category** (string, required)
- Must exactly match one of the predefined category names
- Case-sensitive validation
- No partial matches or variations allowed
- Enum constraint enforced via schema

**confidence** (number, required)
- Range: 0.0 to 1.0 (inclusive)
- Decimal precision supported
- Represents classification certainty
- Higher values indicate stronger evidence

**key_evidence** (string, required)
- Specific content references from input data
- Must align with category conditions
- Must NOT match category exceptions
- Direct quotes or specific observations
- Multiple evidence pieces strengthen classification

**reasoning** (string, required)
- Clear explanation of decision-making process
- Shows connection between evidence and category
- Considers alternative categories and rejection rationale
- Demonstrates logical analysis flow
- Provides transparency for classification decisions

### Quality Metrics and Validation

#### Template Quality Scoring

**Overall Quality Score** (0.0 - 1.0)
- Composite score across all template components
- Weighted average of component scores
- Minimum threshold: 0.7 for production use
- Scores above 0.9 indicate excellent quality

**Component Scoring Breakdown**:
- **System Prompt** (30% weight): Role clarity, expertise definition, behavioral guidelines
- **User Prompt Template** (40% weight): Category definitions, instructions, output format
- **Metadata** (30% weight): Completeness, accuracy, validation results

#### Validation Levels

**Basic Validation**
- Structural completeness check
- Required field presence
- Basic format validation

**Standard Validation** (Default)
- Component quality scoring
- Content validation rules
- Evidence alignment checks
- Recommendation generation

**Comprehensive Validation**
- Advanced quality metrics
- Cross-component consistency
- Performance optimization suggestions
- Integration readiness assessment

## Integration Specifications

### Bedrock Processing Step Integration

#### Template Usage Pattern

```python
# Load generated template
with open('/opt/ml/processing/output/templates/prompts.json', 'r') as f:
    template = json.load(f)

# Use in Bedrock processing
system_prompt = template['system_prompt']
user_prompt = template['user_prompt_template'].format(
    input_data=actual_data,
    context_field=context_value
)

# Process with Bedrock
response = bedrock_client.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 1000,
        'system': system_prompt,
        'messages': [{'role': 'user', 'content': user_prompt}]
    })
)
```

#### INPUT_PLACEHOLDERS Integration Pattern

**Template Generation Configuration:**
```python
# Environment variable configuration for Template Generation step
environ_vars = {
    "INPUT_PLACEHOLDERS": '["input_data", "customer_id", "transaction_amount"]',
    # ... other config
}
```

**Generated Template Output:**
```json
{
  "system_prompt": "You are an expert analyst...",
  "user_prompt_template": "Categories and their criteria:\n\n...\n\nAnalysis Instructions:\n\nPlease analyze:\nInput_data: {input_data}\nCustomer_id: {customer_id}\nTransaction_amount: {transaction_amount}\n\n..."
}
```

**Bedrock Processing Integration:**
```python
# Bedrock Processing step automatically extracts placeholders from template
import re

def extract_placeholders_from_template(user_prompt_template: str) -> List[str]:
    """Extract all placeholders from user prompt template."""
    return re.findall(r'\{(\w+)\}', user_prompt_template)

# Load template and extract placeholders
template = load_prompt_template('/opt/ml/processing/input/templates/')
placeholders = extract_placeholders_from_template(template['user_prompt_template'])
# Result: ['input_data', 'customer_id', 'transaction_amount']

# Process DataFrame with automatic column mapping
def process_dataframe_row(row_data: Dict[str, Any]) -> str:
    """Map DataFrame columns to template placeholders automatically."""
    template_vars = {}
    
    for placeholder in placeholders:
        if placeholder in row_data:
            template_vars[placeholder] = row_data[placeholder]
        else:
            logger.warning(f"Missing placeholder '{placeholder}' in row data")
            template_vars[placeholder] = f"[Missing: {placeholder}]"
    
    return template['user_prompt_template'].format(**template_vars)

# Example DataFrame processing
df = pd.DataFrame({
    'input_data': ['Customer complaint about service'],
    'customer_id': ['CUST_12345'],
    'transaction_amount': [99.99]
})

for _, row in df.iterrows():
    formatted_prompt = process_dataframe_row(row.to_dict())
    # Automatically maps DataFrame columns to template placeholders
```

**Key Benefits of INPUT_PLACEHOLDERS Integration:**
- ✅ **Zero Configuration**: No need to specify column names in Processing step
- ✅ **Automatic Mapping**: DataFrame columns automatically mapped to placeholders
- ✅ **Flexible Data Structure**: Works with any DataFrame that has matching column names
- ✅ **Template-Driven**: All placeholder configuration comes from Template Generation
- ✅ **Validation**: Automatic detection and warning for missing placeholders

#### Enhanced Processing Integration (Template-Driven Response Processing)

**Note**: The enhanced validation schema enables template-driven response processing in Bedrock Processing steps, eliminating the need for redundant `BEDROCK_RESPONSE_FORMAT` configuration.

```python
# Load enhanced validation schema with processing metadata
def load_response_configuration(prompt_templates_path: str) -> Dict[str, Any]:
    """Load response configuration from Template Generation outputs."""
    templates_path = Path(prompt_templates_path)
    
    # Load prompts
    prompts_file = templates_path / "prompts.json"
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Load validation schema with processing metadata
    schema_files = list(templates_path.glob("validation_schema_*.json"))
    if not schema_files:
        raise ValueError("No validation schema found in template output")
    
    schema_file = schema_files[0]  # Use most recent
    with open(schema_file, 'r') as f:
        validation_schema = json.load(f)
    
    return {
        'prompts': prompts,
        'validation_schema': validation_schema,
        'processing_config': validation_schema.get('processing_config', {}),
        'template_metadata': validation_schema.get('template_metadata', {})
    }

# Create dynamic Pydantic model from validation schema
def create_pydantic_model_from_schema(schema: Dict[str, Any]) -> BaseModel:
    """Create Pydantic model from clean validation schema (no redundant fields)."""
    from pydantic import create_model, Field
    from typing import Optional, Literal
    
    fields = {}
    properties = schema.get('properties', {})
    required_fields = schema.get('required', [])
    processing_config = schema.get('processing_config', {})
    
    for field_name, field_schema in properties.items():
        # Convert JSON schema type to Python type
        field_type = convert_json_schema_type_to_python(field_schema)
        
        # Create field with validation
        if field_name in required_fields:
            fields[field_name] = (field_type, Field(..., description=field_schema.get('description', '')))
        else:
            fields[field_name] = (Optional[field_type], Field(None, description=field_schema.get('description', '')))
    
    model_name = processing_config.get('response_model_name', 'DynamicResponseModel')
    return create_model(model_name, **fields)

def convert_json_schema_type_to_python(field_schema: Dict[str, Any]) -> type:
    """Convert JSON schema field definition to Python type."""
    field_type = field_schema.get('type', 'string')
    
    if field_type == 'string':
        if 'enum' in field_schema:
            # Create Literal type for enum fields
            from typing import Literal
            return Literal[tuple(field_schema['enum'])]
        return str
    elif field_type == 'number':
        return float
    elif field_type == 'integer':
        return int
    elif field_type == 'boolean':
        return bool
    elif field_type == 'array':
        return list
    else:
        return str  # Default fallback

# Template-driven Bedrock Processing
class BedrockProcessor:
    def __init__(self, config: Dict[str, Any], response_config: Dict[str, Any]):
        self.config = config
        self.response_config = response_config
        
        # Extract processing configuration from validation schema
        processing_config = response_config.get('processing_config', {})
        
        # Create Pydantic model from JSON schema properties (single source of truth)
        validation_schema = response_config['validation_schema']
        self.response_model_class = create_pydantic_model_from_schema(validation_schema)
        
        # Get format type from processing config
        self.response_format = processing_config.get('format_type', 'structured_json')
        
        # Get validation level
        self.validation_level = processing_config.get('validation_level', 'standard')
        
        # Extract field information directly from schema (no redundant fields)
        self.required_fields = validation_schema.get('required', [])
        self.field_properties = validation_schema.get('properties', {})
        
        # Extract category information dynamically (no redundant storage)
        category_property = self.field_properties.get('category', {})
        if 'enum' in category_property:
            self.category_names = category_property['enum']
            self.category_count = len(self.category_names)
        else:
            self.category_names = []
            self.category_count = 0
```

#### Traditional Output Validation (Backward Compatibility)

**Note**: The validation schema can still be used for traditional JSON schema validation of Bedrock responses.

```python
# Load validation schema (generated by template generation step)
with open('/opt/ml/processing/output/schema/validation_schema_*.json', 'r') as f:
    schema = json.load(f)

# Validate Bedrock response (in downstream processing step)
import jsonschema
try:
    # bedrock_response is the JSON output from Bedrock using our generated template
    jsonschema.validate(bedrock_response, schema)
    print("Bedrock response validation passed")
except jsonschema.ValidationError as e:
    print(f"Bedrock response validation failed: {e.message}")
```

### Pipeline Integration Points

#### Input Dependencies
- **Category Definitions**: From data preparation or configuration steps
- **Optional Schema**: From schema generation or customization steps

#### Output Consumers
- **Bedrock Processing Steps**: Primary consumers of generated templates
- **Validation Steps**: Use schemas for output validation
- **Quality Monitoring**: Use metadata for performance tracking
- **Documentation Systems**: Use metadata for template cataloging

## File Naming Conventions

### Template Files
- **Primary Template**: `prompts.json` (fixed name for consistency)
- **Versioned Templates**: `prompts_v{version}.json` (when versioning needed)

### Metadata Files
- **Standard Format**: `template_metadata_{YYYYMMDD_HHMMSS}.json`
- **Example**: `template_metadata_20251102_162000.json`

### Schema Files
- **Standard Format**: `validation_schema_{YYYYMMDD_HHMMSS}.json`
- **Example**: `validation_schema_20251102_162000.json`

### Timestamp Format
- **Pattern**: `YYYYMMDD_HHMMSS`
- **Timezone**: UTC (consistent across environments)
- **Precision**: Second-level granularity

## Output Size and Performance

### File Size Estimates

**Prompt Templates** (`prompts.json`)
- **Typical Size**: 2-10 KB
- **Large Templates**: 10-50 KB (complex categorization)
- **Maximum Expected**: 100 KB (extensive categories with examples)

**Metadata Files**
- **Typical Size**: 5-15 KB
- **With Validation Details**: 15-30 KB
- **Comprehensive Reports**: 30-100 KB

**Schema Files**
- **Typical Size**: 1-5 KB
- **Complex Schemas**: 5-15 KB
- **Maximum Expected**: 25 KB

### Performance Characteristics

**Generation Time**
- **Simple Templates** (1-5 categories): 1-3 seconds
- **Standard Templates** (5-15 categories): 3-10 seconds
- **Complex Templates** (15+ categories): 10-30 seconds

**Memory Usage**
- **Peak Memory**: 50-200 MB during generation
- **Steady State**: 10-50 MB for template storage
- **Validation Overhead**: 5-20 MB additional

## Error Handling and Recovery

### Output Validation Failures

**Missing Required Files**
- **Detection**: Check for `prompts.json` existence
- **Recovery**: Regenerate with default configuration
- **Logging**: Record generation parameters for debugging

**Invalid Template Structure**
- **Detection**: JSON parsing and structure validation
- **Recovery**: Fallback to simplified template generation
- **Notification**: Alert with specific validation errors

**Quality Score Below Threshold**
- **Detection**: Quality score < 0.7
- **Recovery**: Auto-retry with enhanced configuration
- **Escalation**: Manual review for persistent failures

### File System Issues

**Insufficient Disk Space**
- **Detection**: Monitor available space before generation
- **Recovery**: Cleanup temporary files, retry generation
- **Prevention**: Reserve minimum space requirements

**Permission Errors**
- **Detection**: File write permission validation
- **Recovery**: Adjust permissions or use alternative paths
- **Logging**: Record permission issues for system admin

## Security and Compliance

### Data Privacy

**Template Content**
- **No Sensitive Data**: Templates contain only structural information
- **Category Names**: May contain business-sensitive classifications
- **Sanitization**: Remove any PII from category descriptions

**Metadata Security**
- **Generation Config**: May contain internal process information
- **Access Control**: Restrict metadata access to authorized systems
- **Audit Trail**: Log all template access and usage

### Compliance Considerations

**Data Retention**
- **Template Files**: Retain according to business requirements
- **Metadata Files**: Archive for audit and compliance purposes
- **Schema Files**: Maintain for validation consistency

**Version Control**
- **Template Versioning**: Track template changes and evolution
- **Metadata Versioning**: Maintain generation history
- **Schema Versioning**: Ensure validation consistency across versions

## Best Practices

### Template Usage

**Loading Templates**
```python
import json
from pathlib import Path

def load_prompt_template(template_path: str) -> dict:
    """Load and validate prompt template."""
    template_file = Path(template_path) / "prompts.json"
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    with open(template_file, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    # Validate required fields
    required_fields = ['system_prompt', 'user_prompt_template']
    for field in required_fields:
        if field not in template:
            raise ValueError(f"Missing required field: {field}")
    
    return template
```

**Template Formatting**
```python
def format_user_prompt(template: dict, **kwargs) -> str:
    """Format user prompt template with provided data."""
    user_template = template['user_prompt_template']
    
    try:
        formatted_prompt = user_template.format(**kwargs)
        return formatted_prompt
    except KeyError as e:
        raise ValueError(f"Missing required placeholder: {e}")
```

### Quality Monitoring

**Template Quality Checks**
```python
def validate_template_quality(metadata_path: str, min_score: float = 0.7) -> bool:
    """Validate template meets quality requirements."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    quality_score = metadata['validation_results']['quality_score']
    is_valid = metadata['validation_results']['is_valid']
    
    return is_valid and quality_score >= min_score
```

**Performance Monitoring**
```python
def monitor_template_performance(metadata_path: str) -> dict:
    """Extract performance metrics from metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'generation_time': metadata.get('generation_time', 'N/A'),
        'quality_score': metadata['validation_results']['quality_score'],
        'category_count': metadata['category_count'],
        'template_size': metadata.get('template_size_kb', 'N/A')
    }
```

### Error Handling

**Graceful Degradation**
```python
def load_template_with_fallback(primary_path: str, fallback_path: str) -> dict:
    """Load template with fallback option."""
    try:
        return load_prompt_template(primary_path)
    except (FileNotFoundError, ValueError) as e:
        logging.warning(f"Primary template failed: {e}, using fallback")
        return load_prompt_template(fallback_path)
```

**Validation Recovery**
```python
def validate_and_recover(template: dict, schema_path: str) -> dict:
    """Validate template and attempt recovery if needed."""
    try:
        # Validate against schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        jsonschema.validate(template, schema)
        return template
        
    except jsonschema.ValidationError as e:
        logging.error(f"Template validation failed: {e}")
        # Attempt basic recovery
        return recover_template_structure(template)
```

## Troubleshooting Guide

### Common Issues

**Issue: Template File Not Found**
- **Cause**: Generation step failed or output path incorrect
- **Solution**: Check generation logs, verify output directory permissions
- **Prevention**: Implement file existence checks in downstream steps

**Issue: Invalid JSON Structure**
- **Cause**: Template generation interrupted or corrupted
- **Solution**: Regenerate template with validated input
- **Prevention**: Use atomic file writes and validation checks

**Issue: Low Quality Score**
- **Cause**: Insufficient category definitions or validation rules
- **Solution**: Enhance category descriptions, add examples and exceptions
- **Prevention**: Use comprehensive category validation before generation

**Issue: Schema Validation Failures**
- **Cause**: Mismatch between template and expected output format
- **Solution**: Regenerate schema or update template configuration
- **Prevention**: Maintain schema-template consistency checks

### Debugging Tools

**Template Analysis**
```bash
# Check template structure
jq '.' /opt/ml/processing/output/templates/prompts.json

# Validate JSON syntax
python -m json.tool /opt/ml/processing/output/templates/prompts.json

# Check file sizes
ls -lh /opt/ml/processing/output/templates/
ls -lh /opt/ml/processing/output/metadata/
ls -lh /opt/ml/processing/output/schema/
```

**Quality Assessment**
```bash
# Extract quality score
jq '.validation_results.quality_score' /opt/ml/processing/output/metadata/template_metadata_*.json

# Check validation issues
jq '.validation_results.validation_details[].issues' /opt/ml/processing/output/metadata/template_metadata_*.json

# Review recommendations
jq '.validation_results.recommendations' /opt/ml/processing/output/metadata/template_metadata_*.json
```

## Future Enhancements

### Planned Features

**Template Optimization**
- Automatic template performance tuning
- A/B testing framework for template variants
- Dynamic template adaptation based on usage patterns

**Enhanced Validation**
- Semantic validation of category definitions
- Cross-template consistency checking
- Performance prediction modeling

**Integration Improvements**
- Real-time template updates
- Template versioning and rollback
- Integration with MLOps pipelines

### Extensibility Points

**Custom Validators**
- Plugin architecture for domain-specific validation
- Custom quality metrics definition
- Business rule validation integration

**Output Formats**
- Support for additional output formats (XML, YAML)
- Custom schema generation
- Multi-language template support

**Monitoring Integration**
- CloudWatch metrics integration
- Custom dashboard support
- Alert system for quality degradation

## Conclusion

The Bedrock Prompt Template Generation output design provides a comprehensive, scalable, and maintainable solution for automated prompt template creation. The design ensures:

### Key Benefits

✅ **Integration Ready**: Direct compatibility with Bedrock processing steps
✅ **Quality Assured**: Comprehensive validation and quality scoring
✅ **Maintainable**: Clear structure and comprehensive metadata
✅ **Scalable**: Efficient file formats and performance characteristics
✅ **Reliable**: Robust error handling and recovery mechanisms

### Design Principles

- **Consistency**: Standardized file formats and naming conventions
- **Transparency**: Comprehensive metadata and validation reporting
- **Flexibility**: Configurable validation levels and output formats
- **Reliability**: Error handling and graceful degradation
- **Performance**: Optimized file sizes and generation times

This output design enables seamless integration with downstream Bedrock processing while providing the visibility and control needed for production ML pipelines.
