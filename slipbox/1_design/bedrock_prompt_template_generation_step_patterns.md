---
tags:
  - design
  - step_builders
  - bedrock_steps
  - prompt_generation
  - template_patterns
  - sagemaker
  - llm_processing
  - classification_tasks
keywords:
  - bedrock prompt template generation
  - classification prompt patterns
  - structured prompt templates
  - category definition templates
  - LLM prompt engineering
  - template generation automation
topics:
  - step builder patterns
  - prompt template generation
  - SageMaker LLM processing
  - bedrock integration
  - classification task automation
language: python
date of note: 2025-10-31
updated: 2025-11-07
---

# Bedrock Prompt Template Generation Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock prompt template generation step builder implementations in the cursus framework. The Bedrock prompt template generation step creates **ProcessingStep** instances that generate structured, reusable prompt templates specifically designed for categorization and classification tasks using a **file-based configuration approach**. These templates follow a standardized **5-component architecture pattern** and integrate seamlessly with existing Bedrock processing steps.

## Key Architectural Approach: File-Based Configuration

The real implementation uses a **file-based configuration approach** where category definitions and template settings are provided as **JSON files** in a `prompt_configs` directory, rather than environment variables or programmatic configuration.

**Configuration Files Structure:**
```
prompt_configs/
├── category_definitions.json  (REQUIRED)
├── system_prompt.json         (optional - uses defaults if missing)
├── output_format.json         (optional - uses defaults if missing)
└── instruction.json           (optional - uses defaults if missing)
```

## Integration with Bedrock Ecosystem

Bedrock prompt template generation steps integrate with the broader Bedrock workflow:

1. **Prompt Template Generation Step**: Generates structured prompt templates from category definitions
2. **Tabular Preprocessing Step**: Prepares data in train/val/test splits
3. **Bedrock Processing Step**: Uses generated templates for LLM inference
4. **Bedrock Batch Processing Step**: Optional cost-efficient batch processing

**Integration Flow:**
```
Category Definitions (JSON) → Prompt Template Generation → Prompt Templates (prompts.json)
                                                              ↓
Tabular Data → Tabular Preprocessing → Processed Data → Bedrock Processing → Results
```

## 5-Component Architecture Pattern

The generated prompt templates follow a standardized 5-component structure:

1. **System Prompt**: Role assignment, expertise definition, behavioral guidelines
2. **Category Definitions**: Structured categories with conditions, exceptions, key indicators
3. **Input Placeholders**: Variable placeholders for dynamic content injection (e.g., `{dialogue}`, `{shiptrack}`)
4. **Instructions**: Step-by-step analysis instructions and decision criteria
5. **Output Format**: Structured schema specification with field definitions and validation rules

## Core Implementation Components

### 1. PlaceholderResolver Class

The real implementation includes a sophisticated **PlaceholderResolver** class for dynamic placeholder resolution:

```python
class PlaceholderResolver:
    """
    Resolves placeholders marked with ${} syntax from various data sources.
    Connects category definitions to output format through schema enrichment.
    """
    
    def __init__(self, categories: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None):
        self.categories = categories
        self.schema = schema
        self.placeholder_registry = {}
        self.resolution_status = {}
    
    def resolve_placeholder(self, placeholder: str, field_name: str, 
                          source_hint: Optional[str] = None) -> str:
        """Resolve a placeholder using appropriate strategy."""
        # Extract placeholder name from ${placeholder_name} syntax
        placeholder_name = placeholder.strip("${}")
        
        # Try to resolve using various strategies:
        # 1. Explicit source hint (schema_enum, schema_range, categories)
        # 2. Infer from placeholder name
        # 3. Schema lookup by field name
        
        resolved = self._resolve_by_strategy(placeholder_name, field_name, source_hint)
        return resolved
```

**Placeholder Resolution Strategies:**
- **Schema Enum**: Resolves from validation schema enum values
- **Schema Range**: Resolves numeric ranges from schema min/max
- **Categories**: Directly from category list
- **Generic Schema**: Fallback to schema field descriptions

### 2. Schema Enrichment Pattern

Categories are automatically integrated into the validation schema:

```python
def _enrich_schema_with_categories(self, schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Enrich schema with category enum values from category definitions.
    Creates the connection between category definitions and output format.
    """
    if not schema or not self.categories:
        return schema
    
    enriched_schema = schema.copy()
    
    # Update category field enum if it exists
    if "properties" in enriched_schema and "category" in enriched_schema["properties"]:
        category_names = [cat["name"] for cat in self.categories]
        enriched_schema["properties"]["category"]["enum"] = category_names
    
    return enriched_schema
```

### 3. Tone-Aware System Prompt Generation

System prompts adapt based on tone setting:

```python
def _generate_system_prompt(self) -> str:
    """Generate system prompt with role assignment and tone-appropriate language."""
    system_config = self.config.get("system_prompt_config", DEFAULT_SYSTEM_PROMPT_CONFIG)
    
    tone = system_config.get("tone", "professional")
    tone_adjustments = self._get_tone_adjustments(tone)
    
    # Role assignment with tone-appropriate opener
    system_prompt_parts.append(
        f"{tone_adjustments['opener']} {role_definition} with extensive knowledge..."
    )
    
    return " ".join(system_prompt_parts)

def _get_tone_adjustments(self, tone: str) -> Dict[str, str]:
    """Get tone-appropriate language adjustments."""
    tone_map = {
        "professional": {
            "opener": "You are an",
            "task_connector": "Your task is to",
            "guideline_adverb": "Always"
        },
        "casual": {
            "opener": "Hey! You're a",
            "task_connector": "Your job is to",
            "guideline_adverb": "Make sure to"
        },
        "technical": {
            "opener": "System role: You are a",
            "task_connector": "Core functions include:",
            "guideline_adverb": "Operational guidelines require:"
        },
        "formal": {
            "opener": "You shall function as an",
            "task_connector": "Your responsibilities encompass:",
            "guideline_adverb": "You must consistently"
        }
    }
    
    return tone_map.get(tone.lower(), tone_map["professional"])
```

### 4. Dual Output Format Support

Supports both **structured_json** (default) and **structured_text** formats:

```python
def _generate_output_format_section(self) -> str:
    """Generate output format schema section based on format_type."""
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    format_type = output_config.get("format_type", "structured_json")
    
    if format_type == "structured_text":
        return self._generate_structured_text_output_format_from_config()
    else:
        # Default to JSON schema-based generation
        return self._generate_custom_output_format_from_schema()
```

**Structured Text Format Example:**
```
1. Category: [One of: Positive, Negative, Neutral]

2. Confidence Score: [Number between 0.0 and 1.0]

3. Key Evidence:
   * Message Evidence:
      [sep] [Evidence item 1]
      [sep] [Evidence item 2]
   * Shipping Evidence:
      [sep] [Evidence item 1]
```

### 5. Pydantic Models for Configuration

Strong typing with Pydantic models:

```python
class CategoryDefinition(BaseModel):
    """Pydantic model for a single category definition."""
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    conditions: List[str] = Field(..., min_length=1)
    key_indicators: List[str] = Field(..., min_length=1)
    exceptions: List[str] = Field(default_factory=list)
    examples: Optional[List[str]] = None
    priority: int = Field(default=1, ge=1)
    aliases: Optional[List[str]] = None
    validation_rules: Optional[List[str]] = None

class CategoryDefinitionList(BaseModel):
    """Pydantic model for a list of categories with uniqueness validation."""
    categories: List[CategoryDefinition] = Field(..., min_length=1)
    
    @field_validator("categories")
    @classmethod
    def categories_must_have_unique_names(cls, v):
        """Validate all category names are unique."""
        names = set()
        for i, category in enumerate(v):
            if category.name in names:
                raise ValueError(f'Duplicate category name: "{category.name}"')
            names.add(category.name)
        return v
```

## Configuration-to-Prompt Integration

### Overview of Configuration Sub-Components

The prompt generation process uses **four primary configuration objects** that map directly to the 5-component architecture. Each configuration object is optional with comprehensive defaults, allowing users to customize specific aspects while relying on intelligent defaults for others.

**Configuration Objects:**
1. **SystemPromptConfig** → Component 1: System Prompt
2. **CategoryDefinition** → Component 2: Category Definitions
3. **InstructionConfig** → Component 4: Instructions
4. **OutputFormatConfig** → Component 5: Output Format
5. **INPUT_PLACEHOLDERS** (simple list) → Component 3: Input Placeholders

### 1. SystemPromptConfig → System Prompt Integration

**Purpose**: Defines the AI's role, expertise, responsibilities, and behavioral guidelines.

**Configuration Structure:**
```python
class SystemPromptConfig(BaseModel):
    role_definition: str = "expert analyst"
    expertise_areas: List[str] = ["data analysis", "classification", "pattern recognition"]
    responsibilities: List[str] = ["analyze data accurately", "classify content systematically"]
    behavioral_guidelines: List[str] = ["be precise", "be objective", "be thorough"]
    tone: str = "professional"  # professional, casual, technical, formal
```

**Integration Flow:**
```python
def _generate_system_prompt(self) -> str:
    """Generate system prompt from SystemPromptConfig."""
    system_config = self.config.get("system_prompt_config", DEFAULT_SYSTEM_PROMPT_CONFIG)
    
    # Extract values
    role_definition = system_config.get("role_definition")
    expertise_areas = system_config.get("expertise_areas")
    responsibilities = system_config.get("responsibilities")
    behavioral_guidelines = system_config.get("behavioral_guidelines")
    tone = system_config.get("tone", "professional")
    
    # Get tone-appropriate language
    tone_adjustments = self._get_tone_adjustments(tone)
    
    # Construct system prompt
    parts = []
    parts.append(f"{tone_adjustments['opener']} {role_definition} with extensive knowledge in {', '.join(expertise_areas)}.")
    parts.append(f"{tone_adjustments['task_connector']} {', '.join(responsibilities)}.")
    parts.append(f"{tone_adjustments['guideline_adverb']} {', '.join(behavioral_guidelines)} in your analysis.")
    
    return " ".join(parts)
```

**Generated Output Example:**
```
System Prompt: "You are an expert analyst with extensive knowledge in data analysis, classification, pattern recognition. Your task is to analyze data accurately, classify content systematically, provide clear reasoning. Always be precise, be objective, be thorough, be consistent in your analysis."
```

**Tone Variations:**
- **Professional** (default): "You are an expert analyst... Your task is to... Always be precise..."
- **Casual**: "Hey! You're a expert analyst... Your job is to... Make sure to be precise..."
- **Technical**: "System role: You are a expert analyst... Core functions include:... Operational guidelines require: be precise..."
- **Formal**: "You shall function as an expert analyst... Your responsibilities encompass:... You must consistently be precise..."

### 2. CategoryDefinition → Category Definitions Integration

**Purpose**: Defines categories with conditions, exceptions, key indicators, and examples.

**Configuration Structure:**
```python
class CategoryDefinition(BaseModel):
    name: str  # REQUIRED
    description: str  # REQUIRED
    conditions: List[str]  # REQUIRED - at least 1
    key_indicators: List[str]  # REQUIRED - at least 1
    exceptions: List[str] = []  # optional
    examples: Optional[List[str]] = None  # optional
    priority: int = 1  # optional - for sorting
    aliases: Optional[List[str]] = None  # optional
    validation_rules: Optional[List[str]] = None  # optional
```

**Integration Flow:**
```python
def _generate_category_definitions_section(self) -> str:
    """Generate category definitions from CategoryDefinition list."""
    section_parts = ["Categories and their criteria:"]
    
    for i, category in enumerate(self.categories, 1):
        category_parts = [f"\n{i}. {category['name']}"]
        
        # Description
        category_parts.append(f"    - {category['description']}")
        
        # Key elements/indicators
        if category.get('key_indicators'):
            category_parts.append("    - Key elements:")
            for indicator in category['key_indicators']:
                category_parts.append(f"        * {indicator}")
        
        # Conditions
        if category.get('conditions'):
            category_parts.append("    - Conditions:")
            for condition in category['conditions']:
                category_parts.append(f"        * {condition}")
        
        # Exceptions
        if category.get('exceptions'):
            category_parts.append("    - Must NOT include:")
            for exception in category['exceptions']:
                category_parts.append(f"        * {exception}")
        
        # Examples (if include_examples=true)
        if category.get('examples') and self.config.get('INCLUDE_EXAMPLES') == 'true':
            category_parts.append("    - Examples:")
            for example in category['examples']:
                category_parts.append(f"        * {example}")
        
        section_parts.append('\n'.join(category_parts))
    
    return '\n'.join(section_parts)
```

**Generated Output Example:**
```
Categories and their criteria:

1. TrueDNR
    - Delivered Not Received - Package marked as delivered but buyer claims non-receipt
    - Key elements:
        * delivered but not received
        * tracking shows delivered
        * missing package investigation
    - Conditions:
        * Package marked as delivered (EVENT_301)
        * Buyer claims non-receipt
        * Tracking shows delivery
    - Must NOT include:
        * Buyer received wrong item
        * Package damaged on delivery
    - Examples:
        * Package shows delivered but I never got it
        * Tracking says delivered yesterday but nothing here

2. FalseDNR
    - False Delivered Not Received claim...
```

### 3. INPUT_PLACEHOLDERS → Input Placeholders Integration

**Purpose**: Defines which input fields should be included in the template for dynamic content injection.

**Configuration Structure:**
```python
input_placeholders: List[str] = ["dialogue", "shiptrack", "max_estimated_arrival_date"]
```

**Integration Flow:**
```python
def _generate_input_placeholders_section(self) -> str:
    """Generate input placeholders from list."""
    placeholders = json.loads(self.config.get("INPUT_PLACEHOLDERS", '["input_data"]'))
    
    section_parts = ["Analysis Instructions:", ""]
    section_parts.append("Please analyze:")
    
    for placeholder in placeholders:
        section_parts.append(f"{placeholder.title()}: {{{placeholder}}}")
    
    return '\n'.join(section_parts)
```

**Generated Output Example:**
```
Analysis Instructions:

Please analyze:
Dialogue: {dialogue}
Shiptrack: {shiptrack}
Max_Estimated_Arrival_Date: {max_estimated_arrival_date}
```

These placeholders (`{dialogue}`, `{shiptrack}`, etc.) are later replaced with actual values during Bedrock processing.

### 4. InstructionConfig → Instructions Integration

**Purpose**: Defines which instruction components to include and their format.

**Configuration Structure:**
```python
class InstructionConfig(BaseModel):
    include_analysis_steps: bool = True
    include_decision_criteria: bool = True
    include_reasoning_requirements: bool = True
    step_by_step_format: bool = True  # True=numbered, False=bullets
    include_evidence_validation: bool = True
    
    # Optional: Detailed classification guidelines
    classification_guidelines: Optional[Dict[str, Any]] = None
```

**Integration Flow:**
```python
def _generate_instructions_section(self) -> str:
    """Generate instructions from InstructionConfig."""
    instruction_config = self.config.get("instruction_config", DEFAULT_INSTRUCTION_CONFIG)
    
    instructions = ["Provide your analysis in the following structured format:", ""]
    
    # Analysis steps (if enabled)
    if instruction_config.get("include_analysis_steps", True):
        analysis_steps = [
            "Carefully review all provided data",
            "Identify key patterns and indicators",
            "Match against category criteria",
            "Select the most appropriate category",
            "Validate evidence against conditions and exceptions",
            "Provide confidence assessment and reasoning"
        ]
        
        use_step_by_step = instruction_config.get("step_by_step_format", True)
        if use_step_by_step:
            instructions.extend([f"{i+1}. {step}" for i, step in enumerate(analysis_steps)])
        else:
            instructions.extend([f"- {step}" for step in analysis_steps])
        instructions.append("")
    
    # Decision criteria (if enabled)
    if instruction_config.get("include_decision_criteria", True):
        instructions.extend([
            "Decision Criteria:",
            "- Base decisions on explicit evidence in the data",
            "- Consider all category conditions and exceptions",
            "- Choose the category with the strongest evidence match",
            "- Provide clear reasoning for your classification",
            ""
        ])
    
    # Reasoning requirements (if enabled)
    if instruction_config.get("include_reasoning_requirements", True):
        instructions.extend([
            "Reasoning Requirements:",
            "- Explain WHY the evidence supports the selected category",
            "- Address HOW the evidence aligns with category conditions",
            "- Clarify WHAT makes this category the best match",
            "- Describe WHY other categories were ruled out (if applicable)",
            ""
        ])
    
    # Evidence validation (if enabled)
    if instruction_config.get("include_evidence_validation", True):
        instructions.extend([
            "Key Evidence Validation:",
            "- Evidence MUST align with at least one condition for the selected category",
            "- Evidence MUST NOT match any exceptions listed for the selected category",
            "- Evidence should reference specific content from the input data",
            "- Multiple pieces of supporting evidence strengthen the classification",
            ""
        ])
    
    # Detailed classification guidelines (if provided)
    classification_guidelines = instruction_config.get("classification_guidelines")
    if classification_guidelines:
        guidelines_text = self._generate_classification_guidelines(classification_guidelines)
        if guidelines_text:
            instructions.extend([guidelines_text, ""])
    
    return '\n'.join(instructions)
```

**Generated Output Example (with all components enabled):**
```
Provide your analysis in the following structured format:

1. Carefully review all provided data
2. Identify key patterns and indicators
3. Match against category criteria
4. Select the most appropriate category
5. Validate evidence against conditions and exceptions
6. Provide confidence assessment and reasoning

Decision Criteria:
- Base decisions on explicit evidence in the data
- Consider all category conditions and exceptions
- Choose the category with the strongest evidence match
- Provide clear reasoning for your classification

Reasoning Requirements:
- Explain WHY the evidence supports the selected category
- Address HOW the evidence aligns with category conditions
- Clarify WHAT makes this category the best match
- Describe WHY other categories were ruled out (if applicable)

Key Evidence Validation:
- Evidence MUST align with at least one condition for the selected category
- Evidence MUST NOT match any exceptions listed for the selected category
- Evidence should reference specific content from the input data
- Multiple pieces of supporting evidence strengthen the classification
```

**Classification Guidelines Support:**

The `classification_guidelines` field supports hierarchical structured guidance:

```json
{
  "classification_guidelines": {
    "sections": [
      {
        "title": "## Classification Guidelines",
        "subsections": [
          {
            "title": "### Evidence Requirements",
            "content": [
              "- Must reference specific content from input",
              "- Must align with category conditions",
              "- Must avoid category exceptions"
            ]
          },
          {
            "title": "### Decision Process",
            "content": [
              "- Evaluate all categories systematically",
              "- Prioritize categories by evidence strength"
            ]
          }
        ]
      }
    ]
  }
}
```

This generates structured sections in the prompt with proper hierarchy.

### 5. OutputFormatConfig → Output Format Integration

**Purpose**: Defines the expected output structure, field descriptions, and validation rules.

**Configuration Structure:**
```python
class OutputFormatConfig(BaseModel):
    format_type: str = "structured_json"  # or "structured_text"
    required_fields: List[str] = ["category", "confidence", "key_evidence", "reasoning"]
    field_descriptions: Dict[str, str] = {...}
    json_schema: Optional[Dict[str, Any]] = None  # Custom JSON schema
    validation_requirements: List[str] = [...]
    evidence_validation_rules: List[str] = [...]
    
    # For structured_text format:
    header_text: Optional[str] = None
    structured_text_sections: Optional[List[Dict[str, Any]]] = None
    formatting_rules: Optional[List[str]] = None
    example_output: Optional[Any] = None
```

#### 5a. Structured JSON Format Integration

**Integration Flow:**
```python
def _generate_custom_output_format_from_schema(self) -> str:
    """Generate JSON output format from OutputFormatConfig."""
    schema = self.schema_template  # Enriched with category enums
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    
    format_parts = [
        "## Required Output Format",
        "",
        "**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**",
        ""
    ]
    
    # Check for example_output in config
    example_output = output_config.get("example_output")
    if isinstance(example_output, dict):
        # Use provided example directly
        format_parts.extend([
            json.dumps(example_output, indent=2, ensure_ascii=False),
            ""
        ])
    else:
        # Generate structure from schema
        format_parts.extend(["{"])
        
        properties = schema.get("properties", {})
        required_fields = schema.get("required", list(properties.keys()))
        
        for i, field in enumerate(required_fields):
            field_schema = properties.get(field, {})
            # Generate example value based on field type
            comma = "," if i < len(required_fields) - 1 else ""
            format_parts.append(f'    "{field}": "{example_value}"{comma}')
        
        format_parts.extend(["}", ""])
    
    # Field descriptions
    format_parts.append("Field Descriptions:")
    
    properties = schema.get("properties", {})
    required_fields = schema.get("required", list(properties.keys()))
    
    for field in required_fields:
        field_schema = properties.get(field, {})
        description = field_schema.get("description", f"The {field} value")
        field_type = field_schema.get("type", "string")
        
        # Add type and constraint information
        constraints = []
        if field_type == "number":
            if "minimum" in field_schema:
                constraints.append(f"minimum: {field_schema['minimum']}")
            if "maximum" in field_schema:
                constraints.append(f"maximum: {field_schema['maximum']}")
        elif field_type == "string" and "enum" in field_schema:
            constraints.append(f"must be one of: {', '.join(field_schema['enum'])}")
        
        constraint_text = f" ({', '.join(constraints)})" if constraints else ""
        format_parts.append(f"- **{field}** ({field_type}): {description}{constraint_text}")
    
    # Category validation (if category field exists with enum)
    if "category" in required_fields and properties.get("category", {}).get("enum"):
        category_names = properties["category"]["enum"]
        format_parts.extend([
            "",
            "**Category Validation:**",
            f"- The category field must exactly match one of: {', '.join(category_names)}",
            "- Category names are case-sensitive and must match exactly"
        ])
    
    format_parts.extend([
        "",
        "Do not include any text before or after the JSON object. Only return valid JSON."
    ])
    
    return '\n'.join(format_parts)
```

**Generated Output Example (JSON format with complex nested structures):**

When `json_schema` is provided in output_format.json with nested objects/arrays, the output format reflects that structure:

```
## Required Output Format

**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**

{
  "category": "TrueDNR",
  "confidence_score": 0.92,
  "key_evidence": {
    "message_evidence": [
      "[BUYER]: Package shows delivered but I never got it",
      "[SELLER]: Tracking confirms delivery on Nov 5"
    ],
    "shipping_evidence": [
      "[Event Time]: 2025-02-21T17:40:49Z [Event]: Delivered to customer",
      "No return shipment recorded"
    ],
    "timeline_evidence": [
      "Delivery confirmation on 2025-02-21 17:40",
      "Buyer reports non-receipt starting 2025-02-25"
    ]
  },
  "reasoning": {
    "contradicting_evidence": [],
    "primary_factors": [
      "Tracking shows package was delivered successfully",
      "Buyer explicitly states they did not receive the package"
    ],
    "supporting_evidence": [
      "Buyer requests refund due to missing package",
      "No evidence of wrong/defective item"
    ]
  }
}

Field Descriptions:
- **category** (string): Exactly one category from the predefined list (case-sensitive match required) (must be one of: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, ...)
- **confidence_score** (number): Decimal number between 0.00 and 1.00 indicating classification certainty (minimum: 0.0, maximum: 1.0)
- **key_evidence** (object): Object containing three arrays of evidence from different sources
- **reasoning** (object): Object containing three arrays explaining the classification decision

**Category Validation:**
- The category field must exactly match one of: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, ...
- Category names are case-sensitive and must match exactly

Do not include any text before or after the JSON object. Only return valid JSON.
```

**Note on json_schema Field:**

The `json_schema` field in OutputFormatConfig allows defining **complex nested structures**:

```json
{
  "json_schema": {
    "type": "object",
    "properties": {
      "category": {"type": "string", "enum": []},
      "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
      "key_evidence": {
        "type": "object",
        "properties": {
          "message_evidence": {"type": "array", "items": {"type": "string"}},
          "shipping_evidence": {"type": "array", "items": {"type": "string"}},
          "timeline_evidence": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["message_evidence", "shipping_evidence", "timeline_evidence"]
      },
      "reasoning": {
        "type": "object",
        "properties": {
          "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
          "primary_factors": {"type": "array", "items": {"type": "string"}},
          "supporting_evidence": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["contradicting_evidence", "primary_factors", "supporting_evidence"]
      }
    },
    "required": ["category", "confidence_score", "key_evidence", "reasoning"]
  }
}
```

This allows the template generator to create prompts with structured evidence collection (multiple arrays organized by type) rather than simple string fields, enabling more sophisticated LLM responses.

#### 5b. Structured Text Format Integration

**Integration Flow:**
```python
def _generate_structured_text_output_format_from_config(self) -> str:
    """Generate structured text output format from OutputFormatConfig."""
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    
    format_parts = ["## Required Output Format", ""]
    
    # Header text
    header_text = output_config.get("header_text", 
                                     "**CRITICAL: Follow this exact format for automated parsing**")
    format_parts.append(header_text)
    format_parts.append("")
    
    # Generate sections from config
    structured_text_sections = output_config.get("structured_text_sections", [])
    
    if structured_text_sections:
        format_parts.append("```")
        for section in structured_text_sections:
            section_lines = self._generate_section_from_config(section)
            format_parts.extend(section_lines)
        format_parts.append("```")
        format_parts.append("")
    
    # Field descriptions
    field_descriptions = output_config.get("field_descriptions", {})
    if field_descriptions:
        format_parts.append("**Field Descriptions:**")
        for field, description in field_descriptions.items():
            format_parts.append(f"- **{field}**: {description}")
        format_parts.append("")
    
    # Formatting rules
    formatting_rules = output_config.get("formatting_rules", [])
    if formatting_rules:
        format_parts.append("**Formatting Rules:**")
        for rule in formatting_rules:
            format_parts.append(f"- {rule}")
        format_parts.append("")
    
    # Validation requirements
    validation_requirements = output_config.get("validation_requirements", [])
    if validation_requirements:
        format_parts.append("**Validation Requirements:**")
        for req in validation_requirements:
            format_parts.append(f"- {req}")
        format_parts.append("")
    
    # Evidence validation rules
    evidence_validation_rules = output_config.get("evidence_validation_rules", [])
    if evidence_validation_rules:
        format_parts.append("**Evidence Validation:**")
        for rule in evidence_validation_rules:
            format_parts.append(f"- {rule}")
        format_parts.append("")
    
    return '\n'.join(format_parts)
```

**Generated Output Example (Structured Text format):**
```
## Required Output Format

**CRITICAL: Follow this exact format for automated parsing**

```
1. Category: TrueDNR

2. Confidence Score: 0.95

3. Key Evidence:
   * Message Evidence:
      [sep] Buyer states: "Package shows delivered but I never got it"
      [sep] Tracking confirms delivery at 2:15 PM on Nov 5
   * Shipping Evidence:
      [sep] EVENT_301 (Delivered) recorded in tracking
      [sep] No signature required for delivery
   * Timeline Evidence:
      [sep] Buyer contacted us 3 hours after delivery timestamp
```

**Field Descriptions:**
- **Category**: The classified category name (must match exactly one of the defined categories)
- **Confidence Score**: Number between 0.0 and 1.0 indicating classification certainty
- **Key Evidence**: Structured evidence organized by type (message, shipping, timeline)

**Formatting Rules:**
- Use [sep] prefix for each evidence item
- Maintain three-space indentation for subsections
- Include exact quotes from input data when available

**Evidence Validation:**
- Evidence MUST align with at least one condition for the selected category
- Evidence MUST NOT match any exceptions listed for the selected category
```

### Complete Integration Example

**Input Configuration Files:**

1. **category_definitions.json**:
```json
[{"name": "TrueDNR", "description": "...", "conditions": [...], "key_indicators": [...]}]
```

2. **system_prompt.json**:
```json
{"role_definition": "expert analyst", "expertise_areas": [...], "tone": "professional"}
```

3. **instruction.json**:
```json
{"include_analysis_steps": true, "include_decision_criteria": true, "step_by_step_format": true}
```

4. **output_format.json**:
```json
{"format_type": "structured_json", "required_fields": ["category", "confidence", "key_evidence", "reasoning"]}
```

**Integration Process:**
```
1. Load all JSON configs → Parsed config objects
2. SystemPromptConfig → Generate System Prompt (Component 1)
3. CategoryDefinition list → Generate Category Definitions (Component 2)
4. INPUT_PLACEHOLDERS → Generate Input Placeholders (Component 3)
5. InstructionConfig → Generate Instructions (Component 4)
6. OutputFormatConfig → Generate Output Format (Component 5)
7. Combine all components → Complete user_prompt_template
8. Validate template → Quality score and recommendations
9. Save outputs → prompts.json, metadata, validation schema
```

**Final Generated Template (prompts.json):**
```json
{
  "system_prompt": "You are an expert analyst with extensive knowledge in data analysis, classification, pattern recognition. Your task is to analyze data accurately, classify content systematically, provide clear reasoning. Always be precise, be objective, be thorough, be consistent in your analysis.",
  
  "user_prompt_template": "Categories and their criteria:\n\n1. TrueDNR\n    - Delivered Not Received...\n\nAnalysis Instructions:\n\nPlease analyze:\nDialogue: {dialogue}\nShiptrack: {shiptrack}\n\nProvide your analysis in the following structured format:\n\n1. Carefully review all provided data\n2. Identify key patterns and indicators...\n\n## Required Output Format\n\n**CRITICAL: You must respond with a valid JSON object...",
  
  "input_placeholders": ["dialogue", "shiptrack", "max_estimated_arrival_date"]
}
```

This complete integration ensures that every aspect of the configuration flows into specific parts of the generated prompt template, creating a coherent and effective prompt for LLM classification tasks.

## Configuration Structure

### Three-Tier Configuration Design

The config follows a three-tier pattern:

**Tier 1: Essential User Inputs (Required)**
- `input_placeholders`: List of input field names (e.g., `["dialogue", "shiptrack"]`)

**Tier 2: System Inputs with Defaults (Optional)**
- `prompt_configs_path`: Path to configuration directory (default: `"prompt_configs"`)
- `template_task_type`: Task type (default: `"classification"`)
- `template_style`: Template style (default: `"structured"`)
- `validation_level`: Validation level (default: `"standard"`)
- Typed configuration objects: `system_prompt_settings`, `output_format_settings`, `instruction_settings`
- `category_definitions`: List of CategoryDefinition objects

**Tier 3: Derived Fields (Private with Property Access)**
- `effective_system_prompt_config`: Resolved system prompt configuration
- `effective_output_format_config`: Resolved output format configuration
- `effective_instruction_config`: Resolved instruction configuration
- `resolved_prompt_configs_path`: Absolute path to prompt configs directory

### Configuration File Formats

**category_definitions.json** (REQUIRED):
```json
[
  {
    "name": "TrueDNR",
    "description": "Delivered Not Received - Package marked as delivered but buyer claims non-receipt",
    "conditions": [
      "Package marked as delivered (EVENT_301)",
      "Buyer claims non-receipt"
    ],
    "exceptions": [
      "Buyer received wrong item",
      "Package damaged on delivery"
    ],
    "key_indicators": [
      "delivered but not received",
      "tracking shows delivered"
    ],
    "examples": [
      "Package shows delivered but I never got it"
    ],
    "priority": 1
  }
]
```

**system_prompt.json** (optional):
```json
{
  "role_definition": "expert analyst",
  "expertise_areas": ["data analysis", "classification", "pattern recognition"],
  "responsibilities": ["analyze data accurately", "classify content systematically"],
  "behavioral_guidelines": ["be precise", "be objective", "be thorough"],
  "tone": "professional"
}
```

**output_format.json** (optional):
```json
{
  "format_type": "structured_json",
  "required_fields": ["category", "confidence", "key_evidence", "reasoning"],
  "field_descriptions": {
    "category": "The classified category name",
    "confidence": "Confidence score between 0.0 and 1.0",
    "key_evidence": "Specific evidence from input data",
    "reasoning": "Clear explanation of the decision-making process"
  },
  "json_schema": {
    "type": "object",
    "properties": {
      "category": {"type": "string", "enum": []},
      "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
      "key_evidence": {"type": "string"},
      "reasoning": {"type": "string"}
    },
    "required": ["category", "confidence", "key_evidence", "reasoning"]
  },
  "evidence_validation_rules": [
    "Evidence MUST align with at least one condition for the selected category",
    "Evidence MUST NOT match any exceptions listed for the selected category"
  ]
}
```

**instruction.json** (optional):
```json
{
  "include_analysis_steps": true,
  "include_decision_criteria": true,
  "include_reasoning_requirements": true,
  "step_by_step_format": true,
  "include_evidence_validation": true,
  "classification_guidelines": {
    "sections": [
      {
        "title": "## Classification Guidelines",
        "subsections": [
          {
            "title": "### Evidence Requirements",
            "content": [
              "- Must reference specific content from input",
              "- Must align with category conditions"
            ]
          }
        ]
      }
    ]
  }
}
```

## Template Generation Process

### Main Processing Flow

```python
def main(input_paths, output_paths, environ_vars, job_args, logger):
    """Main logic for prompt template generation."""
    
    # 1. Load configurations from JSON files
    prompt_configs_path = input_paths.get("prompt_configs")
    categories = load_category_definitions(prompt_configs_path, log)
    system_prompt_config = load_config_from_json_file(prompt_configs_path, "system_prompt", 
                                                       DEFAULT_SYSTEM_PROMPT_CONFIG, log)
    output_format_config = load_config_from_json_file(prompt_configs_path, "output_format",
                                                       DEFAULT_OUTPUT_FORMAT_CONFIG, log)
    instruction_config = load_config_from_json_file(prompt_configs_path, "instruction",
                                                     DEFAULT_INSTRUCTION_CONFIG, log)
    
    # 2. Generate or load schema template
    schema_template = output_format_config.get("json_schema") or _generate_default_schema(categories)
    
    # 3. Initialize template generator with schema
    generator = PromptTemplateGenerator(config, schema_template)
    
    # 4. Generate template
    template = generator.generate_template()
    
    # 5. Validate template
    validator = TemplateValidator(config["VALIDATION_LEVEL"])
    validation_results = validator.validate_template(template)
    
    # 6. Save outputs
    # - prompts.json (main template file)
    # - template_metadata_{timestamp}.json
    # - validation_schema_{timestamp}.json (if generate_validation_schema=true)
```

### Output Structure

**prompts.json** (main output):
```json
{
  "system_prompt": "You are an expert analyst with extensive knowledge in...",
  "user_prompt_template": "Categories and their criteria:\n\n1. Category1...",
  "input_placeholders": ["dialogue", "shiptrack", "max_estimated_arrival_date"]
}
```

**template_metadata_{timestamp}.json**:
```json
{
  "template_version": "1.0",
  "generation_timestamp": "2025-11-07T11:00:00",
  "task_type": "classification",
  "template_style": "structured",
  "category_count": 13,
  "category_names": ["TrueDNR", "FalseDNR", ...],
  "output_format": "structured_json",
  "validation_results": {
    "is_valid": true,
    "quality_score": 0.95
  },
  "placeholder_validation": {
    "all_resolved": true,
    "successful": 4
  }
}
```

**validation_schema_{timestamp}.json**:
```json
{
  "title": "Bedrock Response Validation Schema",
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["TrueDNR", "FalseDNR", ...],
      "description": "The classified category name"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "required": ["category", "confidence", "key_evidence", "reasoning"],
  "processing_config": {
    "format_type": "structured_json",
    "response_model_name": "ClassificationResponse",
    "validation_level": "standard"
  }
}
```

## Template Validation

The TemplateValidator performs component-specific quality checks:

```python
class TemplateValidator:
    """Validates generated prompt templates for quality and completeness."""
    
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template and return validation results."""
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "validation_details": [],
            "recommendations": []
        }
        
        # Validate system prompt (checks for role, expertise, task context, guidelines)
        system_validation = self._validate_system_prompt(template.get("system_prompt", ""))
        
        # Validate user prompt template (checks for categories, placeholders, instructions, format)
        user_validation = self._validate_user_prompt_template(template.get("user_prompt_template", ""))
        
        # Validate metadata (checks for required fields)
        metadata_validation = self._validate_metadata(template.get("metadata", {}))
        
        # Calculate overall quality score
        scores = [v["score"] for v in validation_results["validation_details"]]
        validation_results["quality_score"] = sum(scores) / len(scores)
        
        # Generate recommendations for improvements
        validation_results["recommendations"] = self._generate_recommendations(...)
        
        return validation_results
```

**Quality Score Thresholds:**
- **≥ 0.8**: Production ready
- **0.7 - 0.79**: Acceptable with minor improvements
- **< 0.7**: Requires improvement

## Step Builder Implementation

### Standard Builder Pattern

```python
class BedrockPromptTemplateGenerationStepBuilder(StepBuilderBase):
    """Builder for Bedrock Prompt Template Generation Step."""
    
    def __init__(self, config: BedrockPromptTemplateGenerationConfig, ...):
        # Load specification
        spec = BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC
        
        super().__init__(config=config, spec=spec, ...)
        self.config = config
    
    def validate_configuration(self) -> None:
        """Validate prompt template generation configuration."""
        # Validate base processing configuration
        # Validate template generation specific configuration
        # Validate category definitions
        # Validate task type, template style, output format
    
    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearnProcessor for template generation."""
        return SKLearnProcessor(
            framework_version="1.2-1",
            role=self.role,
            instance_type=self._get_instance_type(),
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables()
        )
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Create ProcessingInput objects for template generation."""
        # Input: prompt_configs (required directory with JSON config files)
        return [
            ProcessingInput(
                input_name="prompt_configs",
                source=inputs["prompt_configs"],
                destination="/opt/ml/processing/input/prompt_configs"
            )
        ]
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Create ProcessingOutput objects for generated templates."""
        return [
            ProcessingOutput(
                output_name="prompt_templates",
                source="/opt/ml/processing/output/templates",
                destination=outputs.get("prompt_templates")
            ),
            ProcessingOutput(
                output_name="template_metadata",
                source="/opt/ml/processing/output/metadata",
                destination=outputs.get("template_metadata")
            ),
            ProcessingOutput(
                output_name="validation_schema",
                source="/opt/ml/processing/output/schema",
                destination=outputs.get("validation_schema")
            )
        ]
```

### Environment Variables Pattern

Only configuration settings (NOT large JSON data) are passed via environment variables:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Build environment variables for the processing step."""
    return {
        "TEMPLATE_TASK_TYPE": self.config.template_task_type,
        "TEMPLATE_STYLE": self.config.template_style,
        "VALIDATION_LEVEL": self.config.validation_level,
        "INPUT_PLACEHOLDERS": json.dumps(self.config.input_placeholders),
        "INCLUDE_EXAMPLES": str(self.config.include_examples).lower(),
        "GENERATE_VALIDATION_SCHEMA": str(self.config.generate_validation_schema).lower(),
        "TEMPLATE_VERSION": self.config.template_version
    }
```

## Integration with Bedrock Processing Steps

### Seamless Integration

The generated templates integrate directly with Bedrock processing steps:

```python
# Step 1: Generate prompt templates
template_generation_step = BedrockPromptTemplateGenerationStepBuilder(
    config=template_config
).create_step(
    inputs={
        'prompt_configs': 's3://bucket/prompt_configs/'  # Directory with JSON files
    },
    outputs={
        'prompt_templates': 's3://bucket/templates/'
    }
)

# Step 2: Use generated templates in Bedrock processing
bedrock_processing_step = BedrockProcessingStepBuilder(
    config=bedrock_config
).create_step(
    inputs={
        'input_data': input_data_s3_uri,
        'prompt_config': template_generation_step.properties
                         .ProcessingOutputConfig
                         .Outputs['prompt_templates']
                         .S3Output.S3Uri
    },
    outputs={
        'processed_data': 's3://bucket/results/'
    },
    dependencies=[template_generation_step]
)
```

## Key Design Principles

1. **File-Based Configuration**: Large configurations stored as JSON files, not environment variables
2. **5-Component Architecture**: Consistent template structure for optimal LLM performance
3. **Schema Enrichment**: Automatic category enum integration into validation schemas
4. **Placeholder Resolution**: Dynamic ${} placeholder resolution from multiple sources
5. **Type Safety**: Pydantic models for strong typing and validation
6. **Flexible Output Formats**: Support for both structured_json and structured_text
7. **Quality Validation**: Component-specific quality scoring with recommendations
8. **Tone Awareness**: System prompts adapt to specified communication tone
9. **Evidence Validation**: Built-in rules for validating classification evidence
10. **Integration Ready**: Direct compatibility with Bedrock processing steps

## Summary

The Bedrock Prompt Template Generation step provides a production-ready implementation that:

1. **Uses file-based configuration** for maintainability and clarity
2. **Generates structured templates** following 5-component architecture
3. **Enriches schemas** with category enums automatically
4. **Resolves placeholders** dynamically using sophisticated resolution strategies
5. **Validates templates** with component-specific quality scoring
6. **Supports multiple formats** (structured_json and structured_text)
7. **Integrates seamlessly** with Bedrock processing steps via property references

This design ensures that prompt template generation is reliable, maintainable, and optimized for LLM classification tasks while maintaining full compatibility with the cursus framework patterns.
