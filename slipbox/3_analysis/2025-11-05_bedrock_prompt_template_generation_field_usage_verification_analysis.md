---
tags:
  - analysis
  - bedrock_steps
  - configuration_verification
  - field_usage
  - code_quality
keywords:
  - field usage verification
  - config loading
  - prompt generation
  - unused fields
  - code analysis
topics:
  - configuration field analysis
  - script verification
  - technical debt identification
language: python
date of note: 2025-11-05
---

# Bedrock Prompt Template Generation: Field Usage Verification Analysis

## Purpose

Comprehensive verification that all fields from Pydantic config models (SystemPromptConfig, OutputFormatConfig, InstructionConfig) are properly loaded from JSON files and used correctly in prompt generation.

## Analysis Scope

- Config model field definitions
- JSON loading mechanism
- Field usage in prompt generation script
- Identification of unused fields

## SystemPromptConfig Fields

| Field | Defined in Model | Loaded from JSON | Used in Script | Location |
|-------|-----------------|------------------|----------------|----------|
| role_definition | ✅ | ✅ | ✅ | `_generate_system_prompt()` line 238 |
| expertise_areas | ✅ | ✅ | ✅ | `_generate_system_prompt()` line 239 |
| responsibilities | ✅ | ✅ | ✅ | `_generate_system_prompt()` line 240 |
| behavioral_guidelines | ✅ | ✅ | ✅ | `_generate_system_prompt()` line 241 |
| tone | ✅ | ✅ | ❌ NOT USED | - |
| include_expertise_statement | ✅ | ✅ | ❌ NOT USED | - |
| include_task_context | ✅ | ✅ | ❌ NOT USED | - |

### SystemPromptConfig Usage Analysis

**Fields Used Correctly (4/7):**
```python
def _generate_system_prompt(self):
    system_config = self.config.get("system_prompt_config", DEFAULT_SYSTEM_PROMPT_CONFIG)
    
    role_definition = system_config.get("role_definition")  # ✅ Used
    expertise_areas = system_config.get("expertise_areas")  # ✅ Used
    responsibilities = system_config.get("responsibilities")  # ✅ Used
    behavioral_guidelines = system_config.get("behavioral_guidelines")  # ✅ Used
```

**Fields NOT Used (3/7):**
- `tone`: Loaded but never referenced in prompt generation
- `include_expertise_statement`: Boolean flag ignored
- `include_task_context`: Boolean flag ignored

## OutputFormatConfig Fields

| Field | Defined in Model | Loaded from JSON | Used in Script | Location |
|-------|-----------------|------------------|----------------|----------|
| format_type | ✅ | ✅ | ✅ | `_generate_output_format_section()` line 398 |
| required_fields | ✅ | ✅ | ✅ | Schema generation, line 652 |
| field_descriptions | ✅ | ✅ | ✅ | Multiple locations in output format generation |
| validation_requirements | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 447 |
| include_field_constraints | ✅ | ✅ | ❌ NOT USED | - |
| include_formatting_rules | ✅ | ✅ | ❌ NOT USED | - |
| evidence_validation_rules | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 456 |
| header_text | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 413 |
| structured_text_sections | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 421 |
| formatting_rules | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 439 |
| example_output | ✅ | ✅ | ✅ | `_generate_structured_text_output_format_from_config()` line 467 |

### OutputFormatConfig Usage Analysis

**Fields Used Correctly (9/11):**
Most fields are properly used in the structured text output generation:
```python
def _generate_structured_text_output_format_from_config(self):
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    
    # ✅ All these fields are used
    format_type = output_config.get("format_type")
    header_text = output_config.get("header_text")
    structured_text_sections = output_config.get("structured_text_sections")
    field_descriptions = output_config.get("field_descriptions")
    formatting_rules = output_config.get("formatting_rules")
    validation_requirements = output_config.get("validation_requirements")
    evidence_validation_rules = output_config.get("evidence_validation_rules")
    example_output = output_config.get("example_output")
```

**Fields NOT Used (2/11):**
- `include_field_constraints`: Boolean flag loaded but ignored
- `include_formatting_rules`: Boolean flag loaded but ignored

**Note**: The script always includes field descriptions and formatting rules regardless of these boolean flags.

## InstructionConfig Fields

| Field | Defined in Model | Loaded from JSON | Used in Script | Location |
|-------|-----------------|------------------|----------------|----------|
| include_analysis_steps | ✅ | ✅ | ✅ | `_generate_instructions_section()` line 304 |
| include_decision_criteria | ✅ | ✅ | ✅ | `_generate_instructions_section()` line 317 |
| include_edge_case_handling | ✅ | ✅ | ❌ NOT USED | - |
| include_confidence_guidance | ✅ | ✅ | ❌ NOT USED | - |
| include_reasoning_requirements | ✅ | ✅ | ❌ NOT USED | - |
| step_by_step_format | ✅ | ✅ | ❌ NOT USED | - |
| include_evidence_validation | ✅ | ✅ | ✅ | `_generate_instructions_section()` line 328 |
| classification_guidelines | ✅ | ✅ | ✅ | `_generate_instructions_section()` line 340 |

### InstructionConfig Usage Analysis

**Fields Used Correctly (3/8):**
```python
def _generate_instructions_section(self):
    instruction_config = self.config.get("instruction_config", DEFAULT_INSTRUCTION_CONFIG)
    
    # ✅ These fields are checked and used
    if instruction_config.get("include_analysis_steps", True):
        # Generate analysis steps
    
    if instruction_config.get("include_decision_criteria", True):
        # Generate decision criteria
    
    if instruction_config.get("include_evidence_validation", True):
        # Generate evidence validation rules
    
    # ✅ Critical field: classification_guidelines
    classification_guidelines = instruction_config.get("classification_guidelines")
    if classification_guidelines:
        guidelines_text = self._generate_classification_guidelines(classification_guidelines)
```

**Fields NOT Used (5/8):**
- `include_edge_case_handling`: Boolean flag ignored
- `include_confidence_guidance`: Boolean flag ignored
- `include_reasoning_requirements`: Boolean flag ignored
- `step_by_step_format`: Boolean flag ignored
- **IMPORTANT**: `classification_guidelines` IS used ✅

## JSON Loading Mechanism Verification

### Load Function (Lines 537-556)
```python
def load_config_from_json_file(config_path, config_name, default_config, log):
    """Load configuration from JSON file with fallback to defaults."""
    config_file = Path(config_path) / f"{config_name}.json"
    
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)  # ✅ Loads complete JSON
                log(f"Loaded {config_name} config from {config_file}")
                return {**default_config, **config}  # ✅ Merges with defaults
        except Exception as e:
            log(f"Failed to load {config_name} config from {config_file}: {e}. Using defaults.")
            return default_config
    else:
        log(f"{config_name} config file not found at {config_file}. Using defaults.")
        return default_config
```

**Verification**: ✅ This function loads ALL fields from JSON, including nested structures like `classification_guidelines`

### Loading in main() (Lines 616-628)
```python
# Load configuration files from prompt_configs directory
system_prompt_config = load_config_from_json_file(
    prompt_configs_path, "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, log
)

output_format_config = load_config_from_json_file(
    prompt_configs_path, "output_format", DEFAULT_OUTPUT_FORMAT_CONFIG, log
)

instruction_config = load_config_from_json_file(
    prompt_configs_path, "instruction", DEFAULT_INSTRUCTION_CONFIG, log
)
```

**Verification**: ✅ All three config files are loaded correctly

### Config Assembly (Lines 698-714)
```python
config = {
    "TEMPLATE_TASK_TYPE": environ_vars.get("TEMPLATE_TASK_TYPE", "classification"),
    "TEMPLATE_STYLE": environ_vars.get("TEMPLATE_STYLE", "structured"),
    "VALIDATION_LEVEL": environ_vars.get("VALIDATION_LEVEL", "standard"),
    "category_definitions": json.dumps(categories),
    "system_prompt_config": system_prompt_config,  # ✅ Full config
    "output_format_config": output_format_config,  # ✅ Full config
    "instruction_config": instruction_config,      # ✅ Full config
    ...
}
```

**Verification**: ✅ Complete configs passed to generator

## Critical Finding: classification_guidelines

### Is classification_guidelines Loaded?
**YES** ✅ - Confirmed at multiple levels:

1. **JSON Loading**: `json.load(f)` reads entire JSON structure ✅
2. **Dict Merging**: `{**default_config, **config}` preserves all fields ✅
3. **Config Assembly**: `"instruction_config": instruction_config` includes everything ✅
4. **Field Extraction**: `instruction_config.get("classification_guidelines")` retrieves it ✅
5. **Usage**: `self._generate_classification_guidelines(classification_guidelines)` formats it ✅

### Guidelines Processing (Lines 368-396)
```python
def _generate_classification_guidelines(self, guidelines: Dict[str, Any]) -> str:
    """
    Generate detailed classification guidelines from config structure.
    
    Args:
        guidelines: Dictionary containing sections with hierarchical structure
        
    Returns:
        Formatted guideline text
    """
    guideline_parts = []
    
    sections = guidelines.get("sections", [])  # ✅ Extracts sections
    for section in sections:
        # Add main section title
        section_title = section.get("title", "")
        if section_title:
            guideline_parts.append(section_title)
            guideline_parts.append("")
        
        # Add subsections
        subsections = section.get("subsections", [])
        for subsection in subsections:
            # Add subsection title
            subsection_title = subsection.get("title", "")
            if subsection_title:
                guideline_parts.append(subsection_title)
                guideline_parts.append("")
            
            # Add subsection content
            content = subsection.get("content", [])
            if content:
                guideline_parts.extend(content)  # ✅ Adds all content lines
                guideline_parts.append("")
    
    return "\n".join(guideline_parts)  # ✅ Joins into single string
```

**Verification**: ✅ Complete hierarchical structure is processed and formatted

## Summary Statistics

### Overall Field Usage

| Config Model | Total Fields | Used | Not Used | Usage % |
|--------------|-------------|------|----------|---------|
| SystemPromptConfig | 7 | 4 | 3 | 57% |
| OutputFormatConfig | 11 | 9 | 2 | 82% |
| InstructionConfig | 8 | 3 | 5 | 38% |
| **TOTAL** | **26** | **16** | **10** | **62%** |

### Unused Fields (10 total)

**SystemPromptConfig (3 unused):**
1. `tone` - Loaded but never referenced
2. `include_expertise_statement` - Boolean flag ignored
3. `include_task_context` - Boolean flag ignored

**OutputFormatConfig (2 unused):**
1. `include_field_constraints` - Boolean flag ignored
2. `include_formatting_rules` - Boolean flag ignored

**InstructionConfig (5 unused):**
1. `include_edge_case_handling` - Boolean flag ignored
2. `include_confidence_guidance` - Boolean flag ignored
3. `include_reasoning_requirements` - Boolean flag ignored
4. `step_by_step_format` - Boolean flag ignored
5. ~~classification_guidelines~~ - **IS USED** ✅

## Root Cause Analysis

### Why Are Fields Unused?

**Pattern 1: Boolean Flags Ignored**
Most unused fields are boolean flags that should control section inclusion but are checked and then ignored:

```python
# Pattern: Check returns True, but section is generated anyway
if instruction_config.get("include_analysis_steps", True):
    instructions.extend([...])  # Always happens because default is True
```

**Pattern 2: Missing Implementation**
Some fields exist in the model but have no corresponding logic:
- `tone` has no implementation to adjust language style
- `include_expertise_statement` is never checked

**Pattern 3: Always-On Sections**
Some sections are always generated regardless of flags:
- Field descriptions always included (ignores `include_field_constraints`)
- Formatting rules always added (ignores `include_formatting_rules`)

## Impact Assessment

### Critical Assessment
**All fields ARE loaded from JSON correctly** via `load_config_from_json_file()`. The issue is that many boolean flags are **ignored during prompt generation**.

### Is This a Bug?

**Partially**. Two interpretations:

1. **Design Intent**: Boolean flags exist for future extensibility but aren't implemented yet
   - Allows users to define fields that will be used later
   - Provides forward compatibility

2. **Implementation Gap**: Flags should control section inclusion but logic is incomplete
   - Boolean checks exist but don't actually control output
   - Default values of `True` make sections always appear

### Practical Impact

**Low Impact** because:
- Core functionality works correctly
- All content fields (text, lists, dicts) are used properly
- The ignored flags are mostly "nice-to-have" controls
- Users can achieve desired output by providing or omitting content

**Medium Priority** for improvement:
- Should either implement the boolean logic OR remove the unused flags
- Creates confusion when flags don't work as expected
- Adds unnecessary complexity to config models

## Verification Result

### ✅ classification_guidelines Verification

**CONFIRMED**: `classification_guidelines` field is **FULLY FUNCTIONAL**

1. ✅ Loaded from instruction.json correctly
2. ✅ Passed through config assembly chain
3. ✅ Extracted in `_generate_instructions_section()`
4. ✅ Formatted by `_generate_classification_guidelines()`
5. ✅ Included in generated prompt template

**The field works correctly!** If it's not appearing in output, the issue is:
- File not created (instruction.json doesn't exist)
- Config not passed (instruction_settings was None)
- Wrong directory path

## Recommendations

### Short Term (Quick Wins)

1. **Document Unused Fields**
   - Add comments to Pydantic models indicating which fields are not yet implemented
   - Update docstrings to clarify expected behavior

2. **Test Field Usage**
   - Create unit tests that verify each boolean flag controls output
   - Add integration tests for complex fields like classification_guidelines

### Medium Term (Technical Debt)

1. **Implement Boolean Logic**
   - Make include_* flags actually control section inclusion
   - Add conditional logic for tone and other unused string fields

2. **Remove Unused Fields (Alternative)**
   - If flags won't be implemented, remove them from models
   - Simplify config structure to only include functional fields

### Long Term (Architecture)

1. **Field Usage Validation**
   - Add tooling to detect unused Pydantic fields
   - Create CI checks that fail if new fields are added without corresponding usage

2. **Config Schema Validation**
   - Implement JSON schema validation for config files
   - Provide clear error messages when fields are misspelled or misused

## Field Removal Recommendations

### Should Unused Fields Be Removed?

**Analysis of 10 unused fields:**

#### Recommend REMOVE (6 fields)
These provide no value and should be deleted:

**SystemPromptConfig:**
1. ✂️ `include_expertise_statement` - Always generated, flag does nothing
2. ✂️ `include_task_context` - Always generated, flag does nothing

**OutputFormatConfig:**
3. ✂️ `include_field_constraints` - Always generated, flag does nothing
4. ✂️ `include_formatting_rules` - Always generated, flag does nothing

**InstructionConfig:**
5. ✂️ `include_edge_case_handling` - No corresponding implementation
6. ✂️ `include_confidence_guidance` - No corresponding implementation

**Rationale**: These are non-functional boolean flags that mislead users into thinking they control behavior.

#### Recommend IMPLEMENT (4 fields)
These could provide value if implemented:

**InstructionConfig:**
1. 🔧 `include_reasoning_requirements` - Useful for controlling reasoning section
2. 🔧 `step_by_step_format` - Useful for toggling step-by-step instructions

**SystemPromptConfig:**
3. 🔧 `tone` - Useful for adjusting formality (professional, casual, technical)

**InstructionConfig:**
4. 🔧 `include_analysis_steps` - Already has check but always True, should work properly

**Rationale**: These represent legitimate customization options that would be useful if implemented correctly.

### Implementation Priority

**High Priority (Clean Up):**
- Remove 6 non-functional fields immediately
- Update documentation to reflect actual behavior
- Simplify config models

**Medium Priority (Feature Addition):**
- Implement tone adjustment if there's user demand
- Make boolean flags actually control their sections

**Low Priority:**
- Leave include_analysis_steps as-is (works even if default is True)

### Proposed Clean Config Models

See updated config models in: `src/cursus/steps/configs/config_bedrock_prompt_template_generation_step.py`

## Conclusion

The JSON loading mechanism is **100% correct** and preserves all fields including nested structures. The `classification_guidelines` field specifically is **fully functional** and properly integrated into the prompt generation pipeline.

The main finding is that **10 out of 26 fields** (38%) are loaded but not used, primarily boolean control flags that don't actually control output. **6 of these fields should be removed** as they provide no value and mislead users. **4 fields could be implemented** if there's user demand for those features.

For users wondering why `classification_guidelines` isn't appearing in output, the issue is NOT in the script - it's in the configuration workflow (file creation, path resolution, or config passing).
