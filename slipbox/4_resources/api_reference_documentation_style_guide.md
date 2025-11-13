---
tags:
  - resource
  - documentation
  - style-guide
  - api-reference
keywords:
  - documentation style
  - API reference
  - SageMaker style
  - YAML frontmatter
  - code documentation
topics:
  - documentation standards
  - style guide
  - API reference
language: python
date of note: 2024-12-07
---

# API Reference Documentation Style Guide

Standardized format for creating comprehensive API reference documentation for the cursus project.

## Overview

This document defines the specific style and structure for API reference documentation in the cursus project. The style is inspired by SageMaker's documentation format but adapted for our specific needs and context.

## Documentation Structure

Each API reference document must follow this exact structure:

### 1. YAML Header
- Must follow the `documentation_yaml_frontmatter_standard` format
- Required fields: `tags`, `keywords`, `topics`, `language`, `date`
- Tags should be specific and relevant to the module's functionality
- Keywords should include class names, key concepts, and technical terms
- Topics should represent high-level categories
- Language should be "Python" for code modules
- Date should be in YYYY-MM-DD format

### 2. Overview Section
- Brief description of the module's purpose and functionality
- Context about how it fits into the larger system
- Key concepts and principles implemented
- Should be 2-3 paragraphs maximum

### 3. Classes and Methods List Section
- Organized list of all classes and functions in the module
- Grouped by type (Classes, Functions, Constants, etc.)
- Each item should link to its detailed API reference section
- Use bullet points with descriptive text

### 4. API Reference Section
- Detailed documentation for each class and function
- Follow SageMaker documentation style format
- Structure for each class:
  - Class declaration with full module path
  - Purpose and functionality description
  - Constructor parameters with types and descriptions
  - Methods with signatures, parameters, return types
  - Usage examples with code snippets
  - Properties and attributes where applicable

### 5. Related Documentation Section
- Links to other documentation files that this module imports from or relates to
- Cross-references to design documents
- Links to usage examples or tutorials

## API Reference Format Specification

This section defines the exact format for API reference documentation, following the SageMaker documentation style from https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#steps

### Format Requirements

Each class or function must follow this exact sequence:

1. **Definition Line**: Class or function signature with `_class_` or method name
2. **Purpose Description**: Brief explanation of what the class/method does
3. **Parameters Section**: Detailed list of all input parameters with types and descriptions
4. **Returns Section** (for methods): Description of return value and type
5. **Usage Example**: Code snippet showing typical usage

### Class Documentation Format

```markdown
### ClassName

_class_ module.path.ClassName(_param1_, _param2=default_)

Brief description of the class purpose and main functionality.

**Parameters:**
- **param1** (_Type_) – Description of the parameter, its purpose, and any constraints.
- **param2** (_Type_) – Description with default value explanation.

```python
# Example showing typical usage
instance = ClassName(param1_value)
result = instance.method()
```

#### Methods

##### method_name

method_name(_param1_, _param2=None_)

Description of what the method does.

**Parameters:**
- **param1** (_Type_) – Parameter description.
- **param2** (_Optional[Type]_) – Optional parameter description.

**Returns:**
- **Type** – Description of return value.

```python
result = instance.method_name("value")
```

### Function Documentation Format

```markdown
### function_name

function_name(_param1_, _param2=None_)

Description of the function's purpose and behavior.

**Parameters:**
- **param1** (_Type_) – Parameter description.
- **param2** (_Optional[Type]_) – Optional parameter description.

**Returns:**
- **Type** – Description of return value.

```python
result = function_name("input")
```

### Key Format Rules

1. **No "Usage Example:" headers** - Code examples follow directly after parameters/returns
2. **Consistent parameter formatting** - Always use `**param_name** (_Type_) – Description`
3. **Returns section** - Always include for methods that return values
4. **Code blocks** - Use triple backticks with `python` language specification
5. **Brief descriptions** - Keep class/method descriptions concise and focused on functionality
6. **Type annotations** - Always include type information in italics within parentheses

## Style Guidelines

### Writing Style
- Use clear, concise language
- Write in present tense
- Use active voice where possible
- Be specific about types and constraints
- Include practical examples

### Code Examples
- All code examples must be syntactically correct
- Show realistic usage scenarios
- Include imports when necessary
- Use meaningful variable names
- Keep examples concise but complete

### Cross-References
- Use relative links for internal documentation
- Link to related classes and methods
- Reference design documents where applicable
- Include links to external documentation when relevant

### Type Annotations
- Always include type information for parameters and return values
- Use Python typing conventions
- Indicate optional parameters clearly
- Show union types when applicable

## Template Structure

```markdown
---
tags:
  - code
  - config_fields
  - module_name
  - functionality
keywords:
  - ClassName
  - key concepts
  - technical terms
topics:
  - configuration management
  - module functionality
language: python
date of note: YYYY-MM-DD
---

# Module Title

Brief module description.

## Overview

2-3 paragraphs describing the module's purpose, functionality, and context.

## Classes and Methods

### Classes
- [`ClassName`](#classname) - Brief description

### Functions
- [`function_name`](#function_name) - Brief description

## API Reference

### ClassName

_class_ module.path.ClassName(_param1_, _param2=default_)

Brief description of the class purpose and main functionality.

**Parameters:**
- **param1** (_Type_) – Description of the parameter, its purpose, and any constraints.
- **param2** (_Type_) – Description with default value explanation.

```python
# Example showing typical usage
instance = ClassName(param1_value)
result = instance.method()
```

#### method_name

method_name(_param1_, _param2=None_)

Description of what the method does.

**Parameters:**
- **param1** (_Type_) – Parameter description.
- **param2** (_Optional[Type]_) – Optional parameter description.

**Returns:**
- **Type** – Description of return value.

```python
result = instance.method_name("value")
```

### function_name

function_name(_param1_, _param2=None_)

Description of the function's purpose and behavior.

**Parameters:**
- **param1** (_Type_) – Parameter description.
- **param2** (_Optional[Type]_) – Optional parameter description.

**Returns:**
- **Type** – Description of return value.

```python
result = function_name("input")
```

## Related Documentation

- [Related Module](../path/to/related.md) - Description of relationship
- [Design Document](../../1_design/related_design.md) - Design context
```

## Quality Standards

### Completeness
- Every public class and function must be documented
- All parameters must have type annotations and descriptions
- Return values must be documented
- Usage examples must be provided

### Accuracy
- Code examples must be tested and working
- Type annotations must be correct
- Parameter descriptions must match actual behavior
- Links must be valid and current

### Consistency
- Follow the exact structure outlined above
- Use consistent terminology throughout
- Maintain consistent formatting and style
- Apply the same level of detail across all documentation

## Maintenance

This style guide should be updated when:
- New documentation patterns emerge
- Feedback indicates improvements needed
- The underlying codebase structure changes significantly
- New tools or formats become available

All API reference documentation should be reviewed against this guide during creation and updates.
