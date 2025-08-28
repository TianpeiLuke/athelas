---
tags:
  - design
  - architecture
  - principles
  - software_engineering
  - best_practices
keywords:
  - single source of truth
  - declarative programming
  - type safety
  - explicit design
  - component interfaces
  - configuration management
  - code organization
topics:
  - software design principles
  - system architecture
  - component design
language: python
date of note: 2025-08-27
---

# Core Design Principles for Repository Architecture

## Purpose

This document establishes the foundational design principles that govern the architecture and implementation of our repository. These principles provide a consistent framework for decision-making and ensure that our codebase remains maintainable, robust, and scalable as it evolves. By adhering to these principles, we create a system that is easier to understand, extend, and validate.

## Core Principles

### 1. Single Source of Truth

**Definition**: Centralize validation logic and configuration definitions in their respective component's configuration class to avoid redundancy and conflicts.

**Guiding Maxim**: *"One definition, one place, one truth"*

**Implementation Guidelines**:
- Define component configurations in a single, dedicated class
- Centralize validation logic within configuration classes
- Ensure all instances reference the same configuration source
- Avoid duplicating configuration parameters across multiple files
- Use inheritance or composition to extend configurations, not duplication

**Benefits**:
- Eliminates inconsistencies between different parts of the system
- Makes changes more reliable and predictable
- Simplifies debugging and validation
- Reduces cognitive load when understanding component behavior

**Examples**:

```python
# Single configuration class that serves as the source of truth
class BertModelConfig(BaseModel):
    """Configuration for BERT model with centralized validation."""
    text_name: str
    tokenizer: str = "bert-base-cased"
    is_binary: bool = True
    num_classes: int = 2
    
    @field_validator("num_classes")
    def validate_num_classes(cls, value: int, info: ValidationInfo) -> int:
        if info.data.get("is_binary") and value != 2:
            raise ValueError("For binary classification, num_classes must be 2")
        if not info.data.get("is_binary") and value < 2:
            raise ValueError("For multiclass classification, num_classes must be >= 2")
        return value
```

### 2. Declarative Over Imperative

**Definition**: Favor declarative specifications that describe *what* the component should do rather than *how* to do it. Define dependencies and outputs declaratively rather than through imperative code.

**Guiding Maxim**: *"Describe what you want, not how to get it"*

**Implementation Guidelines**:
- Use configuration objects to define component behavior
- Express relationships between components as declarations
- Define workflows through dependency specifications
- Separate configuration from execution logic
- Use factories and builders to interpret declarative specifications

**Benefits**:
- Makes code more self-documenting
- Easier to reason about component behavior
- Simplifies testing through configuration-based scenarios
- Enables higher-level tools to analyze and manipulate system behavior
- Reduces procedural complexity

**Examples**:

```python
# Imperative approach (avoid)
def create_text_processor(text_column, max_length, tokenizer_name):
    processor = BertTokenizer(tokenizer_name)
    processor.max_length = max_length
    processor.column_name = text_column
    return processor

# Declarative approach (preferred)
text_processor_config = {
    "type": "bert_tokenizer",
    "parameters": {
        "text_column": "description",
        "max_length": 128,
        "tokenizer_name": "bert-base-uncased"
    }
}
processor = ProcessorFactory.create(text_processor_config)
```

### 3. Type-Safe Specifications

**Definition**: Use strongly-typed enums and data structures to prevent configuration errors at definition time rather than discovering them at runtime.

**Guiding Maxim**: *"Catch errors at design time, not runtime"*

**Implementation Guidelines**:
- Use enums for categorical values instead of strings
- Define Pydantic models or dataclasses for configurations
- Add validation logic to configuration classes
- Use type annotations consistently
- Implement runtime type checking for external inputs

**Benefits**:
- Prevents entire classes of runtime errors
- Enables better IDE support and autocompletion
- Makes refactoring safer and more reliable
- Improves documentation through explicit type definitions
- Simplifies debugging by failing fast on invalid configurations

**Examples**:

```python
from enum import Enum, auto
from pydantic import BaseModel, Field

class ModelFramework(Enum):
    PYTORCH = auto()
    LIGHTNING = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()

class ModelType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTICLASS = auto()

class ModelConfig(BaseModel):
    """Type-safe model configuration."""
    framework: ModelFramework
    model_type: ModelType
    hidden_dims: list[int] = Field(..., description="List of hidden dimensions")
    dropout_rate: float = Field(0.1, ge=0.0, le=0.9)
```

### 4. Explicit Over Implicit

**Definition**: Favor explicitly defining connections and passing parameters between components over relying on implicit, automatic matching based on naming conventions.

**Guiding Maxim**: *"Make it obvious, not clever"*

**Implementation Guidelines**:
- Use explicit import statements instead of wildcard imports
- Explicitly declare component dependencies
- Use descriptive parameter and variable names
- Avoid magic methods and hidden behavior
- Document expected inputs and outputs clearly
- Make side effects visible and documented

**Benefits**:
- Improves code readability and understandability
- Reduces subtle bugs from implicit behavior
- Makes code more maintainable by new team members
- Simplifies tracing of data flow through the system
- Reduces "action at a distance" problems

**Examples**:

```python
# Implicit approach (avoid)
from .processors import *  # Implicit imports

def process_data(data):
    # Implicitly depends on specific column names
    return tokenize(data['text'])

# Explicit approach (preferred)
from src.processing.text.bert_tokenize_processor import BertTokenizeProcessor

def process_data(data, text_column: str, tokenizer: BertTokenizeProcessor):
    # Explicitly defines dependencies and parameters
    return tokenizer.process(data[text_column])
```

## Integration with Zettelkasten Principles

These core design principles align naturally with Zettelkasten knowledge management principles:

1. **Single Source of Truth** aligns with the Zettelkasten principle of atomicity, where each note contains one coherent concept.

2. **Declarative Over Imperative** supports the Zettelkasten emphasis on clear, self-contained knowledge units that declare their content rather than procedural instructions.

3. **Type-Safe Specifications** provide structure to Zettelkasten connections, ensuring that relationships between notes are well-defined and validated.

4. **Explicit Over Implicit** reinforces the Zettelkasten principle of connectivity, where connections between notes are explicitly defined rather than implied.

## Implementation in Component Design

When designing components following these principles:

### Configuration Classes

```python
class TextProcessorConfig(BaseModel):
    """Single source of truth for text processor configuration."""
    processor_type: Literal["bert", "gensim", "custom"] = "bert"
    text_column: str
    max_length: int = 128
    tokenizer: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_tokenizer(self):
        if self.processor_type == "bert" and not self.tokenizer:
            self.tokenizer = "bert-base-uncased"
        return self
```

### Component Interfaces

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TextProcessor(Protocol):
    """Explicit interface for text processors."""
    def process(self, text: str) -> dict[str, torch.Tensor]: ...
    def get_vocab_size(self) -> int: ...
```

### Component Implementation

```python
class BertTokenizeProcessor:
    """BERT tokenization processor implementation."""
    
    def __init__(self, config: Union[dict, TextProcessorConfig]):
        # Explicitly convert dict to typed config
        if isinstance(config, dict):
            config = TextProcessorConfig(**config)
        self.config = config
        
        # Initialize tokenizer based on config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
    
    def process(self, text: str) -> dict[str, torch.Tensor]:
        """Process text using the BERT tokenizer."""
        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    def get_vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return self.tokenizer.vocab_size
```

## Benefits in System Architecture

Applying these principles throughout the system architecture yields significant benefits:

1. **Reduced Cognitive Load**: Developers can focus on understanding individual components without needing to understand the entire system.

2. **Improved Maintainability**: Clear boundaries and explicit dependencies make it easier to modify components without unintended side effects.

3. **Enhanced Testability**: Declarative specifications and explicit interfaces facilitate isolated testing of components.

4. **Better Documentation**: Type-safe specifications and explicit connections serve as self-documenting code.

5. **Safer Refactoring**: Strong typing and explicit dependencies reduce the risk of breaking changes during refactoring.

## Conclusion

These core design principles provide a solid foundation for our repository architecture. By adhering to Single Source of Truth, Declarative Over Imperative, Type-Safe Specifications, and Explicit Over Implicit, we create a system that is more maintainable, robust, and comprehensible. These principles work in harmony with Zettelkasten knowledge management approaches to create a codebase that not only functions effectively but also serves as a knowledge repository for the team.

As we evolve the codebase, these principles should guide our decision-making process, ensuring that new components and features maintain the architectural integrity of the system.

## Related Concepts

- **Domain-Driven Design**: Emphasizes explicit boundaries and well-defined models
- **Functional Programming**: Favors declarative approaches and explicit data flow
- **Contract-First Design**: Focuses on defining clear interfaces before implementations
- **Configuration as Code**: Treats configuration as a first-class part of the codebase

## Cross-References

- [Zettelkasten Knowledge Management Principles](zettelkasten_knowledge_management_principles.md)
- [Zettelkasten Repository Structure Design](zettelkasten_repository_structure_design.md)
- [2025-08-27 Repository Restructuring Plan](../2_project_planning/2025-08-27_zettelkasten_repository_restructuring_plan.md)
