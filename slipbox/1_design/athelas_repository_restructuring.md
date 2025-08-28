---
tags:
  - design
  - repository_structure
  - knowledge_management
  - zettelkasten
  - neural_networks
  - architecture
keywords:
  - repository organization
  - model catalog
  - neural networks
  - zettelkasten principles
  - package structure
topics:
  - code organization
  - neural network architecture
  - knowledge discovery
  - model management
  - modular design
language: python
date of note: 2025-08-27
---

# Athelas: Zettelkasten-Inspired Repository Restructuring

## Purpose

This document proposes a comprehensive restructuring of the BuyerAbuseNLP repository based on Zettelkasten knowledge management principles. The goal is to create a central catalog for all models with a clean, discoverable architecture that allows models and preprocessing components to be expanded to cover many use cases.

## Current Structure Analysis

The current repository structure follows a traditional functional organization:

```
src/
├── bedrock/          # Bedrock access for prompting
├── lightning/        # PyTorch Lightning implementations
└── processing/       # Base preprocessing structures
```

### Limitations of Current Structure

1. **Rigid Categorization**: The current structure enforces a strict separation that doesn't reflect the natural connections between components
2. **Hidden Relationships**: Related components across different folders (e.g., a model and its corresponding preprocessing) lack explicit connections
3. **Discovery Challenges**: Finding the right model or preprocessing component requires prior knowledge of where it might be located
4. **Expansion Constraints**: Adding new model types or preprocessing approaches requires fitting them into the existing rigid structure

## Zettelkasten Principles Applied

### 1. Atomicity

"Put things that belong together into a single note, give it an ID, but limit its content to that single topic."

**Application to Repository**: Each model, processor, or component should:
- Focus on a single, well-defined responsibility
- Exist as an independent module that can be understood on its own
- Include everything needed to fulfill its purpose (code, tests, documentation)

### 2. Connectivity

"Different techniques can implement the same underlying principle: links vs note sequences."

**Application to Repository**:
- Explicit imports and dependencies between components rather than implicit relationships
- Registry system to catalog and connect related components
- Discovery mechanisms that don't rely on folder location

### 3. Anti-Categories

"Don't use categories - rigid hierarchical structures inhibit organic knowledge growth."

**Application to Repository**:
- Flatter structure with fewer nested directories
- Multi-dimensional organization (tags, metadata) instead of single-path hierarchy
- Allow components to belong to multiple conceptual areas

### 4. Manual Linking Over Search

"Search alone is not enough - manual connections create more meaningful knowledge networks."

**Application to Repository**:
- Explicit registry of connections between models, preprocessors, and other components
- Clear documentation of relationships and compatibility
- Visualization tools for understanding component relationships

### 5. Dual-Form Structure

"Notes have inner and outer forms - metadata and content serve different purposes."

**Application to Repository**:
- Separation of implementation from metadata
- Consistent registration and discovery mechanisms
- Standardized metadata for all components

## Proposed Structure: Athelas

Drawing inspiration from the Lord of the Rings plant known for its healing properties ("kingsfoil"), the Athelas structure represents a healing and restorative approach to code organization - rediscovering forgotten connections and organizing knowledge in a more natural way.

```
src/
├── athelas/                     # Main package namespace
│   ├── __init__.py              # Package initialization
│   ├── catalog/                 # Central discovery and registration system
│   │   ├── __init__.py
│   │   ├── registry.py          # Component registration system
│   │   └── discovery.py         # Component discovery mechanisms
│   ├── models/                  # All model implementations (regardless of framework)
│   │   ├── __init__.py          # Auto-discovery mechanism
│   │   ├── base.py              # Base classes and interfaces
│   │   ├── bert/                # BERT and variants
│   │   ├── cnn/                 # CNN architectures
│   │   ├── lstm/                # LSTM and RNN variants
│   │   ├── multimodal/          # Multi-modal models
│   │   ├── tabular/             # Tabular data models
│   │   └── registry.py          # Model-specific registry
│   ├── preprocessing/           # All preprocessing implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Base processor interface
│   │   ├── text/                # Text preprocessing
│   │   │   ├── bert_tokenize.py
│   │   │   └── gensim_tokenize.py
│   │   ├── tabular/             # Tabular data preprocessing
│   │   │   ├── categorical_label.py
│   │   │   ├── numerical_binning.py
│   │   │   └── numerical_imputation.py
│   │   ├── multimodal/          # Multi-modal preprocessing
│   │   └── registry.py          # Preprocessing registry
│   ├── framework/               # Framework-specific implementations
│   │   ├── __init__.py
│   │   ├── lightning/           # PyTorch Lightning integration
│   │   │   ├── __init__.py
│   │   │   ├── module_factory.py
│   │   │   └── trainer.py
│   │   └── bedrock/             # Bedrock integration
│   │       ├── __init__.py
│   │       └── client.py
│   ├── pipelines/               # Ready-to-use pipelines
│   │   ├── __init__.py
│   │   ├── classification/
│   │   ├── regression/
│   │   └── generation/
│   └── utils/                   # Shared utilities
│       ├── __init__.py
│       ├── config.py
│       ├── metrics.py
│       └── visualization.py
└── scripts/                     # Entry point scripts (unchanged)
```

## Key Components Explained

### 1. Central Catalog (`athelas/catalog/`)

This implements the "Connection Registry Pattern" from Zettelkasten:

```python
# Example registry implementation
class ComponentRegistry:
    def __init__(self):
        self.components = {}
        self.connections = {}
        
    def register(self, component_id, component, metadata=None):
        """Register a component with the system"""
        self.components[component_id] = {
            "component": component,
            "metadata": metadata or {}
        }
        
    def connect(self, source_id, target_id, relationship_type):
        """Create explicit connection between components"""
        if source_id not in self.connections:
            self.connections[source_id] = {}
        
        if relationship_type not in self.connections[source_id]:
            self.connections[source_id][relationship_type] = []
            
        self.connections[source_id][relationship_type].append(target_id)
        
    def discover(self, filters=None):
        """Find components matching specific criteria"""
        # Implementation of multi-faceted discovery
```

### 2. Model Registry System (`athelas/models/registry.py`)

Models would be registered with metadata to enable discovery:

```python
# Example model registration
from athelas.catalog import registry
from athelas.models.bert.classification import BertClassifier

registry.register(
    "bert_classifier",
    BertClassifier,
    metadata={
        "task": "classification",
        "modality": "text",
        "architecture": "transformer",
        "framework": "pytorch",
        "compatible_preprocessors": ["bert_tokenizer"],
        "tags": ["NLP", "classification", "transformer"]
    }
)
```

### 3. Auto-Discovery Mechanism

```python
# Example auto-discovery implementation
def discover_models():
    """Automatically discover and register model implementations"""
    for module_info in pkgutil.iter_modules(athelas.models.__path__):
        if not module_info.ispkg:
            continue
            
        module = importlib.import_module(f"athelas.models.{module_info.name}")
        if hasattr(module, "register_models"):
            module.register_models()
```

### 4. Preprocessing Registry

```python
# Example preprocessing registration
from athelas.catalog import registry
from athelas.preprocessing.text import BertTokenizer

registry.register(
    "bert_tokenizer",
    BertTokenizer,
    metadata={
        "modality": "text",
        "input_type": "raw_text",
        "output_type": "token_ids",
        "compatible_models": ["bert_classifier", "bert_sequence_labeler"],
        "tags": ["NLP", "tokenization", "transformer"]
    }
)
```

### 5. Framework Integration (`athelas/framework/`)

Framework-specific implementations would adapt the core models and preprocessors to specific frameworks:

```python
# Example Lightning module factory
class LightningModuleFactory:
    @staticmethod
    def create_module(model_id, **kwargs):
        """Create a Lightning module for the specified model"""
        model_info = registry.get(model_id)
        model = model_info["component"](**kwargs)
        
        if model_info["metadata"]["task"] == "classification":
            return LightningClassificationModule(model)
        elif model_info["metadata"]["task"] == "regression":
            return LightningRegressionModule(model)
        # etc.
```

## Implementation Strategy

### 1. Phase 1: Central Registry and Discovery System

1. Implement the core registry in `athelas/catalog/`
2. Create base interfaces for models and preprocessors
3. Establish standardized metadata schemas
4. Develop discovery mechanisms

### 2. Phase 2: Model Migration

1. Move existing models from `lightning/` to appropriate locations in `athelas/models/`
2. Update model implementations to follow the new registration pattern
3. Add comprehensive metadata to each model
4. Create compatibility documentation

### 3. Phase 3: Preprocessing Migration

1. Move preprocessing components from `processing/` to `athelas/preprocessing/`
2. Update with registration and metadata
3. Establish explicit connections to compatible models

### 4. Phase 4: Framework Integration

1. Implement framework adapters in `athelas/framework/`
2. Create Lightning-specific adaptations of the core models
3. Add Bedrock integration components

### 5. Phase 5: Pipeline Development

1. Create ready-to-use pipelines combining models, preprocessors, and framework components
2. Develop comprehensive examples
3. Add visualization and discovery tools

## Benefits of the Athelas Structure

### 1. Discovery Without Prior Knowledge

The central registry and metadata system allows users to discover components based on their requirements without needing to know exactly where they're located:

```python
# Finding all text classification models
text_classifiers = athelas.discover(
    modality="text",
    task="classification"
)

# Finding preprocessors compatible with a specific model
compatible_preprocessors = athelas.discover_compatible(
    model_id="bert_classifier",
    component_type="preprocessor"
)
```

### 2. Organic Growth

The structure enables organic growth in multiple directions:

- New model architectures can be added in `models/` without restructuring
- New preprocessing approaches can be added independently
- New frameworks can be integrated without changing core components
- Connections between components evolve naturally as the system grows

### 3. Explicit Relationships

The connection registry makes relationships between components explicit:

```python
# Example of explicit connections
registry.connect(
    "bert_classifier", 
    "bert_tokenizer", 
    "requires"
)

registry.connect(
    "bert_classifier",
    "distilbert_classifier",
    "alternative"
)
```

### 4. Multi-Dimensional Organization

Components can be discovered through multiple paths (task, modality, architecture, etc.), not just through their location in the file system:

```python
# Multiple ways to discover the same component
models = athelas.discover(task="classification")
models = athelas.discover(architecture="transformer")
models = athelas.discover(tags=["NLP"])
```

## Migration Considerations

### 1. Backward Compatibility

To ensure a smooth transition, we should:
- Maintain old import paths temporarily with deprecation warnings
- Create adapter layers where necessary
- Document migration patterns for existing code

### 2. Testing Strategy

- Write comprehensive tests for the new structure before migrating
- Create comparison tests to ensure behavior consistency
- Test discovery and registration mechanisms thoroughly

### 3. Documentation Updates

- Create clear documentation for the new structure
- Provide migration guides for existing code
- Add examples of common patterns

## Conclusion

The proposed Athelas structure applies Zettelkasten principles to create a more flexible, discoverable, and maintainable repository organization. By focusing on explicit connections between components rather than rigid hierarchies, it enables organic growth while maintaining clarity and discoverability.

The structure prioritizes:
- **Atomicity**: Each component has a clear, single responsibility
- **Connectivity**: Explicit connections between related components
- **Discovery**: Multiple paths to find relevant components
- **Flexibility**: Easy addition of new components in any dimension

This approach aligns perfectly with the goal of creating a central catalog for models that can expand to cover many use cases, while maintaining the clarity and organization needed for effective collaboration.

## Next Steps

1. Review and refine this proposal based on team feedback
2. Create a prototype of the core registry system
3. Develop a detailed migration plan
4. Begin implementing Phase 1 components

## Cross-References

- [Zettelkasten Knowledge Management Principles](zettelkasten_knowledge_management_principles.md)
- [Zettelkasten Package Name Proposals](zettelkasten_package_name_proposals.md)
