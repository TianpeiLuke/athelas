# Athelas: Zettelkasten-Inspired ML Model Catalog

Athelas is a comprehensive machine learning repository organized according to Zettelkasten knowledge management principles. It provides a unified catalog of ML models, data processing components, and intelligent knowledge management tools designed to facilitate model discovery, comparison, and innovation.

## Overview

Athelas transforms traditional ML repositories into a living knowledge system by implementing a dual-layer architecture:

- **Implementation Notes**: Atomic, reusable ML components (models, processors, utilities)
- **Literature Notes**: Contextual documentation that connects and explains components
- **Intelligent Agents**: Knowledge orchestrator and retriever for automated knowledge management

## Key Features

### 🧠 **Intelligent Knowledge Management**
- **Knowledge Orchestrator**: Automatically maintains connections between components, generates documentation, and validates relationships
- **Knowledge Retriever**: Enables semantic search and discovery through RAG-based interfaces and knowledge graph exploration
- **Dual-Layer Architecture**: Combines executable code with rich contextual documentation

### 🔗 **Explicit Component Connectivity**
- Connection registries document relationships between models and processors
- Cross-reference metadata enables discovery of compatible components
- Knowledge graph visualization of component relationships

### ⚡ **Multi-Framework Support**
- **PyTorch Lightning**: Advanced neural network implementations
- **PyTorch**: Native PyTorch models and components
- **XGBoost & LightGBM**: Gradient boosting models
- **Reinforcement Learning**: Actor-critic and bandit algorithms
- **AWS Bedrock**: Cloud-based model integrations

### 🛠 **Comprehensive Processing Pipeline**
- **Text Processing**: BERT tokenization, Gensim processing, text augmentation
- **Tabular Processing**: Numerical imputation, categorical encoding, feature engineering
- **Image Processing**: Computer vision preprocessing and augmentation
- **Multimodal Processing**: Cross-modal fusion and attention mechanisms

## Architecture

### Repository Structure

```
src/
├── models/                # ML Model Implementations
│   ├── lightning/        # PyTorch Lightning models
│   ├── pytorch/          # Native PyTorch implementations
│   ├── xgboost/          # XGBoost models
│   ├── lightgbm/         # LightGBM models
│   └── actor_critic/     # Reinforcement learning models
├── processing/           # Data Processing Components
│   ├── text/             # Text processing (BERT, Gensim, etc.)
│   ├── tabular/          # Tabular data processing
│   ├── image/            # Image processing
│   ├── feature/          # Feature engineering
│   └── augmentation/     # Data augmentation
├── knowledge/            # Knowledge Management System
│   ├── orchestrator/     # Knowledge orchestration agents
│   ├── retriever/        # Intelligent retrieval system
│   ├── connections/      # Component relationship registries
│   └── demonstrations/   # Usage examples and tutorials
└── utils/                # Shared utilities

slipbox/                  # Knowledge Documentation
├── models/               # Model documentation and analysis
├── processing/           # Processing component documentation
└── knowledge/            # Knowledge system documentation
```

### Core Design Principles

1. **Atomicity**: Each component focuses on a single, well-defined responsibility
2. **Explicit Connectivity**: Relationships between components are explicitly documented
3. **Emergent Organization**: Structure evolves naturally from content relationships
4. **Knowledge Preservation**: Implementation and context are preserved together

## Installation

```bash
pip install athelas
```

### Development Installation

```bash
git clone https://github.com/TianpeiLuke/athelas.git
cd athelas
pip install -e .
```

## Quick Start

### Basic Model Usage

```python
from athelas.models.lightning import BertClassifier
from athelas.processing.text import BertTokenizeProcessor

# Initialize components
tokenizer = BertTokenizeProcessor()
model = BertClassifier(config={
    'num_classes': 2,
    'learning_rate': 2e-5
})

# Process data
processed_text = tokenizer("Example text for classification")

# Train model
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_dataloader)
```

### Knowledge System Queries

```python
from athelas.knowledge.retriever import KnowledgeRetriever

# Initialize knowledge retriever
retriever = KnowledgeRetriever()

# Semantic search for components
results = retriever.search("text classification with BERT")

# Explore component relationships
related = retriever.find_related_components("bert_classifier")

# Get recommendations based on context
recommendations = retriever.recommend_components({
    'task': 'text_classification',
    'data_type': 'text',
    'framework': 'lightning'
})
```

### Component Discovery

```python
from athelas.knowledge.orchestrator import KnowledgeOrchestrator

# Initialize orchestrator
orchestrator = KnowledgeOrchestrator()

# Discover compatible processors for a model
compatible = orchestrator.find_compatible_processors("bert_classifier")

# Get alternative models for a task
alternatives = orchestrator.find_alternatives("text_classification")
```

## Available Components

### Models

#### Text Classification
- **BERT Classifier**: Transformer-based classification with Hugging Face integration
- **Text CNN**: Convolutional neural networks for text classification
- **LSTM**: Recurrent neural networks for sequence classification

#### Multimodal Models
- **Multimodal BERT**: Text and tabular data fusion
- **Cross-Attention**: Attention-based multimodal fusion
- **Gate Fusion**: Gated multimodal feature combination
- **Mixture of Experts**: Sparse multimodal processing

#### Traditional ML
- **XGBoost**: Gradient boosting for tabular data
- **LightGBM**: Fast gradient boosting implementation

### Processing Components

#### Text Processing
- **BERT Tokenizer**: Hugging Face BERT tokenization
- **Gensim Processor**: Word2Vec and Doc2Vec processing
- **Text Augmentation**: Data augmentation for text

#### Tabular Processing
- **Numerical Imputation**: Missing value handling
- **Categorical Encoding**: Label and one-hot encoding
- **Feature Engineering**: Automated feature creation

## Knowledge Management

### Intelligent Discovery

Athelas includes intelligent agents that help you discover and connect components:

```python
# Ask natural language questions about the catalog
answer = retriever.ask("What models work best for multimodal classification?")

# Explore the knowledge graph
graph = retriever.get_knowledge_graph()
connections = graph.get_connections("bert_classifier")
```

### Automatic Documentation

The Knowledge Orchestrator automatically:
- Extracts metadata from component implementations
- Maintains connection registries between components
- Generates and updates documentation
- Validates component relationships

## Contributing

Athelas follows Zettelkasten principles for contributions:

1. **Atomic Components**: Create focused, single-purpose implementations
2. **Explicit Connections**: Document relationships with other components
3. **Rich Metadata**: Include structured metadata in component docstrings
4. **Knowledge Documentation**: Provide contextual documentation in the slipbox

### Adding a New Component

```python
"""
---
component_type: model
framework: lightning
task: text_classification
connections:
  requires:
    - "processing.text.bert_tokenize_processor.BertTokenizeProcessor"
  alternatives:
    - "models.lightning.text_cnn.TextCNN"
---
"""

class YourModel(LightningModule):
    """Your model implementation with metadata."""
    pass
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Athelas in your research, please cite:

```bibtex
@software{athelas2025,
  title={Athelas: Zettelkasten-Inspired ML Model Catalog},
  author={Xie, Tianpei},
  year={2025},
  url={https://github.com/TianpeiLuke/athelas}
}
```

## Related Projects

- [Zettelkasten Method](https://zettelkasten.de/): Knowledge management methodology
- [PyTorch Lightning](https://lightning.ai/): Framework for professional AI research
- [Hugging Face Transformers](https://huggingface.co/transformers/): State-of-the-art NLP models

---

**Athelas**: *From the Greek word meaning "healing" - helping researchers and practitioners heal the fragmentation in ML model development through
