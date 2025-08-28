# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-27

### Added
- Initial release of Athelas
- Zettelkasten-inspired ML model catalog architecture
- Dual-layer architecture with implementation and literature notes
- PyTorch Lightning model implementations:
  - BERT classifier and base BERT model
  - Text CNN for text classification
  - LSTM for sequence processing
  - Multimodal models (BERT, CNN, Cross-attention, Gate fusion, Mixture of Experts)
  - Tabular autoencoder
- Data processing components:
  - Text processors: BERT tokenization, Gensim processing, BeautifulSoup processing
  - Tabular processors: Numerical imputation, categorical encoding, risk table processing
  - Base processor abstractions with composition support
- Model training utilities and distributed training support
- ONNX export capabilities for model deployment
- Command-line interface with basic commands
- Comprehensive documentation and design principles
- Type hints support (py.typed marker)
- Modular dependency structure with optional extras

### Features
- **Knowledge Management**: Foundation for intelligent knowledge orchestrator and retriever
- **Multi-Framework Support**: Ready for PyTorch, XGBoost, LightGBM, and RL models
- **Atomic Components**: Single-responsibility model and processor implementations
- **Explicit Connectivity**: Structured metadata for component relationships
- **Emergent Organization**: Flexible structure that evolves with content

### Dependencies
- Core: PyTorch, PyTorch Lightning, Transformers, Pydantic, NumPy, Pandas
- Optional extras: AWS (boto3, SageMaker), Boosting (XGBoost, LightGBM), Visualization (Matplotlib, Plotly), Deployment (ONNX), Jupyter, Development tools

### Documentation
- Comprehensive README with architecture overview and usage examples
- Design documents outlining Zettelkasten principles and knowledge layer architecture
- MIT License for open-source distribution

## [Unreleased]

### Planned
- Knowledge Orchestrator implementation for automated component management
- Knowledge Retriever with semantic search and RAG capabilities
- Additional model frameworks (XGBoost, LightGBM, Reinforcement Learning)
- Enhanced CLI with knowledge system integration
- Component discovery and recommendation system
- Knowledge graph visualization
- Extended processing pipeline components
