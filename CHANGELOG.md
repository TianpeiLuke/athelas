# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Knowledge Orchestrator implementation for automated component management
- Knowledge Retriever with semantic search and RAG capabilities
- Enhanced CLI with knowledge system integration
- Component discovery and recommendation system
- Knowledge graph visualization

## [0.2.0] - 2025-11-13

### Added
- **LightGBM Multi-Task (lightgbmmt) Models**:
  - Base model architecture with training state management
  - Model factory for creating different model configurations
  - Multi-task gradient boosting model (MTGBM) implementation
  - Multiple loss functions: fixed weight, adaptive weight, knowledge distillation
  - Hyperparameter management system
  - Training and inference scripts for SageMaker integration
  - Comprehensive test suite for all components

- **Temporal Self-Attention Models**:
  - Temporal self-attention classification model with PyTorch Lightning
  - Dual sequence TSA model for multi-sequence processing
  - Feature attention and sequential attention components
  - Custom loss functions and metrics for temporal data
  - Learning rate schedulers optimized for TSA models
  - Legacy temporal self-attention implementations

- **PyTorch Lightning Extensions**:
  - Trimodal BERT model for three-modality fusion
  - Trimodal cross-attention model
  - Trimodal gate fusion model
  - Updates and improvements to existing Lightning models

- **Processing Pipeline Enhancements**:
  - Reorganized processing modules by data type (categorical, numerical, temporal, text)
  - Categorical processors: imputation, validation, dictionary encoding
  - Numerical processors: feature normalization, minmax scaling
  - Temporal processors: sequence ordering, padding, masking, time delta
  - Text processors: dialogue processing, CS format processing
  - Processing constants and shared utilities

- **Documentation and Knowledge Base**:
  - Design documents for atomic processing architecture
  - Bedrock processing step patterns and examples
  - Label ruleset generation and execution patterns
  - Multi-task learning design documentation
  - Temporal feature engineering design
  - Analysis documents for model implementations
  - PyTest best practices and troubleshooting guides
  - API reference documentation style guide

### Changed
- Reorganized test structure to mirror src package structure
- Updated requirements.txt with new dependencies
- Moved legacy implementations to dedicated `_legacy` directories
- Improved error handling in loss functions
- Enhanced weight update strategies in tree models

### Fixed
- Missing weight update strategy in tree-based models
- Path resolution for hyperparameters
- Import paths after processing module reorganization

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
