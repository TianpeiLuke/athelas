---
tags:
  - design
  - refactoring
  - temporal-self-attention
  - code-organization
  - modularity
  - reusability
keywords:
  - TSA reorganization
  - pytorch modules
  - atomic components
  - lightning optimization
  - code modularization
  - temporal modeling
topics:
  - software architecture
  - code reorganization
  - temporal attention models
  - modular design
  - component extraction
language: python
date of note: 2026-01-06
---

# TSA PyTorch Code Reorganization Design

## Overview

This document outlines the systematic reorganization of the Temporal Self-Attention (TSA) codebase to extract atomic PyTorch components from PyTorch Lightning modules and optimize the Lightning architecture to focus on training, validation, prediction, and ONNX inference logic. The reorganization addresses component misplacement, improves reusability, enhances atomicity, and establishes a scalable architecture for temporal modeling.

**Project**: Temporal Self-Attention (TSA) - Sequential fraud detection using temporal patterns with order and feature attention

**Current State**: Atomic PyTorch components embedded within Lightning modules in `lightning_models/temporal/`, making them difficult to reuse and test independently

**Target State**: Atomic component library in `pytorch/` with Lightning modules as thin wrappers focused exclusively on training orchestration

## Related Documents

### Foundational Design Principles
- **[PyTorch Module Reorganization Design](pytorch_module_reorganization_design.md)** - Core organizational principles and patterns guiding this refactoring
- **[Zettelkasten Knowledge Management Principles](../6_resources/zettelkasten_knowledge_management_principles.md)** - Atomicity, connectivity, and anti-category principles

### TSA Model Documentation
- **[Temporal Self-Attention Model Design](temporal_self_attention_model_design.md)** - **PRIMARY** - Comprehensive architecture documentation for TSA models
- **[TSA Lightning Line-by-Line Comparison](../4_analysis/2025-12-20_tsa_lightning_refactoring_line_by_line_comparison.md)** - Algorithm preservation analysis

### Related Refactoring Work
- **[Names3Risk PyTorch Reorganization Design](names3risk_pytorch_reorganization_design.md)** - Similar refactoring effort, established pytorch/ base structure
- **[Algorithm Preserving Refactoring SOP](../6_resources/algorithm_preserving_refactoring_sop.md)** - Safe refactoring procedures

## Problem Statement

### Current Architecture Issues

The TSA codebase exhibits architectural problems that impede development velocity and code quality:

#### 1. Components Misplaced in Lightning Modules (Critical)

**Atomic PyTorch Components in `lightning_models/temporal/pl_tsa_components.py` (650+ lines):**

The file contains pure PyTorch modules that have zero dependency on PyTorch Lightning:

1. **TimeEncode** (70 lines) - Temporal position encoding using learnable periodic functions
2. **FeatureAggregation** (40 lines) - Progressive dimensionality reduction via deep MLP
3. **MixtureOfExperts** (60 lines) - Sparse expert routing for specialized processing
4. **TemporalMultiheadAttention** (50 lines) - Time-aware multi-head attention
5. **AttentionLayer** (80 lines) - Multi-head attention with temporal encoding and MoE
6. **AttentionLayerPreNorm** (70 lines) - Pre-normalization attention layer
7. **OrderAttentionModule** (150 lines) - Composite order attention block
8. **FeatureAttentionModule** (120 lines) - Composite feature attention block
9. **compute_fm_parallel** (10 lines) - Factorization machine utility function

**Issues:**
- ❌ Cannot import individual components without bringing in entire file
- ❌ Cannot test atomic components independently
- ❌ Cannot reuse in non-Lightning contexts
- ❌ Mixes atomic components (TimeEncode) with composite blocks (OrderAttentionModule)
- ❌ No clear separation of concerns

#### 2. Code Duplication Within TSA Codebase

**FeatureAttentionModule Duplication:**
- Appears in `pl_tsa_components.py` (120 lines)
- Appears in `pl_feature_attention.py` (130 lines) with `compute_fm_parallel`
- **Impact**: Two versions of same component, unclear which is canonical

#### 3. Lightning Modules Doing Too Much

**Current Lightning Module Responsibilities:**
- ✅ Training step logic (correct)
- ✅ Validation step logic (correct)
- ✅ Optimizer configuration (correct)
- ❌ Implementing atomic components (should delegate to pytorch/)
- ❌ Mixing model logic with training logic (should separate)

**Example from `pl_temporal_self_attention_classification.py`:**
```python
class TSAClassificationModule(pl.LightningModule):
    def __init__(self, config):
        # Should import from pytorch/, not implement here
        self.order_attention = OrderAttentionModule(config)
        self.feature_attention = FeatureAttentionModule(config)
        # ... training logic mixed with model logic
```

#### 4. Unclear Component Dependencies

**Hidden Dependencies in pl_tsa_components.py:**
- `OrderAttentionModule` depends on `AttentionLayer`, `FeatureAggregation`, `TimeEncode` (all in same file)
- `FeatureAttentionModule` depends on `AttentionLayerPreNorm` (same file)
- `AttentionLayer` depends on `TemporalMultiheadAttention`, `MixtureOfExperts` (same file)
- No explicit import statements document these relationships

**Impact:**
- Developers must read entire file to understand dependencies
- Refactoring one component risks breaking dependent components
- Circular dependencies are difficult to detect

### Comparison with Names3Risk Architecture

The Names3Risk project has already established a well-organized `pytorch/` structure:
- ✅ `pytorch/attention/` - attention_head.py, multihead_attention.py
- ✅ `pytorch/embeddings/` - Currently empty, ready for temporal encoding
- ✅ `pytorch/feedforward/` - mlp_block.py, residual_block.py
- ✅ `pytorch/pooling/` - attention_pooling.py
- ✅ `pytorch/blocks/` - transformer_block.py, lstm_encoder.py, transformer_encoder.py
- ✅ `pytorch/fusion/` - Currently empty, ready for MoE

**TSA should leverage and extend this structure** rather than duplicating it.

## Proposed Architecture

### Target Directory Structure

```
projects/temporal_self_attention_pytorch/
├── __init__.py
├── pipeline_configs/
│
└── dockers/
    ├── __init__.py
    │
    ├── pytorch/                         # ✅ Atomic, reusable PyTorch components
    │   ├── __init__.py
    │   │
    │   ├── attention/                   # Attention mechanisms
    │   │   ├── __init__.py
    │   │   ├── attention_head.py       # ✅ EXISTS (Names3Risk)
    │   │   ├── multihead_attention.py  # ✅ EXISTS (Names3Risk)
    │   │   └── temporal_attention.py   # 🆕 EXTRACT: TemporalMultiheadAttention
    │   │
    │   ├── embeddings/                  # Embedding layers
    │   │   ├── __init__.py
    │   │   └── temporal_encoding.py    # 🆕 EXTRACT: TimeEncode
    │   │
    │   ├── feedforward/                 # Feedforward networks
    │   │   ├── __init__.py
    │   │   ├── mlp_block.py            # ✅ EXISTS (Names3Risk - simple MLP)
    │   │   └── residual_block.py       # ✅ EXISTS (Names3Risk)
    │   │
    │   ├── fusion/                      # Multi-modal fusion
    │   │   ├── __init__.py
    │   │   └── mixture_of_experts.py   # 🆕 EXTRACT: MixtureOfExperts
    │   │
    │   ├── pooling/                     # Pooling mechanisms
    │   │   ├── __init__.py
    │   │   ├── attention_pooling.py    # ✅ EXISTS (Names3Risk)
    │   │   └── feature_aggregation.py  # 🆕 EXTRACT: FeatureAggregation + compute_fm_parallel
    │   │
    │   └── blocks/                      # Composite building blocks
    │       ├── __init__.py
    │       ├── transformer_block.py    # ✅ EXISTS (Names3Risk)
    │       ├── lstm_encoder.py         # ✅ EXISTS (Names3Risk)
    │       ├── transformer_encoder.py  # ✅ EXISTS (Names3Risk)
    │       ├── attention_layer.py      # 🆕 EXTRACT: AttentionLayer, AttentionLayerPreNorm
    │       ├── order_attention.py      # 🆕 EXTRACT: OrderAttentionModule
    │       └── feature_attention.py    # 🆕 EXTRACT: FeatureAttentionModule (unified)
    │
    ├── lightning_models/                # ✅ OPTIMIZED PyTorch Lightning modules
    │   ├── __init__.py
    │   ├── temporal/
    │   │   ├── __init__.py
    │   │   ├── pl_tsa_single_seq.py        # 🔄 REFACTOR: Thin wrapper importing from pytorch/
    │   │   ├── pl_tsa_dual_seq.py          # 🔄 REFACTOR: Thin wrapper importing from pytorch/
    │   │   ├── pl_tsa_losses.py            # ✅ KEEP: Loss functions (training-specific)
    │   │   ├── pl_tsa_metrics.py           # ✅ KEEP: Metrics (training-specific)
    │   │   └── pl_schedulers.py            # ✅ KEEP: LR schedulers (training-specific)
    │   │
    │   ├── bimodal/                     # Bimodal models
    │   ├── tabular/                     # Tabular-only models
    │   ├── text/                        # Text-only models
    │   └── utils/                       # Lightning utilities
    │
    ├── processing/                      # Data processing components
    │   ├── processors.py
    │   ├── dataloaders/
    │   ├── datasets/
    │   ├── categorical/
    │   ├── numerical/
    │   ├── temporal/
    │   └── text/
    │
    └── scripts/                         # Data preparation scripts
        ├── order_aggregation.py
        ├── feature_aggregation.py
        └── categorical_preprocessing.py
```

### Design Principles Applied

Following the **Five Core Zettelkasten Principles** from the PyTorch Module Reorganization Design:

#### 1. Principle of Atomicity
- **One Module = One Concept**: Each file contains exactly one conceptual component
- **Example**: `temporal_encoding.py` contains ONLY TimeEncode, not FeatureAggregation or MoE
- **Benefit**: Can import and test TimeEncode without bringing in unrelated dependencies

#### 2. Principle of Connectivity
- **Explicit Dependencies**: All relationships declared via imports
- **Example**: `order_attention.py` explicitly imports from `..embeddings.temporal_encoding`, `..pooling.feature_aggregation`
- **Benefit**: Following imports reveals the dependency graph

#### 3. Principle Against Categories
- **Flat Structure**: Maximum 2 levels deep (`pytorch/attention/` not `pytorch/layers/attention/mechanisms/`)
- **Semantic Groupings**: Folders describe function (attention/, fusion/) not hierarchy
- **Benefit**: Easy navigation without remembering arbitrary taxonomies

#### 4. Principle of Manual Linking
- **Documented Relationships**: Every module's docstring lists dependencies and consumers
- **Usage Examples**: Include concrete examples of how modules connect
- **Benefit**: Understanding "why this connection exists" without searching

#### 5. Principle of Dual-Form Structure
- **Code as Inner Form**: PyTorch implementation
- **Metadata as Outer Form**: Comprehensive docstrings with usage examples
- **Benefit**: Both humans and tools can understand structure

### Key Architectural Decisions

#### Decision 1: Shared vs TSA-Specific Components

**Shared Components (Reuse Names3Risk):**
- ✅ `attention/attention_head.py`, `multihead_attention.py` - Standard attention
- ✅ `feedforward/mlp_block.py`, `residual_block.py` - Standard feedforward
- ✅ `pooling/attention_pooling.py` - Attention-weighted pooling
- ✅ `blocks/transformer_block.py` - Standard transformer

**TSA-Specific Components (Add to pytorch/):**
- 🆕 `attention/temporal_attention.py` - Time-aware attention (unique to TSA)
- 🆕 `embeddings/temporal_encoding.py` - TimeEncode (unique to temporal modeling)
- 🆕 `fusion/mixture_of_experts.py` - MoE routing (TSA uses, Names3Risk doesn't)
- 🆕 `pooling/feature_aggregation.py` - Progressive reduction (TSA-specific pattern)
- 🆕 `blocks/attention_layer.py` - Combined attention + FFN (TSA-specific composition)
- 🆕 `blocks/order_attention.py` - Order attention module (TSA-specific)
- 🆕 `blocks/feature_attention.py` - Feature attention module (TSA-specific)

**Rationale**: Maximize reuse where applicable, isolate TSA-specific patterns where necessary.

#### Decision 2: Lightning Module Optimization

**Lightning Modules Should ONLY Contain:**
- ✅ Training step logic (`training_step`, `training_epoch_end`)
- ✅ Validation step logic (`validation_step`, `validation_epoch_end`)
- ✅ Prediction step logic (`predict_step`, `predict_epoch_end`)
- ✅ Test step logic (`test_step`, `test_epoch_end`)
- ✅ Optimizer configuration (`configure_optimizers`)
- ✅ LR scheduler configuration (part of `configure_optimizers`)
- ✅ Metric computation and logging
- ✅ ONNX export logic (`to_onnx`)
- ✅ Checkpoint loading/saving hooks
- ✅ Callbacks and custom hooks

**Lightning Modules Should NOT Contain:**
- ❌ Atomic PyTorch components (TimeEncode, FeatureAggregation, MoE)
- ❌ Attention mechanism implementations (AttentionLayer, MultiheadAttention)
- ❌ Feedforward network implementations (MLPBlock, ResidualBlock)
- ❌ Pooling mechanism implementations (AttentionPooling, FeatureAggregation)
- ❌ Utility functions (compute_fm_parallel)

**Example Refactored Lightning Module:**
```python
# lightning_models/temporal/pl_tsa_single_seq.py
import pytorch_lightning as pl
from ...pytorch.blocks import OrderAttentionModule, FeatureAttentionModule
from ...pytorch.feedforward import ResidualBlock

class TSASingleSeqModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for single-sequence TSA model.
    
    Focuses on training orchestration, delegates computation to pytorch/ components.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Import atomic components from pytorch/
        self.order_attention = OrderAttentionModule(config)
        self.feature_attention = FeatureAttentionModule(config)
        self.classifier = ResidualBlock(
            dim=2 * config["dim_embedding_table"],
            expansion_factor=4,
            dropout=config.get("dropout", 0.1)
        )
        
        # Training-specific configuration
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.save_hyperparameters()
    
    def forward(self, batch):
        """Delegate to atomic components."""
        x_order = self.order_attention(
            batch["x_cat"], batch["x_num"], batch["time_seq"]
        )
        x_feature = self.feature_attention(
            batch["x_cat"], batch["x_num"], batch["x_engineered"]
        )
        ensemble = torch.cat([x_order, x_feature], dim=-1)
        logits = self.classifier(ensemble)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training logic ONLY."""
        logits = self(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation logic ONLY."""
        logits = self(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss)
        self.log("val_auc", compute_auc(preds, batch["labels"]))
        return loss
    
    def configure_optimizers(self):
        """Optimizer configuration ONLY."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def to_onnx(self, filepath):
        """ONNX export logic."""
        # Export model to ONNX format
        pass
```

#### Decision 3: No Duplication with Names3Risk

**Verification:**
- ✅ `MixtureOfExperts` (TSA) ≠ `MLPBlock` (Names3Risk) - Different architectures
- ✅ `TimeEncode` (TSA) - Not in Names3Risk
- ✅ `FeatureAggregation` (TSA) - Not in Names3Risk  
- ✅ `TemporalMultiheadAttention` (TSA) - Not in Names3Risk
- ✅ `AttentionLayer` (TSA composite) ≠ `TransformerBlock` (Names3Risk) - Different compositions

**Outcome**: No conflicts, TSA components are additive to Names3Risk base.

## Component Migration Map

### Phase 1: Extract TSA-Specific Atomic Components (High Priority)

#### 1.1 TimeEncode → `pytorch/embeddings/temporal_encoding.py`

**Status**: 🔴 **CRITICAL - TSA-SPECIFIC COMPONENT**

**Current Location:** `lightning_models/temporal/pl_tsa_components.py` lines 18-70

**Extraction Strategy:** Direct move with enhanced documentation

**Target Implementation:**
```python
# pytorch/embeddings/temporal_encoding.py
"""
Temporal Position Encoding

Learnable temporal encoding using periodic functions for time-series data.

**Core Concept:**
Encodes temporal information via combination of linear transformation and sinusoidal
functions with learnable parameters. Essential for TSA models to capture both
absolute time values and periodic temporal patterns in transaction sequences.

**Architecture:**
- Linear component: Direct time representation
- Sinusoidal components: Periodic patterns at multiple frequencies
- Learnable parameters: Domain-specific adaptation

**Parameters:**
- time_dim (int): Dimension of temporal encoding
- device: Device for tensor allocation
- dtype: Data type for tensors

**Forward Signature:**
Input:
  - tt: Time tensor [B, L, 1] - Time deltas or timestamps
  
Output:
  - temporal_encoding: [L, B, time_dim] - Encoded temporal information

**Dependencies:**
- torch.nn.Linear → Linear transformation
- torch.nn.functional → Sinusoidal activation

**Used By:**
- temporal_self_attention_pytorch.pytorch.attention.temporal_attention → Time-aware attention
- temporal_self_attention_pytorch.pytorch.blocks.order_attention → Order attention module

**Alternative Approaches:**
- Fixed sinusoidal encoding (Transformer) → Not learnable
- Learned embeddings → No periodic structure
- RNN hidden states → Computational cost

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.embeddings import TimeEncode

time_encoder = TimeEncode(time_dim=128)

# Encode time deltas (days since last transaction)
time_deltas = torch.tensor([[[0.5]], [[1.2]], [[2.0]]])  # [B=3, L=1, 1]
time_encoding = time_encoder(time_deltas)  # [1, 3, 128]
```

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Positional encoding
- "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional


class TimeEncode(nn.Module):
    """
    Learnable temporal position encoding using periodic functions.
    
    Combines linear transformation with sinusoidal functions for
    temporal pattern modeling.
    """
    
    def __init__(self, time_dim: int, device=None, dtype=None):
        """
        Initialize TimeEncode.
        
        Args:
            time_dim: Dimension of temporal encoding
            device: Device for tensor allocation (default: None)
            dtype: Data type for tensors (default: None)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.time_dim = time_dim
        
        # Learnable weight matrix and bias
        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform."""
        # Kaiming uniform initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)
    
    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information.
        
        Args:
            tt: Time tensor [B, L, 1] - Time deltas or timestamps
            
        Returns:
            temporal_encoding: [L, B, time_dim] - Encoded temporal information
        """
        # Ensure 3D shape
        tt = tt.unsqueeze(-1) if tt.dim() == 2 else tt  # [B, L, 1]
        
        # Sinusoidal encoding (all dimensions except first)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))
        
        # Linear encoding (first dimension)
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])
        
        # Combine encodings
        t = torch.cat([out1, out2], -1)  # [B, L, time_dim]
        t = t.squeeze(-2)  # Remove extra dimension if present
        t = t.permute(1, 0, 2)  # [L, B, time_dim]
        
        return t
```

**Migration Strategy:**
1. Create `pytorch/embeddings/temporal_encoding.py` with TimeEncode
2. Add comprehensive docstring and type hints
3. Write unit tests for temporal encoding
4. Update `blocks/order_attention.py` to import from new location
5. Add deprecation warning in `pl_tsa_components.py`

#### 1.2 FeatureAggregation + compute_fm_parallel → `pytorch/pooling/feature_aggregation.py`

**Status**: 🔴 **CRITICAL - TSA-SPECIFIC COMPONENT**

**Current Location:** 
- `pl_tsa_components.py` lines 72-110 (FeatureAggregation)
- `pl_feature_attention.py` lines 115-130 (compute_fm_parallel, duplicated)

**Extraction Strategy:** Combine FeatureAggregation and compute_fm_parallel in single file

**Target Implementation:**
```python
# pytorch/pooling/feature_aggregation.py
"""
Feature Aggregation with Factorization Machine Support

Progressive dimensionality reduction via deep MLP and FM-style feature interactions.

**Core Concept:**
Reduces features from n_features to 1 via progressive halving (n → n/2 → n/4 → ... → 1).
Essential for TSA to aggregate multiple categorical and numerical features before attention.
Includes FM computation for second-order feature interactions.

**Architecture:**
- Progressive MLP: Halves feature dimension at each layer
- Activation: LeakyReLU for non-linearity
- Termination: Stops when dimension reaches 1

**Parameters:**
- num_feature (int): Number of input features to aggregate

**Forward Signature:**
Input:
  - x: [..., num_feature] - Feature embeddings to aggregate
  
Output:
  - aggregated: [..., 1] - Aggregated features

**Dependencies:**
- torch.nn.Linear → Dimensionality reduction layers
- torch.nn.LeakyReLU → Non-linear activation

**Used By:**
- temporal_self_attention_pytorch.pytorch.blocks.order_attention → Feature aggregation before attention
- temporal_self_attention_pytorch.pytorch.blocks.feature_attention → Last order feature aggregation

**Alternative Approaches:**
- Mean pooling → Simpler but loses learned importance
- Max pooling → Takes most salient but discards others
- Single linear projection → Less expressive

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.pooling import FeatureAggregation, compute_fm_parallel

# Create feature aggregation
agg = FeatureAggregation(num_feature=20)

# Aggregate categorical and numerical features
features = torch.randn(32, 50, 20, 128)  # [B, L, n_features, embed_dim]
aggregated = agg(features.permute(0, 1, 3, 2))  # [B, L, embed_dim, n_features] -> [B, L, embed_dim]

# Compute FM interactions
fm_features = compute_fm_parallel(features)  # [B, L, embed_dim]
```

**References:**
- "Factorization Machines" (Rendle, 2010) - FM for feature interactions
- "Neural Factorization Machines" (He & Chua, 2017) - Deep learning + FM
"""

import torch
import torch.nn as nn
from typing import Tuple


class FeatureAggregation(nn.Module):
    """
    Feature aggregation via progressive dimensionality reduction.
    
    Uses deep MLP to aggregate features with progressive halving:
    n → n/2 → n/4 → ... → 1
    """
    
    def __init__(self, num_feature: int):
        """
        Initialize FeatureAggregation.
        
        Args:
            num_feature: Number of input features to aggregate
        """
        super().__init__()
        
        self.dim_embed = num_feature
        
        # Build progressive reduction layers
        layers = []
        current_dim = num_feature
        
        while current_dim > 1:
            next_dim = max(1, current_dim // 2)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LeakyReLU()
            ])
            current_dim = next_dim
        
        # Remove the last LeakyReLU
        if layers:
            layers = layers[:-1]
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features across the feature dimension.
        
        Args:
            x: Input tensor [..., num_feature]
            
        Returns:
            aggregated: [..., 1] - Aggregated features
        """
        return self.encoder(x)


def compute_fm_parallel(feature_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute Factorization Machine (FM) style feature interactions in parallel.
    
    Computes second-order feature interactions using the factorization machine formula:
    FM(x) = 0.5 * [(Σ embeddings)² - Σ (embeddings²)]
    
    This captures pairwise feature interactions without computing all pairs explicitly,
    reducing computational complexity from O(n²) to O(n).
    
    Args:
        feature_embedding: Feature embeddings [B, n_features, embed_dim]
        
    Returns:
        fm_interaction: FM interaction features [B, embed_dim]
        
    Example:
        >>> features = torch.randn(32, 10, 128)  # 32 samples, 10 features, 128 dims
        >>> interactions = compute_fm_parallel(features)  # [32, 128]
    """
    # Sum of embeddings then square: (Σ x_i)²
    summed_features_emb = torch.sum(feature_embedding, dim=-2)  # [B, embed_dim]
    summed_features_emb_square = torch.square(summed_features_emb)  # [B, embed_dim]
    
    # Square of embeddings then sum: Σ (x_i²)
    squared_features_emb = torch.square(feature_embedding)  # [B, n_features, embed_dim]
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)  # [B, embed_dim]
    
    # FM interaction computation: 0.5 * [(Σ x_i)² - Σ (x_i²)]
    fm_interaction = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
    
    return fm_interaction
```

**Migration Strategy:**
1. Create `pytorch/pooling/feature_aggregation.py` with both classes
2. Remove duplication from `pl_feature_attention.py`
3. Write unit tests for feature aggregation and FM computation
4. Update `blocks/order_attention.py` and `blocks/feature_attention.py` to import from new location
5. Add deprecation warnings in both `pl_tsa_components.py` and `pl_feature_attention.py`

#### 1.3 MixtureOfExperts → `pytorch/fusion/mixture_of_experts.py`

**Status**: 🔴 **CRITICAL - TSA-SPECIFIC COMPONENT**

**Current Location:** `pl_tsa_components.py` lines 112-175

**Extraction Strategy:** Direct move, add comprehensive documentation

**Migration Strategy:**
1. Create `pytorch/fusion/mixture_of_experts.py` with MoE implementation
2. Write unit tests for expert routing and gating
3. Update `blocks/attention_layer.py` to import from new location
4. Add deprecation warning in `pl_tsa_components.py`

### Phase 2: Extract Composite Blocks (Medium Priority)

#### 2.1 AttentionLayer + AttentionLayerPreNorm → `pytorch/blocks/attention_layer.py`

**Status**: 🟡 **COMPOSITE BLOCK - TSA-SPECIFIC**

**Current Location:** `pl_tsa_components.py` lines 220-320, 322-410

**Dependencies:** Requires TimeEncode, TemporalMultiheadAttention, MixtureOfExperts

**Extraction Strategy:** Combine both variants in single file after dependencies extracted

**Migration Strategy:**
1. First extract all dependencies (Phase 1)
2. Create `pytorch/blocks/attention_layer.py` with both classes
3. Update imports to reference pytorch/ components
4. Write integration tests
5. Update Lightning modules to import from new location

#### 2.2 OrderAttentionModule → `pytorch/blocks/order_attention.py`

**Status**: 🟡 **COMPOSITE BLOCK - TSA-SPECIFIC**

**Current Location:** `pl_tsa_components.py` lines 412-585

**Dependencies:** Requires FeatureAggregation, AttentionLayer, TimeEncode

**Extraction Strategy:** Move after all dependencies extracted

**Migration Strategy:**
1. Ensure all dependencies in pytorch/ (Phase 1 & 2.1)
2. Create `pytorch/blocks/order_attention.py`
3. Update imports to reference pytorch/ components
4. Write integration tests
5. Update Lightning modules to import from new location

#### 2.3 FeatureAttentionModule → `pytorch/blocks/feature_attention.py`

**Status**: 🟡 **COMPOSITE BLOCK - TSA-SPECIFIC + DUPLICATED**

**Current Location:** 
- `pl_tsa_components.py` lines 588-750
- `pl_feature_attention.py` (entire file, 130 lines)

**Dependencies:** Requires AttentionLayerPreNorm

**Extraction Strategy:** Unify duplicated versions, move to pytorch/blocks/

**Migration Strategy:**
1. Compare both versions, identify canonical implementation
2. Create unified `pytorch/blocks/feature_attention.py`
3. Update imports to reference pytorch/ components
4. Write integration tests
5. Delete `pl_feature_attention.py` file entirely
6. Update all references to import from pytorch/blocks/

### Phase 3: Optimize Lightning Modules (High Priority)

#### 3.1 Refactor pl_tsa_single_seq.py

**Current State:** Imports OrderAttentionModule and FeatureAttentionModule from pl_tsa_components

**Target State:** Thin wrapper importing from pytorch/blocks/

**Refactoring Strategy:**
```python
# BEFORE (current)
from .pl_tsa_components import OrderAttentionModule, FeatureAttentionModule

# AFTER (target)
from ...pytorch.blocks import OrderAttentionModule, FeatureAttentionModule
from ...pytorch.feedforward import ResidualBlock
```

**Focus Areas:**
- ✅ Keep training_step, validation_step, test_step
- ✅ Keep configure_optimizers
- ✅ Keep metric logging
- ✅ Keep ONNX export logic
- ❌ Remove any remaining component implementations

#### 3.2 Refactor pl_tsa_dual_seq.py

**Similar strategy to pl_tsa_single_seq.py**

#### 3.3 Clean Up pl_tsa_components.py

**After all extractions complete:**
1. Add deprecation warnings for all classes
2. Re-export from pytorch/ for backward compatibility
3. Plan for eventual deletion in v2.0

**Backward Compatibility Wrapper:**
```python
# pl_tsa_components.py (after extraction)
import warnings
from ...pytorch.embeddings import TimeEncode as _TimeEncode
from ...pytorch.pooling import FeatureAggregation as _FeatureAggregation
# ... etc

class TimeEncode(_TimeEncode):
    """Deprecated: Use temporal_self_attention_pytorch.pytorch.embeddings.TimeEncode"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Importing TimeEncode from pl_tsa_components is deprecated. "
            "Use 'from temporal_self_attention_pytorch.pytorch.embeddings import TimeEncode' instead. "
            "This wrapper will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

## Implementation Timeline

### Week 1: Extract High-Priority Atomic Components
**Goal:** Extract TimeEncode, FeatureAggregation, MoE, TemporalMultiheadAttention

**Tasks:**
- [x] Audit existing pytorch/ structure
- [ ] Create `pytorch/embeddings/temporal_encoding.py` with TimeEncode
- [ ] Create `pytorch/pooling/feature_aggregation.py` with FeatureAggregation + compute_fm_parallel
- [ ] Create `pytorch/fusion/mixture_of_experts.py` with MixtureOfExperts
- [ ] Create `pytorch/attention/temporal_attention.py` with TemporalMultiheadAttention
- [ ] Write unit tests for all extracted components (100% coverage)
- [ ] Verify no duplication with Names3Risk components

**Success Metrics:**
- ✅ All 4 atomic components extracted
- ✅ 100% test coverage
- ✅ No import errors
- ✅ All tests passing

### Week 2: Extract Composite Blocks
**Goal:** Extract AttentionLayer variants, OrderAttentionModule, FeatureAttentionModule

**Tasks:**
- [ ] Create `pytorch/blocks/attention_layer.py` with AttentionLayer + AttentionLayerPreNorm
- [ ] Create `pytorch/blocks/order_attention.py` with OrderAttentionModule
- [ ] Unify and create `pytorch/blocks/feature_attention.py` with FeatureAttentionModule
- [ ] Update all imports to reference pytorch/ components
- [ ] Write integration tests for composite blocks
- [ ] Delete duplicated `pl_feature_attention.py`

**Success Metrics:**
- ✅ All 3 composite blocks extracted
- ✅ Zero duplication (FeatureAttentionModule unified)
- ✅ Integration tests passing
- ✅ Composite blocks use atomic components via imports

### Week 3: Optimize Lightning Modules
**Goal:** Refactor Lightning modules to become thin wrappers

**Tasks:**
- [ ] Refactor `pl_tsa_single_seq.py` to import from pytorch/
- [ ] Refactor `pl_tsa_dual_seq.py` to import from pytorch/
- [ ] Add backward compatibility wrappers in `pl_tsa_components.py`
- [ ] Add deprecation warnings to all legacy imports
- [ ] Update all internal references
- [ ] End-to-end tests for refactored Lightning modules

**Success Metrics:**
- ✅ Lightning modules are thin wrappers (<200 lines each)
- ✅ All training/validation/prediction logic preserved
- ✅ Backward compatibility maintained
- ✅ End-to-end tests passing

### Week 4: Documentation, Testing, Validation
**Goal:** Complete documentation and validate migration success

**Tasks:**
- [ ] Write comprehensive docstrings for all components
- [ ] Create component usage examples
- [ ] Performance benchmarks (verify no regression)
- [ ] Numerical equivalence tests (legacy vs refactored)
- [ ] Update project README
- [ ] Migration guide for users
- [ ] Code review and feedback incorporation

**Success Metrics:**
- ✅ All components documented
- ✅ Performance within 1% of legacy
- ✅ Numerical equivalence verified (rtol ≤ 1e-6)
- ✅ Migration guide complete
- ✅ Code review approved

## Migration Strategy

### Backward Compatibility Approach

**Phase 1-2: Parallel Structure**
- Keep legacy `pl_tsa_components.py` with deprecation warnings
- Re-export from pytorch/ for backward compatibility
- Gradual migration path for existing code

**Phase 3: Deprecation Period (3 months)**
- All new code uses pytorch/ imports
- Legacy imports show deprecation warnings
- Documentation updated to new structure

**Phase 4: Legacy Removal (v2.0)**
- Remove `pl_tsa_components.py`
- Remove backward compatibility wrappers
- Clean up deprecated imports

### Testing Strategy

#### Unit Tests (Per Atomic Component)
```python
# test/pytorch/embeddings/test_temporal_encoding.py
import torch
import pytest
from temporal_self_attention_pytorch.pytorch.embeddings import TimeEncode

class TestTimeEncode:
    def test_forward_shape(self):
        """Test output shape correctness."""
        encoder = TimeEncode(time_dim=128)
        time_input = torch.randn(32, 50, 1)  # [B, L, 1]
        
        output = encoder(time_input)
        
        assert output.shape == (50, 32, 128)  # [L, B, time_dim]
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        encoder = TimeEncode(time_dim=128)
        
        assert encoder.weight.shape == (128, 1)
        assert encoder.emb_tbl_bias.shape == (128,)
        assert not torch.isnan(encoder.weight).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through encoding."""
        encoder = TimeEncode(time_dim=128)
        time_input = torch.randn(32, 50, 1, requires_grad=True)
        
        output = encoder(time_input)
        loss = output.sum()
        loss.backward()
        
        assert time_input.grad is not None
        assert not torch.isnan(time_input.grad).any()
```

#### Integration Tests (Composite Blocks)
```python
# test/pytorch/blocks/test_order_attention.py
import torch
from temporal_self_attention_pytorch.pytorch.blocks import OrderAttentionModule

class TestOrderAttentionModule:
    def test_module_integration(self):
        """Test OrderAttentionModule with all dependencies."""
        config = {
            "n_cat_features": 10,
            "n_num_features": 5,
            "n_embedding": 1000,
            "dim_embedding_table": 64,
            "dim_attn_feedforward": 256,
            "n_layers_order": 2,
            "num_heads": 4,
            "use_moe": True,
            "num_experts": 3
        }
        
        module = OrderAttentionModule(config)
        
        x_cat = torch.randint(0, 1000, (32, 50, 10))
        x_num = torch.randn(32, 50, 5)
        time_seq = torch.randn(32, 50, 1)
        
        output = module(x_cat, x_num, time_seq)
        
        assert output.shape == (32, 128)  # 2 * dim_embedding_table
        assert not torch.isnan(output).any()
```

#### Numerical Equivalence Tests
```python
# test/compatibility/test_legacy_equivalence.py
import torch
from temporal_self_attention_pytorch.dockers.lightning_models.temporal.pl_tsa_components import TimeEncode as LegacyTimeEncode
from temporal_self_attention_pytorch.pytorch.embeddings import TimeEncode as NewTimeEncode

class TestNumericalEquivalence:
    def test_time_encode_equivalence(self):
        """Verify new TimeEncode produces identical results to legacy."""
        torch.manual_seed(42)
        legacy = LegacyTimeEncode(time_dim=128)
        
        torch.manual_seed(42)
        new = NewTimeEncode(time_dim=128)
        
        # Same input
        time_input = torch.randn(32, 50, 1)
        
        # Compare outputs
        with torch.no_grad():
            legacy_output = legacy(time_input)
            new_output = new(time_input)
        
        # Should be numerically identical (within floating point precision)
        assert torch.allclose(legacy_output, new_output, rtol=1e-6, atol=1e-8)
```

## Benefits Analysis

### Quantitative Improvements

**Code Organization:**
- Before: 650+ lines in 1 monolithic file (`pl_tsa_components.py`)
- After: 9 atomic modules averaging 80 lines each
- **Improvement**: 88% reduction in largest file size

**Component Reusability:**
- Before: 0 atomic components usable independently
- After: 9 atomic components + 3 composite blocks all independently usable
- **Improvement**: 12 reusable components

**Duplication Elimination:**
- Before: FeatureAttentionModule duplicated (250 lines total)
- After: 1 unified implementation (120 lines)
- **Savings**: 52% reduction

**Import Clarity:**
- Before: Importing TimeEncode brings 8 other unrelated classes
- After: Importing TimeEncode brings only TimeEncode
- **Improvement**: 89% reduction in unintended dependencies

### Qualitative Improvements

**Developer Experience:**
- ✅ Easier to find components (semantic folder structure)
- ✅ Faster to understand components (focused modules)
- ✅ Simpler to test components (atomic isolation)
- ✅ Safer to refactor components (explicit dependencies)

**Code Quality:**
- ✅ Consistent interfaces across components
- ✅ Comprehensive documentation with examples
- ✅ Type hints enable IDE support
- ✅ Single source of truth eliminates divergence

**Lightning Module Clarity:**
- ✅ Lightning modules focus on training logic only
- ✅ Clear separation between model and training concerns
- ✅ Easier to add new training features (callbacks, metrics)
- ✅ Better integration with Lightning ecosystem

**Cross-Project Reuse:**
- ✅ TSA components can be used in Names3Risk (e.g., TimeEncode for temporal features)
- ✅ Names3Risk components can be used in TSA (e.g., ResidualBlock, AttentionPooling)
- ✅ Shared pytorch/ base enables consistent patterns across projects
- ✅ Future projects can leverage both TSA and Names3Risk components

## Success Criteria

### Technical Metrics
- [ ] **Zero Component Duplication**: All 9 components have single implementations
- [ ] **100% Test Coverage**: All atomic components have comprehensive unit tests
- [ ] **Performance Parity**: <1% difference in training/inference speed vs legacy
- [ ] **Numerical Equivalence**: Outputs match legacy implementation within rtol ≤ 1e-6
- [ ] **Documentation Complete**: All modules have standardized docstrings with examples
- [ ] **Lightning Modules Optimized**: Each <200 lines, focused on training logic only

### Process Metrics
- [ ] **Migration Timeline**: Complete within 4 weeks
- [ ] **Zero Breaking Changes**: All existing code continues to work with warnings
- [ ] **Team Adoption**: >80% of team comfortable with new structure after 1 week
- [ ] **Code Review Approval**: All phases reviewed and approved by 2+ reviewers

### Outcome Metrics
- [ ] **Reduced Maintenance**: 50% reduction in bug fix time for shared components
- [ ] **Increased Velocity**: 30% faster development for new temporal models
- [ ] **Better Onboarding**: New developers productive with TSA in <3 days
- [ ] **Code Reuse**: TSA components reused in at least 1 other project within 6 months

## Risks and Mitigation

### Risk 1: Breaking Changes During Migration
**Probability**: Medium | **Impact**: High

**Mitigation**:
- Maintain backward compatibility wrappers in `pl_tsa_components.py`
- Add deprecation warnings with clear migration path
- Run comprehensive test suite before each phase
- Keep legacy code as fallback for 2 major versions

**Contingency**: Immediate rollback to legacy implementation if critical issues discovered

### Risk 2: Performance Regression
**Probability**: Low | **Impact**: Medium

**Mitigation**:
- Benchmark performance before and after refactoring
- Profile hot paths to identify bottlenecks
- Use identical algorithms, only restructure code organization
- Validate numerical equivalence with legacy implementation

**Contingency**: Optimize specific components if performance degrades >2%

### Risk 3: Increased Complexity for Simple Use Cases
**Probability**: Low | **Impact**: Low

**Mitigation**:
- Provide high-level Lightning modules that hide complexity
- Create usage examples for common patterns
- Maintain simple import paths for common components
- Document migration from legacy to new structure

**Contingency**: Create convenience wrappers if users struggle with new structure

## References and Related Work

### Design Documents
- **[PyTorch Module Reorganization Design](pytorch_module_reorganization_design.md)** - Foundational principles
- **[Temporal Self-Attention Model Design](temporal_self_attention_model_design.md)** - TSA architecture
- **[Names3Risk PyTorch Reorganization Design](names3risk_pytorch_reorganization_design.md)** - Parallel effort
- **[TSA Lightning Line-by-Line Comparison](../4_analysis/2025-12-20_tsa_lightning_refactoring_line_by_line_comparison.md)** - Algorithm preservation

### Implementation References
- **PyTorch Documentation**: Official PyTorch deep learning framework
- **PyTorch Lightning Documentation**: Lightning training framework
- **Zettelkasten Method**: Knowledge management principles for code organization

### Academic References
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "Factorization Machines" (Rendle, 2010) - Feature interaction modeling
- "Outrageously Large Neural Networks" (Shazeer et al., 2017) - Mixture of Experts

## Appendices

### Appendix A: Complete Component Inventory

| Component | Legacy Location | New Location | Type | Priority |
|-----------|----------------|--------------|------|----------|
| TimeEncode | pl_tsa_components.py:18-70 | pytorch/embeddings/temporal_encoding.py | Atomic | 🔴 HIGH |
| FeatureAggregation | pl_tsa_components.py:72-110 | pytorch/pooling/feature_aggregation.py | Atomic | 🔴 HIGH |
| compute_fm_parallel | pl_feature_attention.py:115-130 | pytorch/pooling/feature_aggregation.py | Utility | 🔴 HIGH |
| MixtureOfExperts | pl_tsa_components.py:112-175 | pytorch/fusion/mixture_of_experts.py | Atomic | 🔴 HIGH |
| TemporalMultiheadAttention | pl_tsa_components.py:177-218 | pytorch/attention/temporal_attention.py | Atomic | 🔴 HIGH |
| AttentionLayer | pl_tsa_components.py:220-320 | pytorch/blocks/attention_layer.py | Composite | 🟡 MEDIUM |
| AttentionLayerPreNorm | pl_tsa_components.py:322-410 | pytorch/blocks/attention_layer.py | Composite | 🟡 MEDIUM |
| OrderAttentionModule | pl_tsa_components.py:412-585 | pytorch/blocks/order_attention.py | Composite | 🟡 MEDIUM |
| FeatureAttentionModule | pl_tsa_components.py:588-750 | pytorch/blocks/feature_attention.py | Composite | 🟡 MEDIUM |
| FeatureAttentionModule (dup) | pl_feature_attention.py | REMOVE (use unified version) | Composite | 🔴 HIGH |

### Appendix B: Dependency Graph

```
Lightning Modules (lightning_models/temporal/)
├── pl_tsa_single_seq.py
│   ├── OrderAttentionModule (blocks/)
│   │   ├── AttentionLayer (blocks/)
│   │   │   ├── TemporalMultiheadAttention (attention/)
│   │   │   │   └── TimeEncode (embeddings/)
│   │   │   └── MixtureOfExperts (fusion/)
│   │   ├── FeatureAggregation (pooling/)
│   │   └── TimeEncode (embeddings/)
│   ├── FeatureAttentionModule (blocks/)
│   │   └── AttentionLayerPreNorm (blocks/)
│   │       └── MixtureOfExperts (fusion/)
│   └── ResidualBlock (feedforward/) [Names3Risk]
│
└── pl_tsa_dual_seq.py
    ├── OrderAttentionModule (blocks/) [2x - CID and CCID]
    ├── FeatureAttentionModule (blocks/)
    └── ResidualBlock (feedforward/) [Names3Risk]
```

### Appendix C: Migration Checklist

#### Pre-Migration Phase
- [x] Review and approve this design document
- [x] Audit existing pytorch/ directory for potential conflicts
- [x] Verify no duplication with Names3Risk components
- [ ] Set up test infrastructure for new components
- [ ] Create migration tracking document

#### Phase 1: Extract Atomic Components (Week 1)
- [ ] Extract TimeEncode to pytorch/embeddings/temporal_encoding.py
- [ ] Extract FeatureAggregation + compute_fm_parallel to pytorch/pooling/feature_aggregation.py
- [ ] Extract MixtureOfExperts to pytorch/fusion/mixture_of_experts.py
- [ ] Extract TemporalMultiheadAttention to pytorch/attention/temporal_attention.py
- [ ] Add unit tests (100% coverage)
- [ ] Verify no import errors

#### Phase 2: Extract Composite Blocks (Week 2)
- [ ] Extract AttentionLayer + AttentionLayerPreNorm to pytorch/blocks/attention_layer.py
- [ ] Extract OrderAttentionModule to pytorch/blocks/order_attention.py
- [ ] Unify and extract FeatureAttentionModule to pytorch/blocks/feature_attention.py
- [ ] Delete duplicated pl_feature_attention.py
- [ ] Add integration tests
- [ ] Verify all tests passing

#### Phase 3: Optimize Lightning Modules (Week 3)
- [ ] Refactor pl_tsa_single_seq.py
- [ ] Refactor pl_tsa_dual_seq.py
- [ ] Add backward compatibility wrappers
- [ ] Add deprecation warnings
- [ ] Update all internal references
- [ ] Run end-to-end tests

#### Phase 4: Documentation & Validation (Week 4)
- [ ] Write comprehensive docstrings
- [ ] Create component usage examples
- [ ] Performance benchmarks
- [ ] Numerical equivalence tests
- [ ] Update project documentation
- [ ] Migration guide
- [ ] Code review and approval

---

**Document Status:** ✅ Ready for Implementation  
**Last Updated:** 2026-01-06  
**Authors:** AI Analysis Team  
**Reviewers:** [Pending]  
**Next Review:** After Week 1 completion

**Related Implementation Documents:**
- [PyTorch Module Reorganization Design](pytorch_module_reorganization_design.md) - Foundational principles
- [Temporal Self-Attention Model Design](temporal_self_attention_model_design.md) - TSA architecture details
- [Algorithm Preserving Refactoring SOP](../6_resources/algorithm_preserving_refactoring_sop.md) - Safe refactoring procedures
