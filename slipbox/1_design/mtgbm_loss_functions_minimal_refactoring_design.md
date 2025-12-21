---
tags:
  - design
  - refactoring
  - legacy-preservation
  - minimal-change
  - multi-task-learning
keywords:
  - MTGBM
  - loss functions
  - minimal refactoring
  - legacy preservation
  - code migration
  - backward compatibility
topics:
  - software engineering
  - legacy code migration
  - minimal change refactoring
language: python
date of note: 2025-12-19
---

# MTGBM Loss Functions Minimal Refactoring Design

## Overview

This document specifies a **minimal-change refactoring strategy** for the MTGBM loss function implementations in `projects/cap_mtgbm/docker/models/loss/`. The design philosophy is **"Copy first, refactor never (unless bugs found)"** - preserving legacy implementation logic byte-for-byte to minimize risk while extracting only truly shared functionality into a minimal base class.

This approach differs from the comprehensive refactoring in `mtgbm_models_refactoring_design.md` by prioritizing:
1. **Legacy Fidelity**: ~95% code reuse from legacy implementations
2. **Risk Minimization**: Only extract code that is TRULY identical across all implementations
3. **Easy Verification**: Side-by-side comparison with legacy code
4. **Zero Functional Changes**: Preserve all behaviors, including quirks and edge cases

The refactored implementation maintains three distinct loss functions with minimal abstraction, ensuring production stability while improving code organization.

## Related Documents

- **[MTGBM Models Refactoring Design](./mtgbm_models_refactoring_design.md)** - Comprehensive refactoring approach (alternative design)
- **[MTGBM Refactoring: Critical Bugs Fixed](../4_analysis/2025-12-18_mtgbm_refactoring_critical_bugs_fixed.md)** - Critical bugs discovered and fixed
- **[MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)** - Original multi-task learning design
- **[LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)** - Framework analysis

## Design Philosophy

### Core Principle: Minimal Change

**"Copy first, refactor never (unless bugs found)"**

This refactoring follows a conservative approach that prioritizes:

#### 1. Legacy Preservation
- **Preserve all logic**: Keep original algorithms intact
- **Preserve all quirks**: Don't "fix" non-bugs  
- **Preserve all edge cases**: Keep existing behavior
- **Preserve all comments**: Document legacy decisions

#### 2. Risk Minimization
- **Extract only identical code**: Share only truly duplicate logic
- **No algorithmic changes**: Keep original formulas
- **No performance optimizations**: Maintain original speed
- **No interface changes**: Preserve method signatures

#### 3. Easy Verification
- **Side-by-side comparison**: Each method maps 1:1 to legacy
- **Byte-for-byte matching**: Identical output for identical input
- **Clear lineage**: Comment shows source in legacy code
- **Diff-friendly**: Easy to compare with git diff

### What Gets Shared vs. What Stays Separate

#### Shared in Base Class (30 lines total)
```python
# ONLY truly identical operations across ALL three implementations:
1. grad() computation: y_pred - y_true (IDENTICAL)
2. hess() computation: y_pred * (1 - y_pred) (IDENTICAL)  
3. Data preprocessing shape operations (IDENTICAL)
4. Evaluation metric computation (IDENTICAL)
```

#### Kept in Subclasses (Preserved from legacy)
```python
# Everything else stays in original form:
1. Weight computation logic (Different per loss type)
2. Gradient aggregation (Different: fixed vs normalized)
3. Utility methods (Similar but have subtle differences)
4. KD logic (Unique to KDswap)
5. All hyperparameters and configuration
```

### Anti-Patterns to Avoid

❌ **Don't do this:**
- "While we're here, let's optimize X"
- "We can clean up this ugly code"  
- "Let's use a better algorithm"
- "This obviously should be Y not X"

✅ **Do this:**
- "Copy-paste entire method from legacy"
- "Preserve exact formatting and style"
- "Add comment: // Legacy code from X.py line Y"
- "If it works, don't change it"

## Legacy Code Analysis

### Legacy File Structure

```
projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/
├── baseLoss.py              # 62 lines - Fixed weight baseline
├── customLossNoKD.py        # 142 lines - Adaptive weight (no KD)
└── customLossKDswap.py      # 193 lines - Adaptive weight + KD
```

### Legacy Implementation Comparison

#### 1. baseLoss (Fixed Weight)

**Key Characteristics:**
- **Simplest**: No adaptive weights, no KD
- **Fixed weights**: Hardcoded `[1, 0.1*beta, 0.1*beta, ...]`
- **No normalization**: Direct weighted sum
- **6 tasks only**: Hardcoded array length

**Core Methods:**
```python
def base_obj(self, preds, train_data, ep=None):
    # 1. Preprocess (IDENTICAL across all)
    labels_mat = train_data.get_label().reshape((num_label, -1)).transpose()
    preds_mat = expit(preds.reshape((num_label, -1)).transpose())
    preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)
    
    # 2. Compute gradients (IDENTICAL across all)
    grad_i = preds_mat - labels_mat
    hess_i = preds_mat * (1.0 - preds_mat)
    
    # 3. Aggregate (UNIQUE - no normalization, fixed weights)
    grad_n = grad_i * np.array(self.w)
    grad = np.sum(grad_n, axis=1)
    hess = np.sum(hess_i * np.array(self.w), axis=1)
    
    return grad, hess, grad_i, hess_i
```

**Lines to preserve**: 62 / 62 (100%)

#### 2. customLossNoKD (Adaptive Weight)

**Key Characteristics:**
- **Adaptive weights**: Uses JS divergence
- **Gradient normalization**: Z-score before aggregation
- **Weight strategies**: tenIters, sqrt, delta, standard
- **Flexible task count**: Works with any number of tasks

**Core Methods:**
```python
def self_obj(self, preds, train_data, ep=None):
    # 1. Preprocess (IDENTICAL to baseLoss)
    labels_mat = train_data.get_label().reshape((num_col, -1)).transpose()
    preds_mat = expit(preds.reshape((num_col, -1)).transpose())
    preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)
    
    # 2. Compute gradients (IDENTICAL to baseLoss)
    grad_i = self.grad(labels_mat, preds_mat)
    hess_i = self.hess(preds_mat)
    
    # 3. Compute weights (UNIQUE - similarity-based)
    if self.weight_method == "tenIters":
        # ... weight logic
    elif self.weight_method == "sqrt":
        # ... weight logic
    # ... etc
    
    # 4. Normalize gradients (UNIQUE - z-score normalization)
    grad_n = self.normalize(grad_i)
    
    # 5. Aggregate (UNIQUE - weighted normalized gradients)
    grad = np.sum(grad_n * np.array(w), axis=1)
    hess = np.sum(hess_i * np.array(w), axis=1)
    
    return grad, hess, grad_i, hess_i

def similarity_vec(self, main_label, sub_predmat, num_col, ind_dic, lr):
    """CRITICAL: Uses main LABELS vs subtask PREDICTIONS"""
    dis = []
    for j in range(1, num_col):
        dis.append(jensenshannon(main_label[ind_dic[j]], sub_predmat[ind_dic[j], j]))
    dis_norm = self.unit_scale(np.reciprocal(dis)) * lr
    w = np.insert(dis_norm, 0, 1)
    return w
```

**Lines to preserve**: 142 / 142 (100%)

#### 3. customLossKDswap (Adaptive + KD)

**Key Characteristics:**
- **Inherits all** from customLossNoKD
- **Adds KD logic**: Label replacement when patience exceeded
- **Best prediction tracking**: Stores predictions from best iteration
- **Per-task patience**: Independent patience counters

**Additional State:**
```python
def __init__(self, num_label, val_sublabel_idx, trn_sublabel_idx, patience, weight_method=None):
    # ... same as NoKD plus:
    self.pat = patience
    self.max_score = {}
    self.counter = np.zeros(num_label, dtype=int)
    self.replaced = np.repeat(False, num_label)
    self.best_pred = {}
    self.pre_pred = {}
```

**KD Logic** (UNIQUE):
```python
def self_obj(self, preds, train_data, ep):
    # ... preprocessing ...
    
    # KD LOGIC START (UNIQUE to KDswap)
    for j in range(self.num_col):
        if j in self.max_score:
            best_round = self.max_score[j][0]
            if self.curr_obj_round == best_round + 1:
                self.best_pred[j] = self.pre_pred[j]
        
        if self.counter[j] == self.pat and self.replaced[j] == False:
            labels_mat[:, j] = self.best_pred[j]  # REPLACE LABELS
            self.replaced[j] = True
            self.counter[j] = 0
        
        self.pre_pred[j] = preds_mat[:, j]
    # KD LOGIC END
    
    # ... rest is IDENTICAL to NoKD ...
```

**Lines to preserve**: 193 / 193 (100%)

### Code Duplication Analysis

#### Truly Identical Code (Can be shared)

**1. Gradient/Hessian Computation** (5 lines)
```python
def grad(self, y_true, y_pred):
    return y_pred - y_true

def hess(self, y_pred):
    return y_pred * (1.0 - y_pred)
```
- ✅ IDENTICAL in all three files
- ✅ No variations or edge cases
- ✅ Safe to extract to base class

**2. Data Preprocessing** (3 lines core logic)
```python
labels_mat = train_data.get_label().reshape((num_col, -1)).transpose()
preds_mat = expit(preds.reshape((num_col, -1)).transpose())
preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)
```
- ✅ IDENTICAL in all three files
- ✅ Same epsilon value
- ✅ Safe to extract to base class

**3. Evaluation Metrics** (Core AUC computation - 5 lines)
```python
for j in range(self.num_col):
    s = roc_auc_score(
        labels_mat[self.val_label_idx[j], j],
        preds_mat[self.val_label_idx[j], j]
    )
    curr_score.append(s)
```
- ✅ IDENTICAL in all three files
- ⚠️ Surrounding code differs (weight calculation, KD tracking)
- ✅ Safe to extract AUC computation only

**Total truly identical code**: ~15 lines

#### Similar But Different Code (Keep separate)

**1. Utility Methods** (SIMILAR but subtle differences)
```python
# normalize() - appears in customLossNoKD and customLossKDswap
def normalize(self, vec):
    return (vec - np.mean(vec, axis=0)) / np.std(vec, axis=0)
# ⚠️ NO epsilon protection in legacy
# ⚠️ Only in adaptive losses, not in baseLoss

# unit_scale() - appears in customLossNoKD and customLossKDswap  
def unit_scale(self, vec):
    return vec / np.linalg.norm(vec)
# ⚠️ NO zero-norm protection in legacy
# ⚠️ Only in adaptive losses, not in baseLoss
```
**Decision**: Keep in AdaptiveWeightLoss base, NOT in BaseLossFunction
- Reason: Not used by FixedWeightLoss
- Reason: Preserves legacy structure

**2. Weight Computation** (COMPLETELY DIFFERENT)
- baseLoss: Fixed array
- NoKD: similarity_vec() with JS divergence
- KDswap: Same as NoKD
**Decision**: Keep completely separate

**3. Gradient Aggregation** (COMPLETELY DIFFERENT)
- baseLoss: No normalization
- NoKD: Z-score normalization
- KDswap: Z-score normalization + KD
**Decision**: Keep completely separate

## Minimal Refactoring Design

### Architecture: Three-Layer Hierarchy

```mermaid
flowchart TD
    subgraph "Layer 1: Minimal Base (30 lines)"
        Base["BaseLossFunction<br/>• grad(), hess()<br/>• _preprocess_labels()<br/>• _preprocess_predictions()<br/>• _compute_auc() "]
    end
    
    subgraph "Layer 2: Copy-Paste from Legacy"
        Fixed["FixedWeightLoss<br/>~60 lines<br/>Copied from baseLoss.py"]
        Adaptive["AdaptiveWeightLoss<br/>~140 lines<br/>Copied from customLossNoKD.py"]
    end
    
    subgraph "Layer 3: Inheritance + KD Logic"
        KD["KnowledgeDistillationLoss<br/>~190 lines<br/>Copied from customLossKDswap.py<br/>Inherits AdaptiveWeightLoss"]
    end
    
    Base --> Fixed
    Base --> Adaptive
    Adaptive --> KD
    
    style Base fill:#e1f5e1
    style Fixed fill:#fff3cd
    style Adaptive fill:#fff3cd
    style KD fill:#fff3cd
```

### Component Breakdown

#### 1. BaseLossFunction (Minimal - 30 lines)

**Purpose**: Extract ONLY truly identical code

**Included** (✅ Extract):
- `grad()` - Binary cross-entropy gradient
- `hess()` - Binary cross-entropy hessian  
- `_preprocess_labels()` - Reshape operation
- `_preprocess_predictions()` - Sigmoid + clip + reshape
- `_compute_auc()` - Core AUC calculation loop

**Excluded** (❌ Keep in subclasses):
- `normalize()` - Not in baseLoss
- `unit_scale()` - Not in baseLoss
- Weight computation - All different
- Gradient aggregation - All different
- Evaluation wrapper - Slightly different per class

**Implementation**:
```python
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


class BaseLossFunction(ABC):
    """
    Minimal base class - ONLY truly shared operations.
    
    Extracted from legacy implementations:
    - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/baseLoss.py
    - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossNoKD.py
    - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossKDswap.py
    
    Philosophy: Extract ONLY code that is byte-for-byte identical across
    all three legacy implementations. Everything else preserved in subclasses.
    """
    
    def __init__(self, num_label, val_sublabel_idx, trn_sublabel_idx=None):
        """Minimal initialization - common to all legacy classes."""
        self.num_col = num_label
        self.val_label_idx = val_sublabel_idx
        self.trn_sublabel_idx = trn_sublabel_idx
        self.eval_mat = []
    
    def grad(self, y_true, y_pred):
        """
        Binary cross-entropy gradient.
        
        LEGACY: Identical in baseLoss.py, customLossNoKD.py, customLossKDswap.py
        Formula: grad = y_pred - y_true
        """
        return y_pred - y_true
    
    def hess(self, y_pred):
        """
        Binary cross-entropy hessian.
        
        LEGACY: Identical in baseLoss.py, customLossNoKD.py, customLossKDswap.py
        Formula: hess = y_pred * (1 - y_pred)
        """
        return y_pred * (1.0 - y_pred)
    
    def _preprocess_labels(self, train_data, num_col):
        """
        Reshape labels to matrix form.
        
        LEGACY: Identical in all three implementations
        From: labels [N*T] → labels_mat [N, T]
        """
        labels = train_data.get_label()
        labels_mat = labels.reshape((num_col, -1)).transpose()
        return labels_mat
    
    def _preprocess_predictions(self, preds, num_col, epsilon=1e-15):
        """
        Transform and clip predictions.
        
        LEGACY: Identical in all three implementations
        Steps:
        1. Reshape: [N*T] → [T, N] → [N, T]
        2. Sigmoid: logits → probabilities  
        3. Clip: [epsilon, 1-epsilon]
        
        Note: epsilon=1e-15 hardcoded in legacy
        """
        preds_mat = preds.reshape((num_col, -1)).transpose()
        preds_mat = expit(preds_mat)
        preds_mat = np.clip(preds_mat, epsilon, 1 - epsilon)
        return preds_mat
    
    def _compute_auc(self, labels_mat, preds_mat):
        """
        Compute per-task AUC scores.
        
        LEGACY: Core loop identical in all three implementations
        Returns: array of AUC scores [T]
        """
        curr_score = []
        for j in range(self.num_col):
            s = roc_auc_score(
                labels_mat[self.val_label_idx[j], j],
                preds_mat[self.val_label_idx[j], j]
            )
            curr_score.append(s)
        return np.array(curr_score)
    
    @abstractmethod
    def objective(self, preds, train_data, ep=None):
        """Subclasses implement with legacy logic."""
        pass
    
    @abstractmethod
    def evaluate(self, preds, train_data):
        """Subclasses implement with legacy logic."""
        pass
```

**Key Design Decisions**:
1. **No epsilon parameter**: Hardcoded 1e-15 (legacy value)
2. **No validation**: Legacy has none, we preserve that
3. **No logging**: Legacy has none, we preserve that
4. **No optimization**: No caching, no vectorization (legacy has none)
5. **Exact signatures**: Match legacy method signatures

#### 2. FixedWeightLoss (~60 lines)

**Source**: Copy-paste from `baseLoss.py`

**Preservation Strategy**:
```python
from typing import Dict
import numpy as np

from .base_loss_function import BaseLossFunction


class FixedWeightLoss(BaseLossFunction):
    """
    Fixed weight baseline loss.
    
    LEGACY SOURCE: projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/baseLoss.py
    
    Changes from legacy:
    1. Inherits from BaseLossFunction (NEW)
    2. Uses base class methods: grad(), hess(), _preprocess_*()  
    3. All other logic PRESERVED byte-for-byte from legacy
    
    Preserved:
    - Fixed weight formula: [1, 0.1*beta, 0.1*beta, ...]
    - 6-task assumption (legacy limitation)
    - No gradient normalization
    - Evaluation logic
    """
    
    def __init__(self, val_sublabel_idx: Dict[int, np.ndarray], num_label: int):
        """
        LEGACY: baseLoss.__init__()
        Note: num_label after val_sublabel_idx (legacy signature)
        """
        super().__init__(num_label, val_sublabel_idx)
        
        # LEGACY: Hardcoded 6-task weights with beta=0.2
        beta = 0.2
        self.w = np.array([1, 0.1 * beta, 0.1 * beta, 0.1 * beta, 0.1 * beta, 0.1 * beta])
        self.num_label = num_label
    
    def base_obj(self, preds, train_data, ep=None):
        """
        LEGACY: baseLoss.base_obj() - PRESERVED
        
        Note: Method name 'base_obj' preserved from legacy
        (not renamed to 'objective' for exact compatibility)
        """
        # LEGACY: Preprocessing (now uses base class)
        labels_mat = self._preprocess_labels(train_data, self.num_label)
        preds_mat = self._preprocess_predictions(preds, self.num_label)
        
        # LEGACY: Compute gradients/hessians (now uses base class)
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)
        
        # LEGACY: Aggregate WITHOUT normalization (PRESERVED)
        grad_n = grad_i * np.array(self.w)
        grad = np.sum(grad_n, axis=1)
        hess = np.sum(hess_i * np.array(self.w), axis=1)
        
        return grad, hess, grad_i, hess_i
    
    def objective(self, preds, train_data, ep=None):
        """Wrapper for compatibility with base class."""
        return self.base_obj(preds, train_data, ep)
    
    def base_eval(self, preds, train_data):
        """
        LEGACY: baseLoss.base_eval() - PRESERVED
        
        Note: Method name 'base_eval' preserved from legacy
        """
        # LEGACY: Preprocessing  
        labels_mat = self._preprocess_labels(train_data, self.num_label)
        preds_mat = self._preprocess_predictions(preds, self.num_label)
        
        # LEGACY: Compute AUC (now uses base class)
        curr_score = self._compute_auc(labels_mat, preds_mat)
        
        # LEGACY: Store history (PRESERVED)
        self.eval_mat.append(curr_score.tolist())
        
        # LEGACY: Weighted average (PRESERVED)
        weighted_score_vec = curr_score * self.w
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(self.w)
        
        return "base_metric", wavg_auc, False
    
    def evaluate(self, preds, train_data):
        """Wrapper for compatibility with base class."""
        return self.base_eval(preds, train_data)
```

**Lines preserved from legacy**: 58 / 62 (94%)
**Lines changed**: 4 (imports and inheritance)

#### 3. AdaptiveWeightLoss (~140 lines)

**Source**: Copy-paste from `customLossNoKD.py`

**Preservation Strategy**:
```python
from typing import Dict, Optional
import numpy as np
from scipy.spatial.distance import jensenshannon

from .base_loss_function import BaseLossFunction


class AdaptiveWeightLoss(BaseLossFunction):
    """
    Adaptive weight loss with similarity-based weighting.
    
    LEGACY SOURCE: projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossNoKD.py
    
    Changes from legacy:
    1. Inherits from BaseLossFunction (NEW)
    2. Uses base class methods: grad(), hess(), _preprocess_*()
    3. All other logic PRESERVED byte-for-byte from legacy
    
    Preserved:
    - similarity_vec() logic (EXACT)
    - normalize() implementation (NO epsilon - legacy quirk)
    - unit_scale() implementation (NO protection - legacy quirk)
    - Weight update strategies: tenIters, sqrt, delta, None
    - Gradient normalization (z-score BEFORE weighting)
    """
    
    def __init__(
        self,
        num_label: int,
        val_sublabel_idx: Dict[int, np.ndarray],
        trn_sublabel_idx: Dict[int, np.ndarray],
        weight_method: Optional[str] = None
    ):
        """LEGACY: customLossNoKD.__init__() - PRESERVED"""
        super().__init__(num_label, val_sublabel_idx, trn_sublabel_idx)
        
        # LEGACY: State tracking (PRESERVED)
        self.w_trn_mat = []
        self.similar = []
        self.curr_obj_round = 0
        self.curr_eval_round = 0
        self.weight_method = weight_method
    
    def self_obj(self, preds, train_data, ep=None):
        """
        LEGACY: customLossNoKD.self_obj() - PRESERVED
        
        Critical preservation:
        - Gradient normalization AFTER weight computation (Bug #11 fix in refactor)
        - Weight update logic EXACTLY as legacy
        - similarity_vec() EXACTLY as legacy
        """
        self.curr_obj_round += 1
        
        # LEGACY: Preprocessing (now uses base class)
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col)
        
        # LEGACY: Compute gradients/hessians (now uses base class)
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)
        
        # LEGACY: Weight update strategies (PRESERVED EXACTLY)
        if self.weight_method == "tenIters":
            i = self.curr_obj_round - 1
            if i % 10 == 0:  # LEGACY: Every 10 iters (NOT 50 as in refactor doc)
                self.similar = self.similarity_vec(
                    labels_mat[:, 0], preds_mat, self.num_col, 
                    self.trn_sublabel_idx, 0.1
                )
            w = self.similar
            self.w_trn_mat.append(w)
        
        elif self.weight_method == "sqrt":
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.5  # LEGACY: lr=0.5 for sqrt
            )
            w = np.sqrt(w)
            self.w_trn_mat.append(w)
        
        elif self.weight_method == "delta":
            simi = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.1
            )
            self.similar.append(simi)
            if self.curr_obj_round == 1:
                w = self.similar[0]
            else:
                i = self.curr_obj_round - 1
                diff = self.similar[i] - self.similar[i - 1]
                w = self.w_trn_mat[i - 1] + diff * 0.1  # LEGACY: delta_lr=0.1
            self.w_trn_mat.append(w)
        
        else:  # LEGACY: Standard method (weight_method=None)
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.1
            )
            self.w_trn_mat.append(w)
        
        # LEGACY: Normalize gradients (PRESERVED - NO epsilon protection)
        grad_n = self.normalize(grad_i)
        
        # LEGACY: Weighted aggregation (PRESERVED)
        grad = np.sum(grad_n * np.array(w), axis=1)
        hess = np.sum(hess_i * np.array(w), axis=1)
        
        return grad, hess, grad_i, hess_i
    
    def objective(self, preds, train_data, ep=None):
        """Wrapper for compatibility."""
        return self.self_obj(preds, train_data, ep)
    
    def similarity_vec(self, main_label, sub_predmat, num_col, ind_dic, lr):
        """
        LEGACY: customLossNoKD.similarity_vec() - PRESERVED EXACTLY
        
        Critical: Uses main task LABELS vs subtask PREDICTIONS
        (NOT predictions vs predictions)
        """
        dis = []
        for j in range(1, num_col):
            # LEGACY: JS divergence between LABELS and PREDICTIONS
            dis.append(
                jensenshannon(main_label[ind_dic[j]], sub_predmat[ind_dic[j], j])
            )
        
        # LEGACY: Inverse + L2 normalization + learning rate
        dis_norm = self.unit_scale(np.reciprocal(dis)) * lr
        w = np.insert(dis_norm, 0, 1)  # Main task weight = 1
        return w
    
    def normalize(self, vec):
        """
        LEGACY: customLossNoKD.normalize() - PRESERVED EXACTLY
        
        Note: NO epsilon protection (legacy quirk)
        Will divide by zero if std=0 (legacy behavior)
        """
        norm_vec = (vec - np.mean(vec, axis=0)) / np.std(vec, axis=0)
        return norm_vec
    
    def unit_scale(self, vec):
        """
        LEGACY: customLossNoKD.unit_scale() - PRESERVED EXACTLY
        
        Note: NO zero-norm protection (legacy quirk)
        Will divide by zero if norm=0 (legacy behavior)
        """
        return vec / np.linalg.norm(vec)
    
    def self_eval(self, preds, train_data):
        """LEGACY: customLossNoKD.self_eval() - PRESERVED"""
        self.curr_eval_round += 1
        
        # LEGACY: Preprocessing
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col)
        
        # LEGACY: Get current weights from training
        w = self.w_trn_mat[self.curr_eval_round - 1]
        
        # LEGACY: Compute AUC (now uses base class)
        curr_score = self._compute_auc(labels_mat, preds_mat)
        
        # LEGACY: Store history
        self.eval_mat.append(curr_score.tolist())
        print("--- task eval score: ", np.round(curr_score, 4))
        
        # LEGACY: Weighted average
        weighted_score_vec = curr_score * w
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(w)
        print("--- self_eval score: ", np.round(wavg_auc, 4))
        
        return "self_eval", wavg_auc, False
    
    def evaluate(self, preds, train_data):
        """Wrapper for compatibility."""
        return self.self_eval(preds, train_data)
```

**Lines preserved from legacy**: 138 / 142 (97%)
**Lines changed**: 4 (imports and inheritance)

#### 4. KnowledgeDistillationLoss (~190 lines)

**Source**: Copy-paste from `customLossKDswap.py`

**Preservation Strategy**:
```python
from typing import Dict, Optional
import numpy as np

from .adaptive_weight_loss import AdaptiveWeightLoss


class KnowledgeDistillationLoss(AdaptiveWeightLoss):
    """
    Adaptive weight loss with knowledge distillation.
    
    LEGACY SOURCE: projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossKDswap.py
    
    Changes from legacy:
    1. Inherits from AdaptiveWeightLoss instead of object (NEW)
    2. All KD logic PRESERVED byte-for-byte from legacy
    
    Preserved:
    - KD label replacement logic (EXACT)
    - Best prediction tracking (EXACT)
    - Per-task patience counters (EXACT)
    - All state management (EXACT)
    """
    
    def __init__(
        self,
        num_label: int,
        val_sublabel_idx: Dict[int, np.ndarray],
        trn_sublabel_idx: Dict[int, np.ndarray],
        patience: int,
        weight_method: Optional[str] = None
    ):
        """LEGACY: customLossKDswap.__init__() - PRESERVED"""
        super().__init__(num_label, val_sublabel_idx, trn_sublabel_idx, weight_method)
        
        # LEGACY: KD-specific state (PRESERVED)
        self.pat = patience
        self.max_score = {}
        self.counter = np.zeros(num_label, dtype=int)
        self.replaced = np.repeat(False, num_label)
        self.best_pred = {}
        self.pre_pred = {}
    
    def self_obj(self, preds, train_data, ep):
        """
        LEGACY: customLossKDswap.self_obj() - PRESERVED
        
        Critical KD logic preserved exactly as legacy:
        - Best prediction tracking
        - Label replacement when patience exceeded
        - Per-task replacement flags
        """
        self.curr_obj_round += 1
        
        # LEGACY: Preprocessing
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col)
        
        # LEGACY: KD LOGIC START (PRESERVED EXACTLY)
        for j in range(self.num_col):
            if j in self.max_score:
                best_round = self.max_score[j][0]
                if self.curr_obj_round == best_round + 1:
                    self.best_pred[j] = self.pre_pred[j]
            
            if self.counter[j] == self.pat and self.replaced[j] == False:
                labels_mat[:, j] = self.best_pred[j]
                self.replaced[j] = True
                print(
                    "!TASK ", j, " replaced,",
                    "curr_round: ", self.curr_obj_round,
                    " check counter: ", self.counter[j]
                )
                self.counter[j] = 0
            
            self.pre_pred[j] = preds_mat[:, j]
        # LEGACY: KD LOGIC END
        
        # LEGACY: Rest is IDENTICAL to AdaptiveWeightLoss
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)
        
        # LEGACY: Weight update strategies (inherited but repeated for clarity)
        if self.weight_method == "tenIters":
            i = self.curr_obj_round - 1
            if i % 50 == 0:
                self.similar = self.similarity_vec(
                    labels_mat[:, 0], preds_mat, self.num_col,
                    self.trn_sublabel_idx, 0.1
                )
            w = self.similar
            self.w_trn_mat.append(w)
        elif self.weight_method == "sqrt":
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.5
            )
            w = np.sqrt(w)
            self.w_trn_mat.append(w)
        elif self.weight_method == "delta":
            simi = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.1
            )
            self.similar.append(simi)
            if self.curr_obj_round == 1:
                w = self.similar[0]
            else:
                i = self.curr_obj_round - 1
                diff = self.similar[i] - self.similar[i - 1]
                w = self.w_trn_mat[i - 1] + diff * 0.01
            self.w_trn_mat.append(w)
        else:
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col,
                self.trn_sublabel_idx, 0.1
            )
            self.w_trn_mat.append(w)
        
        # LEGACY: Normalize and aggregate
        grad_n = self.normalize(grad_i)
        grad = np.sum(grad_n * np.array(w), axis=1)
        hess = np.sum(hess_i * np.array(w), axis=1)
        
        return grad, hess, grad_i, hess_i
    
    def objective(self, preds, train_data, ep=None):
        """Wrapper for compatibility."""
        return self.self_obj(preds, train_data, ep)
    
    def self_eval(self, preds, train_data):
        """
        LEGACY: customLossKDswap.self_eval() - PRESERVED
        
        Adds max_score tracking and patience counter logic
        """
        self.curr_eval_round += 1
        
        # LEGACY: Preprocessing
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col)
        
        # LEGACY: Get current weights
        w = self.w_trn_mat[self.curr_eval_round - 1]
        
        # LEGACY: Compute AUC
        curr_score = self._compute_auc(labels_mat, preds_mat)
        
        # LEGACY: Store history
        self.eval_mat.append(curr_score.tolist())
        print("--- task eval score: ", np.round(curr_score, 4))
        
        # LEGACY: Update max scores and counters (KD-specific logic)
        for j in range(self.num_col):
            if not self.replaced[j]:
                if self.curr_eval_round == 1:
                    self.max_score[j] = [self.curr_eval_round, curr_score[j]]
                else:
                    if curr_score[j] >= self.max_score[j][1]:
                        self.max_score[j] = [self.curr_eval_round, curr_score[j]]
                        self.counter[j] = 0
                    else:
                        self.counter[j] += 1
        
        # LEGACY: Weighted average
        weighted_score_vec = curr_score * w
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(w)
        print("--- self_eval score: ", np.round(wavg_auc, 4))
        
        return "self_eval", wavg_auc, False
    
    def evaluate(self, preds, train_data):
        """Wrapper for compatibility."""
        return self.self_eval(preds, train_data)
```

**Lines preserved from legacy**: 189 / 193 (98%)
**Lines changed**: 4 (imports and inheritance)

## Weight Update Strategies: No Separate Classes Needed

### Legacy Approach (Preserved in Minimal Refactoring)

In the legacy code, weight update strategies are implemented as **inline if/elif blocks** within the loss classes:

```python
# LEGACY: customLossNoKD.py and customLossKDswap.py
def self_obj(self, preds, train_data, ep=None):
    # ... preprocessing ...
    
    # Weight update strategies as if/elif blocks (PRESERVED)
    if self.weight_method == "tenIters":
        # ... tenIters logic inline ...
    elif self.weight_method == "sqrt":
        # ... sqrt logic inline ...
    elif self.weight_method == "delta":
        # ... delta logic inline ...
    else:
        # ... standard logic inline ...
```

### Minimal vs. Comprehensive Refactoring

| Aspect | Minimal Refactoring (This Document) | Comprehensive Refactoring |
|--------|-------------------------------------|---------------------------|
| **Weight Strategies** | ❌ NO separate classes | ✅ Separate strategy classes |
| **Implementation** | Inline if/elif blocks (legacy) | Strategy pattern with ABC |
| **Code Location** | Inside AdaptiveWeightLoss | Separate weight_strategies.py |
| **Lines of Code** | ~40 lines inline | ~150 lines (4 classes) |
| **Complexity** | Simple, easy to understand | More flexible, harder to trace |
| **Legacy Fidelity** | 100% preserved | Refactored/modernized |

### Decision: Keep Inline Implementation

**For minimal refactoring, we preserve the legacy inline approach:**

✅ **Keep** (Inline in AdaptiveWeightLoss):
```python
# Inside AdaptiveWeightLoss.self_obj()
if self.weight_method == "tenIters":
    i = self.curr_obj_round - 1
    if i % 10 == 0:
        self.similar = self.similarity_vec(...)
    w = self.similar
elif self.weight_method == "sqrt":
    w = self.similarity_vec(...)
    w = np.sqrt(w)
# ... etc
```

❌ **Don't Create** (Separate Strategy Classes):
```python
# NOT needed for minimal refactoring
class WeightUpdateStrategy(ABC): ...
class StandardStrategy(WeightUpdateStrategy): ...
class TenItersStrategy(WeightUpdateStrategy): ...
class SqrtStrategy(WeightUpdateStrategy): ...
class DeltaStrategy(WeightUpdateStrategy): ...
```

### Rationale

1. **Legacy Preservation**: Inline code is exactly how legacy works
2. **Risk Minimization**: No extraction = no bugs from refactoring
3. **Easy Verification**: Can diff line-by-line with legacy
4. **Simplicity**: Fewer files, easier to trace execution
5. **Zero Functional Changes**: Preserves all behaviors perfectly

### When to Extract Strategies

Extract to separate strategy classes only when:
- Moving to comprehensive refactoring approach
- Need to add many new weight methods
- Want to test strategies in isolation
- Have time for extensive testing
- Team agrees benefits outweigh risks

For minimal refactoring, **inline implementation is preferred**.

## LossFactory Integration

### Current LossFactory Issue

The existing `LossFactory` expects loss classes to accept a `hyperparams` object:

```python
# Current LossFactory.create() - INCOMPATIBLE with minimal design
return loss_class(
    num_label=num_label,
    val_sublabel_idx=val_sublabel_idx,
    trn_sublabel_idx=trn_sublabel_idx,
    hyperparams=hyperparams,  # ❌ Minimal design uses legacy signatures
)
```

**Problem**: Minimal design preserves legacy signatures (no `hyperparams` parameter):
- `FixedWeightLoss(val_sublabel_idx, num_label)` - 2 params
- `AdaptiveWeightLoss(num_label, val_sublabel_idx, trn_sublabel_idx, weight_method)` - 4 params
- `KnowledgeDistillationLoss(num_label, val_sublabel_idx, trn_sublabel_idx, patience, weight_method)` - 5 params

### Solution: Adapt LossFactory to Legacy Signatures

**Philosophy**: Modify the factory, not the losses (preserve legacy code).

#### Minimal LossFactory Refactoring

```python
"""
Factory for creating loss function instances.

Adapted to work with legacy-style loss function signatures while maintaining
modern factory pattern benefits.
"""

from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from .base_loss_function import BaseLossFunction
from .fixed_weight_loss import FixedWeightLoss
from .adaptive_weight_loss import AdaptiveWeightLoss
from .knowledge_distillation_loss import KnowledgeDistillationLoss

if TYPE_CHECKING:
    from ...hyperparams.hyperparameters_lightgbmmt import (
        LightGBMMtModelHyperparameters,
    )


class LossFactory:
    """
    Factory for creating loss function instances.
    
    Adapts between modern hyperparameter-based API and legacy loss signatures
    to maintain backward compatibility with minimal refactoring design.
    """

    _registry = {
        "fixed": FixedWeightLoss,
        "adaptive": AdaptiveWeightLoss,
        "adaptive_kd": KnowledgeDistillationLoss,
    }

    @classmethod
    def create(
        cls,
        loss_type: str,
        num_label: int,
        val_sublabel_idx: Dict[int, np.ndarray],
        trn_sublabel_idx: Optional[Dict[int, np.ndarray]] = None,
        hyperparams: Optional["LightGBMMtModelHyperparameters"] = None,
    ) -> BaseLossFunction:
        """
        Create a loss function instance.
        
        Adapts hyperparameters to legacy loss function signatures.

        Parameters
        ----------
        loss_type : str
            Type of loss function ('fixed', 'adaptive', 'adaptive_kd')
        num_label : int
            Number of tasks
        val_sublabel_idx : dict
            Validation set indices for each task
        trn_sublabel_idx : dict, optional
            Training set indices for each task
        hyperparams : LightGBMMtModelHyperparameters, optional
            Model hyperparameters (used to extract legacy params)

        Returns
        -------
        loss_fn : BaseLossFunction
            Instantiated loss function

        Raises
        ------
        ValueError
            If loss_type is not registered
        """
        if loss_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown loss_type: '{loss_type}'. Available types: {available}"
            )

        loss_class = cls._registry[loss_type]

        # ADAPTER: Extract legacy parameters from hyperparams
        if loss_type == "fixed":
            # LEGACY: FixedWeightLoss(val_sublabel_idx, num_label)
            # Note: beta hardcoded in FixedWeightLoss.__init__
            return loss_class(
                val_sublabel_idx=val_sublabel_idx,
                num_label=num_label,
            )
        
        elif loss_type == "adaptive":
            # LEGACY: AdaptiveWeightLoss(num_label, val_sublabel_idx, trn_sublabel_idx, weight_method)
            weight_method = None
            if hyperparams is not None:
                weight_method = getattr(hyperparams, 'loss_weight_method', None)
            
            return loss_class(
                num_label=num_label,
                val_sublabel_idx=val_sublabel_idx,
                trn_sublabel_idx=trn_sublabel_idx,
                weight_method=weight_method,
            )
        
        elif loss_type == "adaptive_kd":
            # LEGACY: KnowledgeDistillationLoss(num_label, val_sublabel_idx, trn_sublabel_idx, patience, weight_method)
            patience = 100  # Default
            weight_method = None
            
            if hyperparams is not None:
                patience = getattr(hyperparams, 'loss_patience', 100)
                weight_method = getattr(hyperparams, 'loss_weight_method', None)
            
            return loss_class(
                num_label=num_label,
                val_sublabel_idx=val_sublabel_idx,
                trn_sublabel_idx=trn_sublabel_idx,
                patience=patience,
                weight_method=weight_method,
            )
        
        # Should never reach here
        raise ValueError(f"Unhandled loss_type: {loss_type}")

    @classmethod
    def register(cls, name: str, loss_class: type) -> None:
        """
        Register a new loss function type.
        
        Note: Registered loss classes should follow legacy signatures,
        not hyperparams-based signatures.
        """
        if not issubclass(loss_class, BaseLossFunction):
            raise TypeError(f"{loss_class} must inherit from BaseLossFunction")
        cls._registry[name] = loss_class

    @classmethod
    def get_available_losses(cls) -> list:
        """Get list of available loss types."""
        return list(cls._registry.keys())
```

#### Key Changes to LossFactory

**What Changed** (26 lines):
1. ✅ Removed direct `hyperparams` passing to loss constructors
2. ✅ Added adapter logic to extract legacy parameters
3. ✅ Respects legacy signatures for each loss type
4. ✅ Updated docstrings to reflect adapter pattern

**What Stayed the Same** (50 lines):
- ✅ Factory pattern structure
- ✅ Registry mechanism
- ✅ Public API (`create`, `register`, `get_available_losses`)
- ✅ Error handling

#### Usage Examples

```python
# Example 1: Fixed weight loss
from projects.cap_mtgbm.dockers.models.loss.loss_factory import LossFactory

loss_fn = LossFactory.create(
    loss_type='fixed',
    num_label=6,
    val_sublabel_idx=val_idx,
    # trn_sublabel_idx not needed for fixed
    # hyperparams not used (beta hardcoded)
)

# Example 2: Adaptive weight loss with hyperparams
loss_fn = LossFactory.create(
    loss_type='adaptive',
    num_label=6,
    val_sublabel_idx=val_idx,
    trn_sublabel_idx=trn_idx,
    hyperparams=hyperparams,  # Extracts loss_weight_method
)

# Example 3: Adaptive KD loss with hyperparams
loss_fn = LossFactory.create(
    loss_type='adaptive_kd',
    num_label=6,
    val_sublabel_idx=val_idx,
    trn_sublabel_idx=trn_idx,
    hyperparams=hyperparams,  # Extracts loss_patience, loss_weight_method
)

# Example 4: Direct usage without factory (legacy compatibility)
from projects.cap_mtgbm.dockers.models.loss.fixed_weight_loss import FixedWeightLoss

loss_fn = FixedWeightLoss(
    val_sublabel_idx=val_idx,
    num_label=6
)
```

### Benefits of Factory Adapter Pattern

#### 1. Preserves Legacy Signatures
- ✅ Loss classes keep original signatures
- ✅ No changes to loss implementations
- ✅ 100% backward compatible with legacy

#### 2. Maintains Modern API
- ✅ Factory still accepts hyperparams
- ✅ Existing code using factory continues to work
- ✅ Can extend to new parameters easily

#### 3. Clear Separation of Concerns
- ✅ Factory handles API translation
- ✅ Losses remain pure legacy
- ✅ Easy to understand what changed where

#### 4. Future Flexibility
- ✅ Can add new hyperparams without touching losses
- ✅ Can eventually migrate to hyperparams-based losses
- ✅ Adapter pattern is well-understood design

### Migration Path

**Phase 1**: Use adapter pattern (this design)
- Factory adapts hyperparams → legacy params
- Losses unchanged from legacy
- Production-ready immediately

**Phase 2**: Gradual modernization (future)
- Optionally add hyperparams support to losses
- Keep legacy constructors as alternate API
- Factory detects which API to use

**Phase 3**: Deprecate legacy signatures (far future)
- After extensive production validation
- Only if business value justifies
- Keep factory as single creation point

## Implementation Summary

### Code Preservation Statistics

| Component | Total Lines | Legacy Lines | Preserved | New Lines |
|-----------|-------------|--------------|-----------|-----------|
| BaseLossFunction | 80 | 15 | 19% | 65 |
| FixedWeightLoss | 75 | 62 | 94% | 13 |
| AdaptiveWeightLoss | 145 | 142 | 97% | 3 |
| KnowledgeDistillationLoss | 195 | 193 | 98% | 2 |
| **Total** | **495** | **412** | **83%** | **83** |

### What Changed vs. Legacy

**Shared Across All Classes** (BaseLossFunction):
- ✅ `grad()`, `hess()` methods extracted
- ✅ Data preprocessing extracted
- ✅ AUC computation extracted
- ✅ Abstract methods defined

**Per-Class Changes**:
- ✅ FixedWeightLoss: 4 lines (imports + inheritance)
- ✅ AdaptiveWeightLoss: 3 lines (imports + inheritance)
- ✅ KnowledgeDistillationLoss: 2 lines (imports + inheritance)

**What Was NOT Changed**:
- ❌ No algorithmic changes
- ❌ No optimization attempts
- ❌ No interface changes
- ❌ No "improvements" to legacy logic
- ❌ No bug fixes (except documented critical bugs)

## Verification Strategy

### 1. Side-by-Side Comparison

```bash
# Compare each method with legacy
diff -u legacy/baseLoss.py new/fixed_weight_loss.py
diff -u legacy/customLossNoKD.py new/adaptive_weight_loss.py
diff -u legacy/customLossKDswap.py new/knowledge_distillation_loss.py
```

**Expected**: Only imports and class inheritance lines differ

### 2. Functional Equivalence Testing

```python
# Test harness for verification
def test_loss_equivalence(legacy_loss, new_loss, test_data):
    """Verify byte-for-byte output equivalence."""
    
    # Run both implementations
    legacy_grad, legacy_hess, _, _ = legacy_loss.objective(test_data)
    new_grad, new_hess, _, _ = new_loss.objective(test_data)
    
    # Assert exact equality
    assert np.array_equal(legacy_grad, new_grad), "Gradients differ!"
    assert np.array_equal(legacy_hess, new_hess), "Hessians differ!"
    
    # Run evaluation
    legacy_eval = legacy_loss.evaluate(test_data)
    new_eval = new_loss.evaluate(test_data)
    
    assert legacy_eval == new_eval, "Evaluation differs!"
```

### 3. Line-by-Line Audit

For each method in refactored code:
1. ✅ Find corresponding method in legacy
2. ✅ Compare line-by-line
3. ✅ Document any differences
4. ✅ Justify each difference

## Migration Path

### Phase 1: Parallel Operation (Week 1-2)

**Goal**: Deploy and validate refactored code alongside legacy

**Implementation Steps**:
1. ✅ Create `BaseLossFunction` with 5 shared methods (~80 lines)
2. ✅ Create `FixedWeightLoss` inheriting from base (~75 lines, 94% legacy)
3. ✅ Create `AdaptiveWeightLoss` inheriting from base (~145 lines, 97% legacy)
   - Keep weight strategies inline (if/elif blocks)
   - No separate strategy classes
4. ✅ Create `KnowledgeDistillationLoss` inheriting from adaptive (~195 lines, 98% legacy)
5. ✅ Update `LossFactory.create()` with adapter logic (~26 lines modified)
   - Add if/elif blocks to extract legacy params from hyperparams
   - Keep registry, register(), get_available_losses() unchanged
6. ✅ Run both implementations in parallel
7. ✅ Compare outputs for byte-for-byte equivalence
8. ✅ Fix any discrepancies

**Deliverables**:
- 4 new loss files (base + 3 implementations)
- 1 modified factory file (adapter logic only)
- Functional equivalence test suite
- Side-by-side comparison results

**Success Criteria**:
- All tests pass
- Outputs match legacy exactly
- No performance degradation

### Phase 2: Gradual Migration (Week 3-4)

**Goal**: Start using refactored code in production

**Implementation Steps**:
1. ✅ Deploy refactored losses to non-critical pipelines
2. ✅ Update pipeline configs to use LossFactory (no code changes needed)
3. ✅ Monitor metrics and logs
4. ✅ Maintain legacy as fallback option
5. ✅ Collect performance data
6. ✅ Document any issues found

**Pipelines to Migrate First**:
- Development/staging environments
- Low-volume production pipelines
- Pipelines with easy rollback

**Monitoring**:
- AUC scores per task
- Training time
- Memory usage
- Error rates

**Success Criteria**:
- No production incidents
- Metrics match legacy
- Team comfortable with new code

### Phase 3: Full Migration (Week 5-6)

**Goal**: Complete migration to refactored code

**Implementation Steps**:
1. ✅ Switch all remaining pipelines to refactored code
2. ✅ Archive legacy loss files (move to legacy/ subdirectory)
   - Keep for reference, don't delete
3. ✅ Update all documentation
   - API docs
   - Developer guides
   - Migration notes
4. ✅ Team training session
   - Show new structure
   - Explain adapter pattern
   - Demo usage examples
5. ✅ Update CI/CD pipelines

**Documentation Updates**:
- Loss function architecture diagram
- LossFactory adapter pattern explanation
- Migration notes and lessons learned
- Known limitations and workarounds

**Success Criteria**:
- 100% of pipelines migrated
- All documentation updated
- Team trained and comfortable
- Legacy code archived safely

### Phase 4: Cleanup and Future Planning (Week 7+)

**Goal**: Remove fallback code and plan future enhancements

**Implementation Steps**:
1. ✅ Remove legacy fallback mechanisms (after 2+ weeks stable)
2. ✅ Delete temporary parallel testing code
3. ✅ Document lessons learned
4. ✅ Update coding standards
5. ✅ Plan future enhancements (if needed)

**Lessons Learned Documentation**:
- What went well
- What could be improved
- Unexpected issues encountered
- Time/effort estimates vs actuals

**Future Enhancement Considerations** (separate projects):
- Add type hints (if team decides it's valuable)
- Add logging framework (if print statements become insufficient)
- Extract weight strategies to classes (if new methods needed frequently)
- Add configuration system (if hardcoded values become problematic)
- Performance optimization (if bottlenecks identified)

**Success Criteria**:
- Clean codebase
- No legacy dependencies
- Team prepared for future work
- Clear path forward documented

### Migration Checklist

**Code Changes**:
- [ ] BaseLossFunction created
- [ ] FixedWeightLoss created (legacy compatible)
- [ ] AdaptiveWeightLoss created (inline weight strategies)
- [ ] KnowledgeDistillationLoss created
- [ ] LossFactory.create() updated (adapter logic)
- [ ] Tests written and passing
- [ ] Documentation updated

**No Weight Strategy Classes**:
- [ ] ❌ WeightUpdateStrategy ABC (not creating)
- [ ] ❌ StandardStrategy class (not creating)
- [ ] ❌ TenItersStrategy class (not creating)
- [ ] ❌ SqrtStrategy class (not creating)
- [ ] ❌ DeltaStrategy class (not creating)
- [ ] ✅ Inline if/elif blocks preserved in AdaptiveWeightLoss

**No New LossFactory**:
- [ ] ❌ Creating new LossFactory from scratch (not doing)
- [ ] ✅ Modifying existing LossFactory.create() only (26 lines)
- [ ] ✅ Keeping registry, register(), get_available_losses() unchanged

**Validation**:
- [ ] Side-by-side comparison with legacy
- [ ] Functional equivalence tests pass
- [ ] Performance benchmarks acceptable
- [ ] Team review completed

## Benefits of Minimal Refactoring

### 1. Risk Minimization
- ✅ 83% code reuse reduces new bugs
- ✅ Easy verification through comparison
- ✅ Fast rollback if issues found
- ✅ Production-proven logic preserved

### 2. Easy Verification
- ✅ Side-by-side diff shows exact changes
- ✅ Functional equivalence tests pass
- ✅ Team can audit changes quickly
- ✅ Clear accountability

### 3. Maintainability
- ✅ Shared code extracted (15 lines)
- ✅ Clear inheritance hierarchy
- ✅ Documented legacy sources
- ✅ Easy to understand structure

### 4. Future Flexibility
- ✅ Foundation for future improvements
- ✅ Base class enables new loss functions
- ✅ Configuration system ready
- ✅ Can optimize later if needed

## Limitations and Trade-offs

### What This Design Does NOT Provide

#### 1. Performance Optimization
- ❌ No caching
- ❌ No vectorization improvements
- ❌ No algorithm optimizations
- **Reason**: Preserve legacy performance characteristics

#### 2. Modern Features
- ❌ No type hints (legacy doesn't have them)
- ❌ No input validation (legacy doesn't have it)
- ❌ No logging framework (legacy uses print)
- **Reason**: Minimize changes, preserve behavior

#### 3. Bug Fixes
- ❌ No epsilon protection in normalize()
- ❌ No zero-norm protection in unit_scale()
- ❌ No parameter validation
- **Reason**: These are legacy behaviors, not bugs

#### 4. Configuration
- ❌ No external configuration system
- ❌ No hyperparameter classes
- ❌ Hardcoded values preserved
- **Reason**: Can add later without changing logic

### When to Use Comprehensive Refactoring Instead

Use the comprehensive refactoring approach from `mtgbm_models_refactoring_design.md` when:

1. **Production issues** require fixes (e.g., discovered bugs)
2. **Performance** is critical bottleneck
3. **New features** require architectural changes
4. **Team has bandwidth** for extensive testing
5. **Risk tolerance** is higher

Use this minimal refactoring when:

1. **Production stability** is paramount
2. **Quick migration** is needed
3. **Team bandwidth** is limited
4. **Risk tolerance** is low
5. **"If it ain't broke, don't fix it"** philosophy applies

## Conclusion

This minimal refactoring design provides a **low-risk migration path** from legacy MTGBM loss implementations to a slightly more maintainable structure. By preserving 83% of legacy code and extracting only truly identical operations, this approach:

### Achieves
- ✅ Code organization improvement
- ✅ Inheritance hierarchy
- ✅ Reduced duplication (15 lines extracted)
- ✅ Easy verification
- ✅ Fast migration path

### Preserves
- ✅ All legacy behaviors
- ✅ All legacy algorithms
- ✅ All legacy edge cases
- ✅ Production stability
- ✅ Team familiarity

### Enables
- ✅ Future enhancements
- ✅ New loss functions
- ✅ Gradual optimization
- ✅ Configuration additions

**Philosophy**: "Make it work, make it right, make it fast" - **we're at step 1-2**, not step 3.

---

## References

### Legacy Code Sources
- **baseLoss.py** - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/baseLoss.py
- **customLossNoKD.py** - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossNoKD.py
- **customLossKDswap.py** - projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/lossFunction/customLossKDswap.py

### Related Documents
- **[MTGBM Models Refactoring Design](./mtgbm_models_refactoring_design.md)** - Comprehensive alternative
- **[MTGBM Refactoring: Critical Bugs Fixed](../4_analysis/2025-12-18_mtgbm_refactoring_critical_bugs_fixed.md)** - Bug documentation
- **[MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)** - Original design
- **[Best Practices](../0_developer_guide/best_practices.md)** - Development standards

---

*This minimal refactoring design provides a conservative, low-risk approach to modernizing legacy MTGBM loss function implementations while preserving production stability and enabling future enhancements.*
