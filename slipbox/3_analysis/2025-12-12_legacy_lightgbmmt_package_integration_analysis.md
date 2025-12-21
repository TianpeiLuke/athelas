---
tags:
  - analysis
  - legacy-code
  - multi-task-learning
  - lightgbm
  - c-plus-plus
  - custom-package
  - integration
keywords:
  - lightgbmmt
  - custom LightGBM fork
  - C++ modifications
  - multi-task API
  - package integration
  - legacy analysis
topics:
  - legacy code analysis
  - package customization
  - multi-task learning infrastructure
  - integration strategy
language: python, c++
date of note: 2025-12-12
---

# Legacy LightGBMMT Package Integration Analysis

## Executive Summary

The legacy MTGBM implementation uses a **custom-compiled fork of LightGBM** called `lightgbmmt`, which contains C++ modifications that are **fundamentally incompatible** with standard LightGBM. This analysis documents the custom package architecture, identifies critical components requiring integration, and provides a minimal extraction strategy for the refactored codebase.

**Critical Finding:** Standard LightGBM **CANNOT** support multi-task learning without the C++ modifications present in the `lightgbmmt` fork. The refactored implementation must integrate these custom components while maintaining clean architecture.

**Key Discoveries:**
- 🔴 **Custom C++ API:** 3 new C API functions for multi-task support
- 🔴 **Modified Python Bindings:** Extended Booster and Dataset classes
- 🔴 **Custom Tree Learner:** `"serial2"` tree learner for multi-task
- 🟡 **Dual Gradient Support:** Accepts 4 gradient arrays instead of 2
- 🟢 **Epoch-Aware Loss:** Loss functions receive iteration number

**Integration Requirement:** Minimal forking strategy needed to preserve multi-task functionality while fitting into refactored architecture.

## Related Documents
- **[MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)** - Design specification
- **[MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)** - Refactoring architecture
- **[MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md)** - Model class design
- **[MTGBM Refactoring Functional Equivalence Analysis](./2025-12-10_mtgbm_refactoring_functional_equivalence_analysis.md)** - Functional verification
- **[LightGBMMT Package Architecture Critical Analysis](./2025-12-12_lightgbmmt_package_architecture_critical_analysis.md)** - Architecture deep dive
- **[Model Architecture Design Index](../00_entry_points/model_architecture_design_index.md)** - Architecture documentation

---

## 1. Package Architecture Overview

### 1.1 Component Hierarchy

```
Legacy LightGBMMT Stack
├── C++ Layer (Modified LightGBM Core)
│   ├── lib_lightgbm.so              # Compiled binary with custom APIs
│   ├── LGBM_BoosterUpdateOneIterCustom2()  # NEW: Dual gradient update
│   ├── LGBM_BoosterSaveModel2()     # NEW: Multi-task model save
│   └── LGBM_BoosterSetNumLabels()   # NEW: Configure task count
│
├── Python Wrapper Layer (lightgbmmt package)
│   ├── basic.py                      # Modified Booster & Dataset
│   │   ├── Booster.num_labels__      # NEW: Task count tracking
│   │   ├── Booster.__boost()         # MODIFIED: Dual gradients
│   │   ├── Booster.update()          # MODIFIED: Epoch passing
│   │   ├── Booster.save_model2()     # NEW: Multi-task save
│   │   └── Dataset                   # MODIFIED: Multi-dim labels
│   │
│   └── engine.py                     # Modified training
│       └── train()                   # MODIFIED: Pass epoch to fobj
│
└── Application Layer (User Code)
    ├── Mtgbm.py                      # High-level model class
    └── lossFunction/                 # Custom loss functions
        ├── baseLoss.py               # Fixed weights
        ├── customLossNoKD.py         # Adaptive weights
        └── customLossKDswap.py       # Adaptive + KD
```

### 1.2 Standard LightGBM vs. LightGBMMT Comparison

| Feature | Standard LightGBM | LightGBMMT (Fork) |
|---------|------------------|-------------------|
| **Objective Function** | Single task only | Multi-task support |
| **Gradient Input** | 2 arrays (grad, hess) | 4 arrays (grad_main, hess_main, grad_sub, hess_sub) |
| **Label Format** | 1D array | 2D concatenated array |
| **Update API** | `update(fobj)` | `update(fobj, ep)` with epoch |
| **Booster Parameters** | Standard params | + `num_labels`, `tree_learner="serial2"` |
| **C API** | Standard functions | + 3 custom functions |
| **Python Interface** | `lightgbm` package | `lightgbmmt` package |

---

## 2. C++ Layer Modifications

### 2.1 Custom C API Functions

The compiled library (`lib_lightgbm.so`) adds **3 critical functions** not present in standard LightGBM:

#### Function 1: `LGBM_BoosterUpdateOneIterCustom2`

**Purpose:** Update booster with **dual gradient arrays** for multi-task learning.

**Signature:**
```c
int LGBM_BoosterUpdateOneIterCustom2(
    BoosterHandle handle,
    const float* grad_main,      // Main task gradients
    const float* hess_main,      // Main task hessians
    const float* grad_sub,       // Sub-task gradients
    const float* hess_sub,       // Sub-task hessians
    int* is_finished
);
```

**Standard LightGBM Equivalent:** `LGBM_BoosterUpdateOneIterCustom` (only 2 arrays)

**Why Critical:** Enables simultaneous optimization of main task and sub-tasks with separate gradient control.

---

#### Function 2: `LGBM_BoosterSaveModel2`

**Purpose:** Save multi-task model with task-specific information.

**Signature:**
```c
int LGBM_BoosterSaveModel2(
    BoosterHandle handle,
    int start_iteration,
    int num_iteration,
    int num_labels,              // NEW: Total task count
    int label_id,                // NEW: Specific label to save
    const char* filename
);
```

**Standard LightGBM Equivalent:** `LGBM_BoosterSaveModel` (no task info)

**Why Critical:** Allows per-task model extraction and task-specific model saving.

---

#### Function 3: `LGBM_BoosterSetNumLabels`

**Purpose:** Configure the number of tasks in the booster.

**Signature:**
```c
int LGBM_BoosterSetNumLabels(
    BoosterHandle handle,
    int num_labels               // NEW: Task count
);
```

**Standard LightGBM Equivalent:** None (concept doesn't exist)

**Why Critical:** Initializes internal structures for multi-task processing.

---

### 2.2 Custom Tree Learner

**New Tree Learner Type:** `"serial2"`

**Purpose:** Specialized tree learner that handles multi-task splitting.

**Configuration:**
```python
params = {
    'tree_learner': 'serial2',  # Required for multi-task
    'num_labels': 6             # Number of tasks
}
```

**Standard LightGBM Tree Learners:** `"serial"`, `"feature"`, `"data"`, `"voting"`  
**None support multi-task** without `"serial2"` modification.

---

## 3. Python Wrapper Layer Modifications

### 3.1 Modified Booster Class

Located in: `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/basic.py`

#### Key Modifications

**1. Multi-Task Parameter Tracking**

```python
class Booster:
    def __init__(self, params=None, train_set=None, ...):
        # NEW: Track number of tasks
        if "num_labels" not in params:
            params["num_labels"] = 1
        self.num_labels__ = params["num_labels"]
        # ... rest of initialization
```

**Impact:** Enables task count awareness throughout training lifecycle.

---

**2. Dual Gradient Boosting**

```python
def __boost(self, grad, hess, grad2=None, hess2=None):
    """Modified to accept dual gradients for multi-task."""
    is_finished = ctypes.c_int(0)
    
    if self.num_labels__ > 1 and grad2 is not None:
        # Multi-task path: Use custom C API
        _safe_call(
            _LIB.LGBM_BoosterUpdateOneIterCustom2(
                self.handle,
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grad2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hess2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(is_finished),
            )
        )
    else:
        # Single-task path: Use standard C API
        _safe_call(
            _LIB.LGBM_BoosterUpdateOneIterCustom(
                self.handle,
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(is_finished),
            )
        )
    
    return is_finished.value == 1
```

**Impact:** Enables separate gradient control for main and sub-tasks.

---

**3. Epoch-Aware Updates**

```python
def update(self, train_set=None, fobj=None, ep=None):
    """Modified to pass epoch number to custom loss function."""
    if fobj is None:
        # Standard update without custom objective
        _safe_call(_LIB.LGBM_BoosterUpdateOneIter(...))
    else:
        # Custom objective with multi-task support
        if self.num_labels__ > 1:
            # Pass epoch to loss function (NEW)
            grad, hess, grad2, hess2 = fobj(
                self.__inner_predict(0), 
                self.train_set, 
                ep  # NEW: Epoch parameter
            )
            return self.__boost(grad, hess, grad2, hess2)
        else:
            grad, hess = fobj(self.__inner_predict(0), self.train_set)
            return self.__boost(grad, hess)
```

**Impact:** Enables epoch-dependent loss functions (critical for adaptive weighting and KD).

---

**4. Multi-Task Model Saving**

```python
def save_model2(self, filename, num_iteration=None, num_label=0):
    """Save model with multi-task information."""
    if num_iteration is None:
        num_iteration = self.best_iteration
    
    _safe_call(
        _LIB.LGBM_BoosterSaveModel2(
            self.handle,
            ctypes.c_int(0),  # start_iteration
            ctypes.c_int(num_iteration),
            ctypes.c_int(self.num_labels__),  # NEW: Task count
            ctypes.c_int(num_label),           # NEW: Specific task
            c_str(filename),
        )
    )
```

**Impact:** Enables task-specific model extraction.

---

**5. Task Count Configuration**

```python
def set_num_labels(self, num_labels):
    """Set the number of task labels."""
    _safe_call(
        _LIB.LGBM_BoosterSetNumLabels(
            self.handle, 
            ctypes.c_int(num_labels)
        )
    )
```

**Impact:** Allows dynamic task count changes.

---

### 3.2 Modified Dataset Class

Located in: `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/basic.py`

#### Key Modification: Multi-Dimensional Labels

**Standard LightGBM:**
```python
# Single-task labels (1D)
labels = np.array([0, 1, 0, 1, ...])  # Shape: (n_samples,)
dataset = lgb.Dataset(X, label=labels)
```

**LightGBMMT:**
```python
# Multi-task labels (2D, concatenated)
main_labels = np.array([0, 1, 0, 1, ...]).reshape(-1, 1)
sub_labels = np.array([[0, 1], [1, 0], [0, 0], [1, 1], ...])
all_labels = np.concatenate([main_labels, sub_labels], axis=1)
# Shape: (n_samples, 1 + n_subtasks)

dataset = lgbm.Dataset(X, label=all_labels)
```

**Impact:** Dataset class internally handles 2D label arrays and passes them correctly to C++ layer.

---

### 3.3 Modified Training Function

Located in: `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/engine.py`

#### Key Modification: Epoch Passing

**Standard LightGBM train():**
```python
def train(params, train_set, fobj=None, ...):
    booster = Booster(params, train_set)
    for i in range(num_boost_round):
        booster.update(fobj=fobj)  # No epoch info
```

**LightGBMMT train():**
```python
def train(params, train_set, fobj=None, ...):
    booster = Booster(params, train_set)
    for i in range(num_boost_round):
        booster.update(fobj=fobj, ep=i)  # Pass epoch number
```

**Impact:** Enables epoch-dependent loss computation (adaptive weighting, knowledge distillation).

---

## 4. Application Layer Usage

### 4.1 Multi-Task Loss Function Interface

Custom loss functions in legacy code **must** return 4 arrays for multi-task:

```python
class CustomLoss:
    def self_obj(self, preds, train_data, ep):
        """
        Custom objective for multi-task learning.
        
        Parameters:
        -----------
        preds : array
            Predictions from all tasks, flattened
        train_data : Dataset
            Training dataset with concatenated labels
        ep : int
            Current epoch/iteration number
            
        Returns:
        --------
        grad_main : array
            Gradients for main task
        hess_main : array
            Hessians for main task
        grad_sub : array
            Gradients for all sub-tasks (flattened)
        hess_sub : array
            Hessians for all sub-tasks (flattened)
        """
        # Reshape predictions
        n_samples = len(preds) // self.num_labels
        preds_mat = preds.reshape(n_samples, self.num_labels)
        
        # Get labels
        labels = train_data.get_label()
        labels_mat = labels.reshape(n_samples, self.num_labels)
        
        # Compute gradients for all tasks
        grad_i = sigmoid(preds_mat) - labels_mat
        hess_i = sigmoid(preds_mat) * (1 - sigmoid(preds_mat))
        
        # Separate main and sub-tasks
        grad_main = grad_i[:, 0]
        hess_main = hess_i[:, 0]
        grad_sub = grad_i[:, 1:].flatten()
        hess_sub = hess_i[:, 1:].flatten()
        
        return grad_main, hess_main, grad_sub, hess_sub
```

---

### 4.2 Complete Training Example

```python
import lightgbmmt as lgbm
from lossFunction.customLossNoKD import custom_loss_noKD

# Prepare multi-task labels
main_labels = train_df['main_task'].values.reshape(-1, 1)
sub_labels = train_df[['sub1', 'sub2', 'sub3']].values
all_labels = np.concatenate([main_labels, sub_labels], axis=1)

# Create dataset
train_data = lgbm.Dataset(X_train, label=all_labels)
valid_data = lgbm.Dataset(X_valid, label=all_labels_valid)

# Multi-task parameters
params = {
    'objective': 'custom',       # REQUIRED for custom loss
    'num_labels': 4,             # 1 main + 3 sub-tasks
    'tree_learner': 'serial2',   # REQUIRED for multi-task
    'max_depth': 6,
    'learning_rate': 0.01,
    # ... other params
}

# Create custom loss
loss_fn = custom_loss_noKD(
    num_label=4,
    val_sublabel_idx=validation_indices,
    trn_sublabel_idx=training_indices
)

# Train with custom loss
model = lgbm.train(
    params,
    train_set=train_data,
    valid_sets=valid_data,
    num_boost_round=1000,
    fobj=loss_fn.self_obj,    # Custom objective
    feval=loss_fn.self_eval   # Custom evaluation
)

# Predict returns multi-dimensional output
preds = model.predict(X_test)  # Shape: (n_samples, 4)
main_preds = preds[:, 0]
sub_preds = preds[:, 1:]
```

---

## 5. Why Standard LightGBM Cannot Work

### 5.1 Fundamental Incompatibilities

| Requirement | Standard LightGBM | LightGBMMT | Workaround Possible? |
|-------------|------------------|------------|---------------------|
| **Dual gradient update** | ❌ Single grad/hess pair | ✅ Quad arrays | ❌ No - requires C++ |
| **2D label arrays** | ❌ 1D only | ✅ 2D concatenated | ❌ No - Dataset limitation |
| **Epoch to loss function** | ❌ Not passed | ✅ Passed as `ep` | ❌ No - API limitation |
| **Multi-task parameters** | ❌ No `num_labels` | ✅ `num_labels` param | ❌ No - internal structures |
| **Custom tree learner** | ❌ No `serial2` | ✅ `serial2` for MT | ❌ No - requires C++ |
| **Task-specific save** | ❌ No `save_model2` | ✅ Per-task save | 🟡 Maybe - post-processing |

---

### 5.2 Attempted Workarounds and Failures

#### ❌ **Workaround 1: Train N Separate Models**

**Idea:** Train one model per task independently.

**Failure:** 
- Loses multi-task learning benefits
- No parameter sharing
- No transfer learning between tasks
- Not equivalent to MTGBM

---

#### ❌ **Workaround 2: Concatenate Predictions in Loss**

**Idea:** Use standard LightGBM with clever loss function.

**Failure:**
```python
def attempted_loss(preds, train_data):
    # Standard LightGBM only accepts (grad, hess)
    # Cannot return (grad_main, hess_main, grad_sub, hess_sub)
    grad_all = ...  # Some aggregation
    hess_all = ...
    return grad_all, hess_all  # Loses task separation!
```

**Problem:** Cannot provide separate gradient control for different tasks.

---

#### ❌ **Workaround 3: Custom Callback Approach**

**Idea:** Use callbacks to modify gradients between iterations.

**Failure:**
- Callbacks run **after** gradient computation
- Cannot intercept gradient calculation
- Cannot modify C++ internal structures

---

### 5.3 Conclusion: Custom Fork is Mandatory

**There is NO way to achieve MTGBM functionality with standard LightGBM.**

The custom C++ modifications are **essential and cannot be replicated** in Python alone. The refactored implementation **MUST** integrate the lightgbmmt fork.

---

## 6. Integration Strategy for Refactored Code

### 6.1 Minimal Extraction Approach

**Goal:** Extract only what's necessary, integrate into `projects/cap_mtgbm/dockers/models/`

#### Components to Extract

**1. C++ Compiled Library (REQUIRED)**
```
Source: projects/pfw_lightgbmmt_legacy/dockers/mtgbm/compile/lib_lightgbm.so
Target: projects/cap_mtgbm/dockers/lib/lib_lightgbm.so
```

**2. Core Python Wrapper Classes (REQUIRED)**
```
Source Files:
- lightgbmmt/basic.py (Booster, Dataset classes)
- lightgbmmt/engine.py (train function)
- lightgbmmt/libpath.py (library loading)

Target Structure:
projects/cap_mtgbm/dockers/models/lightgbmmt_wrapper/
├── __init__.py
├── booster.py       # Extract Booster from basic.py
├── dataset.py       # Extract Dataset from basic.py
├── training.py      # Extract train() from engine.py
└── libpath.py       # Copy as-is
```

**3. Loss Function Integration (ADAPT)**

Use unified loss class that wraps the custom API:

```python
# In models/loss/multi_task_loss.py
from models.lightgbmmt_wrapper import MTGBMBooster

class MultiTaskLoss(BaseLossFunction):
    def objective(self, preds, train_data, epoch=None):
        """
        Returns 4 arrays for lightgbmmt compatibility.
        """
        # Compute gradients
        grad_i, hess_i = self._compute_gradients(preds, train_data)
        
        # Separate main and sub-tasks
        grad_main = grad_i[:, 0]
        hess_main = hess_i[:, 0]
        grad_sub = grad_i[:, 1:].flatten()
        hess_sub = hess_i[:, 1:].flatten()
        
        return grad_main, hess_main, grad_sub, hess_sub
```

---

### 6.2 Refactored Folder Structure

```
projects/cap_mtgbm/dockers/
├── lib/
│   └── lib_lightgbm.so                    # 🆕 Custom C++ library
│
├── models/
│   ├── lightgbmmt_wrapper/                # 🆕 Minimal wrapper package
│   │   ├── __init__.py
│   │   ├── booster.py                     # 🆕 MTGBMBooster class
│   │   ├── dataset.py                     # 🆕 MTGBMDataset class
│   │   ├── training.py                    # 🆕 train_mtgbm() function
│   │   └── libpath.py                     # 🆕 Library loader
│   │
│   ├── base/
│   │   ├── base_model.py                  # ✅ Keep existing
│   │   └── training_state.py              # ✅ Keep existing
│   │
│   ├── implementations/
│   │   └── mtgbm_model.py                 # 🔄 Update to use wrapper
│   │
│   ├── loss/
│   │   ├── base_loss_function.py          # ✅ Keep existing
│   │   ├── multi_task_loss.py             # 🔄 Unified loss (returns 4 arrays)
│   │   └── loss_factory.py                # ✅ Keep existing
│   │
│   └── factory/
│       └── model_factory.py               # ✅ Keep existing
│
├── lightgbmmt_training.py                 # 🔄 Update to use wrapper
├── lightgbmmt_model_eval.py              # 🔄 Update to use wrapper
└── requirements.txt                       # 📦 Add wrapper dependency
```

---

### 6.3 Integration Code Example

**In `models/implementations/mtgbm_model.py`:**

```python
from models.base.base_model import BaseModel
from models.lightgbmmt_wrapper import (
    MTGBMBooster, 
    MTGBMDataset, 
    train_mtgbm
)
from models.loss.multi_task_loss import MultiTaskLoss

class MTGBMModel(BaseModel):
    """
    Multi-task GBM model using custom lightgbmmt package.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.booster = None
        
    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        """
        Train multi-task model.
        
        Parameters:
        -----------
        y_train_dict : dict
            {'main': array, 'sub1': array, 'sub2': array, ...}
        """
        # Concatenate labels for multi-task
        main_labels = y_train_dict['main'].reshape(-1, 1)
        sub_labels = np.column_stack([
            y_train_dict[task] 
            for task in self.config.sub_tasks
        ])
        all_labels = np.concatenate([main_labels, sub_labels], axis=1)
        
        # Create multi-task dataset
        train_data = MTGBMDataset(X_train, label=all_labels)
        
        # Multi-task parameters
        params = {
            'objective': 'custom',
            'num_labels': 1 + len(self.config.sub_tasks),
            'tree_learner': 'serial2',
            **self.config.lgbm_params
        }
        
        # Create unified loss function
        loss_fn = MultiTaskLoss(
            num_labels=params['num_labels'],
            validation_indices=self._get_validation_indices(y_val_dict),
            training_indices=self._get_training_indices(y_train_dict),
            use_adaptive_weights=self.config.use_adaptive_weights,
            use_knowledge_distillation=self.config.use_kd
        )
        
        # Train using wrapper
        self.booster = train_mtgbm(
            params=params,
            train_set=train_data,
            num_boost_round=self.config.num_rounds,
            fobj=loss_fn.objective,      # Returns 4 arrays
            feval=loss_fn.evaluation
        )
        
    def predict(self, X_test):
        """Returns multi-dimensional predictions."""
        preds = self.booster.predict(X_test)
        return {
            'main': preds[:, 0],
            'sub_tasks': {
                task: preds[:, i+1]
                for i, task in enumerate(self.config.sub_tasks)
            }
        }
```

---

## 7. C++ Library Deployment Considerations

### 7.1 Platform Compatibility

The compiled `lib_lightgbm.so` is platform-specific:

| Platform | File Extension | Compilation Required |
|----------|---------------|---------------------|
| Linux (AL2023) | `.so` | ✅ Re-compile for target |
| macOS | `.dylib` | ✅ Re-compile for target |
| Windows | `.dll` | ✅ Re-compile for target |

**Critical:** The legacy compiled binary may not work on deployment environment (AL2023). **Recompilation required.**

---

### 7.2 Build Requirements

To rebuild the library for deployment:

**Requirements:**
```bash
# AL2023 build environment
yum install -y gcc-c++ cmake make
yum install -y python3-devel
```

**Build Process:**
```bash
cd projects/pfw_lightgbmmt_legacy/dockers/mtgbm/compile/
mkdir build && cd build
cmake ..
make -j4
# Output: lib_lightgbm.so
```

**Verification:**
```python
import ctypes
lib = ctypes.cdll.LoadLibrary('lib_lightgbm.so')
# Check custom functions exist
assert hasattr(lib, 'LGBM_BoosterUpdateOneIterCustom2')
assert hasattr(lib, 'LGBM_BoosterSaveModel2')
assert hasattr(lib, 'LGBM_BoosterSetNumLabels')
```

---

### 7.3 Runtime Loading

**In `models/lightgbmmt_wrapper/libpath.py`:**

```python
def find_lib_path():
    """Locate the custom LightGBM library."""
    import os
    
    # Try relative to this file
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(curr_dir, '..', '..', 'lib')
    lib_path = os.path.join(lib_dir, 'lib_lightgbm.so')
    
    if os.path.exists(lib_path):
        return [lib_path]
    
    # Try environment variable
    if 'LIGHTGBMMT_LIB' in os.environ:
        return [os.environ['LIGHTGBMMT_LIB']]
    
    raise FileNotFoundError(
        "Cannot find lib_lightgbm.so. "
        "Set LIGHTGBMMT_LIB environment variable."
    )
```

---

## 8. Migration Checklist

### 8.1 Pre-Integration Tasks

- [ ] **Extract wrapper classes** from `lightgbmmt/basic.py`
  - [ ] Extract `Booster` class → `booster.py`
  - [ ] Extract `Dataset` class → `dataset.py`
  - [ ] Extract C API bindings
  - [ ] Extract helper functions

- [ ] **Extract training function** from `lightgbmmt/engine.py`
  - [ ] Extract `train()` function → `training.py`
  - [ ] Maintain epoch passing logic
  - [ ] Preserve callback handling

- [ ] **Copy C++ library**
  - [ ] Copy `lib_lightgbm.so` → `projects/cap_mtgbm/dockers/lib/`
  - [ ] Create library loader (`libpath.py`)
  - [ ] Test library loading

---

### 8.2 Integration Tasks

- [ ] **Create wrapper package structure**
  - [ ] Create `models/lightgbmmt_wrapper/` directory
  - [ ] Create `__init__.py` with exports
  - [ ] Add docstrings and type hints
  - [ ] Test wrapper isolation

- [ ] **Implement unified loss function**
  - [ ] Create `models/loss/multi_task_loss.py`
  - [ ] Ensure returns 4 arrays (grad_main, hess_main, grad_sub, hess_sub)
  - [ ] Add adaptive weighting support
  - [ ] Add knowledge distillation support
  - [ ] Test equivalence with legacy loss functions

- [ ] **Update model implementation**
  - [ ] Update `models/implementations/mtgbm_model.py`
  - [ ] Use wrapper classes instead of legacy imports
  - [ ] Test training pipeline
  - [ ] Verify predictions match legacy

---

### 8.3 Testing & Verification Tasks

- [ ] **Unit tests**
  - [ ] Test wrapper classes independently
  - [ ] Test loss function gradients
  - [ ] Test label concatenation/splitting
  - [ ] Test epoch passing

- [ ] **Integration tests**
  - [ ] End-to-end training test
  - [ ] Compare predictions with legacy
  - [ ] Verify model save/load
  - [ ] Test multi-task inference

- [ ] **Performance tests**
  - [ ] Training time comparison
  - [ ] Memory usage comparison
  - [ ] Inference latency comparison

---

### 8.4 Documentation Tasks

- [ ] **Code documentation**
  - [ ] Document wrapper API
  - [ ] Document loss function interface
  - [ ] Add usage examples
  - [ ] Create migration guide

- [ ] **Design documentation**
  - [ ] Update design documents
  - [ ] Document integration decisions
  - [ ] Add architecture diagrams
  - [ ] Link to analysis documents

---

## 9. Risks and Mitigation Strategies

### 9.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **C++ library incompatibility** | HIGH | MEDIUM | Recompile for target platform, thorough testing |
| **Python wrapper bugs** | HIGH | LOW | Extensive unit tests, gradual rollout |
| **Performance degradation** | MEDIUM | LOW | Benchmark before/after, optimize hotspots |
| **Memory leaks** | HIGH | LOW | Memory profiling, proper resource cleanup |
| **Platform-specific issues** | MEDIUM | MEDIUM | Test on all target platforms |

### 9.2 Integration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Breaking changes to refactored code** | HIGH | MEDIUM | Comprehensive tests, phased integration |
| **Configuration incompatibilities** | MEDIUM | LOW | Unified config validation |
| **Training instability** | HIGH | LOW | Compare with legacy training runs |
| **Model drift** | HIGH | LOW | A/B testing, gradual rollout |

### 9.3 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Deployment failures** | HIGH | MEDIUM | Staging environment testing, rollback plan |
| **Library path issues** | MEDIUM | MEDIUM | Environment variable fallback, clear error messages |
| **Version conflicts** | MEDIUM | LOW | Pinned dependencies, isolated environments |
| **Documentation gaps** | LOW | MEDIUM | Comprehensive docs, runbooks |

---

## 10. Success Criteria

### 10.1 Functional Requirements

✅ **Must Have:**
- [ ] All 3 loss function types working (Fixed, Adaptive, Adaptive+KD)
- [ ] Predictions match legacy implementation (< 0.1% difference)
- [ ] Training completes successfully on test dataset
- [ ] Model save/load works correctly
- [ ] Passes all legacy test cases

✅ **Should Have:**
- [ ] Training time within 10% of legacy
- [ ] Memory usage within 20% of legacy
- [ ] Clean architecture with proper separation of concerns
- [ ] Comprehensive documentation

✅ **Nice to Have:**
- [ ] Better performance than legacy
- [ ] Enhanced error messages
- [ ] Additional validation checks
- [ ] Improved logging

---

### 10.2 Non-Functional Requirements

**Code Quality:**
- [ ] Type hints on all functions
- [ ] Docstrings on all classes/methods
- [ ] < 10 pylint warnings per file
- [ ] 80%+ test coverage

**Performance:**
- [ ] Training time < 1.1x legacy
- [ ] Inference latency < 1.05x legacy
- [ ] Memory usage < 1.2x legacy

**Reliability:**
- [ ] No memory leaks
- [ ] No crashes on edge cases
- [ ] Proper error handling
- [ ] Graceful degradation

---

## 11. Rollout Plan

### 11.1 Phase 1: Extraction & Setup (Week 1)

**Objectives:**
- Extract wrapper classes from legacy code
- Set up folder structure
- Copy C++ library
- Create basic tests

**Deliverables:**
- `models/lightgbmmt_wrapper/` package
- `lib/lib_lightgbm.so` library
- Basic unit tests
- Library loading verification

---

### 11.2 Phase 2: Integration (Week 2)

**Objectives:**
- Implement unified loss function
- Update model implementation
- Integration testing
- Performance benchmarking

**Deliverables:**
- `models/loss/multi_task_loss.py`
- Updated `models/implementations/mtgbm_model.py`
- Integration test suite
- Performance comparison report

---

### 11.3 Phase 3: Validation (Week 3)

**Objectives:**
- Comprehensive testing
- Documentation
- Bug fixes
- Optimization

**Deliverables:**
- Full test coverage
- Complete documentation
- Performance optimization
- Migration guide

---

### 11.4 Phase 4: Deployment (Week 4)

**Objectives:**
- Staging deployment
- Production rollout
- Monitoring setup
- Legacy deprecation

**Deliverables:**
- Staging validation results
- Production deployment
- Monitoring dashboards
- Deprecation timeline

---

## 12. Alternative Approaches Considered

### 12.1 Pure Python Reimplementation

**Idea:** Reimplement multi-task GBDT entirely in Python/PyTorch.

**Pros:**
- No C++ dependency
- Easier to modify
- Better portability

**Cons:**
- ❌ Significantly slower training
- ❌ Higher memory usage
- ❌ Different numerical behavior
- ❌ Months of development time

**Decision:** REJECTED - Not feasible within timeline and performance requirements.

---

### 12.2 Use Standard LightGBM with Workarounds

**Idea:** Use standard LightGBM and simulate multi-task with separate models.

**Pros:**
- No custom fork needed
- Standard LightGBM benefits
- Simpler deployment

**Cons:**
- ❌ Not true multi-task learning
- ❌ No parameter sharing
- ❌ No transfer learning
- ❌ Not functionally equivalent

**Decision:** REJECTED - Loses multi-task learning benefits, not equivalent to legacy.

---

### 12.3 Fork Entire LightGBM Repository

**Idea:** Fork full LightGBM repo and maintain custom branch.

**Pros:**
- Full control over codebase
- Can merge upstream updates
- Clear modification tracking

**Cons:**
- ❌ Large maintenance burden
- ❌ Must track upstream changes
- ❌ Complex merge conflicts
- ❌ Overkill for our needs

**Decision:** REJECTED - Minimal extraction approach is sufficient and easier to maintain.

---

### 12.4 Selected Approach: Minimal Extraction

**Idea:** Extract only custom C++ library and Python wrappers.

**Pros:**
- ✅ Minimal code to maintain
- ✅ Clean integration with refactored code
- ✅ Preserves functionality
- ✅ Manageable scope

**Cons:**
- 🟡 Still depends on custom C++ binary
- 🟡 Must recompile for each platform

**Decision:** SELECTED - Best balance of functionality, maintainability, and implementation effort.

---

## 13. Lessons Learned

### 13.1 From Legacy Analysis

**Key Insights:**
1. **C++ modifications are essential** - Cannot be replicated in Python
2. **Dual gradient API is core** - Enables separate task optimization
3. **Epoch passing is critical** - Required for adaptive/KD loss functions
4. **Custom tree learner needed** - `"serial2"` handles multi-task splits
5. **Label format matters** - 2D arrays required for multi-task

**Best Practices:**
1. Always verify C++ API availability before using
2. Document platform-specific compilation requirements
3. Test library loading early in development
4. Maintain clear separation between wrapper and application code
5. Comprehensive logging for debugging C++/Python interactions

---

### 13.2 For Future Custom Package Integrations

**Recommendations:**
1. **Document dependencies early** - Identify all custom modifications upfront
2. **Minimal extraction strategy** - Extract only what's absolutely necessary
3. **Clean architecture** - Isolate custom components from business logic
4. **Comprehensive testing** - Test at C++/Python boundary extensively
5. **Platform testing** - Test on all target deployment platforms
6. **Version control** - Track custom library versions carefully
7. **Documentation** - Document why custom fork is needed and what it provides

---

## 14. References

### 14.1 Internal Documents

- [MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)
- [MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)
- [MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md)
- [MTGBM Refactoring Functional Equivalence Analysis](./2025-12-10_mtgbm_refactoring_functional_equivalence_analysis.md)
- [LightGBMMT Package Architecture Critical Analysis](./2025-12-12_lightgbmmt_package_architecture_critical_analysis.md)

### 14.2 Code Locations

**Legacy Code:**
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/basic.py`
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/engine.py`
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/compile/lib_lightgbm.so`

**Refactored Code:**
- `projects/cap_mtgbm/dockers/models/`
- `projects/cap_mtgbm/dockers/lib/` (future location)

### 14.3 External Resources

- LightGBM Official Documentation: https://lightgbm.readthedocs.io/
- LightGBM C API Reference: https://lightgbm.readthedocs.io/en/latest/C-API.html
- Multi-Task Learning Overview: Academic papers on MTL

---

## 15. Appendix

### 15.1 Glossary

**Terms:**

- **MTGBM**: Multi-Task Gradient Boosted Machine
- **LightGBMMT**: Custom LightGBM fork with multi-task support
- **Dual Gradients**: Separate gradient arrays for main and sub-tasks
- **Epoch**: Training iteration number
- **serial2**: Custom tree learner for multi-task learning
- **KD**: Knowledge Distillation
- **Adaptive Weighting**: Dynamic task weight adjustment

### 15.2 Common Errors and Solutions

**Error 1: Library not found**
```
FileNotFoundError: Cannot find lib_lightgbm.so
```
**Solution:** Set `LIGHTGBMMT_LIB` environment variable or copy library to expected location.

---

**Error 2: Symbol not found**
```
AttributeError: 'CDLL' object has no attribute 'LGBM_BoosterUpdateOneIterCustom2'
```
**Solution:** Using standard LightGBM instead of custom fork. Ensure custom library is loaded.

---

**Error 3: Wrong number of return values**
```
ValueError: too many values to unpack (expected 2)
```
**Solution:** Loss function must return 4 arrays for multi-task, not 2.

---

**Error 4: Shape mismatch**
```
ValueError: operands could not be broadcast together
```
**Solution:** Check label concatenation format - should be 2D array with shape (n_samples, n_tasks).

---

### 15.3 Quick Start Guide

**Minimal working example:**

```python
# 1. Import wrapper
from models.lightgbmmt_wrapper import MTGBMDataset, train_mtgbm
from models.loss.multi_task_loss import MultiTaskLoss

# 2. Prepare labels
labels = np.concatenate([main, sub1, sub2], axis=1)  # 2D array

# 3. Create dataset
train_data = MTGBMDataset(X, label=labels)

# 4. Configure
params = {
    'objective': 'custom',
    'num_labels': 3,
    'tree_learner': 'serial2'
}

# 5. Create loss
loss = MultiTaskLoss(num_labels=3, ...)

# 6. Train
model = train_mtgbm(params, train_data, fobj=loss.objective)

# 7. Predict
preds = model.predict(X_test)  # Shape: (n_samples, 3)
```

---

## Maintenance Notes

**Last Updated:** 2025-12-12

**Document Owner:** ML Platform Team

**Review Schedule:** Quarterly or when significant changes occur

**Update Triggers:**
- C++ library updates
- Python wrapper modifications
- Integration changes
- New deployment platforms
- Performance optimizations

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-12 | 1.0 | AI Assistant | Initial creation |

---

**End of Document**
