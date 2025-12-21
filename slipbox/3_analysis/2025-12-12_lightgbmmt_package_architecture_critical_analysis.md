---
tags:
  - analysis
  - critical-finding
  - architecture
  - multi-task-learning
  - lightgbm
  - package-dependency
keywords:
  - lightgbmmt package
  - custom LightGBM fork
  - multi-column labels
  - C++ modifications
  - functional equivalence
  - architectural constraint
topics:
  - multi-task learning
  - package architecture
  - C++ extensions
  - label format
  - prediction format
language: python
date of note: 2025-12-12
---

# LightGBMMT Package Architecture: Critical Analysis

## 🚨 CRITICAL FINDING: Custom LightGBM Fork Required

## Executive Summary

**CRITICAL DISCOVERY**: The legacy MTGBM implementation depends on a **custom-modified version of LightGBM** called `lightgbmmt`, which includes **C++ code changes** to support multi-column labels and multi-column predictions. This is **NOT** the standard LightGBM package from pip/conda.

**Impact on Refactoring**: The refactored implementation using standard `lightgbm` **CANNOT** functionally replicate the legacy multi-task prediction behavior without either:
1. Using the custom `lightgbmmt` package, OR
2. Training separate models per task

This finding has **fundamental implications** for the architecture and functional equivalence of the refactored implementation.

## Critical Evidence

### Evidence 1: Custom Package Import

**Legacy Code** (`projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/model/Mtgbm.py:13`):
```python
import lightgbmmt as lgbm  # ← CUSTOM package, NOT standard lightgbm!
```

**NOT**:
```python
import lightgbm as lgb  # ← Standard LightGBM from pip
```

### Evidence 2: Multi-Column Label Format

**Legacy Training** (`Mtgbm.py:109-117`):
```python
d_train = lgbm.Dataset(
    X_tr,
    label=np.concatenate([Y_tr.values.reshape((-1, 1)), Y_tr2.values], axis=1),
    #     ↑ Multi-column label matrix: shape (n_samples, n_tasks)
    #     Standard LightGBM would REJECT this!
)

d_valid = lgbm.Dataset(
    X_vl,
    label=np.concatenate([Y_vl.values.reshape((-1, 1)), Y_vl2.values], axis=1),
    #     ↑ Passes ALL task labels simultaneously
)
```

**Standard LightGBM Requirement**:
```python
# Standard LightGBM ONLY accepts:
dataset = lgb.Dataset(X, label=y)  # y must be 1D: shape (n_samples,)
```

### Evidence 3: Multi-Column Predictions

**Legacy Prediction** (`Mtgbm.py:234-236`):
```python
temp = self.model.predict(self.X_test)
#      ↑ Returns shape: (n_samples, n_tasks) - MULTI-COLUMN output!

self.y_lgbmt = expit(temp[:, 0])        # Main task (column 0)
self.y_lgbmtsub = expit(temp[:, 1:])    # Subtasks (columns 1+)
#                        ↑ Slicing columns proves multi-column output
```

**Standard LightGBM Behavior**:
```python
predictions = model.predict(X_test)
# ↑ Returns shape: (n_samples,) - SINGLE-COLUMN output only!
# Cannot slice columns because there's only one column!
```

### Evidence 4: Custom LightGBM C++ Code

**Legacy Package Structure**:
```
projects/pfw_lightgbmmt_legacy/dockers/mtgbm/
├── lightgbmmt/              # Custom Python package
│   ├── engine.py            # Modified training engine
│   └── basic.py             # Modified Booster class
└── compile/                 # CUSTOM C++ CODE
    ├── include/LightGBM/    # Modified LightGBM headers
    └── src/                 # Modified LightGBM source
        ├── boosting/        # Multi-task boosting modifications
        ├── objective/       # Multi-task objective functions
        └── ...
```

**Key File**: `compile/include/LightGBM/c_api.h` - Modified C API to accept multi-column labels

## Technical Analysis: How lightgbmmt Works

### Architecture Comparison

#### Standard LightGBM (Refactored Implementation)

```
Training:
  Input:  X: (n_samples, n_features)
          y: (n_samples,)              ← Single column ONLY
  
  Model:  Single-output decision trees
          Each leaf: single value
  
  Output: predictions: (n_samples,)    ← Single column
```

#### Custom lightgbmmt (Legacy Implementation)

```
Training:
  Input:  X: (n_samples, n_features)
          y: (n_samples, n_tasks)      ← Multi-column accepted!
  
  Model:  Multi-output decision trees
          Each leaf: vector of n_tasks values
  
  Output: predictions: (n_samples, n_tasks)  ← Multi-column output!
```

### C++ Modifications Required

To support multi-column labels, `lightgbmmt` modifies:

**1. Data Structures**
```cpp
// Standard LightGBM
struct Dataset {
    std::vector<float> labels_;  // 1D array
};

// lightgbmmt (hypothetical)
struct Dataset {
    std::vector<std::vector<float>> labels_;  // 2D array for multi-task
    int num_tasks_;
};
```

**2. Tree Leaf Values**
```cpp
// Standard LightGBM
struct LeafValue {
    double value;  // Single value per leaf
};

// lightgbmmt (hypothetical)
struct LeafValue {
    std::vector<double> values;  // Vector of values (one per task)
};
```

**3. Objective Functions**
```cpp
// Standard LightGBM signature
void GetGradients(const double* scores, const label_t* labels, 
                  score_t* out_gradients, score_t* out_hessians);

// lightgbmmt (hypothetical)
void GetGradients(const double** scores,  // 2D: [n_samples][n_tasks]
                  const label_t** labels,  // 2D: [n_samples][n_tasks]
                  score_t** out_gradients, score_t** out_hessians);
```

**4. Prediction Logic**
```cpp
// Standard LightGBM
double Predict(const Tree& tree, const float* features) {
    int leaf_idx = GetLeafIndex(tree, features);
    return tree.leaf_values[leaf_idx];  // Single value
}

// lightgbmmt (hypothetical)
std::vector<double> Predict(const Tree& tree, const float* features) {
    int leaf_idx = GetLeafIndex(tree, features);
    return tree.leaf_values[leaf_idx];  // Vector of values
}
```

## Implications for Refactored Implementation

### Critical Constraint: Standard LightGBM Limitation

**The refactored implementation uses standard `lightgbm`**, which means:

❌ **CANNOT** pass multi-column labels
❌ **CANNOT** get multi-column predictions from single model
❌ **CANNOT** functionally replicate legacy behavior exactly

### Workaround Analysis

**Current Refactored Approach** (`projects/cap_mtgbm/dockers/models/implementations/mtgbm_model.py`):

```python
# WORKAROUND: Pass single-column label, store multi-task in custom field
train_data = lgb.Dataset(
    X_train, 
    label=y_train[:, main_task_idx],  # ← Single column for validation
)
train_data.set_field('multi_task_labels', y_train.flatten())  # ← Store all tasks
```

**Problem**: This is a **HACK**, not true multi-task learning because:
- Model trees have single-output leaves, not multi-output
- Cannot truly share information between tasks at tree level
- Loss function retrieves labels from custom field, not from C++ core

### True Multi-Task Options

**Option A: Use lightgbmmt Package** ✅ Functionally equivalent
```python
import lightgbmmt as lgb

# Can use original format
train_data = lgb.Dataset(X_train, label=y_train)  # shape: (n, n_tasks) ✓
predictions = model.predict(X_test)  # shape: (n, n_tasks) ✓
```

**Pros:**
- True functional equivalence with legacy
- Multi-task at C++ level
- Single model with multi-output trees

**Cons:**
- Custom package dependency
- Maintenance burden (need to keep fork updated)
- Not widely available (not on pip/conda)

**Option B: Train Separate Models** ⚠️ Different architecture
```python
models = {}
predictions = np.zeros((n_samples, n_tasks))

for task_id in range(n_tasks):
    models[task_id] = lgb.train(
        params, 
        lgb.Dataset(X_train, label=y_train[:, task_id])  # Single task
    )
    predictions[:, task_id] = models[task_id].predict(X_test)
```

**Pros:**
- Uses standard LightGBM
- No custom package needed
- Well understood approach

**Cons:**
- NOT functionally equivalent to legacy
- Multiple separate models (higher memory)
- No shared tree structure between tasks
- Different multi-task learning mechanism

**Option C: Current Workaround** ⚠️ Not true multi-task
```python
# Store multi-task labels in custom field
# Model still single-output at C++ level
# Multi-task handled in Python loss function
```

**Pros:**
- Uses standard LightGBM
- Single model file

**Cons:**
- NOT functionally equivalent at architecture level
- Trees have single-output leaves
- Multi-task information sharing happens in loss, not in trees
- More complex to understand and maintain

## Functional Equivalence Analysis

### What IS Equivalent

✅ **Loss Function Computation**: Gradients and hessians computed the same way
✅ **Weight Update Logic**: Task weighting mechanism identical
✅ **Training Dynamics**: Both use gradient boosting with custom objectives
✅ **Evaluation Metrics**: Same metrics computed the same way

### What is NOT Equivalent

❌ **Tree Structure**: 
   - Legacy: Multi-output trees (each leaf outputs vector)
   - Refactored: Single-output trees (each leaf outputs scalar)

❌ **Label Format**:
   - Legacy: Multi-column labels passed to C++ core
   - Refactored: Single-column label, multi-task stored separately

❌ **Prediction Mechanism**:
   - Legacy: Model.predict() returns multi-column directly
   - Refactored: Model.predict() returns single column (must compute others separately)

❌ **Model File Format**:
   - Legacy: Trees contain multi-output leaf values
   - Refactored: Trees contain single-output leaf values

### Performance Implications

**Legacy (True Multi-Task)**:
- Tree traversal: 1× per sample
- Leaf lookup: 1× per sample
- Output: n_tasks values from single tree

**Refactored (Workaround)**:
- Tree traversal: 1× per sample
- Leaf lookup: 1× per sample
- Output: 1 value (main task)
- **Additional computation needed for subtasks** (unclear how)

## Recommendations

### Immediate Actions

1. **Document Dependency Clearly**
   - Add prominent note in all design docs about lightgbmmt requirement
   - Explain why standard LightGBM cannot replicate behavior

2. **Clarify Architecture Differences**
   - Update refactoring docs to explain tree structure differences
   - Document that refactored approach is NOT exact functional equivalent

3. **Investigate lightgbmmt Availability**
   - Can we use/maintain the lightgbmmt package?
   - Is it compatible with modern Python/LightGBM versions?
   - What's the maintenance burden?

### Strategic Decision Required

**Question**: Should refactored implementation:

**A) Use lightgbmmt package** (true equivalence)
- Pros: Functionally equivalent, true multi-task
- Cons: Custom dependency, maintenance

**B) Accept architectural difference** (current approach)
- Pros: Standard dependencies, simpler
- Cons: Not truly equivalent, different architecture

**C) Train separate models** (clearer alternative)
- Pros: Standard dependencies, well-understood
- Cons: Explicitly different approach, more memory

### Testing Implications

**Cannot compare**:
- Tree structures (fundamentally different)
- Model file formats (incompatible)
- Internal representations (different C++ code)

**Can compare**:
- Final predictions (if workaround correctly implemented)
- Training dynamics (loss curves should match)
- Evaluation metrics (should be similar)

## Conclusion

The discovery that legacy MTGBM uses a **custom-modified LightGBM package** (`lightgbmmt`) is **CRITICAL** for understanding functional equivalence. 

**Key Takeaway**: The refactored implementation using standard LightGBM **CANNOT** achieve exact functional equivalence at the architecture level. The workaround using single-column labels represents a fundamentally different multi-task learning approach.

**Decision Needed**: Choose between:
1. Accepting architectural differences (current path)
2. Using lightgbmmt package (true equivalence)
3. Explicitly training separate models (transparent alternative)

This finding should be prominently documented in all design and analysis documents to set accurate expectations about functional equivalence.

## References

### Legacy Code Locations
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/src/model/Mtgbm.py` - Main training class
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/lightgbmmt/` - Custom package
- `projects/pfw_lightgbmmt_legacy/dockers/mtgbm/compile/` - Custom C++ code

### Standard LightGBM
- https://github.com/microsoft/LightGBM - Official repository
- https://lightgbm.readthedocs.io/ - Official documentation

### Related Documents
- **[MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)**
- **[LightGBMMT Package Correspondence Analysis](./2025-12-10_lightgbmmt_package_correspondence_analysis.md)**
- **[MTGBM Refactoring Functional Equivalence Analysis](./2025-12-10_mtgbm_refactoring_functional_equivalence_analysis.md)**

---

*This critical analysis reveals fundamental architectural constraints that must be acknowledged in all refactoring documentation and design decisions.*
