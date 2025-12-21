---
tags:
  - design
  - integration
  - multi-task-learning
  - lightgbm
  - custom-fork
  - architecture
keywords:
  - lightgbmmt
  - fork integration
  - multi-task prediction
  - C++ library
  - Python wrapper
topics:
  - package integration
  - multi-task learning
  - architecture design
  - custom library integration
language: python, c++
date of note: 2025-12-12
---

# LightGBMMT Fork Integration Design

## Executive Summary

This design document describes the integration of the custom LightGBMT fork into the refactored CAP MTGBM project to enable multi-task predictions with [N_samples, N_tasks] output shape.

**Current Status:** The lightgbmmt Python package and C++ compilation infrastructure have **ALREADY been copied** to `projects/cap_mtgbm/dockers/models/`. The remaining work focuses on updating `mtgbm_model.py` to use the custom package.

**Key Achievement:** Simplified integration - no extraction needed, only model class updates required.

## Related Documents

- **[Legacy LightGBMMT Package Integration Analysis](../4_analysis/2025-12-12_legacy_lightgbmmt_package_integration_analysis.md)** - Complete technical analysis
- **[MTGBM Models Refactoring Design](./mtgbm_models_refactoring_design.md)** - Loss function refactoring
- **[MTGBM Model Classes Refactoring Design](./mtgbm_model_classes_refactoring_design.md)** - Model class design

## Current Project Structure

### Already Present (No Action Needed)

```
projects/cap_mtgbm/dockers/models/
├── compile/                              # ✅ C++ Source & Library
│   ├── lib_lightgbm.so                   # Pre-compiled custom library
│   ├── CMakeLists.txt                    # Build configuration
│   ├── include/                          # C++ headers
│   └── src/                              # C++ source code
│
├── lightgbmmt/                           # ✅ Python Wrapper Package
│   ├── __init__.py                       # Exports: Dataset, Booster, train
│   ├── basic.py                          # Dataset, Booster classes
│   ├── engine.py                         # train(), cv() functions
│   ├── libpath.py                        # Library finder
│   ├── callback.py                       # Callback utilities
│   ├── compat.py                         # Compatibility layer
│   ├── plotting.py                       # Visualization
│   └── sklearn.py                        # Scikit-learn interface
│
├── loss/                                 # ✅ Refactored (No Changes)
│   ├── base_loss_function.py
│   ├── fixed_weight_loss.py
│   ├── adaptive_weight_loss.py
│   ├── knowledge_distillation_loss.py
│   ├── loss_factory.py
│   └── weight_strategies.py
│
├── base/                                 # ✅ Refactored (No Changes)
│   ├── base_model.py
│   └── training_state.py
│
├── implementations/                      # 🔄 Needs Update
│   └── mtgbm_model.py                    # Update imports and methods
│
└── factory/                              # ✅ No Changes
    └── model_factory.py
```

## Integration Task: Update mtgbm_model.py

### Required Changes

The `models/implementations/mtgbm_model.py` file currently uses standard `lightgbm` but needs to use the custom `lightgbmmt` package. Seven specific changes are required:

### 1. Update Imports (Line 8)

```python
# BEFORE
import lightgbm as lgb

# AFTER
from ..lightgbmmt import Dataset, Booster, train
from scipy.special import expit
```

### 2. Update _prepare_data() Method (Lines 91-109)

```python
# BEFORE - Uses standard lgb.Dataset with field hack
train_data = lgb.Dataset(
    X_train,
    label=y_train[:, main_task_idx],  # Only main task
    feature_name=feature_columns,
    ...
)
train_data.set_field("multi_task_labels", y_train.flatten())  # Hack

# AFTER - Uses Dataset with native 2D label support
train_data = Dataset(
    X_train,
    label=y_train,  # Pass full 2D array [N_samples, N_tasks]
    feature_name=feature_columns,
    ...
)
# No field hack needed - native 2D support!
```

### 3. Update _initialize_model() Method (Lines 145-157)

```python
# BEFORE
self.lgb_params = {
    "boosting_type": self.hyperparams.boosting_type,
    "num_leaves": self.hyperparams.num_leaves,
    ...
}

# AFTER - Add multi-task parameters
self.lgb_params = {
    "boosting_type": self.hyperparams.boosting_type,
    "num_leaves": self.hyperparams.num_leaves,
    ...
    "objective": "custom",                      # NEW
    "num_labels": len(task_columns),            # NEW
    "tree_learner": "serial2",                  # NEW (required for multi-task)
}
```

### 4. Update _train_model() Method (Lines 169-177)

```python
# BEFORE - Uses lgb.train
self.model = lgb.train(
    train_params,
    train_data,
    num_boost_round=self.hyperparams.num_iterations,
    ...
)

# AFTER - Uses train() and sets num_labels
self.model = train(
    train_params,
    train_data,
    num_boost_round=self.hyperparams.num_iterations,
    fobj=self.loss_function.objective,
    ...
)
# CRITICAL: Set number of labels after training
self.model.set_num_labels(len(task_columns))
```

### 5. Update _predict() Method (Lines 192-205)

```python
# BEFORE - Assumes single output
predictions = self.model.predict(X)  # Shape: [N_samples]

# AFTER - Handle multi-task output
predictions = self.model.predict(X)  # Shape: [N_samples, N_tasks]
# Apply sigmoid if needed
predictions = expit(predictions)
return predictions
```

### 6. Update _save_model() Method (Lines 216-217)

```python
# BEFORE - Standard save
self.model.save_model(str(model_file))

# AFTER - Custom save preserves multi-task info
self.model.save_model2(str(model_file))
```

### 7. Update _load_model() Method (Lines 239-241)

```python
# BEFORE - Standard load
self.model = lgb.Booster(model_file=str(model_file))

# AFTER - Custom load + restore num_labels
self.model = Booster(model_file=str(model_file))
# Restore num_labels from saved hyperparameters
self.model.set_num_labels(len(self.hyperparams.task_label_names))
```

## Implementation Workflow

### Phase 1: Update mtgbm_model.py (2-3 hours)

1. Update imports
2. Modify _prepare_data() for native 2D labels
3. Add multi-task parameters in _initialize_model()
4. Update _train_model() with set_num_labels() call
5. Handle multi-dimensional output in _predict()
6. Use save_model2() in _save_model()
7. Restore num_labels in _load_model()

### Phase 2: Testing (1-2 hours)

1. **Library Loading Test**
   ```python
   from models.lightgbmmt import find_lib_path
   lib_path = find_lib_path()
   # Verify custom functions exist
   ```

2. **Dataset Creation Test**
   ```python
   from models.lightgbmmt import Dataset
   labels_2d = np.random.rand(100, 4)
   dataset = Dataset(X, label=labels_2d)
   assert dataset.label.shape == (100, 4)
   ```

3. **Training Test**
   ```python
   # Train with refactored loss function
   # Verify convergence
   ```

4. **Prediction Shape Test**
   ```python
   preds = model.predict(X_test)
   assert preds.shape == (len(X_test), num_tasks)
   ```

### Phase 3: Validation (1 hour)

- Compare predictions with legacy implementation
- Verify all loss types work (fixed, adaptive, adaptive_kd)
- Check memory usage and performance
- Run full integration test

## Key Integration Points

### Compatibility with Refactored Loss Functions

**No changes needed to loss functions!** They already return the correct format:

```python
def objective(self, preds, train_data, ep):
    """
    Returns 4 arrays for multi-task:
    - grad_main: Main task gradients
    - hess_main: Main task hessians
    - grad_sub: Sub-task gradients
    - hess_sub: Sub-task hessians
    """
    # Loss function logic unchanged
    return grad_main, hess_main, grad_sub, hess_sub
```

The custom `train()` function in lightgbmmt passes these to the C++ library which handles them correctly.

### Library Path Configuration

The `lightgbmmt/libpath.py` module automatically finds the library:

```python
# Search order:
# 1. Relative path: ../compile/lib_lightgbm.so
# 2. Environment variable: LIGHTGBMMT_LIB
# 3. Raises error if not found
```

**Current Setup:** Library is at `models/compile/lib_lightgbm.so` - automatically found!

## Benefits Summary

### Technical Achievements

1. **Multi-Task Predictions Enabled**
   - Native [N_samples, N_tasks] output
   - No separate models needed
   - Shared tree structures

2. **All Refactoring Preserved**
   - Loss functions unchanged
   - Clean architecture maintained  
   - Code quality improvements kept

3. **Minimal Integration Effort**
   - Only 1 file needs updates (mtgbm_model.py)
   - 7 specific method changes
   - Clear, documented changes

### Operational Benefits

1. **Ready for Production**
   - Pre-compiled library included
   - Complete Python wrapper present
   - Works with existing Cursus framework

2. **Easy to Test**
   - Isolated changes in one file
   - Clear success criteria
   - Gradual rollout possible

3. **Maintainable**
   - Complete package included
   - Can recompile if needed
   - Clear documentation

## Testing Checklist

### Unit Tests
- [ ] Library loads successfully
- [ ] Dataset accepts 2D labels
- [ ] Booster requires multi-task params
- [ ] Multi-dimensional predictions work
- [ ] Save/load preserves num_labels

### Integration Tests
- [ ] Refactored loss functions work
- [ ] All loss types train successfully
- [ ] Predictions match legacy (< 0.1% diff)
- [ ] Model save/load works correctly
- [ ] End-to-end pipeline completes

### Performance Tests
- [ ] Training time within 10% of legacy
- [ ] Memory usage within 20% of legacy
- [ ] Prediction latency acceptable
- [ ] Scales to production data sizes

## Success Criteria

### Must Have
✅ Multi-task training completes
✅ Multi-dimensional predictions work
✅ All 3 loss types functional
✅ Predictions accurate (vs legacy)
✅ Model persistence works

### Should Have
✅ Code coverage > 80%
✅ Documentation complete
✅ Performance acceptable
✅ Integration tests pass

## Migration Plan

### Immediate Actions (Today)
1. Update mtgbm_model.py imports
2. Modify 7 methods as specified
3. Run basic smoke tests
4. Verify library loading

### Short Term (This Week)
1. Comprehensive unit tests
2. Integration test suite
3. Performance benchmarking
4. Documentation updates

### Medium Term (Next Week)
1. Deploy to staging
2. Validate on real data
3. Production deployment
4. Monitor performance

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| C++ library incompatibility | Low | High | Test on target platform, recompile if needed |
| Python API changes | Low | Medium | Minimal wrapper API, well-documented |
| Performance regression | Low | Medium | Benchmark before/after, optimize if needed |
| Integration bugs | Medium | Low | Comprehensive tests, gradual rollout |

## Conclusion

The integration is **significantly simpler than originally planned** because:

1. **lightgbmmt package already present** - No extraction needed
2. **C++ library already compiled** - Ready to use
3. **Only 1 file needs updates** - Focused, manageable scope
4. **Clear change specification** - 7 documented method updates
5. **All refactoring preserved** - No loss of improvements

**Estimated Time:** 4-6 hours total (down from original 1-2 weeks estimate)

**Next Step:** Update `projects/cap_mtgbm/dockers/models/implementations/mtgbm_model.py` following the 7 changes specified above.

---

**Document Version:** 2.0  
**Last Updated:** 2025-12-12  
**Status:** Ready for Implementation  
**Implementation Timeline:** 1 day
