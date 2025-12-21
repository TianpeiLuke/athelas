---
tags:
  - analysis
  - critical-issue
  - design-flaw
  - multi-task-learning
  - lightgbm
  - architecture
keywords:
  - design flaw
  - prediction limitation
  - lightgbmmt dependency
  - multi-task output
  - refactoring blocker
topics:
  - software architecture
  - design validation
  - technical debt
  - system constraints
language: python
date of note: 2025-12-12
---

# MTGBM Refactoring Design Critical Issues

## Executive Summary

**🚨 CRITICAL DESIGN FLAW IDENTIFIED**

The MTGBM refactoring designs ([loss function refactoring](../1_design/mtgbm_models_refactoring_design.md) and [model classes refactoring](../1_design/mtgbm_model_classes_refactoring_design.md)) contain a **fundamental architectural flaw** that makes the proposed designs **non-functional for production use**.

**The Issue:**
Both designs assume that **multi-task prediction works with standard LightGBM**, but this is **architecturally impossible** because:
1. Standard LightGBM models output **single values per prediction**
2. Multi-task prediction requires **multiple outputs per sample**
3. The refactored code uses standard LightGBM (not the custom lightgbmmt fork)

**Impact:**
- ✅ **Training:** Fully functional (custom loss works during training)
- ❌ **Inference:** Completely non-functional (no way to generate multi-task predictions)
- ❌ **Evaluation:** Blocked by inference failure
- ❌ **Production deployment:** Impossible without fixes

**Status:** The refactored designs are **incomplete and require major revisions** before implementation can proceed.

## Related Documents

- **[LightGBMMT Package Architecture Critical Analysis](./2025-12-12_lightgbmmt_package_architecture_critical_analysis.md)** - **PRIMARY** - Detailed technical analysis of the constraint
- **[MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)** - Loss function refactoring design (flawed)
- **[MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md)** - Model architecture refactoring design (flawed)
- **[LightGBMMT Package Correspondence Analysis](./2025-12-10_lightgbmmt_package_correspondence_analysis.md)** - Package vs script correspondence

---

## 1. Critical Flaw: Prediction Method Assumption

### 1.1 Design Assumption (INCORRECT)

Both refactoring designs assume the following workflow works:

```python
# From mtgbm_model_classes_refactoring_design.md
class BaseMultiTaskModel(ABC):
    def predict(self, X: np.ndarray, use_best_epoch: bool = True) -> np.ndarray:
        """Make predictions using trained models."""
        predictions = np.zeros((X.shape[0], self.config.num_tasks))
        
        for e in range(epoch + 1):
            for task_idx in range(self.config.num_tasks):
                model = self.state.models.get((e, task_idx))
                
                # 🚨 THIS ASSUMES model.predict() RETURNS MULTI-TASK OUTPUT
                predictions[:, task_idx] = model.predict(X_task)  # ❌ WRONG
        
        return predictions  # Shape: [N_samples, N_tasks]
```

**Why This Appears to Work:**
- During **training**, the design works perfectly because:
  - Custom loss function receives predictions: `[N_samples * N_tasks]` flattened
  - Loss function reshapes to: `[N_samples, N_tasks]`
  - Custom gradients/hessians guide multi-task learning
  - **Training succeeds** ✅

**Why This Fails:**
- During **inference**, standard LightGBM:
  - Each model outputs: `[N_samples]` - single task only
  - Cannot output: `[N_samples, N_tasks]` - multiple tasks
  - No mechanism to combine outputs into multi-task format
  - **Inference fails** ❌

### 1.2 Reality: Standard LightGBM Constraint

**Standard LightGBM Architecture:**
```python
import lightgbm as lgb

# Train single-output model
