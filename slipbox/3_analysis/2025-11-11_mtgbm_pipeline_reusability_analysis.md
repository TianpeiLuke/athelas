---
tags:
  - analysis
  - pipeline
  - reusability
  - mtgbm
  - lightgbm
  - multi_task
keywords:
  - pipeline reusability
  - XGBoost pipeline
  - MTGBM pipeline
  - step reusability
  - preprocessing
  - calibration
topics:
  - pipeline design
  - component reusability
  - multi-task learning
language: python
date of note: 2025-11-11
---

# MTGBM Pipeline Reusability Analysis

## Executive Summary

This document analyzes the reusability of existing Cursus pipeline components for the MTGBM (Multi-Task Gradient Boosting Machine) implementation. Based on analysis of the XGBoost Complete E2E pipeline and existing step builders, approximately **50-60% of pipeline components can be reused** with varying degrees of adaptation.

## Analysis Scope

### Documents Reviewed
1. **XGBoost Complete E2E DAG**: `src/cursus/pipeline_catalog/shared_dags/xgboost/complete_e2e_dag.py`
2. **Available Step Builders**: 33 builders in `src/cursus/steps/builders/`
3. **Tabular Preprocessing Script**: `src/cursus/steps/scripts/tabular_preprocessing.py`
4. **Model Calibration Script**: `src/cursus/steps/scripts/model_calibration.py`
5. **MTGBM Requirements**: From `2025-11-10_lightgbmmt_multi_task_implementation_analysis.md`

### Key Findings
- ✅ **Data Loading**: Fully reusable (CradleDataLoading)
- ⚠️ **Preprocessing**: Partially reusable (lacks multi-label logic)
- ❌ **Training**: Requires new multi-task training step
- ⚠️ **Evaluation**: Needs multi-task metrics adaptation
- ⚠️ **Calibration**: May need per-task calibration
- ✅ **Packaging/Registration**: Fully reusable

---

## 1. XGBoost Complete E2E Pipeline Structure

### Pipeline Components (10 Steps)

```python
# Training Flow
1. CradleDataLoading_training          # Data acquisition from DAWS
2. TabularPreprocessing_training       # Standard preprocessing
3. XGBoostTraining                     # Model training

# Calibration Flow
4. CradleDataLoading_calibration       # Calibration data
5. TabularPreprocessing_calibration    # Calibration preprocessing

# Evaluation Flow
6. XGBoostModelEval_calibration        # Model evaluation

# Calibration & Output Flow
7. ModelCalibration_calibration        # Probability calibration
8. Package                             # Model artifact packaging
9. Payload                             # Payload generation for testing
10. Registration                       # MIMS model registration
```

### Pipeline Characteristics
- **Node Count**: 10 nodes, 11 edges
- **Entry Points**: CradleDataLoading (training & calibration)
- **Exit Points**: Registration
- **Complexity**: Comprehensive end-to-end workflow
- **Features**: Training, calibration, packaging, registration, evaluation

---

## 2. Reusability Assessment by Component

### 2.1 Data Loading Steps ✅ FULLY REUSABLE

#### Step Information
- **Builder**: `builder_cradle_data_loading_step.py`
- **Contract**: Cradle Data Loading Contract
- **Script**: DAWS integration via Cradle API

#### Capabilities
- ✅ Loads data from DAWS workflows
- ✅ Supports multiple data formats (CSV, TSV, Parquet, JSON)
- ✅ Handles training/calibration/testing variants
- ✅ Integrates with signature files for column definitions

#### MTGBM Usage
**Direct Reuse**: Can load multi-label training data from DAWS workflows without modification.

**Comparison with MTGBM DataDownloader.py**:
```python
# MTGBM DataDownloader.py functionality:
- Download from DAWS workflow
- Support multiple formats
- Handle data partitioning

# CradleDataLoading provides:
- All of the above ✅
- Plus signature file integration ✅
- Plus better error handling ✅
- Plus contract-based validation ✅
```

**✅ CONFIRMED**: DataDownloader.py is **completely replaced** by CradleDataLoading step.
No need to keep DataDownloader.py - CradleDataLoading provides superior functionality.

**Recommendation**: ✅ **Use existing CradleDataLoading step directly, eliminate DataDownloader.py**

---

### 2.2 Preprocessing Steps ⚠️ PARTIALLY REUSABLE

#### Step Information
- **Builder**: `builder_tabular_preprocessing_step.py`
- **Script**: `tabular_preprocessing.py`
- **Contract**: Tabular Preprocess Contract

#### Current Capabilities

**Data Loading**:
```python
- Combines data shards from directory
- Supports CSV, TSV, Parquet, JSON (plain and gzipped)
- Auto-detects separators for CSV/TSV
- Memory-efficient iterative concatenation
- Signature file integration
```

**Label Processing**:
```python
- Handles categorical → numeric conversion
- Supports stratified train/test/val splits
- Handles missing label values
- Can work without labels (for inference)
```

**Format Preservation**:
```python
- Detects input format automatically
- Preserves format in output (CSV/TSV/Parquet)
- Configurable via OUTPUT_FORMAT env variable
```

**Configuration Parameters**:
```python
Environment Variables:
- LABEL_FIELD: Label column name
- TRAIN_RATIO: Training split ratio (default 0.7)
- TEST_VAL_RATIO: Test/validation split ratio (default 0.5)
- OUTPUT_FORMAT: Output format (CSV/TSV/Parquet)

Arguments:
- --job_type: training|validation|testing|calibration
```

#### MTGBM-Specific Requirements

**What MTGBM DataProcessor.py Does**:

1. **Payment Method Filtering** (clean_sample):
```python
def clean_sample(df: pd.DataFrame, paymeth_list: list) -> pd.DataFrame:
    """
    Filter samples to include only specified payment methods.
    
    Example:
    paymeth_list = ['CC', 'DC', 'ACH', 'AMEX', 'INVOICE']
    df = df[df['payment_method'].isin(paymeth_list)]
    """
```

2. **Multi-Label Creation** (create_paymeth_label):
```python
def create_paymeth_label(
    df: pd.DataFrame, 
    target_label: str,
    paymeth_list: list,
    paymeth_col: str = 'payment_method'
) -> pd.DataFrame:
    """
    Create payment-method-specific subtask labels.
    
    Example:
    target_label = 'is_fraud'
    paymeth_list = ['CC', 'DC', 'ACH', 'AMEX', 'INVOICE']
    
    Creates columns:
    - is_fraud_CC
    - is_fraud_DC
    - is_fraud_ACH
    - is_fraud_AMEX
    - is_fraud_INVOICE
    
    Each column contains:
    - 1 if (payment_method == X AND is_fraud == 1)
    - 0 if (payment_method == X AND is_fraud == 0)
    - NaN if payment_method != X
    """
```

3. **Multi-Label Dataset Structure**:
```python
Final DataFrame structure:
- Original features
- Main label: is_fraud (binary)
- Subtask labels: is_fraud_CC, is_fraud_DC, ..., is_fraud_INVOICE
- Each subtask is sparse (mostly NaN for non-matching payment methods)
```

#### Gap Analysis

| Feature | Tabular Preprocessing | MTGBM DataProcessor | Gap |
|---------|----------------------|---------------------|-----|
| Load multiple formats | ✅ Yes | ✅ Yes | None |
| Signature integration | ✅ Yes | ❌ No | Enhancement |
| Format preservation | ✅ Yes | ❌ No | Enhancement |
| Label processing | ✅ Yes (single) | ✅ Yes | None |
| Train/test/val split | ✅ Yes | ❌ No | MTGBM doesn't need |
| Payment method filtering | ❌ No | ✅ Yes | **CRITICAL GAP** |
| Multi-label generation | ❌ No | ✅ Yes | **CRITICAL GAP** |
| Sparse label handling | ❌ No | ✅ Yes | **CRITICAL GAP** |

#### Label-Free Processing Capability

**IMPORTANT DISCOVERY**: `tabular_preprocessing.py` has **conditional** label-free processing:

```python
# From tabular_preprocessing.py __main__ section (lines 397-400):
LABEL_FIELD = os.environ.get("LABEL_FIELD")
if not LABEL_FIELD and args.job_type != "calibration":
    raise RuntimeError("LABEL_FIELD environment variable must be set.")

# From main() function (lines 278-295):
# Only process labels if label_field is provided and exists
if label_field:
    # Label processing: numeric conversion, encoding, stratified splits
    ...
else:
    log("[INFO] No label field provided, skipping label processing")
    
# Splitting logic:
if job_type == "training":
    if label_field:
        # Stratified splits using label
        train_df, holdout_df = train_test_split(..., stratify=df[label_field])
    else:
        # Random splits without stratification
        train_df, holdout_df = train_test_split(..., random_state=42)
```

**Key Constraints**:
- ✅ **Calibration job_type**: Can work WITHOUT LABEL_FIELD (label-free mode)
- ❌ **Training/validation/testing job_types**: REQUIRES LABEL_FIELD (throws error if missing)
- ✅ When LABEL_FIELD not provided (calibration only): Uses random splits, no label processing
- ✅ When LABEL_FIELD provided: Performs label encoding and stratified splits

**Implications for MTGBM**:
- ❌ **Cannot** use TabularPreprocessing in label-free mode for training (requires LABEL_FIELD)
- ✅ Can use TabularPreprocessing for calibration/inference without labels
- ⚠️ For MTGBM training, still need custom preprocessing with multi-label logic
- ⚠️ Still need payment method filtering and multi-label generation

#### Recommendations

**Option A: Create Custom MtgbmPreprocessing Step** ⭐ RECOMMENDED
```python
# New step: builder_mtgbm_preprocessing_step.py
Features:
✅ Reuse tabular_preprocessing.py for data loading (label-free mode)
✅ Add payment method filtering logic
✅ Add multi-label generation logic
✅ Add sparse label handling
✅ Maintain format preservation
✅ Reuse signature integration

Benefits:
+ Clean separation of concerns
+ Leverages existing label-free capability
+ Reusable for other multi-task scenarios
+ Testable independently
+ Follows framework patterns
```

**Option B: Extend Tabular Preprocessing Script**
```python
# Modify tabular_preprocessing.py
Add environment variables:
- ENABLE_MULTI_LABEL: true|false
- PAYMENT_METHOD_COLUMN: column name
- PAYMENT_METHOD_LIST: comma-separated list
- MULTI_LABEL_PREFIX: prefix for subtask columns

Benefits:
+ Single preprocessing step
- Less clear for non-MTGBM users
- More complex configuration
```

**Option C: Package DataProcessor as Separate Script**
```python
# Keep DataProcessor.py as-is, package with training step
Run sequence:
1. TabularPreprocessing (standard preprocessing)
2. DataProcessor.py (multi-label logic in training container)

Benefits:
+ Faster development
- Duplicates preprocessing logic
- Harder to test independently
- Less reusable
```

**Final Recommendation**: ⭐ **Option A** - Create dedicated `MtgbmPreprocessing` step

---

### 2.3 Training Steps ❌ REQUIRES NEW IMPLEMENTATION

#### Existing Steps
- `builder_lightgbm_training_step.py` - Single-task LightGBM (EXISTS)
- `builder_xgboost_training_step.py` - Single-task XGBoost

#### Gap Analysis
| Feature | Existing LightGBM | MTGBM Requirements | Gap |
|---------|------------------|-------------------|-----|
| Single-task training | ✅ Yes | N/A | N/A |
| Multi-task training | ❌ No | ✅ Required | **CRITICAL** |
| Custom loss functions | ❌ No | ✅ Required (base, auto_weight, auto_weight_KD) | **CRITICAL** |
| lightgbmmt library | ❌ No | ✅ Required | **CRITICAL** |
| Adaptive weighting | ❌ No | ✅ Required | **CRITICAL** |
| Knowledge distillation | ❌ No | ✅ Required | **CRITICAL** |
| Multi-label dataset | ❌ No | ✅ Required | **CRITICAL** |

#### Required Implementation
```python
# New step: builder_lightgbmmt_training_step.py
# Status: DESIGNED ✅ (see lightgbm_multi_task_training_step_design.md)

Key Components:
1. Multi-task hyperparameters
   - main_target: Main task label
   - sub_tasks_list: List of subtask labels
   - loss_type: base|auto_weight|auto_weight_KD

2. lightgbmmt library packaging
   - Custom C library (libmtgbm.so)
   - Python wrapper (mtgbm.py)
   - Mtgbm.py training script

3. Multi-task model artifacts
   - model.txt (LightGBM format)
   - metadata.json (task structure, weights)
   - config.json (hyperparameters)
```

**Recommendation**: ❌ **Create new LightgbmmtTraining step** (design completed)

---

### 2.4 Evaluation Steps ⚠️ NEEDS MULTI-TASK ADAPTATION

#### Existing Steps
- `builder_xgboost_model_eval_step.py` - Single-task evaluation
- Script: `xgboost_model_eval.py`

#### Current Capabilities
```python
Single-Task Metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion matrix
- Classification report
- Prediction distribution plots
- ROC curves
```

#### MTGBM Requirements
```python
Multi-Task Metrics (per task):
- Per-task accuracy, precision, recall, F1
- Per-task ROC-AUC
- Per-task confusion matrices
- Main task vs subtask performance comparison
- Weight evolution tracking (for adaptive weighting)
- Task correlation analysis

Visualization Requirements:
- Multiple ROC curves (main task + each subtask)
- Performance comparison bar charts
- Weight evolution over training iterations
- Task-specific prediction distributions
```

#### Gap Analysis
| Feature | XGBoost Eval | MTGBM Requirements | Gap |
|---------|-------------|-------------------|-----|
| Single-task metrics | ✅ Yes | ✅ Required | None |
| Multi-task metrics | ❌ No | ✅ Required | **CRITICAL** |
| Per-task visualizations | ❌ No | ✅ Required | **CRITICAL** |
| Weight tracking | ❌ No | ✅ Optional | Enhancement |
| Task correlation | ❌ No | ✅ Optional | Enhancement |

#### Recommendations

**Option A: Create LightgbmmtModelEval Step** ⭐ RECOMMENDED
```python
# New step: builder_lightgbmmt_model_eval_step.py
Features:
✅ Per-task metric computation
✅ Multi-task visualization
✅ Main task emphasis
✅ Subtask comparison
✅ Weight evolution tracking (optional)

Benefits:
+ Clean separation from single-task evaluation
+ Can optimize for multi-task scenarios
+ Better monitoring capabilities
```

**Option B: Extend XGBoost Eval**
```python
# Modify xgboost_model_eval.py
Add multi-task detection:
- If num_classes > 1 AND has subtasks → multi-task mode
- Generate per-task metrics
- Create comparison visualizations

Benefits:
+ Single evaluation step
- More complex configuration
- Harder to maintain
```

**Final Recommendation**: ⭐ **Option A** - Create dedicated `LightgbmmtModelEval` step

---

### 2.5 Calibration Steps ⚠️ MAY NEED ADAPTATION

#### Step Information
- **Builder**: `builder_model_calibration_step.py`
- **Script**: `model_calibration.py`
- **Alternate**: `builder_percentile_model_calibration_step.py`

#### Current Capabilities

**Calibration Methods**:
```python
1. GAM (Generalized Additive Models)
   - Supports monotonic constraints
   - Configurable number of splines
   - Requires pygam package

2. Isotonic Regression
   - Simple monotonic calibration
   - No hyperparameters
   - Faster than GAM

3. Platt Scaling
   - Logistic regression calibration
   - Minimal regularization
   - Fast and simple
```

**Binary Classification Support**:
```python
- Single probability score calibration
- Reliability diagrams (before/after)
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier score
- AUC preservation check
```

**Multi-Class Classification Support**:
```python
- One-vs-rest calibration approach
- Per-class calibration models
- Probability normalization
- Macro-averaged metrics
- Per-class reliability diagrams
- Multi-class Brier score
```

**Format Preservation**:
```python
- Auto-detects input format (CSV/TSV/Parquet)
- Preserves format in output
- Adds calibrated probability columns
- Example: prob_class_1 → calibrated_prob_class_1
```

**Configuration**:
```python
Environment Variables:
- CALIBRATION_METHOD: gam|isotonic|platt
- LABEL_FIELD: Label column name
- SCORE_FIELD: Probability score column (binary)
- SCORE_FIELD_PREFIX: Prefix for multi-class probabilities
- IS_BINARY: true|false
- NUM_CLASSES: Number of classes
- MULTICLASS_CATEGORIES: Class names as JSON array
- MONOTONIC_CONSTRAINT: true|false
- GAM_SPLINES: Number of splines for GAM
```

#### MTGBM Calibration Requirements

**Multi-Task Calibration Questions**:

1. **Calibration Scope**:
   - ❓ Should we calibrate the main task only?
   - ❓ Should we calibrate all subtasks independently?
   - ❓ Should subtask calibration consider main task?

2. **Calibration Strategy**:
   - ❓ Use one-vs-rest approach (like multi-class)?
   - ❓ Calibrate jointly with task correlations?
   - ❓ Use different methods per task?

3. **Practical Considerations**:
   - Main task fraud detection is critical → must calibrate
   - Subtasks (payment-method specific) may have sparse data
   - Some subtasks may not need calibration if rarely used

#### Gap Analysis

| Feature | Model Calibration | MTGBM Requirements | Gap |
|---------|------------------|-------------------|-----|
| Binary calibration | ✅ Yes | ✅ Required (main task) | None |
| Multi-class calibration | ✅ Yes (one-vs-rest) | N/A | N/A |
| Multi-label calibration | ❌ No | ✅ Required | **CRITICAL GAP** |
| Multi-task calibration | ❌ No | ✅ Required | **CRITICAL GAP** |
| Per-task calibration | ❌ No | ✅ Required | **CRITICAL GAP** |
| Sparse data handling | ⚠️ Limited | ✅ Required for subtasks | **POTENTIAL GAP** |
| Task-aware calibration | ❌ No | ✅ Required | **CRITICAL GAP** |

**⚠️ CRITICAL CLARIFICATION**: 
- Existing `ModelCalibration` is **binary-only** (single probability score)
- Multi-class calibration exists but uses **one-vs-rest**, not suitable for multi-label/multi-task
- MTGBM requires **multi-label calibration** where each task has independent probability
- This is fundamentally different from multi-class (mutually exclusive) calibration

#### Recommendations

**Phase 1: Main Task Only** ⭐ RECOMMENDED FOR MVP
```python
# Use existing ModelCalibration for main task
Configuration:
- IS_BINARY: true
- LABEL_FIELD: is_fraud (main task)
- SCORE_FIELD: prob_class_1 (main task probability)
- CALIBRATION_METHOD: gam (most flexible)

Benefits:
+ Can use existing binary calibration step as-is
+ Focuses on most critical task
+ Faster development
+ Lower complexity
```

**Phase 2: Multi-Label/Multi-Task Calibration** ❌ **REQUIRES NEW STEP**
```python
# Create new MultilabelModelCalibration step
# Status: NEW IMPLEMENTATION REQUIRED

Features needed:
✅ Independent calibration per task (not one-vs-rest)
✅ Sparse label handling (many NaN values per subtask)
✅ Per-task calibration models
✅ Task-specific reliability diagrams
✅ Multi-label metrics (not multi-class metrics)
✅ Handle imbalanced subtask data

Key Difference from Multi-Class:
- Multi-class: Probabilities sum to 1 (mutually exclusive)
- Multi-label: Each task independent (probabilities don't sum to 1)
- Multi-task: Like multi-label but with task correlations

Implementation approach:
1. Train independent calibrator per task (like binary)
2. Handle sparse labels (filter NaN before calibration)
3. No probability normalization (unlike multi-class)
4. Per-task evaluation metrics
```

**Final Recommendation**: 
- ⭐ **MVP**: Use existing `ModelCalibration` for main task only
- ❌ **Production**: Create new `MultilabelModelCalibration` step for all tasks
- 📋 **Design**: Required before Phase 2 implementation

---

### 2.6 Packaging & Registration Steps ✅ FULLY REUSABLE

#### Step Information
- **Package**: `builder_package_step.py` → `package.py`
- **Payload**: `builder_payload_step.py` → `payload.py`
- **Registration**: `builder_registration_step.py` → MIMS registration

#### Why Fully Reusable

**Framework-Agnostic Design**:
```python
Package Step:
- Packages any model artifacts from /opt/ml/model/
- Creates model.tar.gz
- Supports arbitrary model formats
- Works with LightGBM, XGBoost, PyTorch, TensorFlow, etc.

Payload Step:
- Tests model inference endpoint
- Sends sample predictions
- Validates model responsiveness
- Framework-independent

Registration Step:
- Registers with MIMS (Model Inventory Management System)
- Handles model metadata
- Framework-agnostic registration
```

#### MTGBM Model Artifacts
```python
Expected model artifacts from LightgbmmtTraining:
/opt/ml/model/
├── model.txt                  # LightGBM model file
├── metadata.json              # Multi-task structure, weights
├── config.json               # Hyperparameters
└── inference_handler.py      # Custom inference logic (optional)

Package step will:
✅ Package all of these into model.tar.gz
✅ Upload to S3
✅ Provide artifact location to Registration
```

**Recommendation**: ✅ **Use existing Package, Payload, Registration steps directly**

---

## 3. Recommended MTGBM Pipeline Structure

### Option 1: Complete E2E Pipeline ⭐ RECOMMENDED

```python
Pipeline: MTGBM Complete E2E
Nodes: 10 steps
Reusability: 50% fully reusable, 20% adaptable, 30% new

# Training Flow
1. CradleDataLoading_training          # ✅ REUSE (existing)
2. MtgbmPreprocessing_training         # ❌ NEW (custom multi-label)
3. LightgbmmtTraining                  # ❌ NEW (multi-task)

# Testing Flow
4. CradleDataLoading_testing           # ✅ REUSE (existing)
5. MtgbmPreprocessing_testing          # ❌ NEW (custom multi-label)
6. LightgbmmtModelEval_testing         # ❌ NEW (multi-task metrics)

# Calibration Flow (Main Task Only)
7. ModelCalibration_calibration        # ⚠️ ADAPT (main task config)

# Output Flow
8. Package                             # ✅ REUSE (existing)
9. Payload                             # ✅ REUSE (existing)
10. Registration                       # ✅ REUSE (existing)
```

#### Advantages
- ✅ Comprehensive monitoring (separate test evaluation)
- ✅ Production-ready calibration
- ✅ Full artifact management
- ✅ Follows XGBoost pipeline pattern
- ✅ Easy to understand for teams familiar with XGBoost pipeline

#### Disadvantages
- More steps to implement (6 custom/adapted vs 4 reused)
- Longer pipeline execution time
- More complex configuration

---

### Option 2: Simplified Training Pipeline

```python
Pipeline: MTGBM Simple Training
Nodes: 5 steps
Reusability: 60% fully reusable, 0% adaptable, 40% new

# Training Flow
1. CradleDataLoading_training          # ✅ REUSE
2. MtgbmPreprocessing_training         # ❌ NEW
3. LightgbmmtTraining                  # ❌ NEW (includes basic eval in training)

# Output Flow
4. Package                             # ✅ REUSE
5. Registration                        # ✅ REUSE
```

#### Advantages
- ✅ Faster development (fewer steps)
- ✅ Simpler configuration
- ✅ Faster pipeline execution
- ✅ Good for iterative development/experimentation

#### Disadvantages
- ❌ No separate evaluation step (harder to monitor)
- ❌ No calibration (may need for production)
- ❌ Evaluation logic embedded in training (less flexible)

---

### Option 3: Training with Evaluation

```python
Pipeline: MTGBM Training + Eval
Nodes: 7 steps
Reusability: 57% fully reusable, 14% adaptable, 29% new

# Training Flow
1. CradleDataLoading_training          # ✅ REUSE
2. MtgbmPreprocessing_training         # ❌ NEW
3. LightgbmmtTraining                  # ❌ NEW

# Evaluation Flow
4. CradleDataLoading_testing           # ✅ REUSE
5. MtgbmPreprocessing_testing          # ❌ NEW
6. LightgbmmtModelEval_testing         # ❌ NEW

# Output Flow
7. Package                             # ✅ REUSE
8. Registration                        # ✅ REUSE
```

#### Advantages
- ✅ Balance between simplicity and monitoring
- ✅ Separate evaluation for better metrics
- ✅ No calibration complexity for MVP
- ✅ Follows training_with_evaluation_dag pattern

#### Disadvantages
- ⚠️ Missing calibration (may need later)
- ⚠️ No payload testing

**Recommendation for MVP**: ⭐ **Option 3** (Training + Eval)
**Recommendation for Production**: ⭐ **Option 1** (Complete E2E)

---

## 4. Implementation Roadmap

### Phase 1: Core Components (MVP)
```
Priority: CRITICAL
Timeline: 2-3 weeks

1. ❌ MtgbmPreprocessing Step
   - Inherit tabular preprocessing capabilities
   - Add payment method filtering
   - Add multi-label generation
   - Add sparse label handling
   - Status: Design needed

2. ❌ LightgbmmtTraining Step
   - Multi-task training logic
   - lightgbmmt library integration
   - Custom loss functions
   - Status: Design completed ✅

3. ❌ LightgbmmtModelEval Step
   - Per-task metrics
   - Multi-task visualizations
   - Performance comparison
   - Status: Design needed

4. 📋 MTGBM Training + Eval Pipeline
   - DAG definition
   - Config templates
   - Integration tests
   - Status: Design needed
```

### Phase 2: Production Features
```
Priority: HIGH
Timeline: 1-2 weeks

1. ⚠️ ModelCalibration Configuration
   - Main task calibration setup
   - Config templates
   - Integration testing
   - Status: Adaptation needed

2. ✅ Package/Payload/Registration
   - Verify artifact compatibility
   - Test model packaging
   - MIMS registration testing
   - Status: Validation needed

3. 📋 MTGBM Complete E2E Pipeline
   - Full DAG definition
   - Config templates
   - End-to-end testing
   - Status: Design needed
```

### Phase 3: Enhancements (Future)
```
Priority: MEDIUM
Timeline: 2-4 weeks

1. 📋 Multi-Task Calibration
   - Evaluate subtask calibration need
   - Design task-aware calibration
   - Implement if beneficial
   - Status: Research needed

2. 📋 Weight Evolution Monitoring
   - Track adaptive weight changes
   - Visualize weight dynamics
   - Alert on unexpected behavior
   - Status: Design needed

3. 📋 Task Correlation Analysis
   - Analyze main task vs subtask relationships
   - Identify beneficial/harmful correlations
   - Guide model improvements
   - Status: Research needed
```

---

## 5. Key Decisions Required

### Decision 1: Preprocessing Architecture
**Question**: Should MTGBM preprocessing be a separate step or extension of tabular preprocessing?

**Options**:
- A: Dedicated MtgbmPreprocessing step ⭐ RECOMMENDED
- B: Extend TabularPreprocessing with multi-label flag
- C: Keep DataProcessor.py in training container

**Recommendation**: **Option A** - Clean separation, better reusability

---

### Decision 2: Evaluation Strategy
**Question**: How comprehensive should MTGBM evaluation be?

**Options**:
- A: Basic metrics in training script (faster development)
- B: Separate comprehensive evaluation step ⭐ RECOMMENDED
- C: Both (training metrics + detailed evaluation)

**Recommendation**: **Option B** for MVP, **Option C** for production

---

### Decision 3: Calibration Scope
**Question**: What should be calibrated?

**Options**:
- A: Main task only ⭐ RECOMMENDED for MVP
- B: Main task + all subtasks
- C: Main task + selected subtasks based on usage

**Recommendation**: **Option A** for MVP, evaluate **Option C** for production

---

### Decision 4: Pipeline Variant
**Question**: Which pipeline structure to implement first?

**Options**:
- A: Complete E2E (10 steps)
- B: Simplified Training (5 steps)
- C: Training + Eval (7 steps) ⭐ RECOMMENDED

**Recommendation**: **Option C** for MVP, **Option A** for production

---

## 6. Risk Assessment

### High Risk
```
❌ Multi-Task Training Implementation
   Risk: Complex custom loss functions, library integration
   Mitigation: Design completed, follow XGBoost training patterns
   
❌ Multi-Label Preprocessing
   Risk: Sparse label handling, payment method filtering logic
   Mitigation: Clear requirements from DataProcessor.py analysis
```

### Medium Risk
```
⚠️ Model Calibration Adaptation
   Risk: Unclear if subtask calibration needed
   Mitigation: Start with main task only, monitor production metrics
   
⚠️ Evaluation Metrics
   Risk: Per-task metrics may be complex
   Mitigation: Leverage existing evaluation patterns, add multi-task logic
```

### Low Risk
```
✅ Data Loading
   Risk: Minimal, existing step proven
   Mitigation: Direct reuse
   
✅ Packaging/Registration
   Risk: Minimal, framework-agnostic
   Mitigation: Validate artifact structure only
```

---

## 7. Comparison with Original MTGBM Implementation

### What MTGBM Project Has
```python
projects/cap_mtgbm/docker/
├── DataDownloader.py          # → Replace with CradleDataLoading ✅
├── DataProcessor.py           # → Basis for MtgbmPreprocessing ⚠️
├── Mtgbm.py                   # → Core training script ✅
└── models/
    └── mtgbm.py              # → Custom loss implementation ✅
```

### What Cursus Framework Provides
```python
Reusable Components:
✅ CradleDataLoading (better than DataDownloader)
✅ Format preservation (not in MTGBM)
✅ Signature integration (not in MTGBM)
✅ Calibration pipeline (not in MTGBM)
✅ Packaging/Registration (not in MTGBM)
✅ Comprehensive evaluation (not in MTGBM)

Need to Add:
❌ Multi-label preprocessing (from DataProcessor)
❌ Multi-task training step (integrate Mtgbm.py)
❌ Multi-task evaluation step (new)
```

---

## 8. Conclusions

### Reusability Summary

| Component Category | Steps | Fully Reusable | Needs Adaptation | Requires New |
|-------------------|-------|----------------|------------------|--------------|
| Data Loading | 2 | 2 (100%) | 0 (0%) | 0 (0%) |
| Preprocessing | 2 | 0 (0%) | 0 (0%) | 2 (100%) |
| Training | 1 | 0 (0%) | 0 (0%) | 1 (100%) |
| Evaluation | 1 | 0 (0%) | 0 (0%) | 1 (100%) |
| Calibration (MVP) | 1 | 1 (100%) | 0 (0%) | 0 (0%) |
| Calibration (Prod) | 1 | 0 (0%) | 0 (0%) | 1 (100%) |
| Packaging/Output | 3 | 3 (100%) | 0 (0%) | 0 (0%) |
| **MVP TOTAL** | **10** | **6 (60%)** | **0 (0%)** | **4 (40%)** |
| **PRODUCTION TOTAL** | **11** | **5 (45%)** | **0 (0%)** | **5 (55%)** |

**Updated Reusability Assessment**:
- **MVP Pipeline**: 60% fully reusable (main task calibration only)
- **Production Pipeline**: 45% fully reusable (includes multi-label calibration)
