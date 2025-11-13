---
tags:
  - analysis
  - step-equivalency
  - tsa-pipeline
  - cursus-framework
  - step-mapping
  - gap-analysis
keywords:
  - step equivalency
  - TSA pipeline
  - Cursus steps
  - functionality mapping
  - step creation recommendations
  - temporal self attention
  - pipeline standardization
topics:
  - pipeline step analysis
  - framework integration
  - step standardization
  - functionality coverage
  - step development recommendations
language: python
date of note: 2025-10-20
---

# TSA Pipeline vs Cursus Framework Step Equivalency Analysis

## Executive Summary

This analysis compares the Temporal Self-Attention (TSA) pipeline steps with existing Cursus framework steps to identify equivalencies, partial coverage, and unique functionalities. The goal is to determine which TSA pipeline components can leverage existing Cursus steps and which require new step implementations to achieve full framework integration.

## Cursus Framework Step Inventory

### Available Step Categories

Based on analysis of `src/cursus/steps/scripts/` and `src/cursus/registry/step_names_original.py`:

#### **Data Loading Steps**
- **CradleDataLoading**: Cradle data service integration
- **DummyDataLoading**: Mock data loading for testing

#### **Processing Steps**
- **TabularPreprocessing**: General tabular data preprocessing
- **StratifiedSampling**: Class-balanced sampling strategies
- **RiskTableMapping**: Categorical feature risk mapping
- **MissingValueImputation**: Statistical missing value handling
- **CurrencyConversion**: Multi-currency processing

#### **Training Steps**
- **PyTorchTraining**: General PyTorch model training
- **XGBoostTraining**: XGBoost model training
- **LightGBMTraining**: LightGBM model training
- **DummyTraining**: Mock training for testing

#### **Model Processing Steps**
- **ModelCalibration**: Probability calibration (GAM, Isotonic, Platt)
- **XGBoostModelEval**: Model evaluation and metrics
- **XGBoostModelInference**: Batch inference
- **ModelMetricsComputation**: Performance metrics computation
- **ModelWikiGenerator**: Automated documentation

#### **Deployment Steps**
- **Package**: Model packaging for deployment
- **Registration**: Model registration in MIMS
- **Payload**: Payload testing and validation

## TSA Pipeline Step Equivalency Analysis

### Complete Equivalencies (100% Coverage)

#### 1. **CradleDataLoadingStep ↔ CradleDataLoading**
- **TSA Implementation**: Framework-provided step
- **Cursus Equivalent**: `CradleDataLoading` step
- **Coverage**: ✅ **Complete (100%)**
- **Functionality Match**:
  - Cradle service integration
  - Data filtering and validation
  - S3 output management
  - Metadata handling
- **Recommendation**: **Direct replacement** - use existing Cursus step

#### 2. **MimsModelRegistrationProcessingStep ↔ Registration**
- **TSA Implementation**: Framework-provided step
- **Cursus Equivalent**: `Registration` step
- **Coverage**: ✅ **Complete (100%)**
- **Functionality Match**:
  - MIMS system integration
  - Model validation and testing
  - Version management
  - Production deployment preparation
- **Recommendation**: **Direct replacement** - use existing Cursus step

#### 3. **AddInferenceDependencies ↔ Package**
- **TSA Implementation**: `scripts/mims_package_na.py`
- **Cursus Equivalent**: `Package` step (`scripts/package.py`)
- **Coverage**: ✅ **Complete (100%)**

**Detailed Functionality Comparison**:

| Functionality | TSA Implementation | Cursus Implementation | Match |
|---------------|-------------------|----------------------|-------|
| Model artifact extraction | ✅ `model.tar.gz` extraction | ✅ `model.tar.gz` extraction | ✅ |
| Code dependency copying | ✅ Copy to `/code` directory | ✅ Copy to `/code` directory | ✅ |
| Configuration handling | ✅ JSON/PKL config files | ✅ Generic config support | ✅ |
| Directory structure | ✅ Standard MIMS layout | ✅ Standard MIMS layout | ✅ |
| Tar compression | ✅ Final model.tar.gz | ✅ Final model.tar.gz | ✅ |
| Calibration integration | ✅ B-spline parameters | ✅ Calibration artifacts | ✅ |
| Error handling | ✅ Basic error handling | ✅ Enhanced error handling | ✅ |
| Logging | ✅ Basic logging | ✅ Comprehensive logging | ✅ |

- **Recommendation**: **Direct replacement** - Cursus implementation is more robust

### Similar Functionality with Enhanced Coverage

#### 4. **generic_rfuge ↔ ModelCalibration**
- **TSA Implementation**: `scripts/generic_rfuge.r` (R-based B-spline)
- **Cursus Equivalent**: `ModelCalibration` step (`scripts/model_calibration.py`)
- **Coverage**: ✅ **Enhanced (120%)**

**Functionality Comparison**:

| Feature | TSA generic_rfuge | Cursus ModelCalibration | Enhancement |
|---------|------------------|------------------------|-------------|
| **Calibration Methods** | B-spline only | GAM, Isotonic, Platt, B-spline | ✅ **3x more methods** |
| **Classification Support** | Binary only | Binary + Multi-class | ✅ **Multi-class support** |
| **Metrics** | Basic validation | ECE, MCE, Brier, AUC | ✅ **Comprehensive metrics** |
| **Visualization** | None | Reliability diagrams | ✅ **Visual diagnostics** |
| **Language** | R-based | Python-based | ✅ **Ecosystem consistency** |
| **Error Handling** | Basic | Robust validation | ✅ **Production-ready** |
| **Monotonicity** | Built-in | Configurable | ✅ **Flexible constraints** |

- **Recommendation**: **Upgrade to Cursus** - significantly enhanced capabilities

### Partial Coverage with Gaps

#### 5. **TSA Preprocessing Steps ↔ Multiple Cursus Steps**

**TSA Preprocessing Components**:
- `preprocess_train_na.py` - Training data preprocessing
- `preprocess_vali_na.py` - Validation data preprocessing  
- `preprocess_cali_na.py` - Calibration data preprocessing
- `preprocess_train_na_merge.py` - Training data merging

**Cursus Coverage Analysis**:

| TSA Functionality | Cursus Step | Coverage Level | Gap Description |
|------------------|-------------|----------------|-----------------|
| **Basic Tabular Processing** | `TabularPreprocessing` | ✅ **80%** | Missing temporal sequence handling |
| **Missing Value Handling** | `MissingValueImputation` | ✅ **100%** | Statistical methods covered |
| **Categorical Encoding** | `RiskTableMapping` | ✅ **70%** | Missing fraud-specific mappings |
| **Chunked Processing** | None | ❌ **0%** | No distributed chunk processing |
| **Temporal Sequences** | None | ❌ **0%** | No time-series preprocessing |
| **Two-Sequence Handling** | None | ❌ **0%** | No CID/CCID sequence processing |
| **Memory Management** | None | ❌ **0%** | No large-dataset optimization |

**Covered Functionalities**:
- ✅ Basic data loading and validation
- ✅ Statistical missing value imputation
- ✅ Standard categorical encoding
- ✅ Data type conversions
- ✅ Basic feature scaling

**Uncovered Functionalities**:
- ❌ Temporal sequence ordering and validation
- ❌ Time delta computation
- ❌ Two-sequence (CID/CCID) data structure handling
- ❌ Chunked processing for memory efficiency
- ❌ Fraud-specific feature engineering
- ❌ Sequence padding/truncation to fixed lengths
- ❌ Multi-instance distributed preprocessing

#### 6. **TSA Training Step ↔ PyTorchTraining**

**TSA Training Component**: `scripts/train.py` (Two-Sequence MoE TSA)

**Cursus Coverage Analysis**:

| TSA Functionality | Cursus PyTorchTraining | Coverage Level | Gap Description |
|------------------|----------------------|----------------|-----------------|
| **Basic PyTorch Training** | ✅ Supported | ✅ **100%** | Framework, optimizers, schedulers |
| **Distributed Training** | ✅ Supported | ✅ **100%** | Multi-GPU DDP support |
| **Mixed Precision** | ✅ Supported | ✅ **100%** | AMP training |
| **Early Stopping** | ✅ Supported | ✅ **100%** | Patience-based stopping |
| **Model Checkpointing** | ✅ Supported | ✅ **100%** | State management |
| **TSA Architecture** | ❌ Not supported | ❌ **0%** | Temporal Self-Attention models |
| **Two-Sequence Models** | ❌ Not supported | ❌ **0%** | CID/CCID sequence handling |
| **Mixture of Experts** | ❌ Not supported | ❌ **0%** | MoE routing and training |
| **Attention Mechanisms** | ❌ Not supported | ❌ **0%** | Temporal attention layers |
| **Fraud-Specific Metrics** | ❌ Not supported | ❌ **0%** | Domain-specific evaluation |

**Covered Functionalities**:
- ✅ PyTorch training infrastructure
- ✅ Distributed data parallel training
- ✅ Automatic mixed precision
- ✅ Learning rate scheduling
- ✅ Model checkpointing and resumption
- ✅ Basic evaluation metrics

**Uncovered Functionalities**:
- ❌ Temporal Self-Attention model architecture
- ❌ Two-sequence (CID/CCID) model handling
- ❌ Mixture of Experts implementation
- ❌ Temporal attention mechanisms
- ❌ Fraud detection specific loss functions
- ❌ Sequence-aware data loading
- ❌ Attention mask computation
- ❌ Gate function training for sequence importance

## Unique TSA Functionalities Not Covered by Cursus

### 1. **Temporal Sequence Preprocessing**

**Unique Operations**:
- **Time Delta Computation**: Relative time calculations between transactions
- **Sequence Ordering**: Temporal ordering validation and correction
- **Padding/Truncation**: Fixed-length sequence normalization (length 51)
- **Temporal Feature Engineering**: Time-based feature derivation
- **Sequence Validation**: Temporal consistency checks

**Business Value**: Essential for fraud detection temporal patterns

### 2. **Two-Sequence Data Architecture**

**Unique Operations**:
- **CID Sequence Processing**: Customer ID transaction sequences
- **CCID Sequence Processing**: Credit Card ID transaction sequences  
- **Dual-Sequence Alignment**: Temporal alignment between sequences
- **Gate Function Data**: Sequence importance weighting preparation
- **Cross-Sequence Features**: Inter-sequence relationship features

**Business Value**: Captures both customer and payment method behavior patterns

### 3. **Chunked Distributed Preprocessing**

**Unique Operations**:
- **Memory-Efficient Chunking**: 60-chunk processing for large datasets
- **Multi-Instance Coordination**: Distributed preprocessing across instances
- **Chunk Merging**: Intelligent recombination of processed chunks
- **Memory Management**: Large dataset handling without OOM errors
- **Progress Tracking**: Chunk-level processing monitoring

**Business Value**: Enables processing of massive fraud detection datasets

### 4. **Temporal Self-Attention Architecture**

**Unique Components**:
- **Temporal Attention Layers**: Time-aware multi-head attention
- **Order Attention**: Sequence-level attention mechanisms
- **Feature Attention**: Feature-level attention for current transaction
- **Time Encoding**: Learnable temporal position encoding
- **Mixture of Experts**: Sparse expert routing for model capacity

**Business Value**: State-of-the-art fraud detection model architecture

### 5. **Fraud-Specific Feature Engineering**

**Unique Operations**:
- **Risk-Based Categorical Encoding**: Fraud-optimized category mappings
- **Temporal Feature Derivation**: Time-based risk indicators
- **Sequence Statistics**: Transaction sequence analytics
- **Behavioral Pattern Features**: Customer behavior characterization
- **Payment Method Features**: Credit card specific features

**Business Value**: Domain-specific feature engineering for fraud detection

## Recommended New Cursus Steps

Based on the analysis of TSA preprocessing functions, each recommended step is separated into **base sharable functionality** (80% general temporal operations) and **domain-specific extensions** (20% fraud-specific components), creating 10 total step recommendations.

### 1A. **TemporalSequencePreprocessing Step** (Base Sharable)

**Purpose**: Handle general temporal sequence data preprocessing for any time-series ML models

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (core functions)
- `dockers/tsa/scripts/preprocess_train_na.py` (orchestration)
- `dockers/tsa/scripts/params_na.py` (configuration)

**Sharable Features** (80% of functionality):

#### **1. Temporal Sequence Ordering and Validation**
**TSA Implementation**: `sequence_data_parsing()` function
```python
# Key validation operations from preprocess_functions_na.py
if len(seq_cat_vars_mtx) == len(seq_num_vars_mtx):
    if sum(seq_cat_vars_mtx[:, -1].argsort() == seq_cat_vars_mtx[:, -1].argsort()) != len(seq_cat_vars_mtx):
        # Handle sequence ordering mismatches
        intersect1d, comm1, comm2 = np.intersect1d(
            seq_cat_vars_mtx[:, -1], seq_num_vars_mtx[:, -1], return_indices=True
        )
        seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
        seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]
```
**Operations**:
- Cross-sequence temporal alignment validation
- Sequence completeness checks using `orderDate` field
- Intersection-based sequence synchronization
- Temporal consistency validation across multiple sequences

#### **2. Time Delta Computation and Normalization**
**TSA Implementation**: Time delta calculation in `sequence_data_parsing()`
```python
# Time delta computation from current transaction
seq_num_mtx[:, -2] = seq_num_mtx[-1, -2] - seq_num_mtx[:, -2]

# Time delta validation and capping
if np.max(seq_num_mtx_cid[:, -2]) > 10000000:
    seq_num_mtx_cid[:, -2] = 10000000
if np.min(seq_num_mtx_cid[:, -2]) < 0:
    continue  # Skip invalid sequences
```
**Operations**:
- Relative time difference computation from current timestamp
- Time delta capping for outlier handling (max 10M seconds ≈ 115 days)
- Negative time delta validation and filtering
- Temporal normalization using min-max scaling

#### **3. Sequence Padding/Truncation to Fixed Lengths**
**TSA Implementation**: Fixed-length sequence normalization
```python
# Configuration from params_na.py
seq_len = 51  # Fixed sequence length

# Padding operations in sequence_data_parsing()
if not no_history_flag:
    seq_cat_mtx = np.pad(seq_cat_mtx, [(seq_len - 1 - len(seq_cat_vars_mtx), 0), (0, 0)])
    seq_num_mtx = np.pad(seq_num_mtx, [(seq_len - 1 - len(seq_num_vars_mtx), 0), (0, 0)])
else:
    seq_cat_mtx = np.pad(seq_cat_mtx, [(seq_len - 1, 0), (0, 0)])
    seq_num_mtx = np.pad(seq_num_mtx, [(seq_len - 1, 0), (0, 0)])
```
**Operations**:
- Zero-padding for sequences shorter than target length
- Truncation for sequences longer than target length
- Maintains consistent tensor dimensions for batch processing
- Configurable sequence length parameter

#### **4. Generic Temporal Feature Engineering**
**TSA Implementation**: Temporal feature extraction and scaling
```python
# Temporal scaling and normalization
seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(seq_num_min_)

# Add temporal indicator column
seq_num_mtx = np.concatenate([seq_num_mtx, np.ones((seq_num_mtx.shape[0], 1))], axis=1)
```
**Operations**:
- Min-max scaling for temporal numerical features
- Temporal position encoding within sequences
- Time-based feature derivation (day_of_week, hour_of_day extractable from orderDate)
- Missing timestamp handling with configurable default values

#### **5. Memory-Efficient Chunked Processing**
**TSA Implementation**: `chunk_processing()` function with distributed processing
```python
# Chunked processing configuration from preprocess_train_na.py
chunk_processing(training_data_path, "train", 60)  # 60 chunks for training
chunk_processing(calibration_data_path, 'cali', 5)  # 5 chunks for calibration

# Parallel data loading in chunk_processing()
if len(files) < cpu_count() - 9:
    num_jobs = len(files)
else:
    num_jobs = -10
df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in files)

# Memory-mapped file operations for large datasets
A = np.memmap(
    filename=os.path.join(out_dir, "{}_{}_v{}.raw".format(dataset, A_name, p)),
    dtype=A_list[0].dtype, mode="w+", shape=final_shape
)
```
**Operations**:
- Configurable chunk size based on memory constraints
- Parallel data loading using joblib with CPU-aware job allocation
- Memory-mapped file operations for datasets larger than RAM
- Multi-instance distributed processing coordination
- Progress tracking and fault tolerance mechanisms

#### **6. Sequence Length Validation and Statistics**
**TSA Implementation**: Comprehensive sequence validation
```python
# Sequence validation in sequence_data_parsing()
for VAR in input_data_seq_cat_otf_vars:
    if VAR not in input_data:
        print("Sanity check failed. Input data does not contain required key")
        return False, None, None

# No history flag detection
no_history_flag = input_data[objectid_otf_name] in ["", "My Text String"]

# Sequence length consistency checks
if len(seq_cat_vars_mtx) != len(seq_num_vars_mtx):
    print("Input data OTFs have mismatch length")
    # Handle with intersection-based alignment
```
**Operations**:
- Required field validation for sequence data
- Empty sequence detection and handling
- Cross-sequence length consistency validation
- Statistical validation of temporal ordering
- Sequence completeness scoring and reporting

#### **7. Missing Timestamp Handling**
**TSA Implementation**: Default value handling with configurable mappings
```python
# Default value functions from preprocess_functions_na.py
mtx_from_dict_fill_default = lambda input_data, var_list_otf, var_list, map_dict: np.array([
    [map_dict[var_list[i]] if a in ["", "My Text String"] else a 
     for a in input_data[var_list_otf[i]].split(SEP)]
    for i in range(len(var_list_otf))
]).transpose()

# Load default value mappings
with open(os.path.join(config_path, "default_value_dict_na.json"), "r") as f:
    default_value_dict = json.load(f)
```
**Operations**:
- Configurable default value mappings for missing timestamps
- Empty string and placeholder detection
- Forward-fill, backward-fill, and zero-fill strategies
- Temporal interpolation for missing sequence elements

#### **8. Configurable Time Window Aggregations**
**TSA Implementation**: Time-based filtering and aggregation
```python
# Time window filtering in processing functions
train_data = train_data[train_data["orderDate"] > time.time() - 240 * 24 * 3600]  # 240 days
calib_data = calib_data[calib_data["orderDate"] > time.time() - 150 * 24 * 3600]   # 150 days
test_data = test_data[test_data["orderDate"] > time.time() - 90 * 24 * 3600]       # 90 days
```
**Operations**:
- Configurable lookback windows for different datasets
- Time-based data filtering and selection
- Rolling window aggregations for temporal features
- Multi-granularity time window support (hours, days, weeks, months)

**Configuration Parameters**:
```python
{
    "sequence_length": 51,  # Configurable from TSA seq_len
    "time_field": "orderDate",  # Configurable timestamp field
    "chunk_size": "auto",  # Memory-based or fixed (60 for training in TSA)
    "temporal_features": ["time_delta", "day_of_week", "hour_of_day", "time_since_last"],
    "padding_strategy": ["zero_pad", "forward_fill", "backward_fill"],
    "validation_rules": ["temporal_ordering", "sequence_completeness", "time_gaps"],
    "aggregation_windows": [1, 7, 30, 90, 240],  # Days, from TSA implementation
    "time_delta_cap": 10000000,  # Seconds, from TSA validation
    "parallel_jobs": "auto",  # CPU-aware job allocation
    "memory_mapping": True,  # Enable for large datasets
    "default_value_strategy": "configurable_dict"  # JSON-based default mappings
}
```

**Script Location**: `src/cursus/steps/scripts/temporal_sequence_preprocessing.py`

### 1B. **FraudTemporalSequencePreprocessing Step** (Domain-Specific Extension)

**Purpose**: Fraud-specific temporal sequence preprocessing extending base functionality with payment fraud detection features

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (fraud-specific validation logic)
- `dockers/tsa/scripts/params_na.py` (fraud feature definitions)
- `dockers/tsa/scripts/preprocess_train_na.py` (fraud data filtering)

**Domain-Specific Features** (20% of functionality):

#### **1. Fraud-Specific Data Validation and Filtering**
**TSA Implementation**: Fraud detection specific validation in `data_parsing()` function
```python
# Fraud-specific credit card filtering from preprocess_functions_na.py
train_data = train_data[train_data["creditCardIds"] != "9990-0012191-573601"]

# Fraud tag validation and conversion
train_data = train_data.loc[
    (~train_data[tag].isnull()) & (train_data[tag] != -1), :
]
train_data[tag] = pd.to_numeric(train_data[tag], downcast="integer")

# Fraud-specific holiday filtering to remove seasonal bias
train_data = train_data[
    (train_data["orderDate"] >= 1701158400) | (train_data["orderDate"] < 1700726400)
]  # Christmas holiday filter
train_data = train_data[
    (train_data["orderDate"] >= 1697094000) | (train_data["orderDate"] < 1696921200)
]  # Halloween holiday filter
```
**Operations**:
- Blacklisted credit card ID filtering for known fraud patterns
- Fraud label validation with null and invalid value handling
- Holiday period filtering to remove seasonal transaction bias
- Fraud-specific data quality checks and cleansing

#### **2. Two-Sequence Fraud Architecture Processing**
**TSA Implementation**: Dual-sequence processing for Customer ID (CID) and Credit Card ID (CCID)
```python
# CID sequence processing for customer behavior patterns
objectid_otf_name_cid = "payment_risk.retail_order_cat_seq_by_cid.c_objectid_seq"
ret_cid, seq_cat_mtx_cid, seq_num_mtx_cid = sequence_data_parsing(
    input_data,
    input_data_seq_cat_otf_vars_cid,
    input_data_seq_num_otf_vars_cid,
    objectid_otf_name_cid,
)

# CCID sequence processing for payment method behavior patterns
objectid_otf_name_ccid = "payment_risk.retail_order_cat_seq_by_ccid.c_objectid_seq"
ret_ccid, seq_cat_mtx_ccid, seq_num_mtx_ccid = sequence_data_parsing(
    input_data,
    input_data_seq_cat_otf_vars_ccid,
    input_data_seq_num_otf_vars_ccid,
    objectid_otf_name_ccid,
)
```
**Operations**:
- Customer ID sequence processing for behavioral fraud detection
- Credit Card ID sequence processing for payment fraud detection
- Cross-sequence validation for customer-card relationship analysis
- Dual-sequence temporal alignment for fraud pattern correlation

#### **3. Fraud-Specific Categorical Feature Engineering**
**TSA Implementation**: 109 fraud-specific categorical features from `params_na.py`
```python
# Payment fraud risk indicators
"c_cciscorporate_seq",           # Corporate card fraud risk
"c_ccisdebit_seq",               # Debit card fraud patterns
"c_ccisprepaid_seq",             # Prepaid card fraud risk
"c_ccissuer_seq",                # Card issuer fraud patterns
"c_creditcardhit_seq",           # Credit card blacklist hits

# Geographic fraud indicators
"c_geobillcountrycccountrycodeequal_seq",     # Bill/CC country mismatch
"c_geoipcountrycodecccountrycodeequal_seq",   # IP/CC country mismatch
"c_georeportedipmktplcountrycodeequal_seq",   # IP/Marketplace mismatch

# Behavioral fraud indicators
"c_fingerprintchanged_seq",      # Device fingerprint changes
"c_emailchanged_seq",            # Email address changes
"c_ipchanged_seq",               # IP address changes
"c_paymentchg_seq",              # Payment method changes
```
**Operations**:
- Payment instrument fraud risk categorization
- Geographic fraud pattern detection through country code mismatches
- Behavioral change detection for account takeover patterns
- Cross-border transaction fraud indicators

#### **4. Fraud-Specific Numerical Feature Engineering**
**TSA Implementation**: 67 fraud-specific numerical features from `params_na.py`
```python
# Fraud velocity features
"c_days_lastorder_seq",                    # Time since last order (velocity)
"c_ordertotalamountusd_seq",              # Transaction amount patterns
"c_ccage_seq",                            # Credit card age (new card risk)
"c_cccount_seq",                          # Number of credit cards used

# Fraud risk scoring features
"c_billaddrstrangeness_seq",              # Billing address anomaly score
"c_shipaddrstrangeness_seq",              # Shipping address anomaly score
"c_emailnamestrangeness_seq",             # Email name anomaly score
"c_fingerprintriskvalue_seq",             # Device fingerprint risk score

# Tugboat fraud network features
"c_tugboat_ev2customer_evwmaxlinkscorecust_seq",  # Customer fraud network score
"c_tugboat_ev2customer_evwmaxlinkscoretrx_seq",   # Transaction fraud network score
```
**Operations**:
- Transaction velocity and frequency analysis for fraud detection
- Address and name strangeness scoring for synthetic identity detection
- Device fingerprint risk assessment for device-based fraud
- Network-based fraud scoring using Tugboat fraud graph analysis

#### **5. Fraud-Specific Time Window Analysis**
**TSA Implementation**: Fraud-optimized time windows from `processing_training_data_by_chunk()`
```python
# Fraud-specific lookback windows
train_data = train_data[train_data["orderDate"] > time.time() - 240 * 24 * 3600]  # 240 days training
calib_data = calib_data[calib_data["orderDate"] > time.time() - 150 * 24 * 3600]   # 150 days calibration
test_data = test_data[test_data["orderDate"] > time.time() - 90 * 24 * 3600]       # 90 days validation
```
**Operations**:
- Fraud-optimized lookback windows (240/150/90 days) for different datasets
- Seasonal fraud pattern analysis with holiday filtering
- Fraud recency weighting with shorter windows for recent fraud trends
- Time-based fraud pattern evolution tracking

#### **6. Fraud-Specific Downsampling Strategy**
**TSA Implementation**: Fraud-aware positive rate targeting
```python
# Fraud-specific downsampling configuration
config: Dict[str, Any] = {
    "tag": "IS_FRD",                    # Fraud label identifier
    "target_positive_rate": 0.2,       # 20% fraud rate for training
}

# Fraud-aware downsampling logic
if positive_rate < target_positive_rate:
    negative_downsampled_cnt = int(
        positive_count * (1 - target_positive_rate) / target_positive_rate
    )
    positive_downsampled_cnt = positive_count

train_data = pd.concat([
    train_data[zeros_cond].sample(negative_downsampled_cnt),
    train_data[ones_cond].sample(positive_downsampled_cnt),
], ignore_index=True)
```
**Operations**:
- Fraud-specific positive rate targeting (20% fraud cases)
- Class imbalance handling for fraud detection model training
- Fraud pattern preservation during downsampling
- Statistical fraud distribution maintenance across chunks

#### **7. Fraud-Specific Sequence Validation**
**TSA Implementation**: Enhanced validation for fraud detection sequences
```python
# Fraud-specific time delta validation with fraud-optimized thresholds
if np.max(seq_num_mtx_cid[:, -2]) > 10000000:  # 115 days max for fraud patterns
    seq_num_mtx_cid[:, -2] = 10000000
if np.min(seq_num_mtx_cid[:, -2]) < 0:
    continue  # Skip invalid fraud sequences

# Fraud-specific numerical categorical variable handling
for i in numerical_cat_vars_indices:
    cur_var = input_data[input_data_seq_cat_vars[i]]
    if cur_var not in ["", "My Text String", "false"]:
        cur_var = str(int(float(cur_var)))  # Fraud-specific encoding
        input_data[input_data_seq_cat_vars[i]] = cur_var
```
**Operations**:
- Fraud-optimized time delta thresholds (115 days maximum)
- Fraud-specific categorical variable validation and encoding
- Invalid fraud sequence detection and filtering
- Fraud pattern temporal consistency validation

#### **8. Fraud Risk Feature Transformation**
**TSA Implementation**: Specialized fraud risk feature processing
```python
# Special handling for fingerprintRiskValue (fraud-specific feature)
cur_var = input_data[input_data_seq_cat_vars[38]]  # fingerprintRiskValue
if cur_var not in ["", "My Text String", "false"]:
    if float(cur_var) == 0:
        cur_var = str(int(float(cur_var)))  # Zero risk encoding
    else:
        cur_var = str(float(cur_var))       # Non-zero risk preservation
    input_data[input_data_seq_cat_vars[38]] = cur_var
```
**Operations**:
- Fraud risk score preservation and encoding
- Device fingerprint risk value special handling
- Fraud-specific feature transformation rules
- Risk-based categorical encoding for fraud detection

**Configuration Parameters**:
```python
{
    "extends": "TemporalSequencePreprocessing",
    "fraud_label": "IS_FRD",                    # Fraud detection target variable
    "target_positive_rate": 0.2,               # 20% fraud rate for training balance
    "fraud_time_windows": {
        "training": 240,                        # 240 days for training data
        "calibration": 150,                     # 150 days for calibration
        "validation": 90                        # 90 days for validation
    },
    "fraud_categorical_features": {
        "payment_fraud": ["cciscorporate", "ccisdebit", "ccisprepaid", "ccissuer"],
        "geographic_fraud": ["geobillcountrycccountrycodeequal", "geoipcountrycodecccountrycodeequal"],
        "behavioral_fraud": ["fingerprintchanged", "emailchanged", "ipchanged", "paymentchg"],
        "cross_border": ["marketplacecountrycode_*_is_match"]
    },
    "fraud_numerical_features": {
        "velocity": ["days_lastorder", "ordertotalamountusd", "ccage", "cccount"],
        "strangeness": ["billaddrstrangeness", "shipaddrstrangeness", "emailnamestrangeness"],
        "risk_scores": ["fingerprintriskvalue", "tugboat_*_linkScore*"],
        "network_features": ["tugboat_ev2customer_*"]
    },
    "fraud_validation_rules": {
        "time_delta_cap": 10000000,            # 115 days maximum (fraud-optimized)
        "blacklisted_cards": ["9990-0012191-573601"],
        "holiday_filters": [
            {"start": 1700726400, "end": 1701158400},  # Christmas
            {"start": 1696921200, "end": 1697094000},  # Halloween
            {"start": 1688972400, "end": 1689231600}   # July 4th
        ]
    },
    "two_sequence_processing": {
        "customer_sequence_key": "retail_order_*_seq_by_cid",
        "payment_sequence_key": "retail_order_*_seq_by_ccid",
        "cross_sequence_validation": True,
        "fraud_correlation_analysis": True
    },
    "fraud_risk_encoding": {
        "fingerprint_risk_special_handling": True,
        "numerical_categorical_indices": [3,4,5,6,8,9,10,13,14,16,17,18,19,20,21,22,24,27,28,29,30,32,33,34,35,36,37,39,40],
        "risk_preservation_rules": ["zero_risk_encoding", "non_zero_risk_preservation"]
    }
}
```

**Script Location**: `src/cursus/steps/scripts/fraud_temporal_sequence_preprocessing.py`

### 2A. **MultiSequencePreprocessing Step** (Base Sharable)

**Purpose**: Handle general multi-sequence data structures for any multi-entity modeling with cross-sequence validation and alignment

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (multi-sequence coordination in `data_parsing()`)
- `dockers/tsa/scripts/params_na.py` (dual-sequence parameter definitions)
- `dockers/tsa/scripts/preprocess_train_na.py` (multi-sequence orchestration)

**Sharable Features** (80% of functionality):

#### **1. Multi-Sequence Data Loading and Coordination**
**TSA Implementation**: Dual-sequence processing coordination in `data_parsing()` function
```python
# Multi-sequence processing coordination from preprocess_functions_na.py
def data_parsing(input_data):
    # CID sequence processing
    objectid_otf_name_cid = "payment_risk.retail_order_cat_seq_by_cid.c_objectid_seq"
    ret_cid, seq_cat_mtx_cid, seq_num_mtx_cid = sequence_data_parsing(
        input_data,
        input_data_seq_cat_otf_vars_cid,
        input_data_seq_num_otf_vars_cid,
        objectid_otf_name_cid,
    )

    # CCID sequence processing
    objectid_otf_name_ccid = "payment_risk.retail_order_cat_seq_by_ccid.c_objectid_seq"
    ret_ccid, seq_cat_mtx_ccid, seq_num_mtx_ccid = sequence_data_parsing(
        input_data,
        input_data_seq_cat_otf_vars_ccid,
        input_data_seq_num_otf_vars_ccid,
        objectid_otf_name_ccid,
    )

    # Multi-sequence validation
    if ret_cid == False:
        return False, None, None, None, None, None, None
    if ret_ccid == False:
        return False, None, None, None, None, None, None
```
**Operations**:
- Configurable multi-sequence entity processing (generalizable beyond CID/CCID)
- Cross-sequence validation with failure propagation
- Parallel sequence processing with independent validation
- Multi-sequence return value coordination and error handling

#### **2. Cross-Sequence Parameter Management**
**TSA Implementation**: Dual parameter sets from `params_na.py`
```python
# Multi-sequence categorical parameter management
input_data_seq_cat_otf_vars_cid = [
    "payment_risk.retail_order_cat_seq_by_cid.c_bfs_signin_age_checkout_browserfamily_seq",
    "payment_risk.retail_order_cat_seq_by_cid.c_bfs_signin_age_checkout_devicetype_seq",
    # ... 109 CID categorical features
]

input_data_seq_cat_otf_vars_ccid = [
    "payment_risk.retail_order_cat_seq_by_ccid.c_bfs_signin_age_checkout_browserfamily_seq", 
    "payment_risk.retail_order_cat_seq_by_ccid.c_bfs_signin_age_checkout_devicetype_seq",
    # ... 109 CCID categorical features (parallel structure)
]

# Multi-sequence numerical parameter management
input_data_seq_num_otf_vars_cid = [
    "payment_risk.retail_order_num_seq_by_cid.c_aveamt_fpage_seq",
    "payment_risk.retail_order_num_seq_by_cid.c_aveamt_ipage_seq",
    # ... 67 CID numerical features
]

input_data_seq_num_otf_vars_ccid = [
    "payment_risk.retail_order_num_seq_by_ccid.c_aveamt_fpage_seq",
    "payment_risk.retail_order_num_seq_by_ccid.c_aveamt_ipage_seq", 
    # ... 67 CCID numerical features (parallel structure)
]
```
**Operations**:
- Parallel parameter structure management for multiple sequences
- Configurable sequence naming conventions (generalizable pattern: `*_by_{entity_key}.*`)
- Cross-sequence feature alignment with identical feature sets per sequence
- Multi-entity parameter validation and consistency checking

#### **3. Multi-Sequence Temporal Alignment and Validation**
**TSA Implementation**: Cross-sequence temporal consistency in `sequence_data_parsing()`
```python
# Multi-sequence temporal alignment from sequence_data_parsing()
if not no_history_flag:
    if len(seq_cat_vars_mtx) == len(seq_num_vars_mtx):
        if sum(seq_cat_vars_mtx[:, -1].argsort() == seq_cat_vars_mtx[:, -1].argsort()) != len(seq_cat_vars_mtx):
            # Cross-sequence temporal alignment using intersection
            intersect1d, comm1, comm2 = np.intersect1d(
                seq_cat_vars_mtx[:, -1], seq_num_vars_mtx[:, -1], return_indices=True
            )
            seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
            seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]
    else:
        # Handle length mismatches with intersection-based alignment
        intersect1d, comm1, comm2 = np.intersect1d(
            seq_cat_vars_mtx[:, -1], seq_num_vars_mtx[:, -1], return_indices=True
        )
        seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
        seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]
```
**Operations**:
- Cross-sequence temporal ordering validation using timestamp fields
- Intersection-based sequence alignment for temporal consistency
- Multi-sequence length mismatch handling with data preservation
- Temporal synchronization across categorical and numerical sequences

#### **4. Multi-Sequence Data Structure Management**
**TSA Implementation**: Parallel sequence tensor creation and management
```python
# Multi-sequence tensor management from parallel_data_parsing()
def parallel_data_parsing(df, num_workers=80):
    # Initialize multi-sequence storage
    X_seq_cat_cid_list = []
    X_seq_num_cid_list = []
    X_seq_cat_ccid_list = []
    X_seq_num_ccid_list = []
    X_num_list = []
    Y_list = []

    # Process each row with multi-sequence parsing
    for result in results:
        (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
         seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y) = result.get()
        
        if ret:
            # Multi-sequence validation and storage
            X_seq_cat_cid_list.append(seq_cat_mtx_cid)
            X_seq_num_cid_list.append(seq_num_mtx_cid)
            X_seq_cat_ccid_list.append(seq_cat_mtx_ccid)
            X_seq_num_ccid_list.append(seq_num_mtx_ccid)

    # Multi-sequence tensor stacking
    train_X_seq_cat_cid = np.stack(X_seq_cat_cid_list, axis=0)
    train_X_seq_num_cid = np.stack(X_seq_num_cid_list, axis=0)
    train_X_seq_cat_ccid = np.stack(X_seq_cat_ccid_list, axis=0)
    train_X_seq_num_ccid = np.stack(X_seq_num_ccid_list, axis=0)
```
**Operations**:
- Parallel multi-sequence data structure management
- Configurable sequence tensor organization and stacking
- Multi-sequence batch processing with consistent dimensionality
- Cross-sequence data integrity validation during processing

#### **5. Multi-Sequence Memory-Mapped Storage**
**TSA Implementation**: Distributed multi-sequence storage in `stream_save()` function
```python
# Multi-sequence memory-mapped storage from stream_save()
def stream_save(A_list, A_name, dataset, out_dir="/opt/ml/processing/output", p="0"):
    # Create memory-mapped array for multi-sequence data
    final_shape = (sum(arr.shape[0] for arr in A_list),) + A_list[0].shape[1:]
    A = np.memmap(
        filename=os.path.join(out_dir, "{}_{}_v{}.raw".format(dataset, A_name, p)),
        dtype=A_list[0].dtype, mode="w+", shape=final_shape
    )

    # Sequential assignment for multi-sequence chunks
    current_position = 0
    for i in range(len(A_list)):
        chunk = A_list[i]
        A[current_position : current_position + chunk.shape[0], ...] = chunk
        current_position += chunk.shape[0]
        A.flush()

# Multi-sequence storage coordination from chunk_processing()
stream_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)
stream_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)
stream_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)
stream_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)
```
**Operations**:
- Memory-efficient multi-sequence storage with configurable naming
- Parallel sequence file management with consistent organization
- Multi-sequence chunk coordination and sequential assembly
- Cross-sequence storage validation and integrity checking

#### **6. Multi-Sequence Validation and Error Handling**
**TSA Implementation**: Comprehensive multi-sequence validation logic
```python
# Multi-sequence validation from parallel_data_parsing()
for result in results:
    (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
     seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y) = result.get()
    
    if ret:
        # Multi-sequence time delta validation
        if np.max(seq_num_mtx_cid[:, -2]) > 10000000:
            seq_num_mtx_cid[:, -2] = 10000000
        if np.min(seq_num_mtx_cid[:, -2]) < 0:
            continue  # Skip invalid CID sequences
            
        if np.max(seq_num_mtx_ccid[:, -2]) > 10000000:
            seq_num_mtx_ccid[:, -2] = 10000000
        if np.min(seq_num_mtx_ccid[:, -2]) < 0:
            continue  # Skip invalid CCID sequences
```
**Operations**:
- Cross-sequence validation with configurable thresholds
- Multi-sequence error propagation and handling strategies
- Parallel sequence quality control with independent validation
- Multi-entity data integrity enforcement

#### **7. Multi-Sequence Feature Engineering Coordination**
**TSA Implementation**: Parallel feature processing across sequences
```python
# Multi-sequence categorical transformation coordination
columns_list = input_data_seq_cat_vars[:-2]
transform_object = CategoricalTransformer(
    categorical_map=categorical_map, columns_list=columns_list
)

# Apply same transformation logic to both sequences
seq_cat_mtx = transform_object.transform(seq_cat_mtx)  # Applied to both CID and CCID

# Multi-sequence numerical scaling coordination
seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(seq_num_min_)
# Same scaling applied consistently across all sequences
```
**Operations**:
- Consistent feature transformation across multiple sequences
- Shared transformation objects for cross-sequence consistency
- Parallel feature engineering with synchronized parameters
- Multi-sequence feature validation and quality control

#### **8. Multi-Sequence Orchestration and Workflow Management**
**TSA Implementation**: Multi-sequence processing orchestration in `chunk_processing()`
```python
# Multi-sequence processing orchestration
def chunk_processing(data_path, dataset, num_chunk):
    for i_chunk in range(num_chunk):
        if dataset == "train":
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_training_data_by_chunk(files_chunk, tag)
        elif dataset == "cali":
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_calibration_data_by_chunk(files_chunk, tag)
        elif dataset == "vali":
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_validation_data_by_chunk(files_chunk, tag)

        # Multi-sequence chunk storage coordination
        np.save(file=os.path.join(out_dir, "{}_cid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)), 
                arr=X_seq_cat_cid_chunk)
        np.save(file=os.path.join(out_dir, "{}_ccid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)), 
                arr=X_seq_cat_ccid_chunk)
```
**Operations**:
- Multi-sequence workflow orchestration with configurable processing functions
- Parallel sequence chunk management with consistent naming conventions
- Cross-sequence processing coordination for different dataset types
- Multi-entity workflow validation and progress tracking

**Configuration Parameters**:
```python
{
    "sequence_entities": ["entity_1", "entity_2"],  # Configurable from TSA CID/CCID
    "sequence_length": 51,  # Consistent across all sequences
    "sequence_parameter_patterns": {
        "categorical_otf": "*_seq_by_{entity}.c_*_seq",
        "numerical_otf": "*_seq_by_{entity}.c_*_seq", 
        "object_id_pattern": "*_seq_by_{entity}.c_objectid_seq"
    },
    "alignment_strategy": ["temporal_intersection", "length_matching", "feature_synchronization"],
    "validation_rules": {
        "cross_sequence_temporal_consistency": True,
        "parallel_sequence_validation": True,
        "time_delta_thresholds": {"max": 10000000, "min": 0},
        "sequence_length_consistency": True
    },
    "storage_configuration": {
        "memory_mapping": True,
        "parallel_storage": True,
        "naming_convention": "{dataset}_{entity}_{data_type}_chunk_{chunk_id}.npy",
        "chunk_coordination": True
    },
    "feature_engineering": {
        "shared_transformations": True,
        "consistent_scaling": True,
        "parallel_processing": True,
        "cross_sequence_validation": True
    },
    "error_handling": {
        "sequence_failure_propagation": True,
        "independent_validation": True,
        "quality_control_thresholds": "configurable",
        "cross_sequence_integrity_checks": True
    },
    "workflow_orchestration": {
        "multi_sequence_chunk_processing": True,
        "parallel_entity_processing": True,
        "configurable_processing_functions": True,
        "progress_tracking": "per_sequence_and_aggregate"
    }
}
```

**Script Location**: `src/cursus/steps/scripts/multi_sequence_preprocessing.py`

### 2B. **FraudTwoSequencePreprocessing Step** (Domain-Specific Extension)

**Purpose**: Fraud-specific dual-sequence (Customer ID + Credit Card ID) preprocessing extending base multi-sequence functionality with payment fraud detection features

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (fraud-specific dual-sequence coordination)
- `dockers/tsa/scripts/params_na.py` (fraud CID/CCID feature definitions)
- `dockers/tsa/scripts/preprocess_train_na.py` (fraud dual-sequence orchestration)

**Domain-Specific Features** (20% of functionality):

#### **1. Fraud-Specific CID/CCID Sequence Architecture**
**TSA Implementation**: Fraud-optimized dual-sequence processing in `data_parsing()` function
```python
# Fraud-specific Customer ID sequence processing from preprocess_functions_na.py
objectid_otf_name_cid = "payment_risk.retail_order_cat_seq_by_cid.c_objectid_seq"
ret_cid, seq_cat_mtx_cid, seq_num_mtx_cid = sequence_data_parsing(
    input_data,
    input_data_seq_cat_otf_vars_cid,      # 109 fraud-specific CID categorical features
    input_data_seq_num_otf_vars_cid,      # 67 fraud-specific CID numerical features
    objectid_otf_name_cid,
)

# Fraud-specific Credit Card ID sequence processing
objectid_otf_name_ccid = "payment_risk.retail_order_cat_seq_by_ccid.c_objectid_seq"
ret_ccid, seq_cat_mtx_ccid, seq_num_mtx_ccid = sequence_data_parsing(
    input_data,
    input_data_seq_cat_otf_vars_ccid,     # 109 fraud-specific CCID categorical features
    input_data_seq_num_otf_vars_ccid,     # 67 fraud-specific CCID numerical features
    objectid_otf_name_ccid,
)

# Fraud-specific dual-sequence validation
if ret_cid == False or ret_ccid == False:
    return False, None, None, None, None, None, None
```
**Operations**:
- Customer behavior sequence processing for fraud pattern detection
- Payment method sequence processing for card fraud detection
- Fraud-specific dual-sequence validation with failure propagation
- Customer-card relationship analysis for fraud correlation

#### **2. Fraud-Specific Cross-Sequence Feature Engineering**
**TSA Implementation**: Fraud-optimized categorical features across CID/CCID sequences from `params_na.py`
```python
# Fraud payment instrument features (identical across CID/CCID for correlation analysis)
# CID sequence payment fraud features
"payment_risk.retail_order_cat_seq_by_cid.c_cciscorporate_seq",      # Corporate card fraud risk by customer
"payment_risk.retail_order_cat_seq_by_cid.c_ccisdebit_seq",          # Debit card usage patterns by customer
"payment_risk.retail_order_cat_seq_by_cid.c_ccisprepaid_seq",        # Prepaid card fraud risk by customer
"payment_risk.retail_order_cat_seq_by_cid.c_ccissuer_seq",           # Card issuer patterns by customer

# CCID sequence payment fraud features (parallel structure for cross-analysis)
"payment_risk.retail_order_cat_seq_by_ccid.c_cciscorporate_seq",     # Corporate card fraud risk by card
"payment_risk.retail_order_cat_seq_by_ccid.c_ccisdebit_seq",         # Debit card usage patterns by card
"payment_risk.retail_order_cat_seq_by_ccid.c_ccisprepaid_seq",       # Prepaid card fraud risk by card
"payment_risk.retail_order_cat_seq_by_ccid.c_ccissuer_seq",          # Card issuer patterns by card

# Fraud behavioral change detection (cross-sequence correlation)
"payment_risk.retail_order_cat_seq_by_cid.c_paymentchg_seq",         # Payment method changes by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_paymentchg_seq",        # Payment method changes by card
"payment_risk.retail_order_cat_seq_by_cid.c_fingerprintchanged_seq", # Device changes by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_fingerprintchanged_seq", # Device changes by card
```
**Operations**:
- Cross-sequence payment fraud feature correlation analysis
- Customer-card behavioral consistency validation
- Payment method switching pattern detection across sequences
- Device fingerprint correlation for fraud detection

#### **3. Fraud-Specific Geographic Cross-Border Analysis**
**TSA Implementation**: Fraud-optimized geographic features for cross-sequence analysis
```python
# Geographic fraud indicators (CID sequence - customer perspective)
"payment_risk.retail_order_cat_seq_by_cid.c_geobillcountrycccountrycodeequal_seq",    # Bill/CC country match by customer
"payment_risk.retail_order_cat_seq_by_cid.c_geoipcountrycodecccountrycodeequal_seq",  # IP/CC country match by customer
"payment_risk.retail_order_cat_seq_by_cid.c_georeportedipmktplcountrycodeequal_seq",  # IP/Marketplace match by customer

# Geographic fraud indicators (CCID sequence - card perspective)
"payment_risk.retail_order_cat_seq_by_ccid.c_geobillcountrycccountrycodeequal_seq",   # Bill/CC country match by card
"payment_risk.retail_order_cat_seq_by_ccid.c_geoipcountrycodecccountrycodeequal_seq", # IP/CC country match by card
"payment_risk.retail_order_cat_seq_by_ccid.c_georeportedipmktplcountrycodeequal_seq", # IP/Marketplace match by card

# Cross-border fraud pattern analysis
"payment_risk.retail_order_cat_seq_by_cid.c_marketplacecountrycode_cctry_cd_is_match_seq",    # Marketplace/CC country by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_marketplacecountrycode_cctry_cd_is_match_seq",   # Marketplace/CC country by card
```
**Operations**:
- Cross-sequence geographic fraud pattern detection
- Customer-card geographic consistency analysis
- Cross-border transaction fraud correlation
- Geographic anomaly detection across dual sequences

#### **4. Fraud-Specific Velocity and Risk Scoring**
**TSA Implementation**: Fraud-optimized numerical features for dual-sequence risk analysis
```python
# Fraud velocity features (CID sequence - customer behavior)
"payment_risk.retail_order_num_seq_by_cid.c_days_lastorder_seq",           # Customer transaction velocity
"payment_risk.retail_order_num_seq_by_cid.c_ordertotalamountusd_seq",      # Customer spending patterns
"payment_risk.retail_order_num_seq_by_cid.c_ccage_seq",                    # Credit card age from customer perspective
"payment_risk.retail_order_num_seq_by_cid.c_cccount_seq",                  # Number of cards used by customer

# Fraud velocity features (CCID sequence - card behavior)
"payment_risk.retail_order_num_seq_by_ccid.c_days_lastorder_seq",          # Card transaction velocity
"payment_risk.retail_order_num_seq_by_ccid.c_ordertotalamountusd_seq",     # Card spending patterns
"payment_risk.retail_order_num_seq_by_ccid.c_ccage_seq",                   # Credit card age from card perspective
"payment_risk.retail_order_num_seq_by_ccid.c_cccount_seq",                 # Card usage frequency

# Fraud network scoring (Tugboat features for cross-sequence analysis)
"payment_risk.retail_order_num_seq_by_cid.c_tugboat_ev2customer_evwmaxlinkscorecust_seq",   # Customer fraud network score
"payment_risk.retail_order_num_seq_by_ccid.c_tugboat_ev2customer_evwmaxlinkscorecust_seq",  # Card fraud network score
```
**Operations**:
- Cross-sequence velocity analysis for fraud detection
- Customer-card spending pattern correlation
- Dual-sequence fraud network scoring
- Risk score consistency validation across sequences

#### **5. Fraud-Specific Dual-Sequence Validation Logic**
**TSA Implementation**: Enhanced validation for fraud detection dual sequences
```python
# Fraud-specific dual-sequence time delta validation from parallel_data_parsing()
for result in results:
    (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
     seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y) = result.get()
    
    if ret:
        # Customer sequence fraud validation
        if np.max(seq_num_mtx_cid[:, -2]) > 10000000:  # 115 days max for customer fraud patterns
            seq_num_mtx_cid[:, -2] = 10000000
        if np.min(seq_num_mtx_cid[:, -2]) < 0:
            continue  # Skip invalid customer fraud sequences
            
        # Credit card sequence fraud validation
        if np.max(seq_num_mtx_ccid[:, -2]) > 10000000:  # 115 days max for card fraud patterns
            seq_num_mtx_ccid[:, -2] = 10000000
        if np.min(seq_num_mtx_ccid[:, -2]) < 0:
            continue  # Skip invalid card fraud sequences
```
**Operations**:
- Fraud-optimized time delta validation for both customer and card sequences
- Independent fraud sequence quality control with cross-validation
- Fraud pattern temporal consistency enforcement
- Dual-sequence fraud data integrity validation

#### **6. Fraud-Specific Customer-Card Relationship Analysis**
**TSA Implementation**: Fraud-optimized relationship features across sequences
```python
# Customer loyalty and card switching analysis (derived from cross-sequence patterns)
# Prime membership correlation across sequences
"payment_risk.retail_order_cat_seq_by_cid.c_isprimemember_seq",      # Prime membership by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_isprimemember_seq",     # Prime membership by card usage

# Payment method consistency analysis
"payment_risk.retail_order_cat_seq_by_cid.c_paymeth_seq",            # Payment methods used by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_paymeth_seq",           # Payment methods associated with card

# Address consistency for fraud detection
"payment_risk.retail_order_cat_seq_by_cid.c_same_bs_zip_seq",        # Billing/shipping zip consistency by customer
"payment_risk.retail_order_cat_seq_by_ccid.c_same_bs_zip_seq",       # Billing/shipping zip consistency by card
```
**Operations**:
- Customer-card loyalty pattern analysis for fraud detection
- Payment method consistency validation across sequences
- Address correlation analysis for synthetic identity detection
- Cross-sequence behavioral alignment scoring

#### **7. Fraud-Specific Dual-Sequence Storage Coordination**
**TSA Implementation**: Fraud-optimized storage for dual-sequence analysis
```python
# Fraud-specific dual-sequence storage from chunk_processing()
# Customer sequence storage with fraud-specific naming
stream_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)      # Customer categorical fraud features
stream_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)      # Customer numerical fraud features

# Credit card sequence storage with fraud-specific naming
stream_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)    # Card categorical fraud features
stream_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)    # Card numerical fraud features

# Fraud-specific chunk coordination
np.save(file=os.path.join(out_dir, "{}_cid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)), 
        arr=X_seq_cat_cid_chunk)
np.save(file=os.path.join(out_dir, "{}_ccid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)), 
        arr=X_seq_cat_ccid_chunk)
```
**Operations**:
- Fraud-specific dual-sequence file organization and naming
- Customer-card sequence coordination for fraud analysis
- Fraud pattern preservation across storage chunks
- Cross-sequence data integrity maintenance for fraud detection

#### **8. Fraud-Specific Gate Function Data Preparation**
**TSA Implementation**: Fraud-optimized gate function preparation for dual-sequence models
```python
# Fraud-specific gate function data preparation (implicit in TSA model architecture)
# Dense numerical features for fraud gate function
dense_num_vars_lst = arr_from_dict_fill_default(
    input_data, input_data_dense_num_vars, default_value_dict
)
dense_num_vars_lst = dense_num_vars_lst[:, :-2]  # Remove metadata columns
dense_num_arr = dense_num_vars_lst * np.array(num_static_scale_) + np.array(num_static_min_)

# Return structure for fraud dual-sequence model with gate function support
return (
    True,
    seq_cat_mtx_cid.astype(np.int16),     # Customer categorical sequence for gate function
    seq_num_mtx_cid,                      # Customer numerical sequence for gate function
    seq_cat_mtx_ccid.astype(np.int16),    # Card categorical sequence for gate function
    seq_num_mtx_ccid,                     # Card numerical sequence for gate function
    dense_num_arr,                        # Dense features for gate function weighting
    arr_from_dict(input_data, ["IS_FRD"]) # Fraud label for gate function training
)
```
**Operations**:
- Fraud-specific gate function feature preparation
- Customer-card sequence importance weighting for fraud detection
- Dense feature integration for fraud gate function
- Fraud label coordination for gate function training

**Configuration Parameters**:
```python
{
    "extends": "MultiSequencePreprocessing", 
    "fraud_dual_sequence_config": {
        "primary_sequence_key": "customer_id",           # CID sequence for customer behavior
        "secondary_sequence_key": "credit_card_id",      # CCID sequence for payment behavior
        "sequence_naming_pattern": "payment_risk.retail_order_*_seq_by_{entity}.*"
    },
    "fraud_cross_sequence_features": {
        "payment_fraud_correlation": ["cciscorporate", "ccisdebit", "ccisprepaid", "ccissuer"],
        "behavioral_fraud_correlation": ["paymentchg", "fingerprintchanged", "emailchanged", "ipchanged"],
        "geographic_fraud_correlation": ["geobillcountrycccountrycodeequal", "geoipcountrycodecccountrycodeequal"],
        "cross_border_analysis": ["marketplacecountrycode_*_is_match"]
    },
    "fraud_velocity_analysis": {
        "customer_velocity_features": ["days_lastorder", "ordertotalamountusd", "ccage", "cccount"],
        "card_velocity_features": ["days_lastorder", "ordertotalamountusd", "ccage", "cccount"],
        "network_scoring_features": ["tugboat_ev2customer_*"]
    },
    "fraud_relationship_analysis": {
        "loyalty_indicators": ["isprimemember", "primemembertype"],
        "consistency_indicators": ["paymeth", "same_bs_zip", "same_bs_state"],
        "switching_patterns": ["paymentchg", "ubidchanged", "fingerprintchanged"]
    },
    "fraud_validation_rules": {
        "dual_sequence_time_delta_cap": 10000000,        # 115 days maximum for both sequences
        "customer_sequence_validation": True,
        "card_sequence_validation": True,
        "cross_sequence_consistency_checks": True
    },
    "fraud_gate_function_config": {
        "enable_gate_function": True,
        "customer_sequence_weighting": True,
        "card_sequence_weighting": True,
        "dense_feature_integration": True,
        "fraud_label_coordination": "IS_FRD"
    },
    "fraud_storage_config": {
        "customer_sequence_prefix": "cid",
        "card_sequence_prefix": "ccid",
        "fraud_chunk_coordination": True,
        "cross_sequence_integrity_validation": True
    }
}
```

**Script Location**: `src/cursus/steps/scripts/fraud_two_sequence_preprocessing.py`

### 3A. **AttentionModelTraining Step** (Base Sharable)

**Purpose**: Train general attention-based models with configurable architectures for any sequence modeling tasks

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/train.py` (main training orchestration)
- `dockers/tsa/scripts/models.py` (attention model architectures)
- `dockers/tsa/scripts/basic_blocks.py` (attention layer implementations)
- `dockers/tsa/scripts/utilities.py` (training utilities and optimization)

**Sharable Features** (80% of functionality):

#### **1. Multi-Head Attention Architecture Training**
**TSA Implementation**: Configurable multi-head attention from `train.py` and `basic_blocks.py`
```python
# Multi-head attention configuration from train.py get_parser()
parser.add_argument(
    "--num_heads",
    type=int,
    default=1,
    help="number of heads for multi-head attention",
)
parser.add_argument(
    "--dim_attn_feedforward",
    type=int,
    default=64,
    help="feedforward dimension for multi-head attention",
)

# Multi-head attention implementation from basic_blocks.py AttentionLayer
class AttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads=1,
        dropout=0.1,
        use_moe=True,
        num_experts=5,
        use_time_seq=True,
    ):
        # Configurable multi-head attention mechanism
        if self.use_time_seq:
            self.multi_attn = TemporalMultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
        else:
            self.multi_attn = nn.modules.MultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
```
**Operations**:
- Configurable number of attention heads (1-16+ heads supported)
- Flexible feedforward dimensions for attention layers
- Standard and temporal multi-head attention mechanisms
- Dropout regularization for attention weights
- Scalable attention computation for different sequence lengths

#### **2. Layered Attention Architecture with Configurable Depth**
**TSA Implementation**: Multi-layer attention stacking from `basic_blocks.py` and `models.py`
```python
# Configurable attention layer depth from train.py
parser.add_argument(
    "--n_layers_order",
    type=int,
    default=6,
    help="number of layers for multi-head order attention",
)
parser.add_argument(
    "--n_layers_feature",
    type=int,
    default=6,
    help="number of layers for multi-head feature attention",
)

# Multi-layer attention implementation from basic_blocks.py OrderAttentionLayer
self.layer_stack = nn.ModuleList(
    [
        AttentionLayer(
            dim_embed,
            dim_attn_feedforward,
            num_heads,
            dropout=dropout,
            use_moe=use_moe,
            num_experts=num_experts,
            use_time_seq=use_time_seq,
        )
        for _ in range(n_layers_order)  # Configurable layer depth
    ]
)

# Layer-wise attention processing
for att_layer in self.layer_stack:
    x = att_layer(x, time_seq, attn_mask, key_padding_mask)
```
**Operations**:
- Configurable attention layer depth (1-12+ layers supported)
- Sequential attention layer processing with residual connections
- Independent layer configuration for different attention types
- Layer normalization and dropout between attention layers
- Memory-efficient layer stacking for deep attention models

#### **3. Flexible Embedding and Feature Integration**
**TSA Implementation**: Configurable embedding systems from `models.py` and `basic_blocks.py`
```python
# Embedding configuration from train.py
parser.add_argument(
    "--n_embedding",
    type=int,
    default=1352,
    metavar="NE",
    help="size of feature embedding lookup table",
)
parser.add_argument(
    "--dim_embedding_table",
    type=int,
    default=128,
    help="embedding lookup table dimension",
)

# Embedding implementation from models.py OrderFeatureAttentionClassifier
self.embedding = nn.Embedding(
    n_embedding + 2, dim_embedding_table, padding_idx=0
)

# Feature aggregation from basic_blocks.py
self.feature_aggregation_cat = FeatureAggregation(n_cat_features)
self.feature_aggregation_num = FeatureAggregation(n_num_features)

# Embedding processing in OrderAttentionLayer
cat_indices = x_cat.int()
x_cat_all = self.embedding(cat_indices)
x_cat = self.feature_aggregation_cat(x_cat_all.permute(0, 1, 3, 2)).squeeze(-1)
```
**Operations**:
- Configurable embedding table sizes and dimensions
- Categorical and numerical feature embedding integration
- Feature aggregation networks for dimensionality reduction
- Padding-aware embedding with configurable padding indices
- Multi-modal feature fusion for attention input preparation

#### **4. Advanced Attention Mechanisms and Masking**
**TSA Implementation**: Attention masking and temporal encoding from `utilities.py` and `basic_blocks.py`
```python
# Attention mask generation from utilities.py
def get_subsequent_mask(len_s, device):
    """For masking out the subsequent info."""
    subsequent_mask = (
        torch.triu(torch.ones((len_s, len_s), device=device), diagonal=1)
    ).bool()
    return subsequent_mask

# Key padding mask handling from utilities.py iteration functions
if use_key_padding_mask:
    key_padding_mask = torch.logical_not(
        torch.nn.functional.pad(batch["not_padded"], (0, 1), value=True)
    ).to(device)
else:
    key_padding_mask = None

attn_mask = get_subsequent_mask(50, device).to(device) if use_attn_mask else None

# Temporal encoding from basic_blocks.py TimeEncode
class TimeEncode(torch.nn.Module):
    def forward(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])
        t = torch.cat([out1, out2], -1)
        return t
```
**Operations**:
- Causal and bidirectional attention masking strategies
- Key padding mask computation for variable-length sequences
- Temporal position encoding with learnable and sinusoidal components
- Attention mask broadcasting for batch processing
- Configurable masking strategies for different attention patterns

#### **5. Mixture of Experts (MoE) Integration**
**TSA Implementation**: MoE feedforward networks from `basic_blocks.py` and configuration
```python
# MoE configuration from train.py
parser.add_argument(
    "--use_moe",
    type=int,
    default=1,
    metavar="UMOE",
    help="whether to use mixture of experts inside transformer layer",
)
parser.add_argument(
    "--num_experts",
    type=int,
    default=1,
    metavar="NE",
    help="number of experts to use",
)

# MoE implementation in AttentionLayer from basic_blocks.py
if use_moe:
    self.feedforward = MoE(
        dim=dim_embed,
        num_experts=num_experts,
        hidden_dim=dim_attn_feedforward,
        second_policy_train="random",
        second_policy_eval="random",
    )
else:
    self.feedforward = nn.Sequential(
        nn.Linear(dim_embed, dim_attn_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_attn_feedforward, dim_embed),
    )
```
**Operations**:
- Configurable Mixture of Experts feedforward networks
- Expert routing with random and learned policies
- Scalable expert capacity for model complexity control
- Standard feedforward fallback for non-MoE configurations
- Expert load balancing and routing efficiency optimization

#### **6. Distributed Training Infrastructure**
**TSA Implementation**: Multi-GPU distributed training from `train.py`
```python
# Distributed training setup from train.py
parser.add_argument("--local_rank", default=int(os.environ["LOCAL_RANK"]), type=int)

# DDP model wrapping
if args.local_rank >= 0:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
    )

# Distributed evaluation from utilities.py
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

# Distributed synchronization
dist.all_reduce(exit_flag, op=dist.ReduceOp.MAX)  # Synchronize the exit flag
dist.barrier()  # Process synchronization
```
**Operations**:
- Multi-GPU distributed data parallel (DDP) training
- Synchronized batch normalization across GPUs
- Distributed gradient aggregation and parameter updates
- Cross-GPU evaluation result gathering and synchronization
- Process coordination and barrier synchronization for training stability

#### **7. Advanced Optimization and Scheduling**
**TSA Implementation**: Sophisticated optimization from `utilities.py` and `train.py`
```python
# Optimizer configuration from utilities.py create_optimizer()
def create_optimizer(args, model_params):
    if args.optim == "adam":
        optimizer = Adam(
            params=model_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    scheduler = OneCycleLR(
        optimizer=optimizer, max_lr=args.scheduler_maxlr, total_steps=args.max_epoch
    )
    return optimizer, scheduler

# Mixed precision training from train.py
scaler = torch.cuda.amp.GradScaler() if args.use_amp == 1 else None

# Training loop with AMP from utilities.py train()
with torch.cuda.amp.autocast(enabled=use_amp):
    pred, y_ = iteration(batch, model, use_time_seq, use_engineered_features)
    loss = criterion(pred, y_)

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
**Operations**:
- Configurable optimizers (Adam, SGD, RMSprop) with hyperparameter tuning
- Advanced learning rate scheduling (OneCycleLR, CosineAnnealing, CyclicLR)
- Automatic mixed precision (AMP) training for memory efficiency
- Gradient scaling and clipping for training stability
- Flexible optimization strategies for different model architectures

#### **8. Comprehensive Loss Functions and Evaluation**
**TSA Implementation**: Advanced loss functions from `utilities.py` and focal loss implementations
```python
# Loss function configuration from utilities.py set_loss()
def set_loss(args):
    if args.loss == "FocalLoss":
        criterion = FocalLoss(
            gamma=args.loss_gamma, alpha=args.loss_alpha, reduction=args.loss_reduction
        )
    elif args.loss == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(reduction="sum")
    elif args.loss == "Cyclical_FocalLoss":
        criterion = Cyclical_FocalLoss(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            epochs=args.max_epoch,
            gamma_hc=args.gamma_hc,
            factor=args.cyclical_factor,
            reduction="sum",
        )
    return criterion

# Comprehensive evaluation metrics from utilities.py get_performance()
def get_performance(score, label, measures):
    performance = {}
    if "auc" in measures:
        auc = roc_auc_score(
            np.array(label), np.array(softmax(score, axis=1))[:, -1]
        )
        performance["auc"] = round(auc, ndigits)
    
    if "pr_auc" in measures:
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            np.array(label), np.array(softmax(score, axis=1))[:, -1], pos_label=1
        )
        pr_auc = metrics.auc(recalls, precisions)
        performance["pr_auc"] = round(pr_auc, ndigits)
```
**Operations**:
- Multiple loss function support (CrossEntropy, Focal, Cyclical Focal, BCEWithLogits)
- Configurable loss parameters (gamma, alpha, reduction strategies)
- Comprehensive evaluation metrics (AUC, PR-AUC, Precision, Recall, F1, Accuracy)
- Class imbalance handling through advanced loss functions
- Performance tracking and logging for model monitoring

**Configuration Parameters**:
```python
{
    "attention_architecture": {
        "num_heads": 1,                          # Configurable from TSA num_heads
        "dim_attn_feedforward": 64,              # Configurable from TSA dim_attn_feedforward
        "n_layers_order": 6,                     # Configurable from TSA n_layers_order
        "n_layers_feature": 6,                   # Configurable from TSA n_layers_feature
        "dropout": 0.1,                          # Attention dropout rate
        "use_time_seq": True,                    # Enable temporal attention mechanisms
        "attention_type": ["self_attention", "cross_attention", "multi_head"]
    },
    "embedding_configuration": {
        "n_embedding": 1352,                     # Embedding table size from TSA
        "dim_embedding_table": 128,              # Embedding dimension from TSA
        "n_cat_features": 53,                    # Categorical features from TSA
        "n_num_features": 47,                    # Numerical features from TSA
        "emb_tbl_use_bias": True,               # Embedding bias usage
        "padding_idx": 0                         # Padding index for embeddings
    },
    "sequence_configuration": {
        "seq_len": 51,                          # Sequence length from TSA
        "use_key_padding_mask": True,           # Enable padding masks
        "use_attn_mask": False,                 # Causal masking configuration
        "mask_strategy": ["causal", "bidirectional", "custom"]
    },
    "mixture_of_experts": {
        "use_moe": True,                        # Enable MoE from TSA
        "num_experts": 5,                       # Number of experts from TSA
        "expert_policy": ["random", "learned"], # Expert routing policies
        "load_balancing": True                   # Expert load balancing
    },
    "training_configuration": {
        "batch_size": 1024,                     # Training batch size from TSA
        "max_epoch": 70,                        # Maximum epochs from TSA
        "optimizer": ["adam", "sgd", "rmsprop"], # Optimizer choices from TSA
        "lr": 1e-5,                             # Learning rate from TSA
        "scheduler_maxlr": 1e-3,                # Scheduler max LR from TSA
        "weight_decay": 0.0,                    # Weight decay from TSA
        "use_amp": False,                       # Automatic mixed precision
        "patience": 10                          # Early stopping patience from TSA
    },
    "loss_and_evaluation": {
        "loss_functions": ["CrossEntropyLoss", "FocalLoss", "Cyclical_FocalLoss", "BCEWithLogitsLoss"],
        "loss_gamma": 2,                        # Focal loss gamma from TSA
        "loss_alpha": 0.25,                     # Focal loss alpha from TSA
        "evaluation_metrics": ["auc", "pr_auc", "precision", "recall", "f1", "accuracy"],
        "loss_reduction": ["mean", "sum"]       # Loss reduction strategies
    },
    "distributed_training": {
        "enable_ddp": True,                     # Distributed data parallel
        "sync_batchnorm": True,                 # Synchronized batch normalization
        "find_unused_parameters": True,         # DDP unused parameter handling
        "gradient_synchronization": True        # Cross-GPU gradient sync
    },
    "model_checkpointing": {
        "save_best_model": True,                # Save best validation model
        "checkpoint_frequency": "epoch",        # Checkpointing frequency
        "early_stopping": True,                 # Enable early stopping
        "model_versioning": True                # Model version management
    }
}
```

**Script Location**: `src/cursus/steps/scripts/attention_model_training.py`

### 3B. **TemporalSelfAttentionTraining Step** (Domain-Specific Extension)

**Purpose**: Complete Temporal Self-Attention model training with specialized fraud detection architectures and dual-sequence MoE support

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/train.py` (TSA-specific training orchestration and model building)
- `dockers/tsa/scripts/models.py` (TSA model architectures: OrderFeatureAttentionClassifier, TwoSeqMoEOrderFeatureAttentionClassifier)
- `dockers/tsa/scripts/basic_blocks.py` (TSA-specific attention layers: OrderAttentionLayer, FeatureAttentionLayer)
- `dockers/tsa/scripts/dataloaders.py` (TSA dual-sequence data loading)
- `dockers/tsa/scripts/utilities.py` (TSA-specific training utilities and evaluation)

**Domain-Specific Features** (20% of functionality):

#### **1. TSA Model Architecture Selection and Building**
**TSA Implementation**: Specialized TSA model architectures from `train.py` and `models.py`
```python
# TSA model selection from train.py build_model()
def build_model(args, modelname):
    if modelname == "OrderFeature":
        model = OrderFeatureAttentionClassifier(
            args.n_cat_features,           # 53 categorical features for fraud detection
            args.n_num_features,           # 47 numerical features for fraud detection
            args.n_classes,                # 2 classes (fraud/non-fraud)
            args.n_embedding,              # 1352 embedding table size
            args.seq_len,                  # 51 sequence length
            args.n_engineered_num_features, # 0 engineered features
            args.dim_embedding_table,      # 128 embedding dimension
            args.dim_attn_feedforward,     # 64 feedforward dimension
            args.use_mlp,                  # MLP block usage
            args.num_heads,                # Multi-head attention heads
            args.dropout,                  # 0.1 dropout rate
            args.n_layers_order,           # 6 order attention layers
            args.n_layers_feature,         # 6 feature attention layers
            args.emb_tbl_use_bias,         # Embedding bias usage
            args.use_moe,                  # Mixture of Experts usage
            args.num_experts,              # Number of experts
            args.use_time_seq,             # Temporal sequence usage
        ).to(args.device)
    elif modelname == "TwoSeqMoEOrderFeature":
        model = TwoSeqMoEOrderFeatureAttentionClassifier(
            # Same parameters for dual-sequence fraud detection model
            args.n_cat_features, args.n_num_features, args.n_classes,
            args.n_embedding, args.seq_len, args.n_engineered_num_features,
            args.dim_embedding_table, args.dim_attn_feedforward, args.num_heads,
            args.dropout, args.n_layers_order, args.n_layers_feature,
            args.emb_tbl_use_bias, args.use_moe, args.num_experts, args.use_time_seq,
        ).to(args.device)
```
**Operations**:
- Fraud-specific TSA model architecture selection (single vs dual-sequence)
- Specialized parameter configuration for fraud detection (53 cat + 47 num features)
- Temporal Self-Attention layer configuration (6 order + 6 feature layers)
- Mixture of Experts integration for fraud pattern complexity

#### **2. Dual-Sequence TSA Architecture for Fraud Detection**
**TSA Implementation**: Two-sequence MoE model from `models.py` TwoSeqMoEOrderFeatureAttentionClassifier
```python
# Dual-sequence TSA architecture from models.py
class TwoSeqMoEOrderFeatureAttentionClassifier(torch.nn.Module):
    def __init__(self, n_cat_features, n_num_features, n_classes, n_embedding, seq_len, ...):
        # Dual order attention layers for CID and CCID sequences
        self.order_attention_cid = OrderAttentionLayer(
            self.n_cat_features, self.n_num_features, self.n_embedding, self.seq_len,
            self.dim_embed, self.dim_attn_feedforward, self.embedding,
            self.num_heads, self.dropout, self.n_layers_order,
            self.emb_tbl_use_bias, self.use_moe, self.num_experts, self.use_time_seq
        )
        
        self.order_attention_ccid = OrderAttentionLayer(
            # Parallel structure for credit card sequence processing
            self.n_cat_features, self.n_num_features, self.n_embedding, self.seq_len,
            self.dim_embed, self.dim_attn_feedforward, self.embedding,
            self.num_heads, self.dropout, self.n_layers_order,
            self.emb_tbl_use_bias, self.use_moe, self.num_experts, self.use_time_seq
        )

        # Gate function for sequence importance weighting
        self.embedding_gate = nn.Embedding(n_embedding + 2, 16, padding_idx=0)
        self.gate_emb = OrderAttentionLayer(
            self.n_cat_features, self.n_num_features, self.n_embedding, self.seq_len,
            32, 128, self.embedding_gate, 1, self.dropout, 1,
            self.emb_tbl_use_bias, 0, 1, False, False,  # Simplified gate function
        )
        self.gate_score = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 2), nn.Softmax(dim=1),  # CID vs CCID importance
        )
```
**Operations**:
- Parallel Customer ID (CID) and Credit Card ID (CCID) sequence processing
- Gate function for dynamic sequence importance weighting
- Fraud-specific dual-sequence attention coordination
- Customer-card behavior pattern fusion for fraud detection

#### **3. TSA-Specific Temporal Attention Mechanisms**
**TSA Implementation**: Specialized temporal attention from `basic_blocks.py` OrderAttentionLayer and FeatureAttentionLayer
```python
# Order attention for temporal sequence processing from basic_blocks.py
class OrderAttentionLayer(torch.nn.Module):
    def forward(self, x_cat, x_num, time_seq, attn_mask=None, key_padding_mask=None):
        # Temporal feature aggregation for fraud detection
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)
        x_cat = self.feature_aggregation_cat(x_cat_all.permute(0, 1, 3, 2)).squeeze(-1)
        
        # Numerical feature processing with temporal encoding
        num_indices = torch.arange(
            self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
        ).repeat(B, L).view(B, L, -1).to(x_cat.device)
        x_num_all = self.embedding(num_indices) * (x_num[..., None])
        x_num = self.feature_aggregation_num(x_num_all.permute(0, 1, 3, 2)).squeeze(-1)
        
        # Temporal sequence processing with dummy order token
        x = torch.cat([x_cat, x_num], dim=-1)
        dummy = self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)
        
        # Multi-layer temporal attention processing
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)
        
        return x[:, -1, :] if not self.return_seq else x

# Feature attention for current transaction from basic_blocks.py
class FeatureAttentionLayer(torch.nn.Module):
    def forward(self, x_cat, x_num, x_engineered):
        # Current transaction feature attention processing
        x_cat_last = x_cat_all[:, -1, :, :]  # Last transaction categorical features
        x_num_last = x_num_all[:, -1, :, :]  # Last transaction numerical features
        x_last = torch.cat([x_cat_last, x_num_last], dim=1)
        
        # Multi-layer feature attention for current transaction
        for att_layer_feature in self.layer_stack_feature:
            x_last = att_layer_feature(x_last, None, None)
        
        return x_last[:, -1, :]  # Final feature representation
```
**Operations**:
- Temporal order attention for transaction sequence analysis
- Feature attention for current transaction fraud risk assessment
- Dummy order token for sequence-level representation
- Multi-layer attention processing for temporal fraud patterns

#### **4. TSA Gate Function for Dual-Sequence Weighting**
**TSA Implementation**: Gate function mechanism from `models.py` TwoSeqMoEOrderFeatureAttentionClassifier forward method
```python
# Gate function for dual-sequence importance weighting
def forward(self, x_seq_cat_cid, x_seq_num_cid, time_seq_cid, 
           x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid, x_engineered, ...):
    # Gate function embeddings for both sequences
    gate_emb_cid = self.gate_emb(
        x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
    )
    gate_emb_ccid = self.gate_emb(
        x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid, attn_mask, key_padding_mask_ccid
    )
    
    # Gate score computation for sequence importance
    gate_scores_raw = self.gate_score(
        torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1)
    )
    gate_scores = gate_scores_raw.clone()
    
    # Handle empty CCID sequences (no credit card history)
    gate_scores[
        (torch.sum(key_padding_mask_ccid, dim=1) == 50).nonzero().squeeze(-1), 1
    ] = 0
    
    # Dynamic sequence selection based on gate scores
    ccid_keep_idx = (
        (gate_scores[:, 1] > 0.05).nonzero().squeeze(-1).to(x_seq_cat_cid.device)
    )
    
    # Weighted ensemble of dual sequences
    ensemble_order = torch.einsum(
        "i,ij->ij", gate_scores[:, 0], x_cid
    ) + torch.einsum("i,ij->ij", gate_scores[:, 1], x_ccid)
    
    return scores, ensemble
```
**Operations**:
- Dynamic gate function for CID vs CCID sequence importance
- Empty sequence handling for customers without credit card history
- Fraud-specific sequence weighting based on data availability
- Ensemble combination of customer and payment method sequences

#### **5. TSA-Specific Data Loading for Dual Sequences**
**TSA Implementation**: Specialized data loading from `dataloaders.py` and `utilities.py`
```python
# TSA dual-sequence data loading from dataloaders.py
def load_data_two_seq(args, batch_size, data_version):
    # Load dual-sequence fraud detection data
    train_dataset = TwoSeqDataset(
        args.train_data_folder, data_version, "train"
    )
    vali_dataset = TwoSeqDataset(
        args.vali_data_folder, data_version, "vali"
    )
    cali_dataset = TwoSeqDataset(
        args.cali_data_folder, data_version, "cali"
    )
    
    # Distributed samplers for multi-GPU training
    if args.local_rank >= 0:
        train_sampler = DistributedSampler(train_dataset)
        vali_sampler = DistributedSampler(vali_dataset)
        cali_sampler = DistributedSampler(cali_dataset)
    
    return train_sampler, train_dataloader, vali_sampler, vali_dataloader, cali_sampler, cali_dataloader

# TSA dual-sequence iteration from utilities.py
def iteration_two_seq(batch, model, use_time_seq, use_engineered_features, ...):
    # Extract dual-sequence data
    x_seq_cat_cid, x_seq_num_cid = batch["x_seq_cat_cid"].to(device), batch["x_seq_num_cid"].to(device)
    x_seq_cat_ccid, x_seq_num_ccid = batch["x_seq_cat_ccid"].to(device), batch["x_seq_num_ccid"].to(device)
    y = batch["y"].long().view(-1).to(device)
    
    # Dual-sequence padding masks
    key_padding_mask_cid = torch.logical_not(
        torch.nn.functional.pad(batch["not_padded_cid"], (0, 1), value=True)
    ).to(device)
    key_padding_mask_ccid = torch.logical_not(
        torch.nn.functional.pad(batch["not_padded_ccid"], (0, 1), value=True)
    ).to(device)
    
    # Dual-sequence temporal information
    time_seq_cid = batch["time_to_last_cid"].to(device) if use_time_seq else None
    time_seq_ccid = batch["time_to_last_ccid"].to(device) if use_time_seq else None
    
    # TSA dual-sequence model forward pass
    pred, _ = model(
        x_seq_cat_cid, x_seq_num_cid, time_seq_cid,
        x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid,
        x_engineered, attn_mask, key_padding_mask_cid, key_padding_mask_ccid
    )
    
    return pred, y
```
**Operations**:
- Dual-sequence dataset loading for CID and CCID sequences
- Distributed sampling for multi-GPU fraud detection training
- Dual-sequence padding mask computation
- Temporal information coordination across sequences

#### **6. TSA-Specific Loss Functions and Evaluation**
**TSA Implementation**: Fraud-optimized loss functions from `utilities.py` and training loop
```python
# TSA-specific loss configuration from utilities.py set_loss()
def set_loss(args):
    if args.loss == "FocalLoss":
        # Focal loss for fraud class imbalance
        criterion = FocalLoss(
            gamma=args.loss_gamma,      # 2.0 for hard example focus
            alpha=args.loss_alpha,      # 0.25 for class balance
            reduction=args.loss_reduction
        )
    elif args.loss == "Cyclical_FocalLoss":
        # Cyclical focal loss for fraud detection training dynamics
        criterion = Cyclical_FocalLoss(
            gamma_pos=args.gamma_pos,   # 0 for positive examples
            gamma_neg=args.gamma_neg,   # 4 for negative examples (non-fraud)
            epochs=args.max_epoch,      # 70 epochs
            gamma_hc=args.gamma_hc,     # 0 cyclical gamma
            factor=args.cyclical_factor, # 2 cyclical factor
            reduction="sum",
        )
    return criterion

# TSA fraud-specific evaluation from utilities.py get_performance()
def get_performance(score, label, measures):
    # Fraud detection specific metrics
    if "auc" in measures:
        # ROC-AUC for fraud detection performance
        auc = roc_auc_score(
            np.array(label), np.array(softmax(score, axis=1))[:, -1]
        )
    
    if "pr_auc" in measures:
        # Precision-Recall AUC for imbalanced fraud detection
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            np.array(label), np.array(softmax(score, axis=1))[:, -1], pos_label=1
        )
        pr_auc = metrics.auc(recalls, precisions)
    
    if "aps" in measures:
        # Average Precision Score for fraud detection
        aps = average_precision_score(
            np.array(label), np.array(softmax(score, axis=1))[:, -1], pos_label=1
        )
    
    return performance
```
**Operations**:
- Focal loss for fraud class imbalance handling (alpha=0.25, gamma=2.0)
- Cyclical focal loss for dynamic fraud detection training
- ROC-AUC and PR-AUC evaluation for imbalanced fraud datasets
- Average Precision Score for fraud detection performance assessment

#### **7. TSA Training Loop with Early Stopping and Model Selection**
**TSA Implementation**: Fraud-optimized training loop from `train.py` main()
```python
# TSA training loop with fraud-specific early stopping
for epoch in range(epoch_start, args.max_epoch):
    # Distributed training coordination
    dist.all_reduce(exit_flag, op=dist.ReduceOp.MAX)
    
    if exit_flag.item() == 1:
        # Early stopping triggered - load best model for fraud detection
        state_dict = torch.load(model_name, map_location=torch.device("cpu"))
        model.load_state_dict(new_state_dict)
        
        # Generate fraud detection scores and labels
        perf_valid, scores_valid, labels_valid = evaluation_single_seq(
            args, vali_dataloader, model, epoch, vali_sampler, measures, ...
        )
        
        # Save fraud detection results
        if args.local_rank == 0:
            # Fraud probability scores
            score = pd.DataFrame({
                "score": np.array(softmax(scores_valid.data.tolist(), axis=1))[:, -1]
            })
            score.to_csv(args.model_dir + "/score_file.csv", header=False, index=False)
            
            # Fraud labels
            score = pd.DataFrame({"IS_FRD": np.array(labels_valid.data.tolist())})
            score.to_csv(args.model_dir + "/tag_file.csv", header=False, index=False)
        break
    
    # TSA training step
    train(args, train_dataloader, model, epoch, train_sampler, optimizer, scheduler, ...)
    
    # TSA validation and model selection
    perf_valid, _, _ = evaluation_single_seq(args, vali_dataloader, model, epoch, ...)
    
    # Fraud detection model selection based on AUC
    if perf_valid["auc"] > best_auc:
        best_auc = perf_valid["auc"]
        torch.save(model.module.state_dict(), model_name)
        checkpoint = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
        }
        torch.save(checkpoint, checkpoint_name)
```
**Operations**:
- Distributed early stopping coordination across GPUs
- Fraud detection model selection based on validation AUC
- Fraud probability score generation and saving
- Fraud label coordination and result persistence

#### **8. TSA Calibration Data Processing for Fraud Detection**
**TSA Implementation**: Fraud-specific calibration processing from `train.py`
```python
# TSA calibration data processing for fraud detection
perf_cali, scores_cali, labels_cali = evaluation_single_seq(
    args, cali_dataloader, model, epoch, cali_sampler, measures, ...
)

if args.local_rank == 0:
    # Fraud detection calibration scores
    y_score_calib = np.array(softmax(scores_cali.data.tolist(), axis=1))[:, -1]
    
    # Save calibration results for fraud detection
    score = pd.DataFrame({"score": y_score_calib})
    score.to_csv(args.model_dir + "/cali_score_file.csv", header=False, index=False)
    
    score = pd.DataFrame({"IS_FRD": np.array(labels_cali.data.tolist())})
    score.to_csv(args.model_dir + "/cali_tag_file.csv", header=False, index=False)
    
    # Generate percentile mapping for fraud detection deployment
    bins = 1000
    percntls = np.array(range(bins)) / (bins + 0.0)
    percentile_score_sorted_arr = np.percentile(y_score_calib, percntls * 100)
    
    # Save percentile mapping for fraud detection inference
    model_location = args.model_dir + "/percentile_score.pkl"
    pkl.dump(
        np.column_stack((percentile_score_sorted_arr, percntls)).tolist(),
        open(model_location, "wb"),
    )
```
**Operations**:
- Fraud detection calibration data evaluation
- Percentile score mapping generation for fraud detection deployment
- Calibration result persistence for fraud detection inference
- Statistical calibration analysis for fraud detection model deployment

**Configuration Parameters**:
```python
{
    "extends": "AttentionModelTraining",
    "tsa_model_architectures": {
        "single_sequence": "OrderFeatureAttentionClassifier",
        "dual_sequence": "TwoSeqMoEOrderFeatureAttentionClassifier",
        "default_model": "TwoSeqMoEOrderFeature"          # From TSA modelname parameter
    },
    "fraud_detection_features": {
        "n_cat_features": 53,                             # Categorical features from TSA
        "n_num_features": 47,                             # Numerical features from TSA  
        "n_classes": 2,                                   # Binary fraud classification
        "fraud_label": "IS_FRD",                          # Fraud detection target
        "n_engineered_num_features": 0                    # No additional engineered features
    },
    "tsa_attention_configuration": {
        "n_layers_order": 6,                              # Order attention layers from TSA
        "n_layers_feature": 6,                            # Feature attention layers from TSA
        "order_attention_type": "temporal_sequence",      # Temporal order attention
        "feature_attention_type": "current_transaction",  # Current transaction attention
        "dummy_order_token": True                         # Sequence-level representation
    },
    "dual_sequence_configuration": {
        "sequence_types": ["customer_id", "credit_card_id"], # CID and CCID sequences
        "gate_function_enabled": True,                    # Dynamic sequence weighting
        "gate_embedding_dim": 16,                         # Gate function embedding size
        "gate_hidden_dim": 256,                           # Gate function hidden dimension
        "empty_sequence_handling": True,                  # Handle missing CCID sequences
        "sequence_importance_threshold": 0.05             # Minimum gate score threshold
    },
    "tsa_loss_configuration": {
        "primary_loss": "Cyclical_FocalLoss",             # Default TSA loss function
        "loss_gamma": 2,                                  # Focal loss gamma from TSA
        "loss_alpha": 0.25,                               # Focal loss alpha from TSA
        "gamma_pos": 0,                                   # Positive gamma for cyclical focal
        "gamma_neg": 4,                                   # Negative gamma for cyclical focal
        "cyclical_factor": 2,                             # Cyclical factor from TSA
        "gamma_hc": 0                                     # Cyclical gamma from TSA
    },
    "fraud_evaluation_metrics": {
        "primary_metric": "auc",                          # ROC-AUC for model selection
        "fraud_metrics": ["auc", "pr_auc", "aps", "precision", "recall", "f1"],
        "imbalanced_focus": True,                         # Focus on imbalanced metrics
        "calibration_enabled": True                       # Enable calibration processing
    },
    "tsa_training_configuration": {
        "max_epoch": 70,                                  # Maximum epochs from TSA
        "patience": 10,                                   # Early stopping patience from TSA
        "steps_per_epoch": 2000,                          # Steps per epoch from TSA
        "model_selection_metric": "auc",                  # AUC-based model selection
        "save_calibration_results": True,                 # Save calibration data
        "percentile_bins": 1000                           # Percentile mapping resolution
    },
    "fraud_data_loading": {
        "dual_sequence_datasets": True,                   # Enable dual-sequence loading
        "sequence_keys": ["cid", "ccid"],                # Sequence identifier keys
        "padding_mask_coordination": True,                # Coordinate padding across sequences
        "temporal_coordination": True,                    # Coordinate temporal information
        "distributed_sampling": True                      # Multi-GPU distributed sampling
    },
    "tsa_deployment_preparation": {
        "score_file_generation": True,                    # Generate fraud probability scores
        "tag_file_generation": True,                      # Generate fraud labels
        "percentile_mapping": True,                       # Generate percentile mappings
        "calibration_artifacts": True,                    # Save calibration artifacts
        "model_checkpointing": True                       # Save model checkpoints
    }
}
```

**Script Location**: `src/cursus/steps/scripts/temporal_self_attention_training.py`

### 4A. **DistributedProcessing Step** (Base Sharable)

**Purpose**: Handle general large-scale distributed data processing with memory management and chunked processing coordination

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (chunked processing coordination in `chunk_processing()`)
- `dockers/tsa/scripts/preprocess_train_na.py` (distributed processing orchestration)
- `dockers/tsa/scripts/preprocess_vali_na.py` (validation data distributed processing)
- `dockers/tsa/scripts/preprocess_cali_na.py` (calibration data distributed processing)

**Sharable Features** (80% of functionality):

#### **1. Intelligent Memory-Based Data Chunking**
**TSA Implementation**: Adaptive chunking strategy from `chunk_processing()` function
```python
# Memory-efficient chunking from preprocess_functions_na.py chunk_processing()
def chunk_processing(data_path, dataset, num_chunk):
    # Load and partition data files for distributed processing
    files = glob.glob(os.path.join(data_path, "*.csv"))
    files_chunk_list = np.array_split(files, num_chunk)
    
    # Memory-aware chunk size calculation
    for i_chunk in range(num_chunk):
        files_chunk = files_chunk_list[i_chunk]
        print(f"Processing chunk {i_chunk+1}/{num_chunk} with {len(files_chunk)} files")
        
        # Adaptive processing based on dataset type and memory constraints
        if dataset == "train":
            # Training data: 60 chunks for memory efficiency
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_training_data_by_chunk(files_chunk, tag)
        elif dataset == "cali":
            # Calibration data: 5 chunks for smaller dataset
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_calibration_data_by_chunk(files_chunk, tag)
        elif dataset == "vali":
            # Validation data: 10 chunks for medium dataset
            (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
             X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
             X_num_chunk, Y_chunk) = processing_validation_data_by_chunk(files_chunk, tag)
```
**Operations**:
- Adaptive chunk size calculation based on available memory and dataset size
- File-based chunking with configurable chunk counts (60 for training, 5 for calibration, 10 for validation)
- Memory-aware processing function selection based on dataset characteristics
- Dynamic chunk allocation using numpy array splitting for balanced distribution

#### **2. Multi-Instance Parallel Processing Coordination**
**TSA Implementation**: CPU-aware parallel processing from `parallel_data_parsing()` function
```python
# Multi-instance parallel coordination from preprocess_functions_na.py
def parallel_data_parsing(df, num_workers=80):
    # CPU-aware job allocation for distributed processing
    if len(files) < cpu_count() - 9:
        num_jobs = len(files)
    else:
        num_jobs = -10  # Use all CPUs except 10 for system stability
    
    # Parallel processing with joblib for distributed coordination
    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in files)
    df = pd.concat(df_list, ignore_index=True)
    
    # Multi-process data parsing coordination
    pool = multiprocessing.Pool(processes=num_workers)
    results = []
    
    # Distribute data parsing across multiple processes
    for i in range(len(df)):
        result = pool.apply_async(data_parsing, (df.iloc[i].to_dict(),))
        results.append(result)
    
    # Coordinate result collection from distributed processes
    X_seq_cat_cid_list = []
    X_seq_num_cid_list = []
    X_seq_cat_ccid_list = []
    X_seq_num_ccid_list = []
    X_num_list = []
    Y_list = []
    
    # Process results from distributed workers
    for result in results:
        (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
         seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y) = result.get()
        
        if ret:
            X_seq_cat_cid_list.append(seq_cat_mtx_cid)
            X_seq_num_cid_list.append(seq_num_mtx_cid)
            X_seq_cat_ccid_list.append(seq_cat_mtx_ccid)
            X_seq_num_ccid_list.append(seq_num_mtx_ccid)
            X_num_list.append(dense_num_arr)
            Y_list.append(y)
    
    pool.close()
    pool.join()
```
**Operations**:
- CPU-aware job allocation with system resource preservation (leave 10 CPUs for system)
- Multi-process pool coordination with configurable worker counts (default 80 workers)
- Distributed result collection and aggregation across multiple processes
- Process lifecycle management with proper cleanup and synchronization

#### **3. Memory-Mapped File Operations for Large Datasets**
**TSA Implementation**: Memory-efficient storage from `stream_save()` function
```python
# Memory-mapped storage for large-scale distributed processing
def stream_save(A_list, A_name, dataset, out_dir="/opt/ml/processing/output", p="0"):
    # Calculate total memory requirements for distributed data
    final_shape = (sum(arr.shape[0] for arr in A_list),) + A_list[0].shape[1:]
    
    # Create memory-mapped array for efficient large dataset handling
    A = np.memmap(
        filename=os.path.join(out_dir, "{}_{}_v{}.raw".format(dataset, A_name, p)),
        dtype=A_list[0].dtype, 
        mode="w+", 
        shape=final_shape
    )
    
    # Sequential assignment for memory efficiency
    current_position = 0
    for i in range(len(A_list)):
        chunk = A_list[i]
        # Memory-mapped assignment without loading entire dataset into RAM
        A[current_position : current_position + chunk.shape[0], ...] = chunk
        current_position += chunk.shape[0]
        A.flush()  # Ensure data persistence
    
    print(f"Saved {A_name} with shape {final_shape} to memory-mapped file")

# Distributed storage coordination from chunk_processing()
# Save processed chunks using memory-mapped files
stream_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)
stream_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)
stream_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)
stream_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)
stream_save(X_num_chunk_list, "X_num", dataset)
stream_save(Y_chunk_list, "Y", dataset)
```
**Operations**:
- Memory-mapped file creation for datasets larger than available RAM
- Sequential chunk assignment without full dataset loading
- Automatic memory management with flush operations for data persistence
- Distributed file coordination with consistent naming conventions

#### **4. Progress Tracking and Monitoring System**
**TSA Implementation**: Comprehensive progress tracking across distributed processing
```python
# Progress tracking from chunk_processing() function
def chunk_processing(data_path, dataset, num_chunk):
    print(f"Starting distributed processing for {dataset} dataset")
    print(f"Total chunks to process: {num_chunk}")
    
    for i_chunk in range(num_chunk):
        files_chunk = files_chunk_list[i_chunk]
        print(f"Processing chunk {i_chunk+1}/{num_chunk}")
        print(f"Files in current chunk: {len(files_chunk)}")
        
        # Track processing progress for each chunk
        start_time = time.time()
        
        # Process chunk with progress monitoring
        if dataset == "train":
            result = processing_training_data_by_chunk(files_chunk, tag)
        elif dataset == "cali":
            result = processing_calibration_data_by_chunk(files_chunk, tag)
        elif dataset == "vali":
            result = processing_validation_data_by_chunk(files_chunk, tag)
        
        # Calculate and report processing metrics
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Chunk {i_chunk+1} processed in {processing_time:.2f} seconds")
        
        # Memory usage monitoring
        if hasattr(result[0], 'shape'):
            print(f"Chunk {i_chunk+1} output shape: {result[0].shape}")
        
        # Save chunk with progress indication
        chunk_save_start = time.time()
        np.save(file=os.path.join(out_dir, f"{dataset}_chunk_{i_chunk}.npy"), arr=result)
        chunk_save_time = time.time() - chunk_save_start
        print(f"Chunk {i_chunk+1} saved in {chunk_save_time:.2f} seconds")
    
    print(f"Distributed processing completed for {dataset} dataset")
```
**Operations**:
- Real-time progress reporting with chunk-level granularity
- Processing time measurement and performance monitoring
- Memory usage tracking and reporting for optimization
- Comprehensive logging for distributed processing coordination

#### **5. Fault Tolerance and Recovery Mechanisms**
**TSA Implementation**: Error handling and recovery from `parallel_data_parsing()` and `data_parsing()`
```python
# Fault tolerance in parallel processing from parallel_data_parsing()
for result in results:
    try:
        (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
         seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y) = result.get()
        
        # Validate processing result
        if ret:
            # Additional validation for data integrity
            if seq_cat_mtx_cid is not None and seq_num_mtx_cid is not None:
                X_seq_cat_cid_list.append(seq_cat_mtx_cid)
                X_seq_num_cid_list.append(seq_num_mtx_cid)
                X_seq_cat_ccid_list.append(seq_cat_mtx_ccid)
                X_seq_num_ccid_list.append(seq_num_mtx_ccid)
                X_num_list.append(dense_num_arr)
                Y_list.append(y)
            else:
                print(f"Warning: Invalid data in processing result, skipping")
        else:
            print(f"Warning: Processing failed for data item, skipping")
    
    except Exception as e:
        print(f"Error processing data item: {str(e)}")
        continue  # Continue processing other items despite individual failures

# Data validation and recovery from data_parsing()
def data_parsing(input_data):
    try:
        # Validate input data structure
        for VAR in input_data_seq_cat_otf_vars:
            if VAR not in input_data:
                print("Sanity check failed. Input data does not contain required key")
                return False, None, None, None, None, None, None
        
        # Process with error handling
        ret_cid, seq_cat_mtx_cid, seq_num_mtx_cid = sequence_data_parsing(
            input_data, input_data_seq_cat_otf_vars_cid, 
            input_data_seq_num_otf_vars_cid, objectid_otf_name_cid,
        )
        
        # Validate processing results
        if ret_cid == False:
            return False, None, None, None, None, None, None
            
        return True, seq_cat_mtx_cid, seq_num_mtx_cid, seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, y
        
    except Exception as e:
        print(f"Error in data_parsing: {str(e)}")
        return False, None, None, None, None, None, None
```
**Operations**:
- Individual item error handling with graceful degradation
- Data validation and integrity checking at multiple levels
- Exception handling with detailed error logging and recovery
- Robust processing continuation despite individual failures

#### **6. Load Balancing and Resource Management**
**TSA Implementation**: Dynamic load balancing from CPU-aware job allocation
```python
# Dynamic load balancing from parallel_data_parsing()
def parallel_data_parsing(df, num_workers=80):
    # Dynamic CPU allocation based on system resources
    available_cpus = cpu_count()
    
    # Load balancing strategy: preserve system resources
    if len(files) < available_cpus - 9:
        # For small datasets, use fewer workers to avoid overhead
        num_jobs = len(files)
    else:
        # For large datasets, use most CPUs but preserve system stability
        num_jobs = -10  # Use all CPUs except 10
    
    print(f"Using {num_jobs} parallel jobs for load balancing")
    
    # Balanced file distribution for parallel processing
    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in files)
    
    # Dynamic worker allocation based on data size
    data_size = len(df)
    if data_size < 1000:
        actual_workers = min(num_workers, 20)  # Reduce workers for small datasets
    elif data_size < 10000:
        actual_workers = min(num_workers, 40)  # Medium workers for medium datasets
    else:
        actual_workers = num_workers  # Full workers for large datasets
    
    print(f"Allocated {actual_workers} workers for {data_size} data items")
    
    # Create worker pool with balanced allocation
    pool = multiprocessing.Pool(processes=actual_workers)

# Resource-aware chunk allocation from chunk_processing()
def chunk_processing(data_path, dataset, num_chunk):
    # Load balancing across chunks
    files = glob.glob(os.path.join(data_path, "*.csv"))
    total_files = len(files)
    
    # Balanced chunk distribution
    files_chunk_list = np.array_split(files, num_chunk)
    
    # Report load balancing metrics
    chunk_sizes = [len(chunk) for chunk in files_chunk_list]
    print(f"Load balancing: {total_files} files across {num_chunk} chunks")
    print(f"Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.1f}")
```
**Operations**:
- Dynamic CPU allocation based on system resources and dataset size
- Adaptive worker allocation with resource-aware scaling
- Balanced file distribution across processing chunks
- Load balancing metrics and monitoring for optimization

#### **7. Configurable Processing Pipeline Orchestration**
**TSA Implementation**: Flexible processing pipeline from multiple preprocessing scripts
```python
# Configurable processing orchestration from preprocess_train_na.py
def main():
    # Configurable processing parameters
    training_data_path = "/opt/ml/processing/input/training_data"
    calibration_data_path = "/opt/ml/processing/input/calibration_data"
    validation_data_path = "/opt/ml/processing/input/validation_data"
    
    # Configurable chunk counts based on dataset characteristics
    training_chunks = 60    # Large dataset requires more chunks
    calibration_chunks = 5  # Smaller dataset requires fewer chunks
    validation_chunks = 10  # Medium dataset requires medium chunks
    
    # Orchestrate distributed processing pipeline
    print("Starting distributed processing pipeline")
    
    # Training data processing
    print("Processing training data...")
    chunk_processing(training_data_path, "train", training_chunks)
    
    # Calibration data processing
    print("Processing calibration data...")
    chunk_processing(calibration_data_path, "cali", calibration_chunks)
    
    # Validation data processing
    print("Processing validation data...")
    chunk_processing(validation_data_path, "vali", validation_chunks)
    
    print("Distributed processing pipeline completed")

# Configurable processing functions for different data types
def processing_training_data_by_chunk(files_chunk, tag):
    # Training-specific processing configuration
    return parallel_data_parsing(df, num_workers=80)

def processing_calibration_data_by_chunk(files_chunk, tag):
    # Calibration-specific processing configuration
    return parallel_data_parsing(df, num_workers=40)  # Fewer workers for smaller dataset

def processing_validation_data_by_chunk(files_chunk, tag):
    # Validation-specific processing configuration
    return parallel_data_parsing(df, num_workers=60)  # Medium workers for medium dataset
```
**Operations**:
- Configurable processing pipeline with flexible data source paths
- Adaptive chunk count configuration based on dataset characteristics
- Dataset-specific processing function selection and configuration
- Orchestrated pipeline execution with comprehensive logging

#### **8. Distributed Storage Coordination and File Management**
**TSA Implementation**: Coordinated distributed storage from `chunk_processing()` and `stream_save()`
```python
# Distributed storage coordination from chunk_processing()
def chunk_processing(data_path, dataset, num_chunk):
    # Initialize distributed storage coordination
    out_dir = "/opt/ml/processing/output"
    
    # Coordinate storage across multiple data types and chunks
    X_seq_cat_cid_chunk_list = []
    X_seq_num_cid_chunk_list = []
    X_seq_cat_ccid_chunk_list = []
    X_seq_num_ccid_chunk_list = []
    X_num_chunk_list = []
    Y_chunk_list = []
    
    # Process and coordinate storage for each chunk
    for i_chunk in range(num_chunk):
        # Process chunk
        (X_seq_cat_cid_chunk, X_seq_num_cid_chunk,
         X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk,
         X_num_chunk, Y_chunk) = process_chunk(files_chunk, dataset)
        
        # Coordinate individual chunk storage
        np.save(file=os.path.join(out_dir, f"{dataset}_cid_X_seq_cat_chunk_{i_chunk}.npy"), 
                arr=X_seq_cat_cid_chunk)
        np.save(file=os.path.join(out_dir, f"{dataset}_cid_X_seq_num_chunk_{i_chunk}.npy"), 
                arr=X_seq_num_cid_chunk)
        np.save(file=os.path.join(out_dir, f"{dataset}_ccid_X_seq_cat_chunk_{i_chunk}.npy"), 
                arr=X_seq_cat_ccid_chunk)
        np.save(file=os.path.join(out_dir, f"{dataset}_ccid_X_seq_num_chunk_{i_chunk}.npy"), 
                arr=X_seq_num_ccid_chunk)
        np.save(file=os.path.join(out_dir, f"{dataset}_X_num_chunk_{i_chunk}.npy"), 
                arr=X_num_chunk)
        np.save(file=os.path.join(out_dir, f"{dataset}_Y_chunk_{i_chunk}.npy"), 
                arr=Y_chunk)
        
        # Accumulate for consolidated storage
        X_seq_cat_cid_chunk_list.append(X_seq_cat_cid_chunk)
        X_seq_num_cid_chunk_list.append(X_seq_num_cid_chunk)
        X_seq_cat_ccid_chunk_list.append(X_seq_cat_ccid_chunk)
        X_seq_num_ccid_chunk_list.append(X_seq_num_ccid_chunk)
        X_num_chunk_list.append(X_num_chunk)
        Y_chunk_list.append(Y_chunk)
    
    # Coordinate consolidated distributed storage using memory-mapped files
    stream_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)
    stream_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)
    stream_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)
    stream_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)
    stream_save(X_num_chunk_list, "X_num", dataset)
    stream_save(Y_chunk_list, "Y", dataset)
```
**Operations**:
- Coordinated storage across multiple data types and processing chunks
- Consistent file naming conventions for distributed storage management
- Individual chunk storage with consolidated memory-mapped file creation
- Distributed file system coordination with proper directory management

**Configuration Parameters**:
```python
{
    "chunking_strategy": {
        "chunk_method": ["memory_based", "file_based", "size_based", "count_based"],
        "adaptive_chunking": True,                    # Adapt chunk size based on dataset characteristics
        "chunk_counts": {
            "training": 60,                           # Large dataset chunk count from TSA
            "calibration": 5,                         # Small dataset chunk count from TSA
            "validation": 10,                         # Medium dataset chunk count from TSA
            "auto": True                              # Automatic chunk count calculation
        },
        "memory_limit_gb": "auto",                    # Automatic memory limit detection
        "file_distribution": "balanced"               # Balanced file distribution across chunks
    },
    "parallel_processing": {
        "cpu_allocation": "dynamic",                  # Dynamic CPU allocation from TSA
        "worker_counts": {
            "default": 80,                            # Default worker count from TSA
            "small_dataset": 20,                      # Reduced workers for small datasets
            "medium_dataset": 40,                     # Medium workers for medium datasets
            "large_dataset": 80                       # Full workers for large datasets
        },
        "system_resource_preservation": 10,           # Reserve 10 CPUs for system from TSA
        "load_balancing": "adaptive",                 # Adaptive load balancing strategy
        "process_coordination": True                  # Multi-process coordination and synchronization
    },
    "memory_management": {
        "memory_mapping": True,                       # Enable memory-mapped file operations
        "memory_efficient_storage": True,             # Use memory-efficient storage strategies
        "sequential_assignment": True,                # Sequential chunk assignment for memory efficiency
        "automatic_flush": True,                      # Automatic data persistence with flush operations
        "memory_monitoring": True                     # Monitor memory usage during processing
    },
    "fault_tolerance": {
        "error_handling": "graceful_degradation",     # Continue processing despite individual failures
        "data_validation": "multi_level",             # Validate data at multiple processing levels
        "exception_recovery": True,                   # Recover from processing exceptions
        "integrity_checking": True,                   # Check data integrity throughout processing
        "failure_logging": "comprehensive"            # Comprehensive error logging and reporting
    },
    "progress_monitoring": {
        "chunk_level_tracking": True,                 # Track progress at chunk granularity
        "processing_time_measurement": True,          # Measure and report processing times
        "memory_usage_tracking": True,                # Track memory usage for optimization
        "performance_metrics": ["processing_time", "memory_usage", "throughput", "error_rate"],
        "real_time_reporting": True                   # Real-time progress reporting
    },
    "storage_coordination": {
        "distributed_storage": True,                  # Coordinate storage across distributed processing
        "consistent_naming": True,                    # Use consistent file naming conventions
        "individual_chunk_storage": True,             # Store individual chunks for debugging
        "consolidated_storage": True,                 # Create consolidated memory-mapped files
        "storage_validation": True                    # Validate storage operations and file integrity
    },
    "pipeline_orchestration": {
        "configurable_data_paths": True,              # Support configurable input/output paths
        "dataset_specific_processing": True,          # Use dataset-specific processing functions
        "flexible_pipeline_execution": True,          # Support flexible pipeline execution order
        "comprehensive_logging": True,                # Comprehensive pipeline execution logging
        "pipeline_coordination": "sequential"         # Sequential pipeline execution with coordination
    }
}
```

**Script Location**: `src/cursus/steps/scripts/distributed_processing.py`

### 4B. **FraudChunkedDistributedProcessing Step** (Domain-Specific Extension)

**Purpose**: Fraud-specific distributed processing optimized for fraud detection datasets with customer-aware chunking and fraud pattern preservation

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (fraud-specific chunked processing in `processing_training_data_by_chunk()`)
- `dockers/tsa/scripts/preprocess_train_na.py` (fraud data filtering and downsampling)
- `dockers/tsa/scripts/preprocess_vali_na.py` (fraud validation data processing)
- `dockers/tsa/scripts/preprocess_cali_na.py` (fraud calibration data processing)

**Domain-Specific Features** (20% of functionality):

#### **1. Fraud-Specific Dataset Chunking and Filtering**
**TSA Implementation**: Fraud-optimized data filtering from `processing_training_data_by_chunk()` function
```python
# Fraud-specific data filtering from preprocess_functions_na.py
def processing_training_data_by_chunk(files_chunk, tag):
    # Load and filter fraud detection data
    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in files_chunk)
    train_data = pd.concat(df_list, ignore_index=True)
    
    # Fraud-specific credit card filtering
    train_data = train_data[train_data["creditCardIds"] != "9990-0012191-573601"]
    
    # Fraud tag validation and filtering
    train_data = train_data.loc[
        (~train_data[tag].isnull()) & (train_data[tag] != -1), :
    ]
    train_data[tag] = pd.to_numeric(train_data[tag], downcast="integer")
    
    # Fraud-specific time window filtering (240 days for training)
    train_data = train_data[train_data["orderDate"] > time.time() - 240 * 24 * 3600]
    
    # Fraud-specific holiday filtering to remove seasonal bias
    train_data = train_data[
        (train_data["orderDate"] >= 1701158400) | (train_data["orderDate"] < 1700726400)
    ]  # Christmas holiday filter
    train_data = train_data[
        (train_data["orderDate"] >= 1697094000) | (train_data["orderDate"] < 1696921200)
    ]  # Halloween holiday filter
    
    print(f"Fraud data filtering: {len(train_data)} transactions after filtering")
    return parallel_data_parsing(train_data, num_workers=80)
```
**Operations**:
- Blacklisted credit card filtering for known fraud patterns
- Fraud label validation with null and invalid value removal
- Fraud-optimized time window filtering (240 days for training data)
- Holiday period filtering to remove seasonal transaction bias
- Fraud-specific data quality validation and cleansing

#### **2. Fraud-Aware Positive Rate Targeting and Downsampling**
**TSA Implementation**: Fraud-specific downsampling strategy from `processing_training_data_by_chunk()`
```python
# Fraud-aware downsampling from processing_training_data_by_chunk()
def fraud_aware_downsampling(train_data, tag, target_positive_rate=0.2):
    # Calculate current fraud rate
    positive_count = len(train_data[train_data[tag] == 1])
    total_count = len(train_data)
    positive_rate = positive_count / total_count
    
    print(f"Original fraud rate: {positive_rate:.3f} ({positive_count}/{total_count})")
    
    # Fraud-specific downsampling to achieve target positive rate
    if positive_rate < target_positive_rate:
        # Calculate required negative samples for target fraud rate
        negative_downsampled_cnt = int(
            positive_count * (1 - target_positive_rate) / target_positive_rate
        )
        positive_downsampled_cnt = positive_count
        
        # Fraud-aware sampling preserving fraud patterns
        zeros_cond = train_data[tag] == 0
        ones_cond = train_data[tag] == 1
        
        train_data = pd.concat([
            train_data[zeros_cond].sample(negative_downsampled_cnt, random_state=42),
            train_data[ones_cond].sample(positive_downsampled_cnt, random_state=42),
        ], ignore_index=True)
        
        final_positive_rate = positive_downsampled_cnt / len(train_data)
        print(f"Fraud downsampling: achieved {final_positive_rate:.3f} fraud rate")
    
    return train_data
```
**Operations**:
- Fraud rate calculation and monitoring across chunks
- Target positive rate enforcement (20% fraud cases for training balance)
- Fraud pattern preservation during downsampling process
- Statistical fraud distribution maintenance across processing chunks

#### **3. Customer-Aware Fraud Sequence Chunking**
**TSA Implementation**: Customer sequence integrity preservation from distributed processing
```python
# Customer-aware fraud chunking from chunk_processing() coordination
def customer_aware_fraud_chunking(data_path, dataset, num_chunk):
    # Load fraud detection data with customer awareness
    files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # Pre-analyze customer distribution for balanced chunking
    customer_distribution = {}
    for file in files[:5]:  # Sample files for customer analysis
        df_sample = pd.read_csv(file, nrows=1000)
        for customer_id in df_sample['customerId'].unique():
            customer_distribution[customer_id] = customer_distribution.get(customer_id, 0) + 1
    
    # Customer-aware file distribution to preserve fraud sequences
    files_chunk_list = []
    customers_per_chunk = len(customer_distribution) // num_chunk
    
    print(f"Customer-aware chunking: {len(customer_distribution)} customers across {num_chunk} chunks")
    print(f"Target customers per chunk: {customers_per_chunk}")
    
    # Distribute files to maintain customer sequence integrity
    for i_chunk in range(num_chunk):
        chunk_files = []
        chunk_customers = 0
        
        for file in files:
            if chunk_customers < customers_per_chunk or i_chunk == num_chunk - 1:
                chunk_files.append(file)
                # Estimate customers in this file (simplified)
                chunk_customers += customers_per_chunk // len(files) * num_chunk
        
        files_chunk_list.append(chunk_files)
    
    return files_chunk_list
```
**Operations**:
- Customer distribution analysis for balanced fraud sequence chunking
- Customer sequence integrity preservation across distributed processing
- Fraud pattern continuity maintenance during chunk distribution
- Customer-aware load balancing for fraud detection processing

#### **4. Fraud-Specific Temporal Ordering Preservation**
**TSA Implementation**: Temporal fraud pattern preservation from sequence processing
```python
# Fraud temporal ordering preservation from data_parsing()
def preserve_fraud_temporal_patterns(input_data):
    # Fraud-specific temporal validation for customer sequences
    cid_temporal_validation = validate_fraud_sequence_ordering(
        input_data, "payment_risk.retail_order_cat_seq_by_cid.c_objectid_seq"
    )
    
    # Fraud-specific temporal validation for credit card sequences  
    ccid_temporal_validation = validate_fraud_sequence_ordering(
        input_data, "payment_risk.retail_order_cat_seq_by_ccid.c_objectid_seq"
    )
    
    # Cross-sequence fraud pattern temporal consistency
    if cid_temporal_validation and ccid_temporal_validation:
        # Validate fraud pattern temporal alignment
        cid_timestamps = extract_timestamps(input_data, "cid")
        ccid_timestamps = extract_timestamps(input_data, "ccid")
        
        # Fraud-specific time delta validation (115 days maximum)
        max_time_delta = 10000000  # 115 days in seconds
        
        for timestamps in [cid_timestamps, ccid_timestamps]:
            if len(timestamps) > 1:
                time_deltas = np.diff(timestamps)
                if np.any(time_deltas > max_time_delta):
                    print("Warning: Fraud sequence temporal gap exceeds threshold")
                    # Cap time deltas for fraud pattern consistency
                    timestamps[time_deltas > max_time_delta] = max_time_delta
        
        return True
    
    return False

def validate_fraud_sequence_ordering(input_data, sequence_key):
    # Fraud-specific sequence ordering validation
    if sequence_key in input_data and input_data[sequence_key] not in ["", "My Text String"]:
        sequence_data = input_data[sequence_key].split(SEP)
        
        # Validate fraud sequence temporal ordering
        if len(sequence_data) > 1:
            # Check for fraud-specific temporal consistency
            timestamps = [float(x) for x in sequence_data if x.replace('.', '').isdigit()]
            if len(timestamps) > 1:
                return all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        return True
    
    return False
```
**Operations**:
- Fraud-specific temporal sequence validation for customer and card sequences
- Cross-sequence fraud pattern temporal consistency enforcement
- Fraud-optimized time delta validation (115 days maximum)
- Temporal fraud pattern integrity preservation during distributed processing

#### **5. Fraud Detection Memory Optimization**
**TSA Implementation**: Fraud-specific memory management from distributed processing
```python
# Fraud-specific memory optimization from stream_save() and chunk coordination
def fraud_optimized_memory_management(fraud_data_chunks, dataset):
    # Fraud-specific memory-mapped storage with optimized data types
    fraud_categorical_chunks = []
    fraud_numerical_chunks = []
    fraud_labels_chunks = []
    
    for chunk in fraud_data_chunks:
        # Fraud categorical data optimization (int16 for fraud features)
        fraud_cat_chunk = chunk['categorical_features'].astype(np.int16)
        fraud_categorical_chunks.append(fraud_cat_chunk)
        
        # Fraud numerical data optimization (float32 for fraud scores)
        fraud_num_chunk = chunk['numerical_features'].astype(np.float32)
        fraud_numerical_chunks.append(fraud_num_chunk)
        
        # Fraud labels optimization (int8 for binary fraud labels)
        fraud_label_chunk = chunk['fraud_labels'].astype(np.int8)
        fraud_labels_chunks.append(fraud_label_chunk)
    
    # Memory-efficient fraud data storage
    fraud_memory_usage = {
        'categorical': sum(chunk.nbytes for chunk in fraud_categorical_chunks),
        'numerical': sum(chunk.nbytes for chunk in fraud_numerical_chunks),
        'labels': sum(chunk.nbytes for chunk in fraud_labels_chunks)
    }
    
    total_fraud_memory = sum(fraud_memory_usage.values())
    print(f"Fraud data memory optimization: {total_fraud_memory / (1024**3):.2f} GB")
    
    # Fraud-specific memory-mapped file creation
    stream_save(fraud_categorical_chunks, f"fraud_categorical_{dataset}", dataset)
    stream_save(fraud_numerical_chunks, f"fraud_numerical_{dataset}", dataset)
    stream_save(fraud_labels_chunks, f"fraud_labels_{dataset}", dataset)
    
    return fraud_memory_usage
```
**Operations**:
- Fraud-specific data type optimization (int16 for categorical, float32 for numerical, int8 for labels)
- Memory usage monitoring and reporting for fraud detection datasets
- Fraud-optimized memory-mapped file storage with specialized naming
- Memory efficiency tracking for fraud detection processing optimization

#### **6. Fraud-Specific Progress Tracking and Metrics**
**TSA Implementation**: Fraud detection progress monitoring from chunk processing
```python
# Fraud-specific progress tracking from chunk_processing()
def fraud_progress_tracking(dataset, i_chunk, num_chunk, processing_result):
    # Fraud-specific progress metrics
    fraud_metrics = {
        'chunk_id': i_chunk + 1,
        'total_chunks': num_chunk,
        'dataset_type': dataset
    }
    
    if processing_result and len(processing_result) > 0:
        # Extract fraud-specific metrics from processing result
        if hasattr(processing_result[0], 'shape'):
            fraud_metrics['customers_processed'] = processing_result[0].shape[0]
        
        if len(processing_result) > 4 and hasattr(processing_result[4], 'shape'):
            fraud_metrics['transactions_processed'] = processing_result[4].shape[0]
        
        if len(processing_result) > 5 and hasattr(processing_result[5], 'shape'):
            fraud_labels = processing_result[5]
            fraud_metrics['fraud_cases'] = np.sum(fraud_labels == 1)
            fraud_metrics['non_fraud_cases'] = np.sum(fraud_labels == 0)
            fraud_metrics['fraud_rate'] = fraud_metrics['fraud_cases'] / len(fraud_labels)
        
        # Fraud sequence validation metrics
        fraud_metrics['sequences_validated'] = fraud_metrics.get('customers_processed', 0)
        
        # Fraud processing efficiency metrics
        fraud_metrics['processing_efficiency'] = (
            fraud_metrics.get('transactions_processed', 0) / 
            max(fraud_metrics.get('customers_processed', 1), 1)
        )
    
    # Report fraud-specific progress
    print(f"Fraud Processing Progress - Chunk {fraud_metrics['chunk_id']}/{fraud_metrics['total_chunks']}")
    print(f"  Dataset: {fraud_metrics['dataset_type']}")
    print(f"  Customers: {fraud_metrics.get('customers_processed', 0)}")
    print(f"  Transactions: {fraud_metrics.get('transactions_processed', 0)}")
    print(f"  Fraud Rate: {fraud_metrics.get('fraud_rate', 0):.3f}")
    print(f"  Sequences Validated: {fraud_metrics.get('sequences_validated', 0)}")
    
    return fraud_metrics
```
**Operations**:
- Fraud-specific progress metrics tracking (customers, transactions, fraud rate)
- Fraud sequence validation progress monitoring
- Fraud processing efficiency measurement and reporting
- Real-time fraud detection progress visualization and logging

#### **7. Fraud Pattern Integrity Validation**
**TSA Implementation**: Fraud pattern consistency validation across distributed processing
```python
# Fraud pattern integrity validation from parallel_data_parsing()
def validate_fraud_pattern_integrity(processing_results):
    fraud_integrity_metrics = {
        'total_processed': 0,
        'fraud_patterns_preserved': 0,
        'customer_sequences_intact': 0,
        'card_sequences_intact': 0,
        'temporal_consistency_maintained': 0
    }
    
    for result in processing_results:
        if result and len(result) >= 7:
            (ret, seq_cat_mtx_cid, seq_num_mtx_cid, 
             seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, fraud_label) = result
            
            fraud_integrity_metrics['total_processed'] += 1
            
            if ret:
                # Validate customer sequence fraud pattern integrity
                if seq_cat_mtx_cid is not None and seq_num_mtx_cid is not None:
                    # Check fraud sequence temporal consistency
                    if validate_fraud_temporal_consistency(seq_num_mtx_cid):
                        fraud_integrity_metrics['customer_sequences_intact'] += 1
                
                # Validate credit card sequence fraud pattern integrity
                if seq_cat_mtx_ccid is not None and seq_num_mtx_ccid is not None:
                    # Check fraud sequence temporal consistency
                    if validate_fraud_temporal_consistency(seq_num_mtx_ccid):
                        fraud_integrity_metrics['card_sequences_intact'] += 1
                
                # Validate cross-sequence fraud pattern consistency
                if (seq_cat_mtx_cid is not None and seq_cat_mtx_ccid is not None):
                    if validate_cross_sequence_fraud_consistency(seq_cat_mtx_cid, seq_cat_mtx_ccid):
                        fraud_integrity_metrics['temporal_consistency_maintained'] += 1
                
                fraud_integrity_metrics['fraud_patterns_preserved'] += 1
    
    # Calculate fraud pattern integrity rates
    total = fraud_integrity_metrics['total_processed']
    if total > 0:
        integrity_rate = fraud_integrity_metrics['fraud_patterns_preserved'] / total
        customer_integrity_rate = fraud_integrity_metrics['customer_sequences_intact'] / total
        card_integrity_rate = fraud_integrity_metrics['card_sequences_intact'] / total
        
        print(f"Fraud Pattern Integrity Validation:")
        print(f"  Overall Integrity: {integrity_rate:.3f}")
        print(f"  Customer Sequences: {customer_integrity_rate:.3f}")
        print(f"  Card Sequences: {card_integrity_rate:.3f}")
    
    return fraud_integrity_metrics

def validate_fraud_temporal_consistency(sequence_matrix):
    # Fraud-specific temporal consistency validation
    if sequence_matrix is not None and len(sequence_matrix.shape) >= 2:
        # Check time delta column (last column - 1)
        time_deltas = sequence_matrix[:, -2]
        
        # Fraud-specific validation: no negative time deltas, max 115 days
        if np.any(time_deltas < 0) or np.any(time_deltas > 10000000):
            return False
        
        # Check for fraud sequence temporal ordering
        if len(time_deltas) > 1:
            return np.all(np.diff(time_deltas) >= 0)  # Non-decreasing order
    
    return True
```
**Operations**:
- Fraud pattern integrity validation across distributed processing results
- Customer and credit card sequence fraud pattern consistency checking
- Cross-sequence fraud pattern temporal consistency validation
- Fraud integrity rate calculation and reporting for quality assurance

#### **8. Fraud-Specific Chunk Coordination and Synchronization**
**TSA Implementation**: Fraud chunk coordination from distributed processing orchestration
```python
# Fraud-specific chunk coordination from chunk_processing()
def fraud_chunk_coordination(data_path, dataset, num_chunk):
    # Initialize fraud-specific chunk coordination
    fraud_chunk_metadata = {
        'dataset': dataset,
        'total_chunks': num_chunk,
        'fraud_chunks_processed': 0,
        'fraud_patterns_validated': 0,
        'customer_sequences_processed': 0,
        'card_sequences_processed': 0
    }
    
    # Fraud-specific chunk processing coordination
    fraud_processing_results = []
    
    for i_chunk in range(num_chunk):
        print(f"Fraud Chunk Coordination - Processing chunk {i_chunk+1}/{num_chunk}")
        
        # Process fraud chunk with specialized handling
        if dataset == "train":
            fraud_result = processing_training_data_by_chunk(files_chunk, "IS_FRD")
        elif dataset == "cali":
            fraud_result = processing_calibration_data_by_chunk(files_chunk, "IS_FRD")
        elif dataset == "vali":
            fraud_result = processing_validation_data_by_chunk(files_chunk, "IS_FRD")
        
        # Fraud-specific result validation and coordination
        if fraud_result:
            fraud_chunk_metadata['fraud_chunks_processed'] += 1
            
            # Validate fraud patterns in chunk result
            if validate_chunk_fraud_patterns(fraud_result):
                fraud_chunk_metadata['fraud_patterns_validated'] += 1
            
            # Count fraud sequences processed
            if len(fraud_result) >= 3:
                fraud_chunk_metadata['customer_sequences_processed'] += fraud_result[1].shape[0] if fraud_result[1] is not None else 0
                fraud_chunk_metadata['card_sequences_processed'] += fraud_result[3].shape[0] if fraud_result[3] is not None else 0
            
            fraud_processing_results.append(fraud_result)
        
        # Fraud chunk synchronization checkpoint
        if (i_chunk + 1) % 10 == 0:  # Every 10 chunks
            print(f"Fraud Synchronization Checkpoint - {i_chunk+1} chunks processed")
            print(f"  Fraud Patterns Validated: {fraud_chunk_metadata['fraud_patterns_validated']}")
            print(f"  Customer Sequences: {fraud_chunk_metadata['customer_sequences_processed']}")
            print(f"  Card Sequences: {fraud_chunk_metadata['card_sequences_processed']}")
    
    # Final fraud coordination summary
    fraud_success_rate = fraud_chunk_metadata['fraud_patterns_validated'] / fraud_chunk_metadata['fraud_chunks_processed']
    print(f"Fraud Chunk Coordination Complete:")
    print(f"  Success Rate: {fraud_success_rate:.3f}")
    print(f"  Total Customer Sequences: {fraud_chunk_metadata['customer_sequences_processed']}")
    print(f"  Total Card Sequences: {fraud_chunk_metadata['card_sequences_processed']}")
    
    return fraud_processing_results, fraud_chunk_metadata

def validate_chunk_fraud_patterns(chunk_result):
    # Validate fraud patterns in chunk processing result
    if chunk_result and len(chunk_result) >= 6:
        # Check for fraud label presence and validity
        fraud_labels = chunk_result[5] if len(chunk_result) > 5 else None
        if fraud_labels is not None:
            # Validate fraud label distribution
            fraud_rate = np.mean(fraud_labels)
            return 0.05 <= fraud_rate <= 0.5  # Reasonable fraud rate range
    
    return False
```
**Operations**:
- Fraud-specific chunk processing coordination with specialized metadata tracking
- Fraud pattern validation and synchronization across distributed chunks
- Customer and credit card sequence processing coordination
- Fraud chunk synchronization checkpoints with progress reporting
- Fraud processing success rate calculation and quality assurance

**Configuration Parameters**:
```python
{
    "extends": "DistributedProcessing",
    "fraud_chunking_strategy": {
        "chunking_method": "customer_aware",              # Customer-aware chunking for fraud sequences
        "preserve_customer_sequences": True,              # Maintain customer sequence integrity
        "temporal_ordering_preservation": True,           # Preserve fraud temporal patterns
        "fraud_pattern_integrity": True,                  # Validate fraud pattern consistency
        "customer_chunk_alignment": True                  # Align chunks by customer boundaries
    },
    "fraud_data_filtering": {
        "blacklisted_cards": ["9990-0012191-573601"],     # Known fraud card filtering
        "time_window_days": {
            "training": 240,                              # 240 days for training data
            "calibration": 150,                           # 150 days for calibration
            "validation": 90                              # 90 days for validation
        },
        "holiday_filters": [
            {"start": 1700726400, "end": 1701158400},     # Christmas holiday filter
            {"start": 1696921200, "end": 1697094000},     # Halloween holiday filter
            {"start": 1688972400, "end": 1689231600}      # July 4th holiday filter
        ],
        "fraud_label_validation": True                    # Validate fraud labels during filtering
    },
    "fraud_downsampling": {
        "target_positive_rate": 0.2,                     # 20% fraud rate for training balance
        "preserve_fraud_patterns": True,                  # Maintain fraud pattern distribution
        "random_seed": 42,                                # Reproducible fraud sampling
        "stratified_sampling": True                       # Stratified fraud sampling strategy
    },
    "fraud_memory_optimization": {
        "categorical_dtype": "int16",                     # Optimized fraud categorical features
        "numerical_dtype": "float32",                     # Optimized fraud numerical features
        "label_dtype": "int8",                            # Optimized fraud binary labels
        "memory_monitoring": True,                        # Monitor fraud data memory usage
        "specialized_naming": True                        # Fraud-specific file naming
    },
    "fraud_progress_metrics": {
        "customers_processed": True,                      # Track customer processing progress
        "transactions_processed": True,                   # Track transaction processing progress
        "sequences_validated": True,                      # Track sequence validation progress
        "fraud_rate_monitoring": True,                    # Monitor fraud rate across chunks
        "processing_efficiency": True,                    # Track fraud processing efficiency
        "real_time_reporting": True                       # Real-time fraud progress reporting
    },
    "fraud_integrity_validation": {
        "pattern_consistency_checks": True,               # Validate fraud pattern consistency
        "temporal_consistency_validation": True,          # Validate temporal fraud patterns
        "cross_sequence_validation": True,                # Validate cross-sequence fraud patterns
        "integrity_rate_thresholds": {
            "minimum_overall": 0.95,                      # Minimum overall integrity rate
            "minimum_customer": 0.90,                     # Minimum customer sequence integrity
            "minimum_card": 0.90                          # Minimum card sequence integrity
        }
    },
    "fraud_coordination": {
        "chunk_synchronization": True,                    # Enable fraud chunk synchronization
        "checkpoint_frequency": 10,                       # Synchronization checkpoint every 10 chunks
        "success_rate_monitoring": True,                  # Monitor fraud processing success rate
        "metadata_tracking": True,                        # Track fraud-specific metadata
        "quality_assurance": True                         # Enable fraud quality assurance checks
    }
}
```

**Script Location**: `src/cursus/steps/scripts/fraud_chunked_distributed_processing.py`

### 5A. **DomainFeatureEngineering Step** (Base Sharable)

**Purpose**: Handle general domain-specific feature engineering with configurable categorical encoding, temporal feature derivation, and behavioral pattern analysis for any domain-specific ML models

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (feature transformation functions in `data_parsing()`)
- `dockers/tsa/scripts/params_na.py` (feature definitions and mappings)
- `dockers/tsa/scripts/preprocess_train_na.py` (feature engineering orchestration)

**Sharable Features** (80% of functionality):

#### **1. Configurable Categorical Feature Encoding and Transformation**
**TSA Implementation**: Categorical transformation framework from `preprocess_functions_na.py`
```python
# Configurable categorical encoding from preprocess_functions_na.py
class CategoricalTransformer:
    def __init__(self, categorical_map, columns_list):
        self.categorical_map = categorical_map
        self.columns_list = columns_list
    
    def transform(self, seq_cat_mtx):
        # Apply configurable categorical transformations
        for i, column in enumerate(self.columns_list):
            if column in self.categorical_map:
                # Map categorical values using configurable mappings
                seq_cat_mtx[:, i] = [
                    self.categorical_map[column].get(val, 0) 
                    for val in seq_cat_mtx[:, i]
                ]
        return seq_cat_mtx

# Default value handling with configurable mappings from preprocess_functions_na.py
mtx_from_dict_fill_default = lambda input_data, var_list_otf, var_list, map_dict: np.array([
    [map_dict[var_list[i]] if a in ["", "My Text String"] else a 
     for a in input_data[var_list_otf[i]].split(SEP)]
    for i in range(len(var_list_otf))
]).transpose()

# Load configurable default value mappings
with open(os.path.join(config_path, "default_value_dict_na.json"), "r") as f:
    default_value_dict = json.load(f)
```
**Operations**:
- Configurable categorical mapping with domain-specific value transformations
- Default value handling with JSON-based configurable mappings
- Missing value imputation using domain-specific default strategies
- Categorical feature validation and consistency checking

#### **2. Generic Temporal Feature Engineering and Derivation**
**TSA Implementation**: Temporal feature processing from `sequence_data_parsing()` function
```python
# Temporal feature engineering from sequence_data_parsing()
def derive_temporal_features(seq_num_mtx, current_timestamp):
    # Time delta computation for temporal patterns
    seq_num_mtx[:, -2] = current_timestamp - seq_num_mtx[:, -2]
    
    # Temporal feature scaling and normalization
    seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(seq_num_min_)
    
    # Add temporal indicator features
    seq_num_mtx = np.concatenate([seq_num_mtx, np.ones((seq_num_mtx.shape[0], 1))], axis=1)
    
    return seq_num_mtx

# Configurable temporal aggregation windows from processing functions
def apply_temporal_windows(data, time_field, windows=[1, 7, 30, 90]):
    temporal_features = {}
    
    for window_days in windows:
        window_seconds = window_days * 24 * 3600
        window_data = data[data[time_field] > time.time() - window_seconds]
        
        # Derive window-specific temporal features
        temporal_features[f'count_{window_days}d'] = len(window_data)
        temporal_features[f'avg_amount_{window_days}d'] = window_data['amount'].mean()
        temporal_features[f'velocity_{window_days}d'] = len(window_data) / window_days
    
    return temporal_features
```
**Operations**:
- Configurable time delta computation for temporal pattern analysis
- Multi-window temporal aggregation (1, 7, 30, 90 day windows)
- Temporal scaling and normalization with configurable parameters
- Velocity and frequency feature derivation across time windows

#### **3. Behavioral Pattern Analysis and Profiling Framework**
**TSA Implementation**: Behavioral feature extraction from categorical and numerical sequences
```python
# Behavioral pattern analysis from sequence processing
def analyze_behavioral_patterns(seq_cat_mtx, seq_num_mtx, pattern_config):
    behavioral_features = {}
    
    # Categorical behavior pattern analysis
    for feature_idx, feature_name in enumerate(pattern_config['categorical_features']):
        feature_values = seq_cat_mtx[:, feature_idx]
        
        # Pattern consistency analysis
        behavioral_features[f'{feature_name}_consistency'] = calculate_consistency(feature_values)
        
        # Pattern change detection
        behavioral_features[f'{feature_name}_changes'] = count_pattern_changes(feature_values)
        
        # Pattern frequency analysis
        behavioral_features[f'{feature_name}_frequency'] = calculate_frequency_patterns(feature_values)
    
    # Numerical behavior pattern analysis
    for feature_idx, feature_name in enumerate(pattern_config['numerical_features']):
        feature_values = seq_num_mtx[:, feature_idx]
        
        # Statistical behavior patterns
        behavioral_features[f'{feature_name}_trend'] = calculate_trend(feature_values)
        behavioral_features[f'{feature_name}_volatility'] = calculate_volatility(feature_values)
        behavioral_features[f'{feature_name}_seasonality'] = detect_seasonality(feature_values)
    
    return behavioral_features

def calculate_consistency(values):
    # Measure behavioral consistency across sequence
    unique_values = len(set(values))
    total_values = len(values)
    return 1 - (unique_values / total_values)  # Higher = more consistent

def count_pattern_changes(values):
    # Count behavioral pattern changes
    changes = sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
    return changes / max(len(values) - 1, 1)

def calculate_trend(values):
    # Calculate numerical trend using linear regression slope
    if len(values) < 2:
        return 0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope
```
**Operations**:
- Configurable behavioral consistency measurement across sequences
- Pattern change detection and quantification
- Statistical trend analysis for numerical behavioral features
- Seasonality detection and behavioral volatility measurement

#### **4. Generic Risk Scoring and Ranking Framework**
**TSA Implementation**: Risk-based feature transformation and scoring
```python
# Generic risk scoring framework from categorical transformation
def calculate_risk_scores(feature_values, target_values, scoring_method='woe'):
    risk_scores = {}
    
    if scoring_method == 'woe':  # Weight of Evidence
        for unique_val in set(feature_values):
            mask = feature_values == unique_val
            positive_rate = np.mean(target_values[mask])
            negative_rate = 1 - positive_rate
            
            # Calculate Weight of Evidence
            if positive_rate > 0 and negative_rate > 0:
                woe = np.log(positive_rate / negative_rate)
                risk_scores[unique_val] = woe
            else:
                risk_scores[unique_val] = 0
    
    elif scoring_method == 'target_rate':  # Target Rate Encoding
        for unique_val in set(feature_values):
            mask = feature_values == unique_val
            risk_scores[unique_val] = np.mean(target_values[mask])
    
    elif scoring_method == 'frequency':  # Frequency-based Risk
        value_counts = pd.Series(feature_values).value_counts()
        total_count = len(feature_values)
        
        for unique_val in set(feature_values):
            frequency = value_counts[unique_val] / total_count
            risk_scores[unique_val] = 1 - frequency  # Rare = higher risk
    
    return risk_scores

# Risk ranking and percentile calculation
def calculate_risk_percentiles(risk_scores, percentile_bins=100):
    score_values = list(risk_scores.values())
    percentiles = np.percentile(score_values, np.linspace(0, 100, percentile_bins))
    
    risk_percentile_map = {}
    for value, score in risk_scores.items():
        percentile_rank = np.searchsorted(percentiles, score) / len(percentiles)
        risk_percentile_map[value] = percentile_rank
    
    return risk_percentile_map
```
**Operations**:
- Configurable risk scoring methods (Weight of Evidence, Target Rate, Frequency-based)
- Risk percentile ranking and distribution analysis
- Statistical risk assessment with multiple scoring strategies
- Risk score normalization and standardization

#### **5. Feature Interaction Detection and Engineering**
**TSA Implementation**: Cross-feature interaction analysis from multi-feature processing
```python
# Feature interaction detection and engineering
def detect_feature_interactions(categorical_features, numerical_features, interaction_config):
    interaction_features = {}
    
    # Categorical-Categorical interactions
    for cat1_idx, cat1_name in enumerate(interaction_config['categorical_pairs']):
        for cat2_idx, cat2_name in enumerate(interaction_config['categorical_pairs']):
            if cat1_idx < cat2_idx:  # Avoid duplicate pairs
                # Create interaction feature
                interaction_key = f'{cat1_name}_{cat2_name}_interaction'
                cat1_values = categorical_features[:, cat1_idx]
                cat2_values = categorical_features[:, cat2_idx]
                
                # Combine categorical values for interaction
                interaction_values = [f'{v1}_{v2}' for v1, v2 in zip(cat1_values, cat2_values)]
                interaction_features[interaction_key] = interaction_values
    
    # Categorical-Numerical interactions
    for cat_idx, cat_name in enumerate(interaction_config['categorical_features']):
        for num_idx, num_name in enumerate(interaction_config['numerical_features']):
            interaction_key = f'{cat_name}_{num_name}_interaction'
            cat_values = categorical_features[:, cat_idx]
            num_values = numerical_features[:, num_idx]
            
            # Statistical interaction analysis
            interaction_stats = {}
            for unique_cat in set(cat_values):
                mask = cat_values == unique_cat
                if np.sum(mask) > 0:
                    interaction_stats[unique_cat] = {
                        'mean': np.mean(num_values[mask]),
                        'std': np.std(num_values[mask]),
                        'count': np.sum(mask)
                    }
            
            interaction_features[interaction_key] = interaction_stats
    
    # Numerical-Numerical interactions
    for num1_idx, num1_name in enumerate(interaction_config['numerical_pairs']):
        for num2_idx, num2_name in enumerate(interaction_config['numerical_pairs']):
            if num1_idx < num2_idx:
                interaction_key = f'{num1_name}_{num2_name}_interaction'
                num1_values = numerical_features[:, num1_idx]
                num2_values = numerical_features[:, num2_idx]
                
                # Calculate numerical interactions
                interaction_features[f'{interaction_key}_ratio'] = num1_values / (num2_values + 1e-8)
                interaction_features[f'{interaction_key}_product'] = num1_values * num2_values
                interaction_features[f'{interaction_key}_difference'] = num1_values - num2_values
    
    return interaction_features
```
**Operations**:
- Categorical-categorical feature interaction detection and combination
- Categorical-numerical statistical interaction analysis
- Numerical-numerical interaction engineering (ratios, products, differences)
- Configurable interaction pair selection and feature generation

#### **6. Automated Feature Selection and Importance Analysis**
**TSA Implementation**: Feature importance and selection framework
```python
# Automated feature selection framework
def automated_feature_selection(features, target, selection_config):
    selected_features = {}
    feature_importance_scores = {}
    
    # Correlation-based feature selection
    if 'correlation' in selection_config['methods']:
        correlation_threshold = selection_config.get('correlation_threshold', 0.1)
        
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, (list, np.ndarray)):
                correlation = np.corrcoef(feature_values, target)[0, 1]
                if abs(correlation) > correlation_threshold:
                    selected_features[feature_name] = feature_values
                    feature_importance_scores[feature_name] = abs(correlation)
    
    # Statistical significance-based selection
    if 'statistical' in selection_config['methods']:
        from scipy.stats import chi2_contingency, f_oneway
        significance_threshold = selection_config.get('significance_threshold', 0.05)
        
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, (list, np.ndarray)):
                # For categorical features, use chi-square test
                if len(set(feature_values)) < 20:  # Assume categorical if few unique values
                    contingency_table = pd.crosstab(feature_values, target)
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    if p_value < significance_threshold:
                        selected_features[feature_name] = feature_values
                        feature_importance_scores[feature_name] = chi2
                
                # For numerical features, use ANOVA F-test
                else:
                    unique_targets = list(set(target))
                    groups = [feature_values[target == t] for t in unique_targets]
                    f_stat, p_value = f_oneway(*groups)
                    
                    if p_value < significance_threshold:
                        selected_features[feature_name] = feature_values
                        feature_importance_scores[feature_name] = f_stat
    
    # Mutual information-based selection
    if 'mutual_info' in selection_config['methods']:
        from sklearn.feature_selection import mutual_info_classif
        mi_threshold = selection_config.get('mutual_info_threshold', 0.01)
        
        feature_matrix = np.column_stack(list(features.values()))
        mi_scores = mutual_info_classif(feature_matrix, target)
        
        for i, (feature_name, _) in enumerate(features.items()):
            if mi_scores[i] > mi_threshold:
                selected_features[feature_name] = list(features.values())[i]
                feature_importance_scores[feature_name] = mi_scores[i]
    
    return selected_features, feature_importance_scores
```
**Operations**:
- Correlation-based feature selection with configurable thresholds
- Statistical significance testing (Chi-square for categorical, ANOVA for numerical)
- Mutual information-based feature importance calculation
- Multi-method feature selection with importance scoring

#### **7. Configurable Feature Engineering Pipeline Orchestration**
**TSA Implementation**: Feature engineering pipeline coordination from `data_parsing()` orchestration
```python
# Feature engineering pipeline orchestration
def orchestrate_feature_engineering(input_data, engineering_config):
    engineered_features = {}
    
    # Stage 1: Categorical feature engineering
    if 'categorical_encoding' in engineering_config:
        categorical_features = extract_categorical_features(input_data, engineering_config['categorical_features'])
        encoded_features = apply_categorical_encoding(categorical_features, engineering_config['categorical_encoding'])
        engineered_features.update(encoded_features)
    
    # Stage 2: Temporal feature engineering
    if 'temporal_features' in engineering_config:
        temporal_features = derive_temporal_features(input_data, engineering_config['temporal_features'])
        engineered_features.update(temporal_features)
    
    # Stage 3: Behavioral pattern analysis
    if 'behavioral_analysis' in engineering_config:
        behavioral_features = analyze_behavioral_patterns(input_data, engineering_config['behavioral_analysis'])
        engineered_features.update(behavioral_features)
    
    # Stage 4: Risk scoring
    if 'risk_scoring' in engineering_config:
        risk_features = calculate_domain_risk_scores(input_data, engineering_config['risk_scoring'])
        engineered_features.update(risk_features)
    
    # Stage 5: Feature interactions
    if 'feature_interactions' in engineering_config:
        interaction_features = detect_feature_interactions(engineered_features, engineering_config['feature_interactions'])
        engineered_features.update(interaction_features)
    
    # Stage 6: Feature selection
    if 'feature_selection' in engineering_config:
        target = input_data.get(engineering_config['target_variable'])
        selected_features, importance_scores = automated_feature_selection(
            engineered_features, target, engineering_config['feature_selection']
        )
        engineered_features = selected_features
    
    # Stage 7: Feature validation and quality control
    validated_features = validate_engineered_features(engineered_features, engineering_config.get('validation', {}))
    
    return validated_features

def validate_engineered_features(features, validation_config):
    validated_features = {}
    
    for feature_name, feature_values in features.items():
        # Check for missing values
        if isinstance(feature_values, (list, np.ndarray)):
            missing_rate = np.sum(pd.isna(feature_values)) / len(feature_values)
            max_missing_rate = validation_config.get('max_missing_rate', 0.5)
            
            if missing_rate <= max_missing_rate:
                # Check for feature variance
                if len(set(feature_values)) > 1:  # Has variance
                    validated_features[feature_name] = feature_values
                else:
                    print(f"Warning: Feature {feature_name} has no variance, excluding")
            else:
                print(f"Warning: Feature {feature_name} has {missing_rate:.2f} missing rate, excluding")
    
    return validated_features
```
**Operations**:
- Multi-stage feature engineering pipeline with configurable stages
- Sequential feature engineering with dependency management
- Feature validation and quality control with configurable thresholds
- Comprehensive feature engineering orchestration with error handling

#### **8. Domain-Agnostic Feature Engineering Utilities**
**TSA Implementation**: Utility functions for general feature engineering from `preprocess_functions_na.py`
```python
# Domain-agnostic feature engineering utilities
def normalize_features(features, normalization_config):
    normalized_features = {}
    
    for feature_name, feature_values in features.items():
        if isinstance(feature_values, (list, np.ndarray)) and len(set(feature_values)) > 1:
            normalization_method = normalization_config.get('method', 'min_max')
            
            if normalization_method == 'min_max':
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                normalized_values = (feature_values - min_val) / (max_val - min_val + 1e-8)
            
            elif normalization_method == 'z_score':
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                normalized_values = (feature_values - mean_val) / (std_val + 1e-8)
            
            elif normalization_method == 'robust':
                median_val = np.median(feature_values)
                mad_val = np.median(np.abs(feature_values - median_val))
                normalized_values = (feature_values - median_val) / (mad_val + 1e-8)
            
            else:
                normalized_values = feature_values  # No normalization
            
            normalized_features[feature_name] = normalized_values
        else:
            normalized_features[feature_name] = feature_values
    
    return normalized_features

def handle_missing_values(features, missing_config):
    imputed_features = {}
    
    for feature_name, feature_values in features.items():
        if isinstance(feature_values, (list, np.ndarray)):
            missing_mask = pd.isna(feature_values)
            
            if np.any(missing_mask):
                imputation_method = missing_config.get('method', 'median')
                
                if imputation_method == 'median':
                    fill_value = np.nanmedian(feature_values)
                elif imputation_method == 'mean':
                    fill_value = np.nanmean(feature_values)
                elif imputation_method == 'mode':
                    fill_value = pd.Series(feature_values).mode().iloc[0]
                elif imputation_method == 'forward_fill':
                    fill_value = pd.Series(feature_values).fillna(method='ffill')
                else:
                    fill_value = missing_config.get('default_value', 0)
                
                imputed_values = np.where(missing_mask, fill_value, feature_values)
                imputed_features[feature_name] = imputed_values
            else:
                imputed_features[feature_name] = feature_values
        else:
            imputed_features[feature_name] = feature_values
    
    return imputed_features

def create_feature_metadata(features, metadata_config):
    feature_metadata = {}
    
    for feature_name, feature_values in features.items():
        if isinstance(feature_values, (list, np.ndarray)):
            metadata = {
                'feature_type': 'categorical' if len(set(feature_values)) < 20 else 'numerical',
                'unique_values': len(set(feature_values)),
                'missing_rate': np.sum(pd.isna(feature_values)) / len(feature_values),
                'data_type': str(type(feature_values[0]).__name__),
                'min_value': np.nanmin(feature_values) if len(set(feature_values)) >= 20 else None,
                'max_value': np.nanmax(feature_values) if len(set(feature_values)) >= 20 else None,
                'mean_value': np.nanmean(feature_values) if len(set(feature_values)) >= 20 else None,
                'std_value': np.nanstd(feature_values) if len(set(feature_values)) >= 20 else None
            }
            feature_metadata[feature_name] = metadata
    
    return feature_metadata
```
**Operations**:
- Configurable feature normalization (min-max, z-score, robust scaling)
- Multiple missing value imputation strategies (median, mean, mode, forward-fill)
- Automated feature metadata generation and documentation
- Domain-agnostic utility functions for general feature engineering tasks

**Configuration Parameters**:
```python
{
    "categorical_encoding": {
        "methods": ["target_encoding", "frequency_encoding", "woe_encoding", "risk_encoding"],
        "default_value_strategy": "configurable_dict",           # JSON-based default mappings from TSA
        "missing_value_handling": ["default_fill", "mode_fill", "frequency_fill"],
        "encoding_validation": True                              # Validate encoding consistency
    },
    "temporal_features": {
        "time_windows": [1, 7, 30, 90, 240],                   # Configurable windows from TSA (days)
        "temporal_aggregations": ["count", "sum", "mean", "std", "min", "max"],
        "velocity_features": ["transaction_velocity", "amount_velocity", "frequency_velocity"],
        "seasonality_detection": ["daily", "weekly", "monthly", "yearly"],
        "time_delta_features": True                              # Enable time delta computation from TSA
    },
    "behavioral_analysis": {
        "pattern_consistency": True,                             # Behavioral consistency measurement
        "pattern_changes": True,                                 # Pattern change detection
        "trend_analysis": True,                                  # Statistical trend analysis
        "volatility_measurement": True,                          # Behavioral volatility calculation
        "seasonality_detection": True,                           # Seasonal pattern detection
        "anomaly_detection": ["statistical", "isolation_forest", "local_outlier_factor"]
    },
    "risk_scoring": {
        "scoring_methods": ["woe", "target_rate", "frequency", "percentile"],
        "percentile_bins": 100,                                  # Risk percentile resolution
        "risk_normalization": True,                              # Normalize risk scores
        "risk_validation": True                                  # Validate risk score consistency
    },
    "feature_interactions": {
        "categorical_pairs": "auto_detect",                      # Automatic categorical pair detection
        "numerical_pairs": "auto_detect",                        # Automatic numerical pair detection
        "cross_type_interactions": True,                         # Categorical-numerical interactions
        "interaction_methods": ["combination", "statistical", "ratio", "product", "difference"],
        "max_interactions": 1000                                 # Limit interaction feature count
    },
    "feature_selection": {
        "methods": ["correlation", "statistical", "mutual_info", "importance"],
        "correlation_threshold": 0.1,                            # Minimum correlation for selection
        "significance_threshold": 0.05,                          # Statistical significance threshold
        "mutual_info_threshold": 0.01,                           # Mutual information threshold
        "max_features": 500                                      # Maximum selected features
    },
    "pipeline_orchestration": {
        "stages": ["categorical_encoding", "temporal_features", "behavioral_analysis", "risk_scoring", "feature_interactions", "feature_selection"],
        "stage_dependencies": True,                              # Enforce stage dependencies
        "parallel_processing": True,                             # Enable parallel feature engineering
        "error_handling": "graceful_degradation"                # Continue on individual feature failures
    },
    "feature_validation": {
        "max_missing_rate": 0.5,                                # Maximum allowed missing rate
        "min_variance_threshold": 1e-8,                         # Minimum feature variance
        "data_type_validation": True,                            # Validate feature data types
        "range_validation": True,                                # Validate feature value ranges
        "quality_control": True                                  # Enable feature quality control
    },
    "utilities": {
        "normalization_methods": ["min_max", "z_score", "robust", "none"],
        "missing_value_methods": ["median", "mean", "mode", "forward_fill", "default_value"],
        "metadata_generation": True,                             # Generate feature metadata
        "feature_documentation": True                            # Document engineered features
    }
}
```

**Script Location**: `src/cursus/steps/scripts/domain_feature_engineering.py`

### 5B. **FraudFeatureEngineering Step** (Domain-Specific Extension)

**Purpose**: Fraud detection specific feature engineering extending base functionality with payment fraud detection, customer behavior analysis, and risk-based categorical encoding

**Primary TSA Scripts**: 
- `dockers/tsa/scripts/preprocess_functions_na.py` (fraud-specific feature transformations in `data_parsing()`)
- `dockers/tsa/scripts/params_na.py` (fraud feature definitions: 109 categorical + 67 numerical + 297 engineered features)
- `dockers/tsa/scripts/preprocess_train_na.py` (fraud feature engineering orchestration)

**Domain-Specific Features** (20% of functionality):

#### **1. Fraud-Optimized Categorical Risk Encoding**
**TSA Implementation**: Fraud-specific categorical encoding from `preprocess_functions_na.py` and `params_na.py`
```python
# Fraud-specific categorical features from params_na.py (109 total fraud categorical features)
# Payment fraud risk indicators
input_data_seq_cat_vars = [
    "c_cciscorporate_seq",           # Corporate card fraud risk indicator
    "c_ccisdebit_seq",               # Debit card fraud patterns
    "c_ccisprepaid_seq",             # Prepaid card fraud risk (high risk)
    "c_ccissuer_seq",                # Card issuer fraud patterns
    "c_creditcardhit_seq",           # Credit card blacklist hits
    
    # Geographic fraud indicators
    "c_geobillcountrycccountrycodeequal_seq",     # Bill/CC country mismatch (fraud indicator)
    "c_geoipcountrycodecccountrycodeequal_seq",   # IP/CC country mismatch (fraud indicator)
    "c_georeportedipmktplcountrycodeequal_seq",   # IP/Marketplace country mismatch
    
    # Behavioral fraud change indicators
    "c_fingerprintchanged_seq",      # Device fingerprint changes (account takeover)
    "c_emailchanged_seq",            # Email address changes (account takeover)
    "c_ipchanged_seq",               # IP address changes (location fraud)
    "c_paymentchg_seq",              # Payment method changes (fraud pattern)
]

# Fraud-specific categorical transformation from preprocess_functions_na.py
def fraud_categorical_encoding(input_data, fraud_categorical_map):
    # Special handling for fraud risk categorical variables
    numerical_cat_vars_indices = [3,4,5,6,8,9,10,13,14,16,17,18,19,20,21,22,24,27,28,29,30,32,33,34,35,36,37,39,40]
    
    for i in numerical_cat_vars_indices:
        cur_var = input_data[input_data_seq_cat_vars[i]]
        if cur_var not in ["", "My Text String", "false"]:
            # Fraud-specific numerical categorical encoding
            if i == 38:  # fingerprintRiskValue (special fraud feature)
                if float(cur_var) == 0:
                    cur_var = str(int(float(cur_var)))  # Zero risk encoding
                else:
                    cur_var = str(float(cur_var))       # Non-zero risk preservation
            else:
                cur_var = str(int(float(cur_var)))      # Standard numerical categorical encoding
            
            input_data[input_data_seq_cat_vars[i]] = cur_var
    
    # Apply fraud-specific categorical mappings
    fraud_encoded_features = {}
    for feature_name, feature_value in input_data.items():
        if feature_name in fraud_categorical_map:
            # Risk-based encoding for fraud detection
            fraud_encoded_features[feature_name] = fraud_categorical_map[feature_name].get(feature_value, 0)
    
    return fraud_encoded_features
```
**Operations**:
- Payment instrument fraud risk categorization (corporate, debit, prepaid cards)
- Geographic fraud pattern encoding through country code mismatches
- Behavioral change detection encoding for account takeover patterns
- Special handling for fraud risk scores (fingerprintRiskValue) with zero/non-zero distinction

#### **2. Fraud-Specific Temporal Velocity Features**
**TSA Implementation**: Fraud velocity features from `params_na.py` numerical features
```python
# Fraud velocity and temporal features from params_na.py (67 numerical features)
input_data_seq_num_vars = [
    "c_days_lastorder_seq",                    # Transaction velocity (days since last order)
    "c_ordertotalamountusd_seq",              # Spending velocity (transaction amounts)
    "c_ccage_seq",                            # Credit card age (new card fraud risk)
    "c_cccount_seq",                          # Credit card count (multiple card fraud)
    
    # Fraud-specific temporal patterns
    "c_aveamt_fpage_seq",                     # Average amount first page (browsing behavior)
    "c_aveamt_ipage_seq",                     # Average amount item page (browsing behavior)
    "c_numorders_seq",                        # Number of orders (velocity indicator)
    "c_numreturns_seq",                       # Number of returns (fraud pattern)
]

# Fraud velocity calculation from temporal processing
def calculate_fraud_velocity_features(sequence_data, current_timestamp):
    fraud_velocity_features = {}
    
    # Transaction velocity analysis
    if len(sequence_data) > 1:
        time_deltas = np.diff([row[-2] for row in sequence_data])  # Time between transactions
        
        # Fraud velocity indicators
        fraud_velocity_features['avg_transaction_interval'] = np.mean(time_deltas)
        fraud_velocity_features['min_transaction_interval'] = np.min(time_deltas)
        fraud_velocity_features['velocity_spike_count'] = np.sum(time_deltas < 3600)  # Transactions within 1 hour
        
        # Spending velocity analysis
        amounts = [row[0] for row in sequence_data if row[0] > 0]  # ordertotalamountusd
        if len(amounts) > 1:
            amount_changes = np.diff(amounts)
            fraud_velocity_features['spending_acceleration'] = np.mean(amount_changes)
            fraud_velocity_features['spending_volatility'] = np.std(amounts)
            fraud_velocity_features['large_amount_spike_count'] = np.sum(amounts > np.mean(amounts) * 3)
    
    # Credit card age fraud risk (new cards are higher risk)
    cc_ages = [row[2] for row in sequence_data if row[2] >= 0]  # ccage
    if cc_ages:
        fraud_velocity_features['avg_cc_age'] = np.mean(cc_ages)
        fraud_velocity_features['new_card_usage_rate'] = np.sum(np.array(cc_ages) < 30) / len(cc_ages)  # Cards < 30 days
    
    return fraud_velocity_features
```
**Operations**:
- Transaction velocity analysis with fraud-specific thresholds (1-hour spike detection)
- Spending velocity and acceleration measurement for fraud detection
- Credit card age analysis for new card fraud risk assessment
- Velocity spike counting for rapid transaction fraud patterns

#### **3. Payment Fraud Pattern Analysis**
**TSA Implementation**: Payment fraud indicators from categorical and numerical features
```python
# Payment fraud pattern analysis from TSA feature definitions
def analyze_payment_fraud_patterns(categorical_features, numerical_features):
    payment_fraud_patterns = {}
    
    # Credit card type fraud risk analysis
    cc_type_features = {
        'corporate_card_usage': categorical_features.get('c_cciscorporate_seq', 0),
        'debit_card_usage': categorical_features.get('c_ccisdebit_seq', 0),
        'prepaid_card_usage': categorical_features.get('c_ccisprepaid_seq', 0),  # High fraud risk
    }
    
    # Calculate payment method fraud risk score
    payment_fraud_patterns['payment_method_risk'] = (
        cc_type_features['prepaid_card_usage'] * 3 +      # Prepaid cards highest risk
        cc_type_features['corporate_card_usage'] * 1.5 +  # Corporate cards medium risk
        cc_type_features['debit_card_usage'] * 1.2        # Debit cards slightly higher risk
    )
    
    # Card issuer fraud pattern analysis
    issuer_risk = categorical_features.get('c_ccissuer_seq', 0)
    payment_fraud_patterns['issuer_fraud_risk'] = issuer_risk
    
    # Credit card blacklist analysis
    cc_blacklist_hit = categorical_features.get('c_creditcardhit_seq', 0)
    payment_fraud_patterns['blacklist_hit_indicator'] = cc_blacklist_hit
    
    # Payment method switching analysis (fraud indicator)
    payment_changes = categorical_features.get('c_paymentchg_seq', 0)
    payment_fraud_patterns['payment_switching_risk'] = payment_changes
    
    # Multiple credit card usage analysis
    cc_count = numerical_features.get('c_cccount_seq', 1)
    payment_fraud_patterns['multiple_card_risk'] = min(cc_count / 5.0, 1.0)  # Normalize to 0-1
    
    return payment_fraud_patterns

# Payment fraud network analysis from Tugboat features
def analyze_payment_fraud_network(numerical_features):
    network_fraud_features = {}
    
    # Tugboat fraud network scoring (from TSA params_na.py)
    tugboat_customer_score = numerical_features.get('c_tugboat_ev2customer_evwmaxlinkscorecust_seq', 0)
    tugboat_transaction_score = numerical_features.get('c_tugboat_ev2customer_evwmaxlinkscoretrx_seq', 0)
    
    # Network-based fraud risk calculation
    network_fraud_features['customer_network_risk'] = tugboat_customer_score
    network_fraud_features['transaction_network_risk'] = tugboat_transaction_score
    network_fraud_features['combined_network_risk'] = (tugboat_customer_score + tugboat_transaction_score) / 2
    
    # Network fraud pattern indicators
    if tugboat_customer_score > 0.7 or tugboat_transaction_score > 0.7:
        network_fraud_features['high_network_risk_flag'] = 1
    else:
        network_fraud_features['high_network_risk_flag'] = 0
    
    return network_fraud_features
```
**Operations**:
- Credit card type fraud risk scoring (prepaid=3x, corporate=1.5x, debit=1.2x risk multipliers)
- Card issuer fraud pattern analysis and risk assessment
- Credit card blacklist hit detection and flagging
- Payment method switching pattern analysis for fraud detection
- Tugboat fraud network scoring for customer and transaction risk assessment

#### **4. Customer Fraud Behavior Profiling**
**TSA Implementation**: Customer behavioral fraud indicators from categorical features
```python
# Customer fraud behavior profiling from TSA categorical features
def profile_customer_fraud_behavior(categorical_features, numerical_features):
    customer_fraud_profile = {}
    
    # Account takeover indicators
    behavioral_changes = {
        'fingerprint_changed': categorical_features.get('c_fingerprintchanged_seq', 0),
        'email_changed': categorical_features.get('c_emailchanged_seq', 0),
        'ip_changed': categorical_features.get('c_ipchanged_seq', 0),
        'payment_changed': categorical_features.get('c_paymentchg_seq', 0),
    }
    
    # Calculate account takeover risk score
    takeover_risk_score = sum(behavioral_changes.values())
    customer_fraud_profile['account_takeover_risk'] = min(takeover_risk_score / 4.0, 1.0)  # Normalize to 0-1
    
    # Customer loyalty and fraud correlation
    prime_membership = categorical_features.get('c_isprimemember_seq', 0)
    customer_fraud_profile['prime_member_protection'] = prime_membership  # Prime members lower fraud risk
    
    # Address consistency fraud indicators
    address_consistency = {
        'same_billing_shipping_zip': categorical_features.get('c_same_bs_zip_seq', 0),
        'same_billing_shipping_state': categorical_features.get('c_same_bs_state_seq', 0),
    }
    
    address_consistency_score = sum(address_consistency.values()) / len(address_consistency)
    customer_fraud_profile['address_consistency_score'] = address_consistency_score
    
    # Customer strangeness scoring (synthetic identity detection)
    strangeness_scores = {
        'billing_address_strangeness': numerical_features.get('c_billaddrstrangeness_seq', 0),
        'shipping_address_strangeness': numerical_features.get('c_shipaddrstrangeness_seq', 0),
        'email_name_strangeness': numerical_features.get('c_emailnamestrangeness_seq', 0),
    }
    
    avg_strangeness = sum(strangeness_scores.values()) / len(strangeness_scores)
    customer_fraud_profile['synthetic_identity_risk'] = avg_strangeness
    
    # Device fingerprint fraud analysis
    fingerprint_risk = numerical_features.get('c_fingerprintriskvalue_seq', 0)
    customer_fraud_profile['device_fraud_risk'] = fingerprint_risk
    
    return customer_fraud_profile

# Customer transaction pattern analysis
def analyze_customer_transaction_patterns(numerical_features):
    transaction_patterns = {}
    
    # Transaction frequency analysis
    num_orders = numerical_features.get('c_numorders_seq', 0)
    num_returns = numerical_features.get('c_numreturns_seq', 0)
    
    # Return rate fraud indicator (high returns can indicate fraud)
    if num_orders > 0:
        return_rate = num_returns / num_orders
        transaction_patterns['return_rate_fraud_indicator'] = min(return_rate * 2, 1.0)  # Cap at 1.0
    else:
        transaction_patterns['return_rate_fraud_indicator'] = 0
    
    # Order frequency fraud analysis
    transaction_patterns['order_frequency_risk'] = min(num_orders / 100.0, 1.0)  # Normalize high frequency
    
    return transaction_patterns
```
**Operations**:
- Account takeover risk scoring based on behavioral changes (fingerprint, email, IP, payment)
- Customer loyalty analysis (Prime membership as fraud protection indicator)
- Address consistency scoring for synthetic identity detection
- Strangeness scoring for billing, shipping, and email name anomalies
- Device fingerprint fraud risk assessment and scoring
- Transaction pattern analysis including return rate fraud indicators

#### **5. Geographic Cross-Border Fraud Detection**
**TSA Implementation**: Geographic fraud indicators from categorical features
```python
# Geographic cross-border fraud detection from TSA categorical features
def detect_geographic_fraud_patterns(categorical_features):
    geographic_fraud_features = {}
    
    # Country code mismatch analysis (major fraud indicators)
    country_mismatches = {
        'billing_cc_country_mismatch': 1 - categorical_features.get('c_geobillcountrycccountrycodeequal_seq', 1),
        'ip_cc_country_mismatch': 1 - categorical_features.get('c_geoipcountrycodecccountrycodeequal_seq', 1),
        'ip_marketplace_country_mismatch': 1 - categorical_features.get('c_georeportedipmktplcountrycodeequal_seq', 1),
    }
    
    # Calculate geographic fraud risk score
    geographic_risk_score = sum(country_mismatches.values())
    geographic_fraud_features['geographic_fraud_risk'] = min(geographic_risk_score / 3.0, 1.0)
    
    # Cross-border transaction indicators
    marketplace_cc_mismatch = 1 - categorical_features.get('c_marketplacecountrycode_cctry_cd_is_match_seq', 1)
    geographic_fraud_features['cross_border_transaction_risk'] = marketplace_cc_mismatch
    
    # High-risk geographic combination detection
    if (country_mismatches['billing_cc_country_mismatch'] and 
        country_mismatches['ip_cc_country_mismatch']):
        geographic_fraud_features['high_geographic_risk_flag'] = 1
    else:
        geographic_fraud_features['high_geographic_risk_flag'] = 0
    
    # Geographic consistency score
    consistency_indicators = [
        categorical_features.get('c_geobillcountrycccountrycodeequal_seq', 1),
        categorical_features.get('c_geoipcountrycodecccountrycodeequal_seq', 1),
        categorical_features.get('c_georeportedipmktplcountrycodeequal_seq', 1),
        categorical_features.get('c_marketplacecountrycode_cctry_cd_is_match_seq', 1),
    ]
    
    geographic_fraud_features['geographic_consistency_score'] = sum(consistency_indicators) / len(consistency_indicators)
    
    return geographic_fraud_features
```
**Operations**:
- Country code mismatch detection (billing/CC, IP/CC, IP/marketplace mismatches)
- Cross-border transaction risk assessment and scoring
- High-risk geographic combination flagging for enhanced fraud detection
- Geographic consistency scoring across multiple location indicators

#### **6. Fraud-Specific Feature Interaction Engineering**
**TSA Implementation**: Cross-feature fraud pattern detection
```python
# Fraud-specific feature interactions from TSA multi-feature analysis
def engineer_fraud_feature_interactions(categorical_features, numerical_features):
    fraud_interactions = {}
    
    # Payment method + geographic fraud interaction
    payment_risk = categorical_features.get('c_ccisprepaid_seq', 0)  # Prepaid card risk
    geographic_risk = 1 - categorical_features.get('c_geobillcountrycccountrycodeequal_seq', 1)  # Country mismatch
    fraud_interactions['payment_geographic_interaction'] = payment_risk * geographic_risk
    
    # Behavioral change + velocity interaction
    behavioral_changes = (
        categorical_features.get('c_fingerprintchanged_seq', 0) +
        categorical_features.get('c_emailchanged_seq', 0) +
        categorical_features.get('c_paymentchg_seq', 0)
    )
    transaction_velocity = numerical_features.get('c_days_lastorder_seq', 365)
    # High behavioral changes + low time since last order = high fraud risk
    fraud_interactions['behavior_velocity_interaction'] = behavioral_changes * (1 / max(transaction_velocity, 1))
    
    # Device risk + amount interaction
    device_risk = numerical_features.get('c_fingerprintriskvalue_seq', 0)
    transaction_amount = numerical_features.get('c_ordertotalamountusd_seq', 0)
    fraud_interactions['device_amount_interaction'] = device_risk * (transaction_amount / 1000.0)  # Normalize amount
    
    # New card + high amount interaction
    cc_age = numerical_features.get('c_ccage_seq', 365)
    new_card_indicator = 1 if cc_age < 30 else 0  # Card less than 30 days old
    fraud_interactions['new_card_amount_interaction'] = new_card_indicator * (transaction_amount / 1000.0)
    
    # Strangeness + network risk interaction
    strangeness_avg = (
        numerical_features.get('c_billaddrstrangeness_seq', 0) +
        numerical_features.get('c_shipaddrstrangeness_seq', 0) +
        numerical_features.get('c_emailnamestrangeness_seq', 0)
    ) / 3
    
    network_risk = numerical_features.get('c_tugboat_ev2customer_evwmaxlinkscorecust_seq', 0)
    fraud_interactions['strangeness_network_interaction'] = strangeness_avg * network_risk
    
    return fraud_interactions
```
**Operations**:
- Payment method and geographic fraud risk interaction analysis
- Behavioral change and transaction velocity cross-correlation
- Device risk and transaction amount interaction scoring
- New credit card and high amount transaction interaction detection
- Customer strangeness and fraud network risk correlation analysis

#### **7. Fraud Risk Score Aggregation and Normalization**
**TSA Implementation**: Comprehensive fraud risk scoring framework
```python
# Fraud risk score aggregation from all fraud indicators
def aggregate_fraud_risk_scores(fraud_features_dict):
    aggregated_scores = {}
    
    # Payment fraud risk aggregation
    payment_risk_components = [
        fraud_features_dict.get('payment_method_risk', 0),
        fraud_features_dict.get('issuer_fraud_risk', 0),
        fraud_features_dict.get('blacklist_hit_indicator', 0) * 5,  # High weight for blacklist hits
        fraud_features_dict.get('multiple_card_risk', 0),
    ]
    aggregated_scores['payment_fraud_score'] = sum(payment_risk_components) / len(payment_risk_components)
    
    # Behavioral fraud risk aggregation
    behavioral_risk_components = [
        fraud_features_dict.get('account_takeover_risk', 0),
        fraud_features_dict.get('synthetic_identity_risk', 0),
        fraud_features_dict.get('device_fraud_risk', 0),
        fraud_features_dict.get('return_rate_fraud_indicator', 0),
    ]
    aggregated_scores['behavioral_fraud_score'] = sum(behavioral_risk_components) / len(behavioral_risk_components)
    
    # Geographic fraud risk aggregation
    geographic_risk_components = [
        fraud_features_dict.get('geographic_fraud_risk', 0),
        fraud_features_dict.get('cross_border_transaction_risk', 0),
        fraud_features_dict.get('high_geographic_risk_flag', 0),
    ]
    aggregated_scores['geographic_fraud_score'] = sum(geographic_risk_components) / len(geographic_risk_components)
    
    # Network fraud risk aggregation
    network_risk_components = [
        fraud_features_dict.get('customer_network_risk', 0),
        fraud_features_dict.get('transaction_network_risk', 0),
        fraud_features_dict.get('high_network_risk_flag', 0),
    ]
    aggregated_scores['network_fraud_score'] = sum(network_risk_components) / len(network_risk_components)
    
    # Overall fraud risk score calculation
    overall_risk_components = [
        aggregated_scores['payment_fraud_score'] * 0.3,      # 30% weight
        aggregated_scores['behavioral_fraud_score'] * 0.25,  # 25% weight
        aggregated_scores['geographic_fraud_score'] * 0.25,  # 25% weight
        aggregated_scores['network_fraud_score'] * 0.2,      # 20% weight
    ]
    
    aggregated_scores['overall_fraud_risk_score'] = sum(overall_risk_components)
    
    # Normalize all scores to 0-1 range
    for score_name, score_value in aggregated_scores.items():
        aggregated_scores[score_name] = min(max(score_value, 0), 1)  # Clamp to [0, 1]
    
    return aggregated_scores

# Fraud risk percentile calculation for deployment
def calculate_fraud_risk_percentiles(fraud_scores, percentile_bins=1000):
    # Similar to TSA percentile calculation for fraud detection deployment
    fraud_score_values = [score for score in fraud_scores if score is not None]
    
    if len(fraud_score_values) > 0:
        percentiles = np.linspace(0, 100, percentile_bins)
        percentile_values = np.percentile(fraud_score_values, percentiles)
        
        # Create percentile mapping for fraud detection inference
        percentile_mapping = list(zip(percentile_values, percentiles / 100.0))
        
        return percentile_mapping
    
    return []
```
**Operations**:
- Multi-component fraud risk score aggregation with weighted combinations
- Payment, behavioral, geographic, and network fraud score calculation
- Overall fraud risk score with configurable component weights (30%, 25%, 25%, 20%)
- Score normalization and clamping to [0, 1] range for consistency
- Percentile mapping generation for fraud detection deployment (similar to TSA approach)

#### **8. Fraud Feature Engineering Pipeline Orchestration**
**TSA Implementation**: Complete fraud feature engineering workflow coordination
```python
# Fraud feature engineering pipeline orchestration
def orchestrate_fraud_feature_engineering(input_data, fraud_config):
    fraud_engineered_features = {}
    
    # Stage 1: Extract base categorical and numerical features
    categorical_features = extract_fraud_categorical_features(input_data, fraud_config)
    numerical_features = extract_fraud_numerical_features(input_data, fraud_config)
    
    # Stage 2: Fraud-specific categorical encoding
    encoded_categorical = fraud_categorical_encoding(input_data, fraud_config['categorical_mappings'])
    fraud_engineered_features.update(encoded_categorical)
    
    # Stage 3: Fraud velocity and temporal features
    velocity_features = calculate_fraud_velocity_features(input_data, fraud_config['current_timestamp'])
    fraud_engineered_features.update(velocity_features)
    
    # Stage 4: Payment fraud pattern analysis
    payment_patterns = analyze_payment_fraud_patterns(categorical_features, numerical_features)
    fraud_engineered_features.update(payment_patterns)
    
    # Stage 5: Customer fraud behavior profiling
    customer_profile = profile_customer_fraud_behavior(categorical_features, numerical_features)
    fraud_engineered_features.update(customer_profile)
    
    # Stage 6: Geographic fraud detection
    geographic_patterns = detect_geographic_fraud_patterns(categorical_features)
    fraud_engineered_features.update(geographic_patterns)
    
    # Stage 7: Fraud feature interactions
    interaction_features = engineer_fraud_feature_interactions(categorical_features, numerical_features)
    fraud_engineered_features.update(interaction_features)
    
    # Stage 8: Fraud risk score aggregation
    aggregated_scores = aggregate_fraud_risk_scores(fraud_engineered_features)
    fraud_engineered_features.update(aggregated_scores)
    
    # Stage 9: Fraud feature validation and quality control
    validated_features = validate_fraud_features(fraud_engineered_features, fraud_config.get('validation', {}))
    
    return validated_features

def validate_fraud_features(fraud_features, validation_config):
    validated_features = {}
    
    # Fraud-specific validation rules
    for feature_name, feature_value in fraud_features.items():
        # Check for valid fraud score ranges
        if 'score' in feature_name.lower():
            if isinstance(feature_value, (int, float)) and 0 <= feature_value <= 1:
                validated_features[feature_name] = feature_value
            else:
                print(f"Warning: Fraud score {feature_name} out of range [0,1]: {feature_value}")
        
        # Check for valid fraud indicators
        elif 'indicator' in feature_name.lower() or 'flag' in feature_name.lower():
            if feature_value in [0, 1]:
                validated_features[feature_name] = feature_value
            else:
                print(f"Warning: Fraud indicator {feature_name} not binary: {feature_value}")
        
        # General feature validation
        else:
            if feature_value is not None and not (isinstance(feature_value, float) and np.isnan(feature_value)):
                validated_features[feature_name] = feature_value
    
    return validated_features
```
**Operations**:
- Multi-stage fraud feature engineering pipeline with sequential processing
- Fraud-specific categorical encoding, velocity analysis, and pattern detection
- Customer profiling, geographic analysis, and feature interaction engineering
- Comprehensive fraud risk score aggregation and validation
- Fraud feature quality control with domain-specific validation rules

**Configuration Parameters**:
```python
{
    "extends": "DomainFeatureEngineering",
    "fraud_categorical_encoding": {
        "payment_fraud_features": ["cciscorporate", "ccisdebit", "ccisprepaid", "ccissuer", "creditcardhit"],
        "geographic_fraud_features": ["geobillcountrycccountrycodeequal", "geoipcountrycodecccountrycodeequal"],
        "behavioral_fraud_features": ["fingerprintchanged", "emailchanged", "ipchanged", "paymentchg"],
        "numerical_categorical_indices": [3,4,5,6,8,9,10,13,14,16,17,18,19,20,21,22,24,27,28,29,30,32,33,34,35,36,37,39,40],
        "special_encoding_rules": {
            "fingerprintRiskValue": "zero_nonzero_distinction",  # Special fraud risk encoding
            "default_fraud_mapping": "risk_based_encoding"
        }
    },
    "fraud_velocity_features": {
        "temporal_indicators": ["days_lastorder", "ordertotalamountusd", "ccage", "cccount"],
        "velocity_thresholds": {
            "transaction_spike_hours": 1,                    # Transactions within 1 hour
            "new_card_days": 30,                            # Cards less than 30 days old
            "large_amount_multiplier": 3                     # Amounts > 3x average
        },
        "velocity_calculations": ["avg_interval", "min_interval", "spike_count", "acceleration", "volatility"]
    },
    "payment_fraud_analysis": {
        "card_type_risk_weights": {
            "prepaid": 3.0,                                  # Highest fraud risk
            "corporate": 1.5,                                # Medium fraud risk
            "debit": 1.2                                     # Slightly higher risk
        },
        "payment_fraud_indicators": ["issuer_risk", "blacklist_hits", "payment_switching", "multiple_cards"],
        "network_fraud_features": ["tugboat_customer_score", "tugboat_transaction_score"],
        "network_risk_threshold": 0.7                       # High network risk threshold
    },
    "customer_behavior_analysis": {
        "account_takeover_indicators": ["fingerprintchanged", "emailchanged", "ipchanged", "paymentchg"],
        "loyalty_indicators": ["isprimemember", "primemembertype"],
        "address_consistency_features": ["same_bs_zip", "same_bs_state"],
        "strangeness_features": ["billaddrstrangeness", "shipaddrstrangeness", "emailnamestrangeness"],
        "device_fraud_features": ["fingerprintriskvalue"],
        "takeover_risk_normalization": 4.0,                 # Normalize by max possible changes
        "strangeness_threshold": 0.5                        # Synthetic identity detection threshold
    },
    "geographic_fraud_detection": {
        "country_mismatch_features": [
            "geobillcountrycccountrycodeequal",
            "geoipcountrycodecccountrycodeequal", 
            "georeportedipmktplcountrycodeequal",
            "marketplacecountrycode_cctry_cd_is_match"
        ],
        "cross_border_risk_weights": {
            "billing_cc_mismatch": 1.0,                     # High fraud indicator
            "ip_cc_mismatch": 1.0,                          # High fraud indicator
            "ip_marketplace_mismatch": 0.8,                 # Medium fraud indicator
            "marketplace_cc_mismatch": 0.6                  # Lower fraud indicator
        },
        "high_risk_combination_threshold": 2               # Multiple geographic mismatches
    },
    "fraud_feature_interactions": {
        "interaction_pairs": [
            ["payment_method_risk", "geographic_fraud_risk"],
            ["behavioral_changes", "transaction_velocity"],
            ["device_risk", "transaction_amount"],
            ["new_card_indicator", "transaction_amount"],
            ["strangeness_score", "network_risk"]
        ],
        "interaction_weights": {
            "payment_geographic": 1.0,
            "behavior_velocity": 2.0,                       # High weight for rapid behavioral changes
            "device_amount": 1.5,                           # Device risk with high amounts
            "new_card_amount": 2.0,                         # New cards with high amounts
            "strangeness_network": 1.2                      # Synthetic identity + network risk
        },
        "normalization_factors": {
            "amount_normalization": 1000.0,                 # Normalize amounts to thousands
            "velocity_normalization": 365.0,                # Normalize days to yearly scale
            "age_normalization": 365.0                      # Normalize card age to yearly scale
        }
    },
    "fraud_risk_aggregation": {
        "component_weights": {
            "payment_fraud_score": 0.30,                    # 30% weight for payment fraud
            "behavioral_fraud_score": 0.25,                 # 25% weight for behavioral fraud
            "geographic_fraud_score": 0.25,                 # 25% weight for geographic fraud
            "network_fraud_score": 0.20                     # 20% weight for network fraud
        },
        "score_normalization": "clamp_0_1",                 # Clamp all scores to [0, 1] range
        "percentile_bins": 1000,                            # Percentile mapping resolution for deployment
        "risk_thresholds": {
            "low_risk": 0.3,                               # Below 0.3 = low fraud risk
            "medium_risk": 0.7,                            # 0.3-0.7 = medium fraud risk
            "high_risk": 0.7                               # Above 0.7 = high fraud risk
        }
    },
    "fraud_validation_rules": {
        "score_range_validation": True,                     # Validate fraud scores in [0, 1] range
        "binary_indicator_validation": True,                # Validate fraud flags are 0 or 1
        "missing_value_tolerance": 0.1,                     # Max 10% missing values per feature
        "feature_consistency_checks": True,                 # Cross-feature consistency validation
        "temporal_consistency_validation": True,            # Validate temporal fraud patterns
        "quality_control_thresholds": {
            "min_fraud_rate": 0.01,                        # Minimum 1% fraud rate in data
            "max_fraud_rate": 0.50,                        # Maximum 50% fraud rate in data
            "feature_completeness": 0.90                    # Minimum 90% feature completeness
        }
    },
    "fraud_pipeline_orchestration": {
        "processing_stages": [
            "categorical_encoding",
            "velocity_analysis", 
            "payment_pattern_analysis",
            "customer_behavior_profiling",
            "geographic_fraud_detection",
            "feature_interactions",
            "risk_aggregation",
            "validation"
        ],
        "stage_dependencies": True,                         # Enforce sequential stage processing
        "error_handling": "graceful_degradation",           # Continue processing on individual feature failures
        "progress_monitoring": True,                        # Track fraud feature engineering progress
        "comprehensive_logging": True                       # Log all fraud feature engineering operations
    }
}
```

**Script Location**: `src/cursus/steps/scripts/fraud_feature_engineering.py`

## Implementation Priority and Dependencies

### **Phase 1: Base Infrastructure (High Priority)**
1. **DistributedProcessing** - Foundation for large-scale processing
2. **TemporalSequencePreprocessing** - Essential for time-series data
3. **MultiSequencePreprocessing** - Multi-entity data handling foundation
4. **DomainFeatureEngineering** - Configurable feature engineering framework

### **Phase 2: Base Model Training (Medium Priority)**
5. **AttentionModelTraining** - General attention mechanism training

### **Phase 3: Specialized Extensions (Lower Priority)**
6. **FraudChunkedDistributedProcessing** - Fraud-optimized distributed processing
7. **FraudTemporalSequencePreprocessing** - Fraud temporal features
8. **FraudTwoSequencePreprocessing** - CID/CCID specific processing
9. **FraudFeatureEngineering** - Fraud detection features
10. **TemporalSelfAttentionTraining** - End-to-end TSA model architecture

### **Dependencies**:
- **Base Steps**: All base steps (1-5) are independent and can be developed in parallel
- **Extension Steps**: All fraud-specific extensions (6-10) depend on their corresponding base steps
- **Cross-Dependencies**:
  - `MultiSequencePreprocessing` builds on `TemporalSequencePreprocessing`
  - `AttentionModelTraining` benefits from temporal preprocessing capabilities
  - `FraudTwoSequencePreprocessing` requires both `FraudTemporalSequencePreprocessing` and `MultiSequencePreprocessing`
  - `TemporalSelfAttentionTraining` requires `AttentionModelTraining` and temporal preprocessing steps

### **Development Strategy**:
- **80/20 Rule**: Base steps provide 80% sharable functionality, extensions add 20% domain-specific features
- **Reusability**: Base steps can be extended for other domains beyond fraud detection
- **Modularity**: Each extension inherits and extends its base step's configuration and functionality

## Integration Benefits

### **Immediate Benefits**:
- ✅ **50% step reuse** through direct equivalencies (CradleDataLoading, Registration, Package)
- ✅ **Enhanced calibration** through superior ModelCalibration step
- ✅ **Standardized configuration** through Cursus config management
- ✅ **Improved monitoring** through Cursus step tracking

### **Long-term Benefits**:
- ✅ **Unified pipeline framework** for all ML workflows
- ✅ **Reusable temporal processing** for other time-series models
- ✅ **Standardized fraud detection** components across teams
- ✅ **Enhanced maintainability** through framework consistency
- ✅ **Improved testing** through Cursus validation framework

## Conclusion

The TSA pipeline demonstrates **60% compatibility** with existing Cursus steps, with complete equivalencies for deployment-related steps and enhanced functionality for model calibration. The unique temporal sequence processing and specialized model architecture represent **40% novel functionality** that requires new step implementations.

**Key Recommendations**:

1. **Immediate Migration**: Replace TSA's CradleDataLoading, Registration, Package, and generic_rfuge steps with Cursus equivalents
2. **Phased Development**: Implement the 5 recommended new steps in priority order
3. **Framework Integration**: Leverage Cursus configuration, validation, and monitoring infrastructure
4. **Knowledge Transfer**: Use TSA implementation as reference for new step development

This analysis provides a roadmap for integrating TSA pipeline capabilities into the Cursus framework while preserving specialized fraud detection functionality and enhancing overall pipeline standardization.
