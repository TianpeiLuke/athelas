---
tags:
  - project
  - implementation
  - lightgbmmt
  - multi_task_learning
  - training_script
  - xgboost_alignment
keywords:
  - lightgbmmt training script
  - multi-task gradient boosting
  - script alignment
  - preprocessing artifacts
  - inference evaluation
topics:
  - lightgbmmt script refinement
  - xgboost pattern alignment
  - multi-task evaluation
language: python
date of note: 2025-11-12
---

# LightGBMMT Implementation Part 2: Training Script Alignment with XGBoost Pattern

## Overview

This document covers the refinement of the LightGBMMT training script to align with the XGBoost training pattern, focusing on:
1. **Main Function Signature**: Standard signature matching XGBoost
2. **Data Loading**: Format detection and preservation
3. **Preprocessing Artifacts**: Integration with upstream steps
4. **Inference & Evaluation**: Multi-task specific metrics and plots
5. **Model Artifacts**: Complete artifact saving structure

**Timeline**: 1 week
**Prerequisites**: Part 1 completed (refactored loss functions, model architecture, hyperparameters)

## Executive Summary

### Objectives
- Align `lightgbmmt_training.py` with `xgboost_training.py` pattern
- Add preprocessing artifact handling (imputation, risk tables, features)
- Implement multi-task inference and evaluation
- Add format detection and preservation
- Standardize model artifact saving

### Success Metrics
- ✅ Main function matches XGBoost signature
- ✅ Supports all preprocessing artifact types
- ✅ Per-task and aggregate evaluation
- ✅ Format preservation (csv/tsv/parquet)
- ✅ Complete model artifact packaging

## Current State Analysis

### Existing Implementation (Part 1)
✅ Refactored loss functions (7 files, 67% code reduction)
✅ Model architecture with template method pattern
✅ Training script with basic structure
✅ Testability main for local development
✅ Hyperparameters class with all parameters

### Gap Analysis

**Missing from Current Implementation**:
1. Main function signature (uses simple train() instead)
2. Format detection helpers
3. Preprocessing artifact loading
4. Numerical imputation integration
5. Risk table mapping integration
6. Feature selection support
7. Multi-task inference functions
8. Multi-task evaluation and plotting
9. Complete model artifact saving
10. Standard __main__ setup

## Phase 1: Helper Functions (Day 1)

### 1.1 Format Detection Helpers

**File**: `projects/cap_mtgbm/docker/lightgbmmt_training.py` (add functions)

```python
def _detect_file_format(file_path: str) -> str:
    """
    Detect the format of a data file based on its extension.
    
    Returns: 'csv', 'tsv', or 'parquet'
    """
    from pathlib import Path
    suffix = Path(file_path).suffix.lower()
    
    if suffix == '.csv':
        return 'csv'
    elif suffix == '.tsv':
        return 'tsv'
    elif suffix == '.parquet':
        return 'parquet'
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load DataFrame and detect its format.
    
    Returns: Tuple of (DataFrame, format_string)
    """
    detected_format = _detect_file_format(file_path)
    
    if detected_format == 'csv':
        df = pd.read_csv(file_path)
    elif detected_format == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif detected_format == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")
    
    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame,
    output_path: str,
    format_str: str
) -> str:
    """
    Save DataFrame in specified format.
    
    Returns: Path to saved file
    """
    from pathlib import Path
    output_path = Path(output_path)
    
    if format_str == 'csv':
        file_path = output_path.with_suffix('.csv')
        df.to_csv(file_path, index=False)
    elif format_str == 'tsv':
        file_path = output_path.with_suffix('.tsv')
        df.to_csv(file_path, sep='\t', index=False)
    elif format_str == 'parquet':
        file_path = output_path.with_suffix('.parquet')
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")
    
    return str(file_path)


def find_first_data_file(data_dir: str) -> str:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(('.csv', '.parquet', '.json', '.tsv')):
            return os.path.join(data_dir, fname)
    return None
```

**Success Criteria**:
- ✅ Supports csv, tsv, parquet formats
- ✅ Preserves format through pipeline
- ✅ Clear error messages

### 1.2 Preprocessing Artifact Loaders

**File**: `projects/cap_mtgbm/docker/lightgbmmt_training.py` (add functions)

```python
def load_precomputed_artifacts(
    model_artifacts_dir: Optional[str],
    use_imputation: bool,
    use_risk_tables: bool,
    use_features: bool
) -> Dict[str, Any]:
    """
    Auto-detect and load pre-computed artifacts from model_artifacts_input.
    
    Returns:
        Dictionary with:
        - 'impute_dict': dict or None
        - 'risk_tables': dict or None
        - 'selected_features': list or None
        - 'loaded': {'imputation': bool, 'risk_tables': bool, 'features': bool}
    """
    result = {
        'impute_dict': None,
        'risk_tables': None,
        'selected_features': None,
        'loaded': {
            'imputation': False,
            'risk_tables': False,
            'features': False
        }
    }
    
    if not model_artifacts_dir or not os.path.exists(model_artifacts_dir):
        logger.warning(f"Model artifacts directory not found: {model_artifacts_dir}")
        return result
    
    logger.info(f"Loading pre-computed artifacts from: {model_artifacts_dir}")
    
    # 1. Try to load imputation dictionary
    if use_imputation:
        impute_path = os.path.join(model_artifacts_dir, 'impute_dict.pkl')
        if os.path.exists(impute_path):
            try:
                with open(impute_path, 'rb') as f:
                    result['impute_dict'] = pkl.load(f)
                result['loaded']['imputation'] = True
                logger.info(f"✓ Loaded pre-computed imputation from {impute_path}")
            except Exception as e:
                logger.warning(f"Failed to load imputation: {e}")
        else:
            logger.warning(f"Imputation dict not found at {impute_path}")
    
    # 2. Try to load risk tables
    if use_risk_tables:
        risk_path = os.path.join(model_artifacts_dir, 'risk_table_map.pkl')
        if os.path.exists(risk_path):
            try:
                with open(risk_path, 'rb') as f:
                    result['risk_tables'] = pkl.load(f)
                result['loaded']['risk_tables'] = True
                logger.info(f"✓ Loaded pre-computed risk tables from {risk_path}")
            except Exception as e:
                logger.warning(f"Failed to load risk tables: {e}")
        else:
            logger.warning(f"Risk tables not found at {risk_path}")
    
    # 3. Try to load selected features
    if use_features:
        features_path = os.path.join(model_artifacts_dir, 'selected_features.json')
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    fs_data = json.load(f)
                result['selected_features'] = fs_data.get('selected_features', [])
                result['loaded']['features'] = True
                logger.info(f"✓ Loaded pre-computed features from {features_path}")
            except Exception as e:
                logger.warning(f"Failed to load features: {e}")
        else:
            logger.warning(f"Features not found at {features_path}")
    
    return result


def validate_precomputed_data_state(
    train_df: pd.DataFrame,
    config: dict,
    imputation_used: bool,
    risk_tables_used: bool
) -> None:
    """
    Validate that data state matches the pre-computed artifact flags.
    
    Raises ValueError if data state doesn't match expected state.
    """
    if imputation_used:
        # Verify data has no NaN in numerical columns
        tab_fields = config.get('tab_field_list', [])
        if tab_fields:
            nan_cols = train_df[tab_fields].columns[
                train_df[tab_fields].isna().any()
            ].tolist()
            if nan_cols:
                raise ValueError(
                    f"USE_PRECOMPUTED_IMPUTATION=true but data contains NaN in: {nan_cols}. "
                    "Data must be pre-imputed when using pre-computed imputation artifacts."
                )
            logger.info("✓ Validated: No NaN values (consistent with pre-computed imputation)")
    
    if risk_tables_used:
        # Verify categorical columns are numeric (risk-mapped)
        cat_fields = config.get('cat_field_list', [])
        for col in cat_fields:
            if col in train_df.columns:
                if not pd.api.types.is_numeric_dtype(train_df[col]):
                    raise ValueError(
                        f"USE_PRECOMPUTED_RISK_TABLES=true but column '{col}' is not numeric. "
                        "Data must be pre-transformed when using pre-computed risk table artifacts."
                    )
        if cat_fields:
            logger.info("✓ Validated: Categorical columns are numeric (consistent with pre-computed risk tables)")
```

**Success Criteria**:
- ✅ Auto-detects available artifacts
- ✅ Validates data state
- ✅ Clear logging of loaded artifacts

## Phase 2: Preprocessing Integration (Days 2-3)

### 2.1 Add Numerical Imputation

**Import**: Add at top of file
```python
from processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor
)
```

**Function**: Add to script
```python
def apply_numerical_imputation(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple:
    """
    Applies numerical imputation using single-column architecture.
    
    Returns: (train_df_imputed, val_df_imputed, test_df_imputed, impute_dict)
    """
    imputation_processors = {}
    train_df_imputed = train_df.copy()
    val_df_imputed = val_df.copy()
    test_df_imputed = test_df.copy() if test_df is not None else None
    
    # Create one processor per numerical column
    for var in config['tab_field_list']:
        proc = NumericalVariableImputationProcessor(
            column_name=var,
            strategy='mean'
        )
        proc.fit(train_df[var])
        imputation_processors[var] = proc
        
        # Transform each split
        train_df_imputed[var] = proc.transform(train_df_imputed[var])
        val_df_imputed[var] = proc.transform(val_df_imputed[var])
        if test_df_imputed is not None:
            test_df_imputed[var] = proc.transform(test_df_imputed[var])
    
    # Build imputation dictionary for artifact saving
    impute_dict = {
        var: proc.get_imputation_value()
        for var, proc in imputation_processors.items()
    }
    
    return (train_df_imputed, val_df_imputed, test_df_imputed, impute_dict)
```

### 2.2 Add Risk Table Mapping

**Import**: Add at top of file
```python
from processing.categorical.risk_table_processor import RiskTableMappingProcessor
```

**Function**: Add to script
```python
def fit_and_apply_risk_tables(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple:
    """
    Fits risk tables on training data and applies to all splits.
    
    Returns: (train_df_transformed, val_df_transformed, test_df_transformed, risk_tables)
    """
    risk_processors = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy() if test_df is not None else None
    
    for var in config['cat_field_list']:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config['label_name'],
            smooth_factor=config.get('smooth_factor', 0.0),
            count_threshold=config.get('count_threshold', 0)
        )
        proc.fit(train_df)
        risk_processors[var] = proc
        
        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
        if test_df_transformed is not None:
            test_df_transformed[var] = proc.transform(test_df_transformed[var])
    
    consolidated_risk_tables = {
        var: proc.get_risk_tables()
        for var, proc in risk_processors.items()
    }
    
    return (
        train_df_transformed,
        val_df_transformed,
        test_df_transformed,
        consolidated_risk_tables
    )
```

**Success Criteria**:
- ✅ Single-column architecture (one processor per column)
- ✅ Fits on training data only
- ✅ Applies to all splits
- ✅ Returns artifacts for saving

## Phase 3: Multi-Task Inference & Evaluation (Days 4-5)

### 3.1 Multi-Task Prediction Function

```python
def predict_multitask(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_columns: List[str]
) -> np.ndarray:
    """
    Generate multi-task predictions.
    
    Returns: np.ndarray of shape (n_samples, n_tasks) with probabilities
    """
    # Prepare features
    X = df[feature_columns]
    
    # Get predictions from model
    # Model should return predictions for all tasks
    predictions = model.predict(X)  # Shape: (n_samples, n_tasks)
    
    return predictions
```

### 3.2 Multi-Task Metrics Computation

```python
def compute_multitask_metrics(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_columns: List[str]
) -> Dict[str, Any]:
    """
    Compute per-task and aggregate metrics.
    
    Args:
        y_true_tasks: Dict mapping task_id to true labels
        y_pred_tasks: Array of shape (n_samples, n_tasks) with predicted probabilities
        task_columns: List of task column names
    
    Returns:
        Dictionary with per-task and aggregate metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    metrics = {}
    n_tasks = len(task_columns)
    
    # Per-task metrics
    auc_rocs = []
    aps = []
    f1s = []
    
    for i, task_name in enumerate(task_columns):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]
        
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred > 0.5)
            
            metrics[f'task_{i}_{task_name}'] = {
                'auc_roc': float(auc_roc),
                'average_precision': float(ap),
                'f1_score': float(f1)
            }
            
            auc_rocs.append(auc_roc)
            aps.append(ap)
            f1s.append(f1)
            
        except ValueError as e:
            # Handle case where only one class present
            logger.warning(f"Task {i} ({task_name}): {e}")
            metrics[f'task_{i}_{task_name}'] = {
                'auc_roc': 0.5,
                'average_precision': 0.5,
                'f1_score': 0.0
            }
    
    # Aggregate metrics
    if auc_rocs:
        metrics['aggregate'] = {
            'mean_auc_roc': float(np.mean(auc_rocs)),
            'median_auc_roc': float(np.median(auc_rocs)),
            'mean_average_precision': float(np.mean(aps)),
            'median_average_precision': float(np.median(aps)),
            'mean_f1_score': float(np.mean(f1s)),
            'median_f1_score': float(np.median(f1s))
        }
    
    return metrics
```

### 3.3 Multi-Task Plotting

```python
def plot_multitask_curves(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_columns: List[str],
    out_dir: str,
    prefix: str
) -> None:
    """
    Generate ROC and PR curves for each task.
    
    Args:
        y_true_tasks: Dict mapping task_id to true labels
        y_pred_tasks: Array of shape (n_samples, n_tasks)
        task_columns: List of task column names
        out_dir: Output directory for plots
        prefix: Prefix for plot filenames (e.g., 'val_', 'test_')
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    
    for i, task_name in enumerate(task_columns):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            logger.warning(f"Task {i} ({task_name}): Only one class present, skipping plots")
            continue
        
        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.title(f'{prefix}Task {i} ({task_name}) ROC')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(os.path.join(out_dir, f'{prefix}task_{i}_{task_name}_roc.jpg'))
            plt.close()
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            
            plt.figure()
            plt.plot(recall, precision, label=f'AP={ap:.3f}')
            plt.title(f'{prefix}Task {i} ({task_name}) PR')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig(os.path.join(out_dir, f'{prefix}task_{i}_{task_name}_pr.jpg'))
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting task {i} ({task_name}): {e}")
```

### 3.4 Complete Evaluation Function

```python
def evaluate_split_multitask(
    name: str,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_columns: List[str],
    model,
    cfg: dict,
    output_format: str = 'csv',
    prefix: str = '/opt/ml/output/data'
) -> None:
    """
    Evaluate a data split for multi-task learning.
    
    Generates:
    - Predictions file with per-task probabilities
    - Metrics JSON with per-task and aggregate metrics
    - Per-task ROC and PR curves
    - Packaged tar.gz archive
    """
    import tarfile
    
    logger.info(f"Evaluating {name} split...")
    
    # Extract task labels
    y_true_tasks = {}
    for i, task_col in enumerate(task_columns):
        y_true_tasks[i] = df[task_col].astype(int).values
    
    # Get predictions
    y_pred_tasks = predict_multitask(model, df, feature_columns, task_columns)
    
    # Compute metrics
    metrics = compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_columns)
    
    # Save predictions
    out_base = os.path.join(prefix, name)
    os.makedirs(out_base, exist_ok=True)
    
    # Build predictions DataFrame
    id_col = cfg.get('id_name', 'id')
    ids = df.get(id_col, np.arange(len(df)))
    
    pred_df = pd.DataFrame({id_col: ids})
    for i, task_col in enumerate(task_columns):
        pred_df[f'{task_col}_true'] = y_true_tasks[i]
        pred_df[f'{task_col}_prob'] = y_pred_tasks[:, i]
    
    output_base = os.path.join(out_base, 'predictions')
    saved_path = save_dataframe_with_format(pred_df, output_base, output_format)
    logger.info(f"Saved predictions (format={output_format}): {saved_path}")
    
    # Save metrics
    metrics_file = os.path.join(out_base, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_file}")
    
    # Generate plots
    out_metrics = os.path.join(prefix, f'{name}_metrics')
    plot_multitask_curves(
        y_true_tasks,
        y_pred_tasks,
        task_columns,
        out_metrics,
        f'{name}_'
    )
    
    # Package into tar.gz
    tar_path = os.path.join(prefix, f'{name}.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as t:
        t.add(out_base, arcname=name)
        t.add(out_metrics, arcname=f'{name}_metrics')
    
    logger.info(f"{name} outputs packaged → {tar_path}")
```

**Success Criteria**:
- ✅ Per-task predictions saved
- ✅ Per-task and aggregate metrics
- ✅ Per-task ROC/PR curves
- ✅ Packaged tar.gz output

## Phase 4: Model Artifact Saving (Day 6)

### 4.1 Update save_artifacts Function

```python
def save_artifacts(
    model,
    risk_tables: dict,
    impute_dict: dict,
    model_path: str,
    feature_columns: List[str],
    hyperparams: LightGBMMtModelHyperparameters,
    training_state: TrainingState
) -> None:
    """
    Saves trained model and all preprocessing artifacts.
    
    Saves:
    - lightgbmmt_model.txt (LightGBM text format)
    - risk_table_map.pkl
    - impute_dict.pkl
    - training_state.json
    - feature_columns.txt (with ordering comments)
    - hyperparameters.json
    - weight_evolution.json (if available)
    """
    os.makedirs(model_path, exist_ok=True)
    
    # 1. Save LightGBM model
    model_file = os.path.join(model_path, 'lightgbmmt_model.txt')
    model.save(model_file)
    logger.info(f"Saved LightGBMMT model to {model_file}")
    
    # 2. Save risk tables
    risk_map_file = os.path.join(model_path, 'risk_table_map.pkl')
    with open(risk_map_file, 'wb') as f:
        pkl.dump(risk_tables, f)
    logger.info(f"Saved risk table map to {risk_map_file}")
    
    # 3. Save imputation dictionary
    impute_file = os.path.join(model_path, 'impute_dict.pkl')
    with open(impute_file, 'wb') as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")
    
    # 4. Save training state (for checkpointing)
    state_file = os.path.join(model_path, 'training_state.json')
    with open(state_file, 'w') as f:
        json.dump(training_state.to_checkpoint_dict(), f, indent=2)
    logger.info(f"Saved training state to {state_file}")
    
    # 5. Save feature columns with ordering
    feature_columns_file = os.path.join(model_path, 'feature_columns.txt')
    with open(feature_columns_file, 'w') as f:
        f.write("# Feature columns in exact order required for model inference\n")
        f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
        f.write("# Each line contains: <column_index>,<column_name>\n")
        for idx, column in enumerate(feature_columns):
            f.write(f"{idx},{column}\n")
    logger.info(f"Saved feature columns to {feature_columns_file}")
    
    # 6. Save hyperparameters
    hyperparams_file = os.path.join(model_path, 'hyperparameters.json')
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams.model_dump(), f, indent=2, sort_keys=True)
    logger.info(f"Saved hyperparameters to {hyperparams_file}")
    
    # 7. Save weight evolution (multi-task specific)
    if training_state.weight_evolution:
        weight_file = os.path.join(model_path, 'weight_evolution.json')
        with open(weight_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            weight_evolution_list = [
                w.tolist() for w in training_state.weight_evolution
            ]
            json.dump(weight_evolution_list, f, indent=2)
        logger.info(f"Saved weight evolution to {weight_file}")
```

**Success Criteria**:
- ✅ All artifacts saved
- ✅ Multi-task specific artifacts (weight_evolution, training_state)
- ✅ Complete for inference

## Phase 5: Main Function Restructure (Day 7)

### 5.1 New main() Function Signature

```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function to execute LightGBMMT training logic.
    
    Args:
        input_paths: Dictionary of input paths
            - "input_path": Directory containing train/val/test data
            - "hyperparameters_s3_uri": Path to hyperparameters directory
            - "model_artifacts_input": (Optional) Pre-computed artifacts
        output_paths: Dictionary of output paths
            - "model_output": Directory to save model artifacts
            - "evaluation_output": Directory to save evaluation outputs
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    try:
        logger.info("====== STARTING MAIN EXECUTION ======")
        
        # Extract paths from parameters
        data_dir = input_paths["input_path"]
        model_dir = output_paths["model_output"]
        output_dir = output_paths["evaluation_output"]
        model_artifacts_input_dir = input_paths.get("model_artifacts_input")
        
        # Priority-based hyperparameters path resolution
        hparam_path = "/opt/ml/code/hyperparams/hyperparameters.json"
        if not os.path.exists(hparam_path):
            if "hyperparameters_s3_uri" in input_paths:
                hparam_path = input_paths["hyperparameters_s3_uri"]
                if not hparam_path.endswith("hyperparameters.json"):
                    hparam_path = os.path.join(hparam_path, "hyperparameters.json")
        
        logger.info(f"Loading configuration from {hparam_path}")
        with open(hparam_path, 'r') as f:
            hyperparams_dict = json.load(f)
        hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)
        
        # Load datasets with format detection
        logger.info("Loading datasets...")
        train_df, val_
