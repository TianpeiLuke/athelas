#!/usr/bin/env python3
"""
LightGBMMT Multi-Task Training Script

Aligns with XGBoost training pattern for consistency across training scripts.
Integrates refactored loss functions and model architecture for multi-task gradient boosting.
"""

import os
import sys
from subprocess import check_call
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """Get CodeArtifact access token for secure PyPI."""
    import boto3

    try:
        os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
        sts = boto3.client("sts", region_name="us-east-1")
        caller_identity = sts.get_caller_identity()
        assumed_role_object = sts.assume_role(
            RoleArn="arn:aws:iam::675292366480:role/SecurePyPIReadRole_"
            + caller_identity["Account"],
            RoleSessionName="SecurePypiReadRole",
        )
        credentials = assumed_role_object["Credentials"]
        code_artifact_client = boto3.client(
            "codeartifact",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name="us-west-2",
        )
        token = code_artifact_client.get_authorization_token(
            domain="amazon", domainOwner="149122183214"
        )["authorizationToken"]

        logger.info("Successfully retrieved secure PyPI access token")
        return token

    except Exception as e:
        logger.error(f"Failed to retrieve secure PyPI access token: {e}")
        raise


def install_packages_from_public_pypi(packages: list) -> None:
    """Install packages from standard public PyPI."""
    logger.info(f"Installing {len(packages)} packages from public PyPI")
    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        logger.info("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """Install packages from secure CodeArtifact PyPI."""
    logger.info(f"Installing {len(packages)} packages from secure PyPI")
    try:
        token = _get_secure_pypi_access_token()
        index_url = f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"
        check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                index_url,
                *packages,
            ]
        )
        logger.info("✓ Successfully installed packages from secure PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from secure PyPI: {e}")
        raise


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """Install packages from PyPI source based on configuration."""
    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    logger.info(f"Number of packages: {len(packages)}")
    logger.info("=" * 70)

    try:
        if use_secure:
            install_packages_from_secure_pypi(packages)
        else:
            install_packages_from_public_pypi(packages)

        logger.info("=" * 70)
        logger.info("✓ PACKAGE INSTALLATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ PACKAGE INSTALLATION FAILED")
        logger.error("=" * 70)
        raise


# ============================================================================
# INSTALL REQUIRED PACKAGES
# ============================================================================

required_packages = [
    "scikit-learn>=0.23.2,<1.0.0",
    "pandas>=1.2.0,<2.0.0",
    "pyarrow>=4.0.0,<6.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "typing-extensions>=4.2.0",
    "matplotlib>=3.0.0",
    "scipy>=1.7.0",
    "lightgbm>=3.3.0",
]

install_packages(required_packages)

print("***********************Package Installation Complete*********************")

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import pickle as pkl
import tarfile
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

from models.loss.loss_factory import LossFactory
from models.factory.model_factory import ModelFactory
from models.base.training_state import TrainingState

from hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters

# Preprocessing imports
from processing.categorical.risk_table_processor import RiskTableMappingProcessor
from processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)


# -------------------------------------------------------------------------
# Logging setup - Updated for CloudWatch compatibility
# -------------------------------------------------------------------------
def setup_logging():
    """Configure logging for CloudWatch compatibility."""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    sys.stdout.flush()

    return logger


logger = setup_logging()


# -------------------------------------------------------------------------
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# -------------------------------------------------------------------------


def _detect_file_format(file_path: str) -> str:
    """Detect the format of a data file based on its extension."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: str) -> Tuple[pd.DataFrame, str]:
    """Load DataFrame and detect its format."""
    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: str, format_str: str
) -> str:
    """Save DataFrame in specified format."""
    output_path = Path(output_path)

    if format_str == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format_str == "tsv":
        file_path = output_path.with_suffix(".tsv")
        df.to_csv(file_path, sep="\t", index=False)
    elif format_str == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")

    return str(file_path)


def find_first_data_file(data_dir: str) -> str:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json", ".tsv")):
            return os.path.join(data_dir, fname)
    return None


# -------------------------------------------------------------------------
# PREPROCESSING ARTIFACT LOADERS
# -------------------------------------------------------------------------


def load_precomputed_artifacts(
    model_artifacts_dir: Optional[str],
    use_imputation: bool,
    use_risk_tables: bool,
    use_features: bool,
) -> Dict[str, Any]:
    """
    Auto-detect and load pre-computed artifacts from model_artifacts_input.

    Returns dictionary with loaded artifacts and status flags.
    """
    result = {
        "impute_dict": None,
        "risk_tables": None,
        "selected_features": None,
        "loaded": {"imputation": False, "risk_tables": False, "features": False},
    }

    if not model_artifacts_dir or not os.path.exists(model_artifacts_dir):
        logger.warning(f"Model artifacts directory not found: {model_artifacts_dir}")
        return result

    logger.info(f"Loading pre-computed artifacts from: {model_artifacts_dir}")

    # 1. Try to load imputation dictionary
    if use_imputation:
        impute_path = os.path.join(model_artifacts_dir, "impute_dict.pkl")
        if os.path.exists(impute_path):
            try:
                with open(impute_path, "rb") as f:
                    result["impute_dict"] = pkl.load(f)
                result["loaded"]["imputation"] = True
                logger.info(f"✓ Loaded pre-computed imputation from {impute_path}")
            except Exception as e:
                logger.warning(f"Failed to load imputation: {e}")
        else:
            logger.warning(f"Imputation dict not found at {impute_path}")

    # 2. Try to load risk tables
    if use_risk_tables:
        risk_path = os.path.join(model_artifacts_dir, "risk_table_map.pkl")
        if os.path.exists(risk_path):
            try:
                with open(risk_path, "rb") as f:
                    result["risk_tables"] = pkl.load(f)
                result["loaded"]["risk_tables"] = True
                logger.info(f"✓ Loaded pre-computed risk tables from {risk_path}")
            except Exception as e:
                logger.warning(f"Failed to load risk tables: {e}")
        else:
            logger.warning(f"Risk tables not found at {risk_path}")

    # 3. Try to load selected features
    if use_features:
        features_path = os.path.join(model_artifacts_dir, "selected_features.json")
        if os.path.exists(features_path):
            try:
                with open(features_path, "r") as f:
                    fs_data = json.load(f)
                result["selected_features"] = fs_data.get("selected_features", [])
                result["loaded"]["features"] = True
                logger.info(f"✓ Loaded pre-computed features from {features_path}")
            except Exception as e:
                logger.warning(f"Failed to load features: {e}")
        else:
            logger.warning(f"Features not found at {features_path}")

    return result


def validate_precomputed_data_state(
    train_df: pd.DataFrame,
    hyperparams: LightGBMMtModelHyperparameters,
    imputation_used: bool,
    risk_tables_used: bool,
) -> None:
    """Validate that data state matches the pre-computed artifact flags."""
    if imputation_used:
        tab_fields = hyperparams.tab_field_list
        if tab_fields:
            nan_cols = (
                train_df[tab_fields].columns[train_df[tab_fields].isna().any()].tolist()
            )
            if nan_cols:
                raise ValueError(
                    f"USE_PRECOMPUTED_IMPUTATION=true but data contains NaN in: {nan_cols}. "
                    "Data must be pre-imputed when using pre-computed imputation artifacts."
                )
            logger.info(
                "✓ Validated: No NaN values (consistent with pre-computed imputation)"
            )

    if risk_tables_used:
        cat_fields = hyperparams.cat_field_list
        for col in cat_fields:
            if col in train_df.columns:
                if not pd.api.types.is_numeric_dtype(train_df[col]):
                    raise ValueError(
                        f"USE_PRECOMPUTED_RISK_TABLES=true but column '{col}' is not numeric. "
                        "Data must be pre-transformed when using pre-computed risk table artifacts."
                    )
        if cat_fields:
            logger.info(
                "✓ Validated: Categorical columns are numeric (consistent with pre-computed risk tables)"
            )


# -------------------------------------------------------------------------
# DATA LOADING WITH FORMAT DETECTION
# -------------------------------------------------------------------------


def load_datasets(
    input_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Loads training, validation, and test datasets with format detection.

    Returns: (train_df, val_df, test_df, detected_format)
    """
    train_file = find_first_data_file(os.path.join(input_path, "train"))
    val_file = find_first_data_file(os.path.join(input_path, "val"))
    test_file = find_first_data_file(os.path.join(input_path, "test"))

    if not train_file or not val_file:
        raise FileNotFoundError(
            "Training or validation data file not found in expected subfolders."
        )

    train_df, train_format = load_dataframe_with_format(train_file)
    val_df, val_format = load_dataframe_with_format(val_file)

    test_df = None
    if test_file:
        test_df, test_format = load_dataframe_with_format(test_file)

    detected_format = train_format
    logger.info(f"Detected input format: {detected_format}")

    if test_df is not None:
        logger.info(
            f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
        )
    else:
        logger.info(
            f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: None"
        )

    return train_df, val_df, test_df, detected_format


# -------------------------------------------------------------------------
# PREPROCESSING FUNCTIONS
# -------------------------------------------------------------------------


def apply_numerical_imputation(
    hyperparams: LightGBMMtModelHyperparameters,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
) -> tuple:
    """
    Applies numerical imputation using single-column architecture.

    Returns: (train_df_imputed, val_df_imputed, test_df_imputed, impute_dict)
    """
    imputation_processors = {}
    train_df_imputed = train_df.copy()
    val_df_imputed = val_df.copy()
    test_df_imputed = test_df.copy() if test_df is not None else None

    for var in hyperparams.tab_field_list:
        proc = NumericalVariableImputationProcessor(column_name=var, strategy="mean")
        proc.fit(train_df[var])
        imputation_processors[var] = proc

        train_df_imputed[var] = proc.transform(train_df_imputed[var])
        val_df_imputed[var] = proc.transform(val_df_imputed[var])
        if test_df_imputed is not None:
            test_df_imputed[var] = proc.transform(test_df_imputed[var])

    impute_dict = {
        var: proc.get_imputation_value() for var, proc in imputation_processors.items()
    }

    return (train_df_imputed, val_df_imputed, test_df_imputed, impute_dict)


def fit_and_apply_risk_tables(
    hyperparams: LightGBMMtModelHyperparameters,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
) -> tuple:
    """
    Fits risk tables on training data and applies to all splits.

    Returns: (train_df_transformed, val_df_transformed, test_df_transformed, risk_tables)
    """
    risk_processors = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy() if test_df is not None else None

    for var in hyperparams.cat_field_list:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=hyperparams.label_name,
            smooth_factor=0.0,  # Can be added to hyperparams if needed
            count_threshold=0,
        )
        proc.fit(train_df)
        risk_processors[var] = proc

        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
        if test_df_transformed is not None:
            test_df_transformed[var] = proc.transform(test_df_transformed[var])

    consolidated_risk_tables = {
        var: proc.get_risk_tables() for var, proc in risk_processors.items()
    }

    return (
        train_df_transformed,
        val_df_transformed,
        test_df_transformed,
        consolidated_risk_tables,
    )


# -------------------------------------------------------------------------
# MULTI-TASK SPECIFIC FUNCTIONS
# -------------------------------------------------------------------------


def identify_task_columns(
    df: pd.DataFrame, hyperparams: LightGBMMtModelHyperparameters
) -> List[str]:
    """
    Identify task label columns with priority order:
    1. hyperparams.task_label_names (explicit configuration)
    2. Auto-detection (fallback for backward compatibility)
    """
    # Priority 1: Use explicit task_label_names from hyperparameters
    if hyperparams.task_label_names:
        task_cols = hyperparams.task_label_names
        logger.info(f"✓ Using task_label_names from hyperparameters: {task_cols}")

        # Validate all columns exist in dataframe
        missing = set(task_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Task label columns specified in hyperparameters not found in data: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

        return task_cols

    # Priority 2: Auto-detection (backward compatibility fallback)
    logger.warning(
        "task_label_names not specified in hyperparameters, using auto-detection"
    )

    # Strategy 1: Look for columns starting with 'task_'
    task_cols = [col for col in df.columns if col.startswith("task_")]

    if not task_cols:
        # Strategy 2: Look for common fraud task patterns
        fraud_patterns = [
            "isFraud",
            "isCCfrd",
            "isDDfrd",
            "isGCfrd",
            "isLOCfrd",
            "isCimfrd",
        ]
        task_cols = [col for col in df.columns if col in fraud_patterns]

    if not task_cols:
        raise ValueError(
            "Could not auto-detect task columns. Expected columns starting with 'task_' or common fraud patterns. "
            "Please specify 'task_label_names' explicitly in hyperparameters."
        )

    logger.info(f"Auto-detected {len(task_cols)} task columns: {task_cols}")

    # Validate against num_tasks if provided
    if hyperparams.num_tasks is not None and len(task_cols) != hyperparams.num_tasks:
        logger.warning(
            f"Auto-detected {len(task_cols)} task columns but num_tasks={hyperparams.num_tasks}. "
            f"Using detected columns: {task_cols}"
        )

    return task_cols


def create_task_indices(
    train_df: pd.DataFrame, val_df: pd.DataFrame, task_columns: List[str]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Create task-specific indices for main task and subtasks."""
    trn_sublabel_idx = {}
    val_sublabel_idx = {}

    for i, task_col in enumerate(task_columns):
        trn_sublabel_idx[i] = np.where(train_df[task_col] == 1)[0]
        val_sublabel_idx[i] = np.where(val_df[task_col] == 1)[0]

    logger.info(f"Created indices for {len(task_columns)} tasks:")
    for i in range(len(task_columns)):
        logger.info(
            f"  Task {i} ({task_columns[i]}): "
            f"train_pos={len(trn_sublabel_idx[i])}, val_pos={len(val_sublabel_idx[i])}"
        )

    return trn_sublabel_idx, val_sublabel_idx


# -------------------------------------------------------------------------
# MULTI-TASK INFERENCE & EVALUATION
# -------------------------------------------------------------------------


def predict_multitask(
    model, df: pd.DataFrame, feature_columns: List[str]
) -> np.ndarray:
    """
    Generate multi-task predictions.

    Returns: np.ndarray of shape (n_samples, n_tasks) with probabilities
    """
    X = df[feature_columns]
    predictions = model.predict(X)
    return predictions


def compute_multitask_metrics(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_columns: List[str],
) -> Dict[str, Any]:
    """Compute per-task and aggregate metrics."""
    metrics = {}
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

            metrics[f"task_{i}_{task_name}"] = {
                "auc_roc": float(auc_roc),
                "average_precision": float(ap),
                "f1_score": float(f1),
            }

            auc_rocs.append(auc_roc)
            aps.append(ap)
            f1s.append(f1)

        except ValueError as e:
            logger.warning(f"Task {i} ({task_name}): {e}")
            metrics[f"task_{i}_{task_name}"] = {
                "auc_roc": 0.5,
                "average_precision": 0.5,
                "f1_score": 0.0,
            }

    # Aggregate metrics
    if auc_rocs:
        metrics["aggregate"] = {
            "mean_auc_roc": float(np.mean(auc_rocs)),
            "median_auc_roc": float(np.median(auc_rocs)),
            "mean_average_precision": float(np.mean(aps)),
            "median_average_precision": float(np.median(aps)),
            "mean_f1_score": float(np.mean(f1s)),
            "median_f1_score": float(np.median(f1s)),
        }

    return metrics


def plot_multitask_curves(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_columns: List[str],
    out_dir: str,
    prefix: str,
) -> None:
    """Generate ROC and PR curves for each task."""
    os.makedirs(out_dir, exist_ok=True)

    for i, task_name in enumerate(task_columns):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        if len(np.unique(y_true)) < 2:
            logger.warning(
                f"Task {i} ({task_name}): Only one class present, skipping plots"
            )
            continue

        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.title(f"{prefix}Task {i} ({task_name}) ROC")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"{prefix}task_{i}_{task_name}_roc.jpg"))
            plt.close()

            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)

            plt.figure()
            plt.plot(recall, precision, label=f"AP={ap:.3f}")
            plt.title(f"{prefix}Task {i} ({task_name}) PR")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"{prefix}task_{i}_{task_name}_pr.jpg"))
            plt.close()

        except Exception as e:
            logger.warning(f"Error plotting task {i} ({task_name}): {e}")


def evaluate_split_multitask(
    name: str,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_columns: List[str],
    model,
    hyperparams: LightGBMMtModelHyperparameters,
    output_format: str = "csv",
    prefix: str = "/opt/ml/output/data",
) -> None:
    """Evaluate a data split for multi-task learning."""
    logger.info(f"Evaluating {name} split...")

    # Extract task labels
    y_true_tasks = {}
    for i, task_col in enumerate(task_columns):
        y_true_tasks[i] = df[task_col].astype(int).values

    # Get predictions
    y_pred_tasks = predict_multitask(model, df, feature_columns)

    # Compute metrics
    metrics = compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_columns)

    # Save predictions
    out_base = os.path.join(prefix, name)
    os.makedirs(out_base, exist_ok=True)

    # Build predictions DataFrame
    id_col = hyperparams.id_name
    ids = df.get(id_col, np.arange(len(df)))

    pred_df = pd.DataFrame({id_col: ids})
    for i, task_col in enumerate(task_columns):
        pred_df[f"{task_col}_true"] = y_true_tasks[i]
        pred_df[f"{task_col}_prob"] = y_pred_tasks[:, i]

    output_base = os.path.join(out_base, "predictions")
    saved_path = save_dataframe_with_format(pred_df, output_base, output_format)
    logger.info(f"Saved predictions (format={output_format}): {saved_path}")

    # Save metrics
    metrics_file = os.path.join(out_base, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_file}")

    # Generate plots
    out_metrics = os.path.join(prefix, f"{name}_metrics")
    plot_multitask_curves(
        y_true_tasks, y_pred_tasks, task_columns, out_metrics, f"{name}_"
    )

    # Package into tar.gz
    tar_path = os.path.join(prefix, f"{name}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(out_base, arcname=name)
        t.add(out_metrics, arcname=f"{name}_metrics")

    logger.info(f"{name} outputs packaged → {tar_path}")


# -------------------------------------------------------------------------
# MODEL ARTIFACT SAVING
# -------------------------------------------------------------------------


def save_artifacts(
    model,
    risk_tables: dict,
    impute_dict: dict,
    model_path: str,
    feature_columns: List[str],
    hyperparams: LightGBMMtModelHyperparameters,
    training_state: TrainingState,
) -> None:
    """Saves trained model and all preprocessing artifacts."""
    os.makedirs(model_path, exist_ok=True)

    # 1. Save LightGBM model
    model_file = os.path.join(model_path, "lightgbmmt_model.txt")
    model.save(model_file)
    logger.info(f"Saved LightGBMMT model to {model_file}")

    # 2. Save risk tables
    risk_map_file = os.path.join(model_path, "risk_table_map.pkl")
    with open(risk_map_file, "wb") as f:
        pkl.dump(risk_tables, f)
    logger.info(f"Saved risk table map to {risk_map_file}")

    # 3. Save imputation dictionary
    impute_file = os.path.join(model_path, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")

    # 4. Save training state (for checkpointing)
    state_file = os.path.join(model_path, "training_state.json")
    with open(state_file, "w") as f:
        json.dump(training_state.to_checkpoint_dict(), f, indent=2)
    logger.info(f"Saved training state to {state_file}")

    # 5. Save feature columns with ordering
    feature_columns_file = os.path.join(model_path, "feature_columns.txt")
    with open(feature_columns_file, "w") as f:
        f.write("# Feature columns in exact order required for model inference\n")
        f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
        f.write("# Each line contains: <column_index>,<column_name>\n")
        for idx, column in enumerate(feature_columns):
            f.write(f"{idx},{column}\n")
    logger.info(f"Saved feature columns to {feature_columns_file}")

    # 6. Save hyperparameters
    hyperparams_file = os.path.join(model_path, "hyperparameters.json")
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams.model_dump(), f, indent=2, sort_keys=True)
    logger.info(f"Saved hyperparameters to {hyperparams_file}")

    # 7. Save weight evolution (multi-task specific)
    if training_state.weight_evolution:
        weight_file = os.path.join(model_path, "weight_evolution.json")
        with open(weight_file, "w") as f:
            weight_evolution_list = [
                w.tolist() for w in training_state.weight_evolution
            ]
            json.dump(weight_evolution_list, f, indent=2)
        logger.info(f"Saved weight evolution to {weight_file}")


# -------------------------------------------------------------------------
# MAIN TRAINING ORCHESTRATOR
# -------------------------------------------------------------------------


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
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    try:
        logger.info("====== STARTING MAIN EXECUTION ======")

        # Extract paths
        data_dir = input_paths["input_path"]
        model_dir = output_paths["model_output"]
        output_dir = output_paths["evaluation_output"]
        model_artifacts_input_dir = input_paths.get("model_artifacts_input")

        # Priority-based hyperparameters loading
        hparam_path = "/opt/ml/code/hyperparams/hyperparameters.json"
        if not os.path.exists(hparam_path):
            if "hyperparameters_s3_uri" in input_paths:
                hparam_path = input_paths["hyperparameters_s3_uri"]
                if not hparam_path.endswith("hyperparameters.json"):
                    hparam_path = os.path.join(hparam_path, "hyperparameters.json")

        logger.info(f"Loading configuration from {hparam_path}")
        with open(hparam_path, "r") as f:
            hyperparams_dict = json.load(f)
        hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)

        # Load datasets with format detection
        logger.info("Loading datasets...")
        train_df, val_df, test_df, input_format = load_datasets(data_dir)

        # Extract environment variables for preprocessing control
        use_precomputed_imputation = environ_vars.get(
            "USE_PRECOMPUTED_IMPUTATION", False
        )
        use_precomputed_risk_tables = environ_vars.get(
            "USE_PRECOMPUTED_RISK_TABLES", False
        )
        use_precomputed_features = environ_vars.get("USE_PRECOMPUTED_FEATURES", False)

        logger.info("=" * 70)
        logger.info("PREPROCESSING ARTIFACT CONTROL")
        logger.info("=" * 70)
        logger.info(f"USE_PRECOMPUTED_IMPUTATION: {use_precomputed_imputation}")
        logger.info(f"USE_PRECOMPUTED_RISK_TABLES: {use_precomputed_risk_tables}")
        logger.info(f"USE_PRECOMPUTED_FEATURES: {use_precomputed_features}")
        logger.info("=" * 70)

        # Try to load pre-computed artifacts
        precomputed = load_precomputed_artifacts(
            model_artifacts_input_dir,
            use_precomputed_imputation,
            use_precomputed_risk_tables,
            use_precomputed_features,
        )

        # Validate data state
        validate_precomputed_data_state(
            train_df,
            hyperparams,
            precomputed["loaded"]["imputation"],
            precomputed["loaded"]["risk_tables"],
        )

        # ===== 1. Numerical Imputation =====
        if precomputed["loaded"]["imputation"]:
            impute_dict = precomputed["impute_dict"]
            logger.info("✓ Using pre-computed imputation (data already transformed)")
        else:
            logger.info("Computing numerical imputation inline...")
            train_df, val_df, test_df, impute_dict = apply_numerical_imputation(
                hyperparams, train_df, val_df, test_df
            )
            logger.info("✓ Numerical imputation completed")

        # ===== 2. Risk Table Mapping =====
        if precomputed["loaded"]["risk_tables"]:
            risk_tables = precomputed["risk_tables"]
            logger.info("✓ Using pre-computed risk tables (data already transformed)")
        else:
            logger.info("Computing risk tables inline...")
            train_df, val_df, test_df, risk_tables = fit_and_apply_risk_tables(
                hyperparams, train_df, val_df, test_df
            )
            logger.info("✓ Risk table mapping completed")

        # ===== 3. Feature Columns =====
        feature_columns = hyperparams.tab_field_list + hyperparams.cat_field_list
        logger.info(f"Using {len(feature_columns)} features: {feature_columns}")

        # ===== 4. Identify Task Columns =====
        logger.info("Identifying task columns...")
        task_columns = identify_task_columns(train_df, hyperparams)

        # num_tasks is now automatically derived from len(task_label_names)
        logger.info(
            f"Number of tasks: {hyperparams.num_tasks} (derived from task_label_names)"
        )

        # ===== 5. Create Task Indices =====
        logger.info("Creating task-specific indices...")
        trn_sublabel_idx, val_sublabel_idx = create_task_indices(
            train_df, val_df, task_columns
        )

        # ===== 6. Create Loss Function =====
        logger.info("Initializing loss function...")
        loss_fn = LossFactory.create(
            loss_type=hyperparams.loss_type,
            num_label=len(task_columns),
            val_sublabel_idx=val_sublabel_idx,
            trn_sublabel_idx=trn_sublabel_idx,
            hyperparams=hyperparams,
        )

        # ===== 7. Create Training State =====
        training_state = TrainingState()

        # ===== 8. Create Model =====
        logger.info("Creating model...")
        model = ModelFactory.create(
            model_type="mtgbm",
            loss_function=loss_fn,
            training_state=training_state,
            hyperparams=hyperparams,
        )

        # ===== 9. Train Model =====
        logger.info("Training model...")
        results = model.train(train_df, val_df, test_df)

        # ===== 10. Save Model Artifacts =====
        logger.info("Saving model artifacts...")
        save_artifacts(
            model=model,
            risk_tables=risk_tables,
            impute_dict=impute_dict,
            model_path=model_dir,
            feature_columns=feature_columns,
            hyperparams=hyperparams,
            training_state=training_state,
        )

        # ===== 11. Evaluation =====
        logger.info("====== STARTING EVALUATION PHASE ======")

        # Validation evaluation
        logger.info("Starting inference & evaluation on validation set")
        try:
            evaluate_split_multitask(
                "val",
                val_df,
                feature_columns,
                task_columns,
                model,
                hyperparams,
                input_format,
                output_dir,
            )
            logger.info("✓ Validation evaluation completed")
        except Exception as e:
            logger.error(f"ERROR in validation evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        # Test evaluation (if available)
        if test_df is not None:
            logger.info("Starting inference & evaluation on test set")
            try:
                evaluate_split_multitask(
                    "test",
                    test_df,
                    feature_columns,
                    task_columns,
                    model,
                    hyperparams,
                    input_format,
                    output_dir,
                )
                logger.info("✓ Test evaluation completed")
            except Exception as e:
                logger.error(f"ERROR in test evaluation: {str(e)}")
                logger.error(traceback.format_exc())

        logger.info("====== MAIN EXECUTION COMPLETED SUCCESSFULLY ======")

    except Exception as e:
        logger.error(f"FATAL ERROR in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# -------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Script starting...")

    # Container path constants
    CONTAINER_PATHS = {
        "INPUT_DATA": "/opt/ml/input/data",
        "MODEL_DIR": "/opt/ml/model",
        "OUTPUT_DATA": "/opt/ml/output/data",
        "CONFIG_DIR": "/opt/ml/code/hyperparams",
        "MODEL_ARTIFACTS_INPUT": "/opt/ml/input/data/model_artifacts_input",
    }

    # Define input and output paths using contract logical names
    input_paths = {
        "input_path": CONTAINER_PATHS["INPUT_DATA"],
        "hyperparameters_s3_uri": CONTAINER_PATHS["CONFIG_DIR"],
    }

    # Add model_artifacts_input only if the directory exists (optional)
    if os.path.exists(CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]):
        input_paths["model_artifacts_input"] = CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]
        logger.info(
            f"Found model artifacts input directory: {CONTAINER_PATHS['MODEL_ARTIFACTS_INPUT']}"
        )

    output_paths = {
        "model_output": CONTAINER_PATHS["MODEL_DIR"],
        "evaluation_output": CONTAINER_PATHS["OUTPUT_DATA"],
    }

    # Collect environment variables for preprocessing artifact control
    environ_vars = {
        "USE_PRECOMPUTED_IMPUTATION": os.environ.get(
            "USE_PRECOMPUTED_IMPUTATION", "false"
        ).lower()
        == "true",
        "USE_PRECOMPUTED_RISK_TABLES": os.environ.get(
            "USE_PRECOMPUTED_RISK_TABLES", "false"
        ).lower()
        == "true",
        "USE_PRECOMPUTED_FEATURES": os.environ.get(
            "USE_PRECOMPUTED_FEATURES", "false"
        ).lower()
        == "true",
    }

    # Create empty args namespace to maintain function signature
    args = argparse.Namespace()

    try:
        logger.info(f"Starting main process with paths:")
        logger.info(f"  Data directory: {input_paths['input_path']}")
        logger.info(f"  Config directory: {input_paths['hyperparameters_s3_uri']}")
        logger.info(f"  Model directory: {output_paths['model_output']}")
        logger.info(f"  Output directory: {output_paths['evaluation_output']}")

        # Call the main function
        main(input_paths, output_paths, environ_vars, args)

        logger.info("LightGBMMT training script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
