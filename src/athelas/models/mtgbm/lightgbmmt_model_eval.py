#!/usr/bin/env python
"""
LightGBMMT Multi-Task Model Evaluation Script

Evaluates trained LightGBMMT models on evaluation datasets.
Generates per-task and aggregate metrics, predictions, and visualizations.
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
    "pyarrow>=10.0.0",
    "lightgbm>=3.3.0",
]

install_packages(required_packages)

print("***********************Package Installation Complete*********************")

# Now import packages after installation
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import time
import tarfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

# Model factory imports
from models.factory.model_factory import ModelFactory
from models.base.training_state import TrainingState
from hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters

# Embedded processor classes to remove external dependencies


class RiskTableMappingProcessor:
    """
    A processor that performs risk-table-based mapping on a specified categorical variable.
    The 'process' method (called via __call__) handles single values.
    The 'transform' method handles pandas Series or DataFrames.
    """

    def __init__(
        self,
        column_name: str,
        label_name: str,
        smooth_factor: float = 0.0,
        count_threshold: int = 0,
        risk_tables: Optional[Dict] = None,
    ):
        """
        Initialize RiskTableMappingProcessor.

        Args:
            column_name: Name of the categorical column to be binned.
            label_name: Name of label/target variable (expected to be binary 0 or 1).
            smooth_factor: Smoothing factor for risk calculation (0 to 1).
            count_threshold: Minimum count for considering a category's calculated risk.
            risk_tables: Optional pre-computed risk tables.
        """
        self.processor_name = "risk_table_mapping_processor"
        self.function_name_list = ["process", "transform", "fit"]

        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name must be a non-empty string.")
        self.column_name = column_name
        self.label_name = label_name
        self.smooth_factor = smooth_factor
        self.count_threshold = count_threshold

        self.is_fitted = False
        if risk_tables:
            self._validate_risk_tables(risk_tables)
            self.risk_tables = risk_tables
            self.is_fitted = True
        else:
            self.risk_tables = {}

    def get_name(self) -> str:
        return self.processor_name

    def _validate_risk_tables(self, risk_tables: Dict) -> None:
        if not isinstance(risk_tables, dict):
            raise ValueError("Risk tables must be a dictionary.")
        if "bins" not in risk_tables or "default_bin" not in risk_tables:
            raise ValueError("Risk tables must contain 'bins' and 'default_bin' keys.")
        if not isinstance(risk_tables["bins"], dict):
            raise ValueError("Risk tables 'bins' must be a dictionary.")
        if not isinstance(
            risk_tables["default_bin"], (int, float, np.floating, np.integer)
        ):
            raise ValueError(
                f"Risk tables 'default_bin' must be a number, got {type(risk_tables['default_bin'])}."
            )

    def set_risk_tables(self, risk_tables: Dict) -> None:
        self._validate_risk_tables(risk_tables)
        self.risk_tables = risk_tables
        self.is_fitted = True

    def process(self, input_value: Any) -> float:
        """
        Process a single input value (for the configured 'column_name'),
        mapping it to its binned risk value.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before processing."
            )
        str_value = str(input_value)
        return self.risk_tables["bins"].get(str_value, self.risk_tables["default_bin"])

    def transform(
        self, data: Union[pd.DataFrame, pd.Series, Any]
    ) -> Union[pd.DataFrame, pd.Series, float]:
        """
        Transform data using the computed risk tables.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before transforming."
            )

        if isinstance(data, pd.DataFrame):
            if self.column_name not in data.columns:
                raise ValueError(
                    f"Column '{self.column_name}' not found in input DataFrame for transform operation."
                )
            output_data = data.copy()
            output_data[self.column_name] = (
                data[self.column_name]
                .astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
            return output_data
        elif isinstance(data, pd.Series):
            return (
                data.astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
        else:
            return self.process(data)

    def get_risk_tables(self) -> Dict:
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor has not been fitted or initialized with risk tables."
            )
        return self.risk_tables


class NumericalVariableImputationProcessor:
    """
    A processor that performs imputation on numerical variables using predefined or computed values.
    Supports mean, median, and mode imputation strategies.
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        imputation_dict: Optional[Dict[str, Union[int, float]]] = None,
        strategy: str = "mean",
    ):
        self.processor_name = "numerical_variable_imputation_processor"
        self.function_name_list = ["fit", "process", "transform"]

        self.variables = variables
        self.strategy = strategy
        self.is_fitted = False

        if imputation_dict:
            self._validate_imputation_dict(imputation_dict)
            self.imputation_dict = imputation_dict
            self.is_fitted = True
        else:
            self.imputation_dict = None

    def get_name(self) -> str:
        return self.processor_name

    def __call__(self, input_data):
        return self.process(input_data)

    def _validate_imputation_dict(self, imputation_dict: Dict[str, Any]) -> None:
        if not isinstance(imputation_dict, dict):
            raise ValueError("imputation_dict must be a dictionary")
        if not imputation_dict:
            raise ValueError("imputation_dict cannot be empty")
        for k, v in imputation_dict.items():
            if not isinstance(k, str):
                raise ValueError(f"All keys must be strings, got {type(k)} for key {k}")
            if not isinstance(v, (int, float, np.number)):
                raise ValueError(
                    f"All values must be numeric, got {type(v)} for key {k}"
                )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        output_data = input_data.copy()
        for var, value in input_data.items():
            if var in self.imputation_dict and pd.isna(value):
                output_data[var] = self.imputation_dict[var]
        return output_data

    def transform(
        self, X: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform input data by imputing missing values.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        # Handle Series input
        if isinstance(X, pd.Series):
            if X.name not in self.imputation_dict:
                raise ValueError(f"No imputation value found for series name: {X.name}")
            return X.fillna(self.imputation_dict[X.name])

        # Handle DataFrame input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas Series or DataFrame")

        # Make a copy to avoid modifying the input
        df = X.copy()

        # Apply imputation only to variables in imputation_dict and only to NaN values
        for var, impute_value in self.imputation_dict.items():
            if var in df.columns:
                # Create mask for NaN values
                nan_mask = df[var].isna()
                # Only replace NaN values
                df.loc[nan_mask, var] = impute_value

        return df

    def get_params(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "imputation_dict": self.imputation_dict,
            "strategy": self.strategy,
        }


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/eval",
    "OUTPUT_METRICS_DIR": "/opt/ml/processing/output/metrics",
    "OUTPUT_PLOTS_DIR": "/opt/ml/processing/output/plots",
}


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# ============================================================================


def _detect_file_format(file_path: Path) -> str:
    """Detect the format of a data file based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: Path) -> Tuple[pd.DataFrame, str]:
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
    df: pd.DataFrame, output_path: Path, format_str: str
) -> Path:
    """Save DataFrame in specified format."""
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

    return file_path


def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract tar file, preventing path traversal attacks (zip slip).

    Validates that each member's extracted path stays within the target directory.

    Args:
        tar: Open TarFile object
        path: Target extraction directory

    Raises:
        ValueError: If a member would extract outside the target directory
    """

    def is_within_directory(directory: str, target: str) -> bool:
        """Check if target path is within directory."""
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise ValueError(f"Attempted path traversal in tar file: {member.name}")

    # If all paths are safe, extract
    tar.extractall(path=path)


def decompress_model_artifacts(model_dir: str):
    """Extract model.tar.gz if it exists with path traversal protection."""
    model_tar_path = Path(model_dir) / "model.tar.gz"
    if model_tar_path.exists():
        logger.info(f"Found model.tar.gz at {model_tar_path}. Extracting...")
        with tarfile.open(model_tar_path, "r:gz") as tar:
            safe_extract_tar(tar, model_dir)
        logger.info("Extraction complete.")
    else:
        logger.info("No model.tar.gz found. Assuming artifacts are directly available.")


# ============================================================================
# MULTI-TASK LABEL PARSING
# ============================================================================


def parse_task_label_names(env_value: str) -> List[str]:
    """
    Parse TASK_LABEL_NAMES from environment variable.

    Supports:
    - Comma-separated: "isFraud,isCCfrd,isDDfrd"
    - JSON array: '["isFraud","isCCfrd","isDDfrd"]'

    Args:
        env_value: Environment variable value

    Returns:
        List of task label names
    """
    if not env_value or env_value.strip() == "":
        raise ValueError("TASK_LABEL_NAMES environment variable is empty")

    # Try JSON format first
    if env_value.strip().startswith("["):
        try:
            task_names = json.loads(env_value)
            if not isinstance(task_names, list):
                raise ValueError("JSON value must be an array")
            return [str(t).strip() for t in task_names]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for TASK_LABEL_NAMES: {e}")

    # Comma-separated format
    task_names = [t.strip() for t in env_value.split(",") if t.strip()]
    if not task_names:
        raise ValueError("TASK_LABEL_NAMES contains no valid task names")

    return task_names


# ============================================================================
# MODEL ARTIFACT LOADING
# ============================================================================


def load_model_artifacts(
    model_dir: str,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Load trained LightGBMMT model using MtgbmModel wrapper.

    Returns: model, risk_tables, impute_dict, feature_columns, hyperparameters
    """
    logger.info(f"Loading model artifacts from {model_dir}")

    # Decompress if needed
    decompress_model_artifacts(model_dir)

    # Load preprocessing artifacts
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    logger.info("Loaded risk_table_map.pkl")

    with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
        impute_dict = pkl.load(f)
    logger.info("Loaded impute_dict.pkl")

    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [
            line.strip().split(",")[1] for line in f if not line.startswith("#")
        ]
    logger.info(f"Loaded feature_columns.txt: {len(feature_columns)} features")

    # Load hyperparameters
    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams_dict = json.load(f)
    hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)
    logger.info("Loaded hyperparameters.json")

    # Create model using factory (NO loss_function or training_state for inference)
    model = ModelFactory.create(
        model_type="mtgbm",
        loss_function=None,  # Not needed for inference
        training_state=None,  # Not needed for inference
        hyperparams=hyperparams,
    )

    # Load model artifacts
    model.load(model_dir)
    logger.info(f"Loaded model with {hyperparams.num_tasks} tasks")

    return model, risk_tables, impute_dict, feature_columns, hyperparams_dict


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def preprocess_eval_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_tables: Dict[str, Any],
    impute_dict: Dict[str, Any],
) -> pd.DataFrame:
    """Apply risk table mapping and numerical imputation to evaluation data."""
    result_df = df.copy()

    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(
        f"Found {len(available_features)} out of {len(feature_columns)} expected features"
    )

    # Risk table mapping
    logger.info("Applying risk table mapping")
    for feature, risk_table in risk_tables.items():
        if feature in available_features:
            proc = RiskTableMappingProcessor(
                column_name=feature, label_name="label", risk_tables=risk_table
            )
            result_df[feature] = proc.transform(df[feature])

    # Numerical imputation
    logger.info("Applying numerical imputation")
    feature_df = result_df[available_features].copy()
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    imputed_df = imputer.transform(feature_df)
    for col in available_features:
        if col in imputed_df:
            result_df[col] = imputed_df[col]

    # Ensure numeric
    result_df[available_features] = (
        result_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    logger.info(f"Preprocessed data shape: {result_df.shape}")
    return result_df


def load_eval_data(eval_data_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load evaluation data from directory."""
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".tsv", ".parquet"]
        ]
    )
    if not eval_files:
        raise RuntimeError("No eval data file found in eval_data input.")

    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")

    df, input_format = load_dataframe_with_format(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}, format: {input_format}")
    return df, input_format


def get_id_column(df: pd.DataFrame, id_field: str) -> str:
    """Determine ID column."""
    id_col = id_field if id_field in df.columns else df.columns[0]
    logger.info(f"Using id_col: {id_col}")
    return id_col


# ============================================================================
# MULTI-TASK INFERENCE
# ============================================================================


def predict_multitask(
    model: Any, df: pd.DataFrame, feature_columns: List[str]
) -> np.ndarray:
    """
    Generate multi-task predictions using MtgbmModel wrapper.

    Returns: np.ndarray of shape (n_samples, n_tasks) with probabilities
    """
    # Pass full DataFrame - model handles feature extraction
    predictions = model.predict(df, feature_columns)

    logger.info(f"Generated predictions shape: {predictions.shape}")
    return predictions


# ============================================================================
# MULTI-TASK MODEL COMPARISON
# ============================================================================


def compute_task_comparison_metrics(
    y_true: np.ndarray,
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    task_name: str,
) -> Dict[str, float]:
    """
    Compute comparison metrics between new and previous model for a single task.
    Pattern from xgboost_model_eval.py compute_comparison_metrics
    """
    logger.info(f"Computing comparison metrics for task: {task_name}")

    comparison_metrics = {}

    # Correlation metrics
    try:
        pearson_corr, pearson_p = pearsonr(y_new_score, y_prev_score)
        spearman_corr, spearman_p = spearmanr(y_new_score, y_prev_score)
    except (TypeError, AttributeError) as e:
        logger.warning(f"SciPy correlation failed: {e}. Using numpy fallback.")
        pearson_corr = float(np.corrcoef(y_new_score, y_prev_score)[0, 1])
        pearson_p = np.nan
        spearman_corr = pearson_corr
        spearman_p = np.nan

    comparison_metrics.update(
        {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
        }
    )

    # Performance comparison
    new_auc = roc_auc_score(y_true, y_new_score)
    prev_auc = roc_auc_score(y_true, y_prev_score)
    new_ap = average_precision_score(y_true, y_new_score)
    prev_ap = average_precision_score(y_true, y_prev_score)

    comparison_metrics.update(
        {
            "new_model_auc": new_auc,
            "previous_model_auc": prev_auc,
            "auc_delta": new_auc - prev_auc,
            "auc_lift_percent": ((new_auc - prev_auc) / prev_auc) * 100
            if prev_auc > 0
            else 0,
            "new_model_ap": new_ap,
            "previous_model_ap": prev_ap,
            "ap_delta": new_ap - prev_ap,
            "ap_lift_percent": ((new_ap - prev_ap) / prev_ap) * 100
            if prev_ap > 0
            else 0,
        }
    )

    # F1 comparison at thresholds
    for threshold in [0.3, 0.5, 0.7]:
        new_f1 = f1_score(y_true, (y_new_score >= threshold).astype(int))
        prev_f1 = f1_score(y_true, (y_prev_score >= threshold).astype(int))
        comparison_metrics[f"new_model_f1_at_{threshold}"] = new_f1
        comparison_metrics[f"previous_model_f1_at_{threshold}"] = prev_f1
        comparison_metrics[f"f1_delta_at_{threshold}"] = new_f1 - prev_f1

    # Score distribution comparison
    comparison_metrics.update(
        {
            "new_score_mean": float(np.mean(y_new_score)),
            "previous_score_mean": float(np.mean(y_prev_score)),
            "new_score_std": float(np.std(y_new_score)),
            "previous_score_std": float(np.std(y_prev_score)),
            "score_mean_delta": float(np.mean(y_new_score) - np.mean(y_prev_score)),
        }
    )

    # Agreement metrics
    for threshold in [0.3, 0.5, 0.7]:
        new_pred = (y_new_score >= threshold).astype(int)
        prev_pred = (y_prev_score >= threshold).astype(int)
        agreement = np.mean(new_pred == prev_pred)
        comparison_metrics[f"prediction_agreement_at_{threshold}"] = agreement

    logger.info(
        f"Task {task_name}: AUC delta={comparison_metrics['auc_delta']:.4f}, "
        f"Correlation={comparison_metrics['pearson_correlation']:.4f}"
    )

    return comparison_metrics


def perform_task_statistical_tests(
    y_true: np.ndarray,
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    task_name: str,
) -> Dict[str, float]:
    """
    Perform statistical tests for a single task.
    Pattern from xgboost_model_eval.py perform_statistical_tests
    """
    logger.info(f"Performing statistical tests for task: {task_name}")

    test_results = {}

    # McNemar's test
    new_pred = (y_new_score >= 0.5).astype(int)
    prev_pred = (y_prev_score >= 0.5).astype(int)

    correct_both = np.sum((new_pred == y_true) & (prev_pred == y_true))
    new_correct_prev_wrong = np.sum((new_pred == y_true) & (prev_pred != y_true))
    new_wrong_prev_correct = np.sum((new_pred != y_true) & (prev_pred == y_true))
    wrong_both = np.sum((new_pred != y_true) & (prev_pred != y_true))

    if (new_correct_prev_wrong + new_wrong_prev_correct) > 0:
        mcnemar_stat = (
            (abs(new_correct_prev_wrong - new_wrong_prev_correct) - 1) ** 2
        ) / (new_correct_prev_wrong + new_wrong_prev_correct)
        mcnemar_p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    else:
        mcnemar_stat = 0.0
        mcnemar_p_value = 1.0

    test_results.update(
        {
            "mcnemar_statistic": mcnemar_stat,
            "mcnemar_p_value": mcnemar_p_value,
            "mcnemar_significant": bool(mcnemar_p_value < 0.05),
            "correct_both": int(correct_both),
            "new_correct_prev_wrong": int(new_correct_prev_wrong),
            "new_wrong_prev_correct": int(new_wrong_prev_correct),
            "wrong_both": int(wrong_both),
        }
    )

    # Paired t-test
    t_stat, t_p_value = stats.ttest_rel(y_new_score, y_prev_score)
    test_results.update(
        {
            "paired_t_statistic": t_stat,
            "paired_t_p_value": t_p_value,
            "paired_t_significant": bool(t_p_value < 0.05),
        }
    )

    # Wilcoxon signed-rank test
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(y_new_score, y_prev_score)
        test_results.update(
            {
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "wilcoxon_significant": bool(wilcoxon_p < 0.05),
            }
        )
    except ValueError as e:
        logger.warning(f"Could not perform Wilcoxon test for task {task_name}: {e}")
        test_results.update(
            {
                "wilcoxon_statistic": np.nan,
                "wilcoxon_p_value": np.nan,
                "wilcoxon_significant": False,
            }
        )

    logger.info(
        f"Task {task_name}: McNemar p={test_results['mcnemar_p_value']:.4f}, "
        f"Paired t-test p={test_results['paired_t_p_value']:.4f}"
    )

    return test_results


def plot_task_comparison_roc(
    y_true: np.ndarray,
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """Plot comparison ROC curves for a single task."""
    logger.info(f"Creating comparison ROC curves for task: {task_name}")

    fpr_new, tpr_new, _ = roc_curve(y_true, y_new_score)
    fpr_prev, tpr_prev, _ = roc_curve(y_true, y_prev_score)

    auc_new = roc_auc_score(y_true, y_new_score)
    auc_prev = roc_auc_score(y_true, y_prev_score)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr_new, tpr_new, "b-", linewidth=2, label=f"New Model (AUC = {auc_new:.3f})"
    )
    plt.plot(
        fpr_prev,
        tpr_prev,
        "r--",
        linewidth=2,
        label=f"Previous Model (AUC = {auc_prev:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task_name} ROC Comparison (Δ AUC = {auc_new - auc_prev:+.3f})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{task_name}_comparison_roc.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison ROC to {out_path}")


def plot_task_comparison_pr(
    y_true: np.ndarray,
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """Plot comparison PR curves for a single task."""
    logger.info(f"Creating comparison PR curves for task: {task_name}")

    precision_new, recall_new, _ = precision_recall_curve(y_true, y_new_score)
    precision_prev, recall_prev, _ = precision_recall_curve(y_true, y_prev_score)

    ap_new = average_precision_score(y_true, y_new_score)
    ap_prev = average_precision_score(y_true, y_prev_score)

    plt.figure(figsize=(10, 6))
    plt.plot(
        recall_new,
        precision_new,
        "b-",
        linewidth=2,
        label=f"New Model (AP = {ap_new:.3f})",
    )
    plt.plot(
        recall_prev,
        precision_prev,
        "r--",
        linewidth=2,
        label=f"Previous Model (AP = {ap_prev:.3f})",
    )

    baseline = np.mean(y_true)
    plt.axhline(
        y=baseline,
        color="k",
        linestyle=":",
        alpha=0.6,
        label=f"Random (AP = {baseline:.3f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{task_name} PR Comparison (Δ AP = {ap_new - ap_prev:+.3f})")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{task_name}_comparison_pr.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison PR to {out_path}")


def plot_task_score_scatter(
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    y_true: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """Plot score scatter for a single task."""
    logger.info(f"Creating score scatter for task: {task_name}")

    plt.figure(figsize=(10, 8))

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    plt.scatter(
        y_prev_score[neg_mask],
        y_new_score[neg_mask],
        c="lightcoral",
        alpha=0.6,
        s=20,
        label="Negative (0)",
    )
    plt.scatter(
        y_prev_score[pos_mask],
        y_new_score[pos_mask],
        c="lightblue",
        alpha=0.6,
        s=20,
        label="Positive (1)",
    )

    min_score = min(np.min(y_prev_score), np.min(y_new_score))
    max_score = max(np.max(y_prev_score), np.max(y_new_score))
    plt.plot(
        [min_score, max_score],
        [min_score, max_score],
        "k--",
        alpha=0.8,
        label="Perfect Correlation",
    )

    try:
        correlation = pearsonr(y_new_score, y_prev_score)[0]
    except (TypeError, AttributeError):
        correlation = float(np.corrcoef(y_new_score, y_prev_score)[0, 1])

    plt.xlabel("Previous Model Score")
    plt.ylabel("New Model Score")
    plt.title(f"{task_name} Score Comparison (Correlation = {correlation:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{task_name}_score_scatter.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score scatter to {out_path}")


def plot_task_score_distributions(
    y_new_score: np.ndarray,
    y_prev_score: np.ndarray,
    y_true: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """Plot score distributions for a single task."""
    logger.info(f"Creating score distributions for task: {task_name}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    # New model distributions
    axes[0, 0].hist(
        y_new_score[neg_mask],
        bins=30,
        alpha=0.7,
        color="lightcoral",
        label="Negative (0)",
        density=True,
    )
    axes[0, 0].hist(
        y_new_score[pos_mask],
        bins=30,
        alpha=0.7,
        color="lightblue",
        label="Positive (1)",
        density=True,
    )
    axes[0, 0].set_title(f"{task_name} New Model Score Distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Previous model distributions
    axes[0, 1].hist(
        y_prev_score[neg_mask],
        bins=30,
        alpha=0.7,
        color="lightcoral",
        label="Negative (0)",
        density=True,
    )
    axes[0, 1].hist(
        y_prev_score[pos_mask],
        bins=30,
        alpha=0.7,
        color="lightblue",
        label="Positive (1)",
        density=True,
    )
    axes[0, 1].set_title(f"{task_name} Previous Model Score Distribution")
    axes[0, 1].set_xlabel("Score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Score difference
    score_diff = y_new_score - y_prev_score
    axes[1, 0].hist(
        score_diff[neg_mask],
        bins=30,
        alpha=0.7,
        color="lightcoral",
        label="Negative (0)",
        density=True,
    )
    axes[1, 0].hist(
        score_diff[pos_mask],
        bins=30,
        alpha=0.7,
        color="lightblue",
        label="Positive (1)",
        density=True,
    )
    axes[1, 0].axvline(x=0, color="black", linestyle="--", alpha=0.8)
    axes[1, 0].set_title(f"{task_name} Score Difference (New - Previous)")
    axes[1, 0].set_xlabel("Score Difference")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Box plots
    box_data = [
        y_prev_score[neg_mask],
        y_new_score[neg_mask],
        y_prev_score[pos_mask],
        y_new_score[pos_mask],
    ]
    box_labels = ["Prev (Neg)", "New (Neg)", "Prev (Pos)", "New (Pos)"]
    box_colors = ["lightcoral", "lightcoral", "lightblue", "lightblue"]

    bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1, 1].set_title(f"{task_name} Score Distribution Comparison")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{task_name}_score_distributions.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distributions to {out_path}")


def create_multitask_comparison_report(
    metrics: Dict[str, Any],
    task_names: List[str],
    output_dir: str,
) -> None:
    """Create comprehensive multi-task comparison report."""
    logger.info("Creating multi-task comparison report")

    report_path = os.path.join(output_dir, "multitask_comparison_report.txt")

    with open(report_path, "w") as f:
        f.write("MULTI-TASK MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Per-task summaries
        for i, task_name in enumerate(task_names):
            task_key = f"task_{i}_{task_name}_comparison"
            if task_key not in metrics:
                continue

            task_metrics = metrics[task_key]

            f.write(f"\nTASK: {task_name}\n")
            f.write("-" * 70 + "\n")

            # Performance
            new_auc = task_metrics.get("new_model_auc", "N/A")
            prev_auc = task_metrics.get("previous_model_auc", "N/A")
            auc_delta = task_metrics.get("auc_delta", "N/A")

            f.write(f"  New Model AUC:      {new_auc:.4f}\n")
            f.write(f"  Previous Model AUC: {prev_auc:.4f}\n")
            f.write(f"  AUC Delta:          {auc_delta:+.4f}\n\n")

            # Statistical significance
            mcnemar_p = task_metrics.get("mcnemar_p_value", "N/A")
            mcnemar_sig = task_metrics.get("mcnemar_significant", False)
            f.write(
                f"  McNemar p-value:    {mcnemar_p:.4f} {'(Significant)' if mcnemar_sig else '(Not Significant)'}\n"
            )

            # Recommendation
            if isinstance(auc_delta, (int, float)):
                if auc_delta > 0.01:
                    f.write("  ✓ NEW MODEL RECOMMENDED\n")
                elif auc_delta > -0.005:
                    f.write("  ≈ SIMILAR PERFORMANCE\n")
                else:
                    f.write("  ✗ PREVIOUS MODEL PREFERRED\n")

        # Aggregate summary
        if "aggregate_comparison" in metrics:
            agg = metrics["aggregate_comparison"]
            f.write("\n" + "=" * 70 + "\n")
            f.write("AGGREGATE SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Mean AUC Delta:     {agg.get('mean_auc_delta', 'N/A'):.4f}\n")
            f.write(f"Median AUC Delta:   {agg.get('median_auc_delta', 'N/A'):.4f}\n")
            f.write(f"Tasks Improved:     {agg.get('tasks_improved', 'N/A')}\n")
            f.write(f"Tasks Degraded:     {agg.get('tasks_degraded', 'N/A')}\n")

    logger.info(f"Saved multi-task comparison report to {report_path}")


# ============================================================================
# MULTI-TASK METRICS
# ============================================================================


def compute_multitask_metrics(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive per-task and aggregate metrics.
    Enhanced with threshold analysis matching xgboost_model_eval.py pattern.
    """
    logger.info("Computing multi-task metrics with threshold analysis")
    metrics = {}
    auc_rocs = []
    aps = []
    f1s = []
    max_f1s = []
    optimal_thresholds = []

    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        try:
            # Core metrics
            auc_roc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred > 0.5)

            # Precision-Recall curve analysis
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)

            # Threshold-based metrics (matching xgboost_model_eval.py)
            task_metrics = {
                "auc_roc": float(auc_roc),
                "average_precision": float(ap),
                "f1_score": float(f1),
                "precision_at_threshold_0.5": float(
                    precision_score(y_true, (y_pred > 0.5).astype(int))
                ),
                "recall_at_threshold_0.5": float(
                    recall_score(y_true, (y_pred > 0.5).astype(int))
                ),
            }

            # Metrics at multiple thresholds
            for threshold in [0.3, 0.5, 0.7]:
                y_pred_thresh = (y_pred >= threshold).astype(int)
                task_metrics[f"f1_score_at_{threshold}"] = float(
                    f1_score(y_true, y_pred_thresh)
                )
                task_metrics[f"precision_at_{threshold}"] = float(
                    precision_score(y_true, y_pred_thresh)
                )
                task_metrics[f"recall_at_{threshold}"] = float(
                    recall_score(y_true, y_pred_thresh)
                )

            # Max F1 score and optimal threshold
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            max_f1 = np.max(f1_scores)
            task_metrics["max_f1_score"] = float(max_f1)

            # ROC curve for optimal threshold
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = (
                roc_thresholds[optimal_idx]
                if len(roc_thresholds) > optimal_idx
                else 0.5
            )
            task_metrics["optimal_threshold"] = float(optimal_threshold)

            metrics[f"task_{i}_{task_name}"] = task_metrics

            # Collect for aggregation
            auc_rocs.append(auc_roc)
            aps.append(ap)
            f1s.append(f1)
            max_f1s.append(max_f1)
            optimal_thresholds.append(optimal_threshold)

            logger.info(
                f"Task {i} ({task_name}): AUC={auc_roc:.4f}, AP={ap:.4f}, F1={f1:.4f}, "
                f"Max F1={max_f1:.4f}, Optimal Threshold={optimal_threshold:.3f}"
            )

        except ValueError as e:
            logger.warning(f"Task {i} ({task_name}): {e}")
            metrics[f"task_{i}_{task_name}"] = {
                "auc_roc": 0.5,
                "average_precision": 0.5,
                "f1_score": 0.0,
                "max_f1_score": 0.0,
                "optimal_threshold": 0.5,
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
            "mean_max_f1_score": float(np.mean(max_f1s)),
            "median_max_f1_score": float(np.median(max_f1s)),
            "mean_optimal_threshold": float(np.mean(optimal_thresholds)),
            "median_optimal_threshold": float(np.median(optimal_thresholds)),
        }

        logger.info("Aggregate Metrics:")
        logger.info(f"  Mean AUC-ROC: {metrics['aggregate']['mean_auc_roc']:.4f}")
        logger.info(f"  Mean AP: {metrics['aggregate']['mean_average_precision']:.4f}")
        logger.info(f"  Mean F1: {metrics['aggregate']['mean_f1_score']:.4f}")
        logger.info(f"  Mean Max F1: {metrics['aggregate']['mean_max_f1_score']:.4f}")

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_multitask_curves(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    output_dir: str,
) -> None:
    """Generate ROC and PR curves for each task."""
    logger.info("Generating multi-task curves")
    os.makedirs(output_dir, exist_ok=True)

    for i, task_name in enumerate(task_names):
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
            plt.title(f"Task {i} ({task_name}) ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"task_{i}_{task_name}_roc.jpg"), dpi=300
            )
            plt.close()

            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)

            plt.figure()
            plt.plot(recall, precision, label=f"AP={ap:.3f}")
            plt.title(f"Task {i} ({task_name}) PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"task_{i}_{task_name}_pr.jpg"), dpi=300
            )
            plt.close()

            logger.info(f"Generated plots for task {i} ({task_name})")

        except Exception as e:
            logger.warning(f"Error plotting task {i} ({task_name}): {e}")


def plot_task_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """
    Plot score distribution for a single task, separated by class.
    Pattern from xgboost_model_eval.py and model_metrics_computation.py
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        y_score[y_true == 0],
        bins=50,
        alpha=0.7,
        label="Negative (0)",
        density=True,
        color="lightcoral",
    )
    plt.hist(
        y_score[y_true == 1],
        bins=50,
        alpha=0.7,
        label="Positive (1)",
        density=True,
        color="lightblue",
    )
    plt.xlabel("Prediction Score")
    plt.ylabel("Density")
    plt.title(f"{task_name} - Score Distribution by Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{task_name}_score_distribution.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distribution plot to {out_path}")


def plot_task_threshold_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_name: str,
    optimal_threshold: float,
    output_dir: str,
) -> None:
    """
    Plot F1, precision, recall vs threshold for a single task.
    Pattern from model_metrics_computation.py
    """
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            f1_scores.append(f1_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
        else:
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)
    plt.plot(thresholds, precisions, label="Precision", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", linewidth=2)
    plt.axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold ({optimal_threshold:.3f})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{task_name} - Threshold Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{task_name}_threshold_analysis.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved threshold analysis plot to {out_path}")


def plot_combined_multitask_roc(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    output_dir: str,
) -> None:
    """
    Plot all task ROC curves on a single plot.
    Pattern from model_metrics_computation.py multiclass_roc_curves
    """
    logger.info("Creating combined multi-task ROC curves")

    plt.figure(figsize=(10, 8))

    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, linewidth=2, label=f"{task_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Task ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "multitask_combined_roc_curves.jpg")
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined multi-task ROC curves to {out_path}")


def generate_comprehensive_visualizations(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    metrics: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Generate comprehensive visualizations for multi-task evaluation.
    Includes per-task and aggregate plots.
    """
    logger.info("Generating comprehensive multi-task visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Per-task visualizations
    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        if len(np.unique(y_true)) < 2:
            logger.warning(
                f"Task {task_name}: Only one class present, skipping visualizations"
            )
            continue

        try:
            # Score distribution
            plot_task_score_distribution(y_true, y_pred, task_name, output_dir)

            # Threshold analysis
            task_metrics = metrics.get(f"task_{i}_{task_name}", {})
            optimal_threshold = task_metrics.get("optimal_threshold", 0.5)
            plot_task_threshold_analysis(
                y_true, y_pred, task_name, optimal_threshold, output_dir
            )

        except Exception as e:
            logger.warning(f"Error generating visualizations for task {task_name}: {e}")

    # Combined multi-task visualization
    try:
        plot_combined_multitask_roc(y_true_tasks, y_pred_tasks, task_names, output_dir)
    except Exception as e:
        logger.warning(f"Error generating combined multi-task ROC curves: {e}")

    logger.info("Comprehensive visualization generation complete")


# ============================================================================
# OUTPUT SAVING
# ============================================================================


def save_predictions(
    ids: np.ndarray,
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    id_col: str,
    output_dir: str,
    input_format: str = "csv",
) -> None:
    """Save multi-task predictions preserving input format."""
    logger.info(f"Saving predictions to {output_dir} in {input_format} format")

    # Build predictions DataFrame
    pred_df = pd.DataFrame({id_col: ids})

    for i, task_name in enumerate(task_names):
        pred_df[task_name] = y_true_tasks[i]
        pred_df[f"{task_name}_prob"] = y_pred_tasks[:, i]

    output_base = Path(output_dir) / "eval_predictions"
    output_path = save_dataframe_with_format(pred_df, output_base, input_format)
    logger.info(f"Saved predictions (format={input_format}): {output_path}")


def save_metrics(
    metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str
) -> None:
    """Save computed metrics as JSON."""
    out_path = os.path.join(output_metrics_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path}")

    # Create summary
    summary_path = os.path.join(output_metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MULTI-TASK METRICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Per-task metrics
        f.write("PER-TASK METRICS\n")
        f.write("-" * 50 + "\n")
        for key, value in sorted(metrics.items()):
            if key.startswith("task_") and isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for metric_name, metric_value in value.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")

        # Aggregate metrics
        if "aggregate" in metrics:
            f.write("\nAGGREGATE METRICS\n")
            f.write("-" * 50 + "\n")
            for metric_name, metric_value in metrics["aggregate"].items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")

    logger.info(f"Saved metrics summary to {summary_path}")


def create_health_check_file(output_path: str) -> str:
    """Create health check file to signal script completion."""
    with open(output_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return output_path


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def evaluate_model_with_comparison(
    model: Any,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_names: List[str],
    previous_scores: Dict[str, np.ndarray],
    id_col: str,
    output_eval_dir: str,
    output_metrics_dir: str,
    comparison_metrics: str,
    statistical_tests: bool,
    comparison_plots: bool,
    input_format: str = "csv",
) -> None:
    """
    Run multi-task model evaluation with comparison to previous model.
    Pattern from xgboost_model_eval.py evaluate_model_with_comparison
    """
    logger.info("Starting multi-task evaluation with comparison mode")

    # Extract task labels
    y_true_tasks = {}
    for i, task_name in enumerate(task_names):
        if task_name not in df.columns:
            raise ValueError(f"Task label '{task_name}' not found in data")
        y_true_tasks[i] = df[task_name].astype(int).values

    # Get IDs
    ids = df[id_col].values

    # Generate new model predictions
    y_pred_tasks = predict_multitask(model, df, feature_columns)

    # Compute standard metrics
    metrics = compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_names)

    # Per-task comparison
    auc_deltas = []
    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_new_score = y_pred_tasks[:, i]
        y_prev_score = previous_scores[task_name]

        # Compute comparison metrics
        if comparison_metrics in ["all", "basic"]:
            comp_metrics = compute_task_comparison_metrics(
                y_true, y_new_score, y_prev_score, task_name
            )
            metrics[f"task_{i}_{task_name}_comparison"] = comp_metrics
            auc_deltas.append(comp_metrics["auc_delta"])

        # Perform statistical tests
        if statistical_tests:
            stat_results = perform_task_statistical_tests(
                y_true, y_new_score, y_prev_score, task_name
            )
            metrics[f"task_{i}_{task_name}_comparison"].update(stat_results)

        # Generate comparison plots
        if comparison_plots:
            plot_task_comparison_roc(
                y_true, y_new_score, y_prev_score, task_name, output_metrics_dir
            )
            plot_task_comparison_pr(
                y_true, y_new_score, y_prev_score, task_name, output_metrics_dir
            )
            plot_task_score_scatter(
                y_new_score, y_prev_score, y_true, task_name, output_metrics_dir
            )
            plot_task_score_distributions(
                y_new_score, y_prev_score, y_true, task_name, output_metrics_dir
            )

    # Aggregate comparison metrics
    if auc_deltas:
        metrics["aggregate_comparison"] = {
            "mean_auc_delta": float(np.mean(auc_deltas)),
            "median_auc_delta": float(np.median(auc_deltas)),
            "tasks_improved": int(sum(1 for d in auc_deltas if d > 0.01)),
            "tasks_degraded": int(sum(1 for d in auc_deltas if d < -0.01)),
        }

    # Save enhanced predictions with comparison
    save_predictions_with_comparison(
        ids,
        y_true_tasks,
        y_pred_tasks,
        previous_scores,
        task_names,
        id_col,
        output_eval_dir,
        input_format,
    )

    # Save metrics
    save_metrics(metrics, output_metrics_dir)

    # Create comparison report
    create_multitask_comparison_report(metrics, task_names, output_metrics_dir)

    logger.info("Multi-task evaluation with comparison complete")


def save_predictions_with_comparison(
    ids: np.ndarray,
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    previous_scores: Dict[str, np.ndarray],
    task_names: List[str],
    id_col: str,
    output_dir: str,
    input_format: str = "csv",
) -> None:
    """Save multi-task predictions with comparison to previous model."""
    logger.info(
        f"Saving predictions with comparison to {output_dir} in {input_format} format"
    )

    # Build predictions DataFrame
    pred_df = pd.DataFrame({id_col: ids})

    for i, task_name in enumerate(task_names):
        pred_df[task_name] = y_true_tasks[i]
        pred_df[f"{task_name}_new_prob"] = y_pred_tasks[:, i]
        pred_df[f"{task_name}_prev_prob"] = previous_scores[task_name]
        pred_df[f"{task_name}_score_diff"] = (
            y_pred_tasks[:, i] - previous_scores[task_name]
        )

    output_base = Path(output_dir) / "eval_predictions_with_comparison"
    output_path = save_dataframe_with_format(pred_df, output_base, input_format)
    logger.info(
        f"Saved predictions with comparison (format={input_format}): {output_path}"
    )


def evaluate_model(
    model: Any,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_names: List[str],
    id_col: str,
    output_eval_dir: str,
    output_metrics_dir: str,
    input_format: str = "csv",
    generate_plots: bool = True,
) -> None:
    """Run multi-task model evaluation using MtgbmModel wrapper."""
    logger.info("Starting multi-task evaluation")

    # Extract task labels
    y_true_tasks = {}
    for i, task_name in enumerate(task_names):
        if task_name not in df.columns:
            raise ValueError(f"Task label '{task_name}' not found in data")
        y_true_tasks[i] = df[task_name].astype(int).values

    # Get IDs
    ids = df[id_col].values

    # Generate predictions
    y_pred_tasks = predict_multitask(model, df, feature_columns)

    # Compute metrics
    metrics = compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_names)

    # Generate visualizations
    if generate_plots:
        logger.info("Generating visualizations (enabled via GENERATE_PLOTS)")
        # Basic ROC/PR curves per task
        plot_multitask_curves(
            y_true_tasks, y_pred_tasks, task_names, output_metrics_dir
        )
        # Comprehensive visualizations
        generate_comprehensive_visualizations(
            y_true_tasks, y_pred_tasks, task_names, metrics, output_metrics_dir
        )
    else:
        logger.info("Visualization generation skipped (GENERATE_PLOTS=false)")

    # Save outputs
    save_predictions(
        ids,
        y_true_tasks,
        y_pred_tasks,
        task_names,
        id_col,
        output_eval_dir,
        input_format,
    )
    save_metrics(metrics, output_metrics_dir)

    logger.info("Multi-task evaluation complete")


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """Main entry point for LightGBMMT model evaluation."""
    # Extract paths
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_eval_dir = output_paths.get(
        "eval_output", output_paths.get("output_eval_dir")
    )
    output_metrics_dir = output_paths.get(
        "metrics_output", output_paths.get("output_metrics_dir")
    )

    # Extract environment variables
    # Basic field configuration
    id_field = environ_vars.get("ID_FIELD", "id")

    # Multi-task configuration
    task_label_names_str = environ_vars.get("TASK_LABEL_NAMES", "")
    previous_score_fields_str = environ_vars.get("PREVIOUS_SCORE_FIELDS", "")

    # Visualization configuration
    generate_plots = environ_vars.get("GENERATE_PLOTS", "true").lower() == "true"

    # Comparison mode configuration (reserved for future implementation)
    comparison_mode = environ_vars.get("COMPARISON_MODE", "false").lower() == "true"
    comparison_metrics = environ_vars.get("COMPARISON_METRICS", "all")
    statistical_tests = environ_vars.get("STATISTICAL_TESTS", "true").lower() == "true"
    comparison_plots = environ_vars.get("COMPARISON_PLOTS", "true").lower() == "true"

    # Guard rail: If PREVIOUS_SCORE_FIELDS is empty, disable comparison mode
    if comparison_mode and (
        not previous_score_fields_str or previous_score_fields_str.strip() == ""
    ):
        logger.warning(
            "COMPARISON_MODE is enabled but PREVIOUS_SCORE_FIELDS is empty. Disabling comparison mode."
        )
        comparison_mode = False

    # Parse task label names
    task_names = parse_task_label_names(task_label_names_str)
    logger.info(f"Parsed task names: {task_names}")

    # Log configuration
    logger.info(f"Visualization plots: {generate_plots}")
    logger.info(f"Comparison mode: {comparison_mode}")
    if comparison_mode:
        logger.info(f"Previous score fields: {previous_score_fields_str}")
        logger.info(f"Comparison metrics: {comparison_metrics}")
        logger.info(f"Statistical tests: {statistical_tests}")
        logger.info(f"Comparison plots: {comparison_plots}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info("Starting LightGBMMT model evaluation")
    logger.info(f"Job type: {job_args.job_type}")

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = (
        load_model_artifacts(model_dir)
    )

    # Verify task names match hyperparameters
    hp_task_names = hyperparams.get("task_label_names", [])
    if hp_task_names and hp_task_names != task_names:
        logger.warning(
            f"Environment TASK_LABEL_NAMES {task_names} differs from "
            f"hyperparameters task_label_names {hp_task_names}. "
            f"Using environment variable."
        )

    # Load and preprocess data
    df, input_format = load_eval_data(eval_data_dir)
    id_col = get_id_column(df, id_field)

    # Preprocess
    df = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)

    # Get available features
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features for inference")

    # Evaluate with or without comparison
    if comparison_mode and previous_score_fields_str:
        # Parse previous score fields
        prev_score_fields = [
            f.strip() for f in previous_score_fields_str.split(",") if f.strip()
        ]

        # Validate and load previous scores
        previous_scores = {}
        for i, task_name in enumerate(task_names):
            if i < len(prev_score_fields):
                prev_field = prev_score_fields[i]
                if prev_field not in df.columns:
                    logger.error(
                        f"Previous score field '{prev_field}' not found in data for task '{task_name}'"
                    )
                    raise ValueError(
                        f"Previous score field '{prev_field}' not found in data"
                    )
                previous_scores[task_name] = df[prev_field].values
                logger.info(
                    f"Loaded previous scores for task '{task_name}' from field '{prev_field}'"
                )
            else:
                logger.error(
                    f"No previous score field specified for task '{task_name}' (index {i})"
                )
                raise ValueError(f"PREVIOUS_SCORE_FIELDS must have one field per task")

        # Run comparison evaluation
        evaluate_model_with_comparison(
            model,
            df,
            available_features,
            task_names,
            previous_scores,
            id_col,
            output_eval_dir,
            output_metrics_dir,
            comparison_metrics,
            statistical_tests,
            comparison_plots,
            input_format,
        )
    else:
        # Standard evaluation
        evaluate_model(
            model,
            df,
            available_features,
            task_names,
            id_col,
            output_eval_dir,
            output_metrics_dir,
            input_format,
            generate_plots,
        )

    logger.info("LightGBMMT model evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
        "metrics_output": CONTAINER_PATHS["OUTPUT_METRICS_DIR"],
    }

    # Collect environment variables (aligned with xgboost_model_eval)
    environ_vars = {
        # Basic field configuration
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        # Multi-task configuration
        "TASK_LABEL_NAMES": os.environ.get(
            "TASK_LABEL_NAMES", ""
        ),  # Explicit task labels (comma-separated or JSON)
        "PREVIOUS_SCORE_FIELDS": os.environ.get(
            "PREVIOUS_SCORE_FIELDS", ""
        ),  # Comma-separated previous score fields for multi-task comparison
        # Visualization configuration
        "GENERATE_PLOTS": os.environ.get("GENERATE_PLOTS", "true"),
        # Comparison mode configuration (reserved for future implementation)
        "COMPARISON_MODE": os.environ.get("COMPARISON_MODE", "false"),
        "COMPARISON_METRICS": os.environ.get("COMPARISON_METRICS", "all"),
        "STATISTICAL_TESTS": os.environ.get("STATISTICAL_TESTS", "true"),
        "COMPARISON_PLOTS": os.environ.get("COMPARISON_PLOTS", "true"),
    }

    try:
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["metrics_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check
        health_path = os.path.join(output_paths["metrics_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("metrics_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
