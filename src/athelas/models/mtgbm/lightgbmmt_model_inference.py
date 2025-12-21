#!/usr/bin/env python
"""
LightGBMMT Multi-Task Model Inference Script

Generates multi-task predictions using trained LightGBM models.
Pure inference only - NO evaluation, NO metrics, NO plots.
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
import tarfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

# Import unified model architecture
from models.implementations.mtgbm_model import MtgbmModel
from models.factory.model_factory import ModelFactory
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
) -> Tuple[MtgbmModel, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Load trained LightGBMMT model and preprocessing artifacts using unified architecture.

    Returns: model, risk_tables, impute_dict, feature_columns, hyperparameters
    """
    logger.info(f"Loading model artifacts from {model_dir}")

    # Decompress if needed
    decompress_model_artifacts(model_dir)

    # Load hyperparameters first to create model instance
    hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = json.load(f)
    logger.info("✓ Loaded hyperparameters.json")

    # Create hyperparameters object
    hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)

    # Create model instance using factory (no loss function needed for inference)
    model = ModelFactory.create(
        model_type="mtgbm",
        loss_function=None,  # Not needed for inference
        training_state=None,  # Not needed for inference
        hyperparams=hyperparams,
    )
    logger.info("✓ Created MtgbmModel instance")

    # Load model artifacts using unified interface
    model.load(model_dir)
    logger.info("✓ Loaded model using MtgbmModel.load()")

    # Load preprocessing artifacts
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    logger.info("✓ Loaded risk_table_map.pkl")

    with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
        impute_dict = pkl.load(f)
    logger.info("✓ Loaded impute_dict.pkl")

    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [
            line.strip().split(",")[1] for line in f if not line.startswith("#")
        ]
    logger.info(f"✓ Loaded feature_columns.txt: {len(feature_columns)} features")

    logger.info("All model artifacts loaded successfully")
    return model, risk_tables, impute_dict, feature_columns, hyperparams_dict


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def preprocess_inference_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_tables: Dict[str, Any],
    impute_dict: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply risk table mapping and numerical imputation to inference data.
    Preserves all original columns while ensuring features are model-ready.
    """
    # Preserve original dataframe structure
    result_df = df.copy()

    # Get available feature columns
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(
        f"Found {len(available_features)} out of {len(feature_columns)} expected feature columns"
    )

    # Apply risk table mapping for categorical features
    logger.info("Starting risk table mapping for categorical features")
    for feature, risk_table in risk_tables.items():
        if feature in available_features:
            logger.info(f"Applying risk table mapping for feature: {feature}")
            processor = RiskTableMappingProcessor(
                column_name=feature, label_name="label", risk_tables=risk_table
            )
            result_df[feature] = processor.transform(df[feature])
    logger.info("Risk table mapping complete")

    # Apply numerical imputation
    logger.info("Starting numerical imputation")
    feature_df = result_df[available_features].copy()
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    imputed_df = imputer.transform(feature_df)

    # Update feature columns in result dataframe
    for col in available_features:
        if col in imputed_df:
            result_df[col] = imputed_df[col]
    logger.info("Numerical imputation complete")

    # Ensure feature columns are numeric
    logger.info("Ensuring feature columns are numeric")
    result_df[available_features] = (
        result_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    logger.info(
        f"Preprocessed data shape: {result_df.shape} (preserving all original columns)"
    )
    return result_df


# ============================================================================
# MULTI-TASK INFERENCE
# ============================================================================


def generate_multitask_predictions(
    model: MtgbmModel, df: pd.DataFrame, feature_columns: List[str]
) -> np.ndarray:
    """
    Generate multi-task predictions using the unified MtgbmModel.

    Args:
        model: Trained MtgbmModel instance
        df: DataFrame with preprocessed features
        feature_columns: List of feature column names

    Returns:
        np.ndarray of shape (n_samples, n_tasks) with probabilities for each task
    """
    # Get available features for prediction
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features for prediction")

    # Generate predictions using unified model interface
    predictions = model.predict(df, feature_columns=available_features)
    logger.info(f"Model prediction shape: {predictions.shape}")

    return predictions


# ============================================================================
# DATA LOADING
# ============================================================================


def load_eval_data(eval_data_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Load the first data file found in the evaluation data directory.
    Returns a pandas DataFrame and the detected format.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".tsv", ".parquet"]
        ]
    )
    if not eval_files:
        logger.error("No eval data file found in eval_data input.")
        raise RuntimeError("No eval data file found in eval_data input.")
    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")

    df, input_format = load_dataframe_with_format(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}, format: {input_format}")
    return df, input_format


def get_id_column(df: pd.DataFrame, id_field: str) -> str:
    """
    Determine the ID column in the DataFrame.
    Falls back to the first column if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    logger.info(f"Using id_col: {id_col}")
    return id_col


# ============================================================================
# OUTPUT SAVING
# ============================================================================


def save_multitask_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    task_names: List[str],
    output_dir: str,
    input_format: str = "csv",
    id_col: str = "id",
    json_orient: str = "records",
) -> str:
    """
    Save multi-task predictions with original data preserving input format.
    Supports CSV, TSV, Parquet, and JSON formats.

    If task label columns exist in input data, they are preserved (for downstream calibration).

    Args:
        df: Original DataFrame (with ID and optional label columns)
        predictions: Prediction array of shape (n_samples, n_tasks)
        task_names: List of task names
        output_dir: Output directory path
        input_format: Format to save in ('csv', 'tsv', 'parquet', 'json')
        id_col: Name of ID column
        json_orient: JSON orientation (for JSON format only)

    Returns:
        Path to saved file
    """
    logger.info(f"Saving predictions to {output_dir} in {input_format} format")

    # Create output dataframe with original data
    output_df = df.copy()

    # Add prediction columns for each task
    # Also check if true labels exist (for downstream calibration)
    n_tasks = predictions.shape[1]
    for i in range(n_tasks):
        if i < len(task_names):
            task_name = task_names[i]

            # If label column exists in input, it will be preserved in output_df
            # (needed for model calibration downstream)
            if task_name in df.columns:
                logger.info(
                    f"Label column '{task_name}' found in input data - preserving for calibration"
                )

            # Add prediction column
            output_df[f"{task_name}_prob"] = predictions[:, i]
        else:
            # Fallback if task names list is shorter than prediction array
            output_df[f"task_{i}_prob"] = predictions[:, i]

    # Save in specified format
    os.makedirs(output_dir, exist_ok=True)

    if input_format.lower() == "json":
        # Special handling for JSON (not using save_dataframe_with_format)
        output_path = os.path.join(output_dir, "predictions.json")
        # Convert numpy types to native Python types for JSON serialization
        output_df_json = output_df.copy()
        for col in output_df_json.columns:
            if output_df_json[col].dtype == "object":
                continue
            elif "int" in str(output_df_json[col].dtype):
                output_df_json[col] = output_df_json[col].astype(int)
            elif "float" in str(output_df_json[col].dtype):
                output_df_json[col] = output_df_json[col].astype(float)

        # Save as JSON with specified orientation
        output_df_json.to_json(output_path, orient=json_orient, indent=2)
    else:
        # Use format-preserving save for CSV, TSV, Parquet
        output_base = Path(output_dir) / "predictions"
        output_path = save_dataframe_with_format(output_df, output_base, input_format)
        output_path = str(output_path)

    logger.info(f"Saved predictions to {output_path}")
    return output_path


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for LightGBMMT model inference script.
    Loads model and data, runs inference, and saves predictions.

    Args:
        input_paths: Dictionary of input paths
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_eval_dir = output_paths.get(
        "eval_output", output_paths.get("output_eval_dir")
    )

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "id")
    task_label_names_str = environ_vars.get("TASK_LABEL_NAMES", "")
    output_format = environ_vars.get("OUTPUT_FORMAT", "csv")
    json_orient = environ_vars.get("JSON_ORIENT", "records")

    # Parse task label names
    task_names = parse_task_label_names(task_label_names_str)
    logger.info(f"Parsed task names: {task_names}")

    # Log job info
    job_type = job_args.job_type
    logger.info(f"Running multi-task model inference with job_type: {job_type}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)

    logger.info("Starting multi-task model inference script")

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = (
        load_model_artifacts(model_dir)
    )

    # Verify task names match hyperparameters (if available)
    hp_task_names = hyperparams.get("task_label_names", [])
    if hp_task_names and hp_task_names != task_names:
        logger.warning(
            f"Environment TASK_LABEL_NAMES {task_names} differs from "
            f"hyperparameters task_label_names {hp_task_names}. "
            f"Using environment variable."
        )

    # Load and preprocess data with format detection
    df, input_format = load_eval_data(eval_data_dir)

    # Get ID column
    id_col = get_id_column(df, id_field)

    # Process the data - preserves all columns including id
    df = preprocess_inference_data(df, feature_columns, risk_tables, impute_dict)

    logger.info(f"Final inference DataFrame shape: {df.shape}")

    # Get the available features (those that exist in the DataFrame)
    available_features = [col for col in feature_columns if col in df.columns]

    # Log inference strategy
    logger.info("INFERENCE STRATEGY:")
    logger.info(f"  → Using {len(available_features)} features for model inference")
    logger.info(f"  → These are the exact features the model was trained on")
    logger.info(f"  → Input data will be filtered to these features only")

    # Generate multi-task predictions
    predictions = generate_multitask_predictions(model, df, available_features)

    # Save predictions with original data preserving format
    # Override with OUTPUT_FORMAT env var if set, otherwise use input format
    final_format = output_format if output_format != "csv" else input_format
    output_path = save_multitask_predictions(
        df,
        predictions,
        task_names,
        output_eval_dir,
        input_format=final_format,
        id_col=id_col,
        json_orient=json_orient,
    )

    logger.info("Multi-task model inference script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
    }

    # Collect environment variables - ID_FIELD and TASK_LABEL_NAMES are required per contract
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),  # Fallback for testing
        "TASK_LABEL_NAMES": os.environ.get(
            "TASK_LABEL_NAMES", ""
        ),  # Required for multi-task
        "OUTPUT_FORMAT": os.environ.get(
            "OUTPUT_FORMAT", "csv"
        ),  # csv, parquet, or json
        "JSON_ORIENT": os.environ.get("JSON_ORIENT", "records"),  # JSON orientation
    }

    try:
        # Call main function with testability parameters
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["eval_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["eval_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(output_paths.get("eval_output", "/tmp"), "_FAILURE")
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
