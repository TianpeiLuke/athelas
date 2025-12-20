#!/usr/bin/env python3
"""
LightGBMMT Multi-Task Model Inference Script

Provides focused inference engine for multi-task gradient boosting models.
Handles model loading, preprocessing, and multi-task prediction generation.
"""

# Standard library imports
import os
import sys
from subprocess import check_call
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI (DEFAULT)
# Set USE_SECURE_PYPI=false to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """
    Get CodeArtifact access token for secure PyPI.

    Returns:
        str: Authorization token for CodeArtifact

    Raises:
        Exception: If token retrieval fails
    """
    # Local import to avoid loading boto3 before package installation
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
    """
    Install packages from standard public PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from public PyPI")
    logger.info(f"Packages: {packages}")

    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        logger.info("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """
    Install packages from secure CodeArtifact PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from secure PyPI")
    logger.info(f"Packages: {packages}")

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
    """
    Install packages from PyPI source based on configuration.

    This is the main installation function that delegates to either public or
    secure PyPI based on the USE_SECURE_PYPI environment variable.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
        use_secure: If True, use secure CodeArtifact PyPI; if False, use public PyPI.
                   Defaults to USE_SECURE_PYPI environment variable.

    Environment Variables:
        USE_SECURE_PYPI: Set to "true" to use secure PyPI (DEFAULT), "false" for public PyPI

    Example:
        # Install from secure PyPI (default)
        install_packages(["pandas==1.5.0", "numpy"])

        # Install from public PyPI
        os.environ["USE_SECURE_PYPI"] = "false"
        install_packages(["pandas==1.5.0", "numpy"])
    """
    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    logger.info(
        f"Environment Variable USE_SECURE_PYPI: {os.environ.get('USE_SECURE_PYPI', 'not set (defaults to true)')}"
    )
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

# Define required packages for this script
# NOTE: Removed heavy calibration dependencies (pygam, scipy, matplotlib)
# since we use lookup table calibration instead of model objects
# NOTE: numpy is already available in SKLearn processor framework 1.2-1
required_packages = [
    "lightgbm>=3.3.0",  # Required for LightGBMMT
    "pydantic>=2.0.0",  # Required for TrainingState in model classes
]

# Install packages using unified installation function
install_packages(required_packages)

print("***********************Package Installation Complete*********************")

import json
import pickle as pkl
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, Optional
from io import StringIO, BytesIO

# Third-party imports
import pandas as pd
import numpy as np

# Model wrapper imports (for consistency with training/eval)
from models.factory.model_factory import ModelFactory
from models.implementations.mtgbm_model import MtgbmModel
from hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters

# Local imports
from processing.categorical.risk_table_processor import RiskTableMappingProcessor
from processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)

# File names
MODEL_FILE = "lightgbmmt_model.txt"  # LightGBM text format
RISK_TABLE_FILE = "risk_table_map.pkl"
IMPUTE_DICT_FILE = "impute_dict.pkl"
FEATURE_IMPORTANCE_FILE = "feature_importance.json"
FEATURE_COLUMNS_FILE = "feature_columns.txt"
HYPERPARAMETERS_FILE = "hyperparameters.json"

# Calibration model files (per-task structure)
CALIBRATION_DIR = "calibration"
# Per-task calibration: task_0_calibration_model.pkl, task_1_calibration_model.pkl, etc.

# Content types
CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_PARQUET = "application/x-parquet"


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


# --------------------------------------------------------------------------------
#                           MODEL LOADING BLOCK
# --------------------------------------------------------------------------------


def validate_model_files(model_dir: str) -> None:
    """
    Validate that all required model files exist.

    Args:
        model_dir: Directory containing model artifacts

    Raises:
        FileNotFoundError: If any required file is missing
    """
    required_files = [
        MODEL_FILE,
        RISK_TABLE_FILE,
        IMPUTE_DICT_FILE,
        FEATURE_COLUMNS_FILE,
        HYPERPARAMETERS_FILE,  # Required for multi-task configuration
    ]
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file} not found in {model_dir}")
        logger.info(f"Found required file: {file}")


def read_feature_columns(model_dir: str) -> List[str]:
    """
    Read feature columns in correct order from feature_columns.txt

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        List[str]: Ordered list of feature column names

    Raises:
        FileNotFoundError: If feature_columns.txt is not found
        ValueError: If file format is invalid
    """
    feature_file = os.path.join(model_dir, FEATURE_COLUMNS_FILE)
    ordered_features = []

    try:
        with open(feature_file, "r") as f:
            for line in f:
                # Skip comments
                if line.startswith("#"):
                    continue
                # Parse "<index>,<column_name>" format
                try:
                    idx, column = line.strip().split(",")
                    ordered_features.append(column)
                except ValueError:
                    continue

        if not ordered_features:
            raise ValueError(f"No valid feature columns found in {feature_file}")

        logger.info(f"Loaded {len(ordered_features)} ordered feature columns")
        return ordered_features
    except Exception as e:
        logger.error(f"Error reading feature columns file: {e}", exc_info=True)
        raise


def load_lightgbmmt_model(model_dir: str) -> MtgbmModel:
    """
    Load LightGBMMT model using MtgbmModel wrapper.

    Consistent with training/eval approach - uses ModelFactory pattern.
    The wrapper handles multi-task specifics (num_labels, predictions, etc.).

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        MtgbmModel: Loaded multi-task model
    """
    # Load hyperparameters
    hyperparams_file = os.path.join(model_dir, HYPERPARAMETERS_FILE)
    with open(hyperparams_file, "r") as f:
        hyperparams_dict = json.load(f)
    hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)

    # Create model using factory (no loss_function/training_state for inference)
    model = ModelFactory.create(
        model_type="mtgbm",
        loss_function=None,  # Not needed for inference
        training_state=None,  # Not needed for inference
        hyperparams=hyperparams,
    )

    # Load model artifacts
    model.load(model_dir)

    num_tasks = len(hyperparams.task_label_names) if hyperparams.task_label_names else 1
    logger.info(f"Loaded MtgbmModel with {num_tasks} tasks from {model_dir}")

    return model


def load_risk_tables(model_dir: str) -> Dict[str, Any]:
    """Load risk tables from pickle file."""
    with open(os.path.join(model_dir, RISK_TABLE_FILE), "rb") as f:
        return pkl.load(f)


def create_risk_processors(
    risk_tables: Dict[str, Any],
) -> Dict[str, RiskTableMappingProcessor]:
    """Create risk table processors for each categorical feature."""
    risk_processors = {}
    for feature, risk_table in risk_tables.items():
        processor = RiskTableMappingProcessor(
            column_name=feature,
            label_name="label",  # Not used during inference
            risk_tables=risk_table,
        )
        risk_processors[feature] = processor
    return risk_processors


def load_imputation_dict(model_dir: str) -> Dict[str, Any]:
    """Load imputation dictionary from pickle file."""
    with open(os.path.join(model_dir, IMPUTE_DICT_FILE), "rb") as f:
        return pkl.load(f)


def create_numerical_processors(
    impute_dict: Dict[str, Any],
) -> Dict[str, NumericalVariableImputationProcessor]:
    """
    Create numerical imputation processors for each numerical feature.

    Uses single-column architecture - one processor per column.
    """
    numerical_processors = {}
    for feature, imputation_value in impute_dict.items():
        processor = NumericalVariableImputationProcessor(
            column_name=feature, imputation_value=imputation_value
        )
        numerical_processors[feature] = processor
    return numerical_processors


def load_feature_importance(model_dir: str) -> Dict[str, Any]:
    """Load feature importance from JSON file."""
    try:
        with open(os.path.join(model_dir, FEATURE_IMPORTANCE_FILE), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(
            f"{FEATURE_IMPORTANCE_FILE} not found, skipping feature importance"
        )
        return {}


def load_hyperparameters(model_dir: str) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON file.

    For multi-task models, hyperparameters contain:
    - task_label_names: List of task names
    - main_task_index: Index of the main task
    """
    try:
        with open(os.path.join(model_dir, HYPERPARAMETERS_FILE), "r") as f:
            hyperparams = json.load(f)

        # Validate multi-task configuration
        if "task_label_names" not in hyperparams:
            raise ValueError("Missing 'task_label_names' in hyperparameters")
        if "main_task_index" not in hyperparams:
            raise ValueError("Missing 'main_task_index' in hyperparameters")

        logger.info(
            f"Loaded multi-task configuration: {len(hyperparams['task_label_names'])} tasks"
        )
        logger.info(f"Main task index: {hyperparams['main_task_index']}")

        return hyperparams
    except Exception as e:
        logger.error(f"Could not load {HYPERPARAMETERS_FILE}: {e}")
        raise


def load_multitask_calibration_models(
    model_dir: str, hyperparams: Dict[str, Any]
) -> Optional[Dict]:
    """
    Load per-task calibration models from single dictionary file.
    Uses hyperparameters.task_label_names to ensure correct task ordering.

    Expected structure:
    calibration/
      calibration_models_dict.pkl  # Dictionary with {score_field: calibrator}

    Args:
        model_dir: Directory containing model artifacts
        hyperparams: Model hyperparameters containing task_label_names

    Returns:
        Dictionary with 'type': 'multitask' and 'data': {task_idx: model}
        or None if no calibration models found

    Raises:
        ValueError: If calibrators don't match expected tasks
    """
    calibration_dir = os.path.join(model_dir, CALIBRATION_DIR)
    if not os.path.exists(calibration_dir):
        logger.info("No calibration directory found")
        return None

    # Get authoritative task order from hyperparameters
    task_label_names = hyperparams.get("task_label_names")
    if not task_label_names:
        raise ValueError("Missing 'task_label_names' in hyperparameters")

    # Try loading from single dictionary file (new format)
    dict_file = os.path.join(calibration_dir, "calibration_models_dict.pkl")
    if os.path.exists(dict_file):
        try:
            with open(dict_file, "rb") as f:
                calibrators_dict = pkl.load(f)

            # Map tasks to calibrators using task names (NOT alphabetical sorting)
            calibrators = {}
            for task_idx, label_name in enumerate(task_label_names):
                # Convert label name to score field name
                # e.g., "is_abuse" → "is_abuse_prob" or "is_abuse_true" → "is_abuse_prob"
                score_field = f"{label_name.replace('_true', '')}_prob"

                if score_field not in calibrators_dict:
                    logger.warning(
                        f"Missing calibrator for task {task_idx}: {score_field}. "
                        f"Available: {list(calibrators_dict.keys())}"
                    )
                    continue

                calibrators[task_idx] = calibrators_dict[score_field]
                logger.info(f"✓ Task {task_idx}: {label_name} → {score_field}")

            if calibrators:
                if len(calibrators) != len(task_label_names):
                    logger.warning(
                        f"Calibrator count mismatch: expected {len(task_label_names)}, "
                        f"got {len(calibrators)}. Proceeding with available calibrators."
                    )
                logger.info(
                    f"Loaded {len(calibrators)} calibration models from dictionary file"
                )
                return {"type": "multitask", "data": calibrators}
        except Exception as e:
            logger.warning(f"Failed to load calibration dictionary: {e}")

    # Fallback: Try loading from separate files (legacy format)
    logger.info("Dictionary file not found, trying legacy format (separate files)")
    num_tasks = len(task_label_names)
    calibrators = {}
    for i in range(num_tasks):
        model_file = os.path.join(calibration_dir, f"task_{i}_calibration_model.pkl")
        if os.path.exists(model_file):
            try:
                with open(model_file, "rb") as f:
                    calibrators[i] = pkl.load(f)
                logger.info(f"Loaded calibration model for task {i} (legacy format)")
            except Exception as e:
                logger.warning(f"Failed to load calibration model for task {i}: {e}")

    if not calibrators:
        logger.info("No calibration models found in calibration directory")
        return None

    return {"type": "multitask", "data": calibrators}


def apply_multitask_calibration(
    predictions: np.ndarray, calibrators: Dict[int, Any]
) -> np.ndarray:
    """
    Apply per-task calibration to multi-task predictions.
    Supports both lookup table format (List[Tuple[float, float]]) and legacy model objects.

    Args:
        predictions: Raw predictions (n_samples, n_tasks)
        calibrators: Dictionary mapping task index to calibration model or lookup table

    Returns:
        Calibrated predictions (n_samples, n_tasks)
    """
    calibrated = predictions.copy()

    # Reusable interpolation function for lookup tables
    def interpolate_score(
        raw_score: float, mapping: List[Tuple[float, float]]
    ) -> float:
        """Interpolate calibrated score for a single raw score."""
        if raw_score <= mapping[0][0]:
            return mapping[0][1]
        if raw_score >= mapping[-1][0]:
            return mapping[-1][1]

        for i in range(len(mapping) - 1):
            if mapping[i][0] <= raw_score <= mapping[i + 1][0]:
                x1, y1 = mapping[i]
                x2, y2 = mapping[i + 1]
                if x2 == x1:
                    return y1
                return y1 + (y2 - y1) * (raw_score - x1) / (x2 - x1)
        return mapping[-1][1]

    for task_idx, calibrator in calibrators.items():
        if task_idx < predictions.shape[1]:
            task_probs = predictions[:, task_idx]

            try:
                # Check if calibrator is a lookup table (new optimized format)
                if isinstance(calibrator, list):
                    logger.info(
                        f"Using lookup table calibration for task {task_idx} (optimized)"
                    )
                    # Apply lookup table calibration (FAST: ~2-5 μs per prediction)
                    for i in range(len(task_probs)):
                        calibrated[i, task_idx] = interpolate_score(
                            task_probs[i], calibrator
                        )

                # Legacy model object format (backward compatibility)
                elif hasattr(calibrator, "transform"):
                    # Isotonic regression
                    logger.info(
                        f"Using Isotonic model calibration for task {task_idx} (legacy)"
                    )
                    calibrated[:, task_idx] = calibrator.transform(task_probs)
                elif hasattr(calibrator, "predict_proba"):
                    # GAM or Platt scaling
                    logger.info(
                        f"Using GAM/Platt model calibration for task {task_idx} (legacy)"
                    )
                    calibrated[:, task_idx] = calibrator.predict_proba(
                        task_probs.reshape(-1, 1)
                    )
                else:
                    logger.warning(
                        f"Unknown calibrator type for task {task_idx}: {type(calibrator)}"
                    )
            except Exception as e:
                logger.warning(f"Failed to calibrate task {task_idx}: {e}")

    return calibrated


def create_model_config(
    model: MtgbmModel,
    feature_columns: List[str],
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create model configuration dictionary for multi-task model.

    Args:
        model: MtgbmModel wrapper
        feature_columns: List of feature column names
        hyperparameters: Model hyperparameters (includes task configuration)

    Returns:
        Configuration dictionary
    """
    task_label_names = hyperparameters.get("task_label_names", [])
    main_task_index = hyperparameters.get("main_task_index", 0)

    return {
        "is_multiclass": False,  # Multi-task, not multiclass
        "is_multitask": True,  # NEW FLAG
        "num_tasks": len(task_label_names),
        "task_names": task_label_names,
        "main_task_index": main_task_index,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
    }


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and preprocessing artifacts from model_dir.

    Uses MtgbmModel wrapper for consistency with training/eval scripts.

    Args:
        model_dir (str): Directory containing model artifacts

    Returns:
        Dict[str, Any]: Dictionary containing model, processors, and configuration

    Raises:
        FileNotFoundError: If required model files are missing
        Exception: For other loading errors
    """
    logger.info(f"Loading LightGBMMT model from {model_dir}")

    try:
        # Validate all required files exist
        validate_model_files(model_dir)

        # Load model using wrapper (handles hyperparameters internally)
        model = load_lightgbmmt_model(model_dir)

        # Load preprocessing artifacts
        risk_tables = load_risk_tables(model_dir)
        risk_processors = create_risk_processors(risk_tables)

        impute_dict = load_imputation_dict(model_dir)
        numerical_processors = create_numerical_processors(impute_dict)

        feature_importance = load_feature_importance(model_dir)
        feature_columns = read_feature_columns(model_dir)

        # Get hyperparameters from model wrapper
        hyperparameters = model.hyperparams.model_dump()

        # Create configuration
        config = create_model_config(model, feature_columns, hyperparameters)

        # Load calibration models with hyperparameters for correct task ordering
        calibrator = load_multitask_calibration_models(model_dir, hyperparameters)
        if calibrator:
            logger.info(
                f"Calibration models loaded for {len(calibrator['data'])} tasks"
            )
        else:
            logger.info("No calibration models found - will use raw predictions")

        return {
            "model": model,  # MtgbmModel wrapper
            "risk_processors": risk_processors,
            "numerical_processors": numerical_processors,
            "feature_importance": feature_importance,
            "config": config,
            "calibrator": calibrator,
        }

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


# --------------------------------------------------------------------------------
#                           INPUT BLOCK
# --------------------------------------------------------------------------------


def input_fn(
    request_body: Union[str, bytes],
    request_content_type: str,
    context: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Deserialize the Invoke request body into an object we can perform prediction on.

    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        context: Additional context (optional)

    Returns:
        Union[pd.DataFrame, Response]: Parsed DataFrame or error Response
    """
    logger.info(f"Received request with Content-Type: {request_content_type}")
    try:
        if request_content_type == CONTENT_TYPE_CSV:
            logger.info("Processing content type: text/csv")
            decoded = (
                request_body.decode("utf-8")
                if isinstance(request_body, bytes)
                else request_body
            )
            logger.debug(f"Decoded CSV data:\n{decoded[:500]}...")
            try:
                df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
                if df.empty:
                    raise ValueError("Empty CSV input provided")
                logger.info(
                    f"Successfully parsed CSV into DataFrame. Shape: {df.shape}"
                )
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse CSV data: {parse_error}")
                raise

        elif request_content_type == CONTENT_TYPE_JSON:
            logger.info("Processing content type: application/json")
            decoded = (
                request_body.decode("utf-8")
                if isinstance(request_body, bytes)
                else request_body
            )
            try:
                if "\n" in decoded:
                    # Multi-record JSON (NDJSON) handling
                    records = [
                        json.loads(line)
                        for line in decoded.strip().splitlines()
                        if line.strip()
                    ]
                    df = pd.DataFrame(records)
                else:
                    json_obj = json.loads(decoded)
                    if isinstance(json_obj, dict):
                        df = pd.DataFrame([json_obj])
                    elif isinstance(json_obj, list):
                        df = pd.DataFrame(json_obj)
                    else:
                        raise ValueError("Unsupported JSON structure")

                if df.empty:
                    raise ValueError("Empty JSON input provided")
                logger.info(
                    f"Successfully parsed JSON into DataFrame. Shape: {df.shape}"
                )
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON data: {parse_error}")
                raise

        elif request_content_type == CONTENT_TYPE_PARQUET:
            logger.info("Processing content type: application/x-parquet")
            df = pd.read_parquet(BytesIO(request_body))
            if df.empty:
                raise ValueError("Empty Parquet input provided")
            logger.info(
                f"Successfully parsed Parquet into DataFrame. Shape: {df.shape}"
            )
            return df

        else:
            logger.warning(f"Unsupported content type: {request_content_type}")
            raise ValueError(
                f"This predictor only supports CSV, JSON, or Parquet data. Received: {request_content_type}"
            )
    except Exception as e:
        logger.error(
            f"Failed to parse input ({request_content_type}). Error: {e}", exc_info=True
        )
        raise ValueError(
            f"Invalid input format or corrupted data. Error during parsing: {e}"
        ) from e


# --------------------------------------------------------------------------------
#                           PREDICT BLOCK
# --------------------------------------------------------------------------------


def validate_input_data(input_data: pd.DataFrame, feature_columns: List[str]) -> None:
    """
    Validate input data meets requirements.

    Args:
        input_data: Input DataFrame
        feature_columns: Expected feature columns

    Raises:
        ValueError: If validation fails
    """
    if input_data.empty:
        raise ValueError("Input DataFrame is empty")

    # If input is headerless CSV, validate column count
    if all(isinstance(col, int) for col in input_data.columns):
        if len(input_data.columns) != len(feature_columns):
            raise ValueError(
                f"Input data has {len(input_data.columns)} columns but model expects {len(feature_columns)} features"
            )
    else:
        # Validate required features present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


def assign_column_names(
    input_data: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    """
    Assign column names to headerless input data.

    Args:
        input_data: Input DataFrame
        feature_columns: Feature column names to assign

    Returns:
        DataFrame with assigned column names
    """
    df = input_data.copy()
    if all(isinstance(col, int) for col in df.columns):
        df.columns = feature_columns
    return df


def preprocess_single_record_fast(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_processors: Dict[str, Any],
    numerical_processors: Dict[str, Any],
) -> np.ndarray:
    """
    Fast path for single-record preprocessing.

    Bypasses pandas DataFrame operations for 10-100x speedup.
    Uses processor.process() method for direct value processing.

    Note: Preprocessing is identical for single-task and multi-task models.
    Multi-task only affects prediction output, not feature preprocessing.

    Args:
        df: Single-row DataFrame with feature values
        feature_columns: Ordered feature column names
        risk_processors: Risk table processors for categorical features
        numerical_processors: Imputation processors for numerical features

    Returns:
        Processed feature values ready for model [n_features]
    """
    processed = np.zeros(len(feature_columns), dtype=np.float32)

    for i, col in enumerate(feature_columns):
        val = df[col].iloc[0]

        # Apply risk table mapping if categorical
        if col in risk_processors:
            # Uses optimized process() method (direct dict lookup)
            val = risk_processors[col].process(val)

        # Apply numerical imputation
        if col in numerical_processors:
            # Uses optimized process() method (simple null check)
            val = numerical_processors[col].process(val)

        # Convert to float with error handling
        try:
            val = float(val)
        except (ValueError, TypeError):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Could not convert {col}={val} to float, using 0.0")
            val = 0.0

        processed[i] = val

    return processed


def apply_preprocessing(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_processors: Dict[str, Any],
    numerical_processors: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply preprocessing steps to input data.

    Note: Preprocessing is identical to XGBoost - same transformations apply
    regardless of single-task vs multi-task model architecture.

    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        risk_processors: Dictionary of risk table processors
        numerical_processors: Dictionary of numerical imputation processors

    Returns:
        Preprocessed DataFrame
    """
    # Conditional logging (only if DEBUG enabled)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Initial data types and unique values:")
        for col in feature_columns:
            logger.debug(
                f"{col}: dtype={df[col].dtype}, unique values={df[col].unique()}"
            )

    # Apply risk table mapping
    for feature, processor in risk_processors.items():
        if feature in df.columns:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Applying risk table mapping for feature: {feature}")
            df[feature] = processor.transform(df[feature])

    # Apply numerical imputation (one processor per column)
    for feature, processor in numerical_processors.items():
        if feature in df.columns:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Applying numerical imputation for feature: {feature}")
            df[feature] = processor.transform(df[feature])

    return df


def safe_numeric_conversion(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    """
    Safely convert a series to numeric values.

    Args:
        series: Input pandas Series
        default_value: Value to use for non-numeric entries

    Returns:
        Converted numeric series
    """
    # If series is already numeric, return as is
    if pd.api.types.is_numeric_dtype(series):
        return series

    # Replace string 'Default' with default_value
    series = series.replace("Default", str(default_value))

    # Try converting to numeric, forcing errors to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Fill NaN with default_value
    numeric_series = numeric_series.fillna(default_value)

    return numeric_series


def convert_to_numeric(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Convert all columns to numeric type.

    Args:
        df: Input DataFrame
        feature_columns: Columns to convert

    Returns:
        DataFrame with numeric columns

    Raises:
        ValueError: If conversion fails
    """
    for col in feature_columns:
        logger.debug(f"Converting {col} to numeric. Current values: {df[col].unique()}")
        df[col] = safe_numeric_conversion(df[col])
        logger.debug(
            f"After conversion {col}: unique values={df[col].unique()}, dtype={df[col].dtype}"
        )

    # Verify numeric conversion
    non_numeric_cols = (
        df[feature_columns].select_dtypes(exclude=["int64", "float64"]).columns
    )
    if not non_numeric_cols.empty:
        logger.error("Non-numeric columns found after preprocessing:")
        for col in non_numeric_cols:
            logger.error(
                f"{col}: dtype={df[col].dtype}, unique values={df[col].unique()}"
            )
        raise ValueError(
            f"Following columns contain non-numeric values after preprocessing: {list(non_numeric_cols)}"
        )

    # Convert to float type
    df[feature_columns] = df[feature_columns].astype(float)
    return df


def generate_multitask_predictions(
    model: MtgbmModel,
    df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    """
    Generate multi-task predictions using MtgbmModel wrapper.

    The wrapper's predict() method handles:
    - Feature extraction
    - Multi-task prediction
    - Sigmoid transformation

    Args:
        model: MtgbmModel wrapper
        df: Preprocessed DataFrame
        feature_columns: List of feature column names

    Returns:
        np.ndarray of shape (n_samples, n_tasks) with probabilities (sigmoid applied)
    """
    # MtgbmModel.predict() handles everything internally
    predictions = model.predict(df, feature_columns)

    logger.info(f"Generated multi-task predictions: shape {predictions.shape}")
    return predictions


def predict_fn(
    input_data: pd.DataFrame, model_artifacts: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Generate predictions from preprocessed input data.

    Optimized for single-record inference with fast path detection.
    Multi-task model returns predictions for all tasks simultaneously.

    Args:
        input_data: DataFrame containing the preprocessed input
        model_artifacts: Dictionary containing model and preprocessing objects

    Returns:
        Dict[str, np.ndarray]: Dictionary with raw and calibrated predictions

    Raises:
        ValueError: If input data is invalid or missing required features
    """
    try:
        # Extract configuration
        model = model_artifacts["model"]
        risk_processors = model_artifacts["risk_processors"]
        numerical_processors = model_artifacts["numerical_processors"]
        config = model_artifacts["config"]
        feature_columns = config["feature_columns"]
        num_tasks = config["num_tasks"]
        calibrator = model_artifacts.get("calibrator")

        # Validate input
        validate_input_data(input_data, feature_columns)

        # FAST PATH: Single-record inference (10-100x faster preprocessing)
        if len(input_data) == 1:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Using fast path for single-record multi-task inference")

            # Assign column names if needed
            df = assign_column_names(input_data, feature_columns)

            # Process single record with fast path (bypasses pandas operations)
            processed_values = preprocess_single_record_fast(
                df=df,
                feature_columns=feature_columns,
                risk_processors=risk_processors,
                numerical_processors=numerical_processors,
            )

            # Create DataFrame for MtgbmModel.predict()
            preprocessed_df = pd.DataFrame(
                processed_values.reshape(1, -1), columns=feature_columns
            )
        else:
            # BATCH PATH: Original DataFrame processing for multiple records
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using batch path for {len(input_data)} records")

            # Assign column names if needed
            df = assign_column_names(input_data, feature_columns)

            # Apply preprocessing
            df = apply_preprocessing(
                df, feature_columns, risk_processors, numerical_processors
            )

            # Convert to numeric
            df = convert_to_numeric(df, feature_columns)

            # Prepare DataFrame for model
            preprocessed_df = df[feature_columns]

        # Generate predictions using wrapper (sigmoid already applied)
        raw_predictions = generate_multitask_predictions(
            model, preprocessed_df, feature_columns
        )

        # Validate output shape matches expected task count
        task_names = config["task_names"]
        expected_tasks = len(task_names)
        actual_tasks = raw_predictions.shape[1]

        if actual_tasks != expected_tasks:
            raise ValueError(
                f"Prediction shape mismatch! Expected {expected_tasks} tasks, "
                f"got {actual_tasks} outputs. Task names: {task_names}"
            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"✓ Predictions validated: {actual_tasks} tasks match "
                f"{expected_tasks} task names: {task_names}"
            )

        # Apply calibration if available, otherwise use raw predictions
        if calibrator is not None:
            try:
                calibrated_predictions = apply_multitask_calibration(
                    raw_predictions, calibrator["data"]
                )
                if logger.isEnabledFor(logging.INFO):
                    logger.info("Applied per-task calibration to predictions")
            except Exception as e:
                logger.warning(
                    f"Failed to apply calibration, using raw predictions: {e}"
                )
                calibrated_predictions = raw_predictions.copy()
        else:
            # No calibrator available, use raw predictions
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "No calibration models found, using raw predictions for calibrated output"
                )
            calibrated_predictions = raw_predictions.copy()

        return {
            "raw_predictions": raw_predictions,
            "calibrated_predictions": calibrated_predictions,
            "task_names": task_names,  # Pass task names for output formatting
        }

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        if logger.isEnabledFor(logging.ERROR):
            logger.error("Input data types and unique values:")
            for col in feature_columns:
                if col in input_data.columns:
                    logger.error(
                        f"{col}: dtype={input_data[col].dtype}, unique values={input_data[col].unique()}"
                    )
        raise


# --------------------------------------------------------------------------------
#                           OUTPUT BLOCK
# --------------------------------------------------------------------------------


def normalize_predictions(
    prediction_output: Union[np.ndarray, List, Dict[str, np.ndarray]],
) -> Tuple[List[List[float]], List[List[float]], int]:
    """
    Normalize prediction output into a consistent format for multi-task models.

    Args:
        prediction_output: Raw prediction output from model or dict with raw and calibrated predictions

    Returns:
        Tuple of (raw scores list, calibrated scores list, num_tasks)

    Raises:
        ValueError: If prediction format is invalid
    """
    # Handle the new dictionary output format
    if isinstance(prediction_output, dict):
        raw_predictions = prediction_output.get("raw_predictions")
        calibrated_predictions = prediction_output.get("calibrated_predictions")

        if raw_predictions is None:
            raise ValueError("Missing raw predictions in output dictionary")

        # Convert raw predictions to list format
        if isinstance(raw_predictions, np.ndarray):
            raw_scores_list = raw_predictions.tolist()
        elif isinstance(raw_predictions, list):
            raw_scores_list = raw_predictions
        else:
            msg = f"Unsupported raw prediction type: {type(raw_predictions)}"
            logger.error(msg)
            raise ValueError(msg)

        # Convert calibrated predictions to list format
        if calibrated_predictions is not None:
            if isinstance(calibrated_predictions, np.ndarray):
                calibrated_scores_list = calibrated_predictions.tolist()
            elif isinstance(calibrated_predictions, list):
                calibrated_scores_list = calibrated_predictions
            else:
                msg = f"Unsupported calibrated prediction type: {type(calibrated_predictions)}"
                logger.error(msg)
                calibrated_scores_list = raw_scores_list  # Fallback to raw scores
        else:
            # If no calibrated predictions, use raw scores
            calibrated_scores_list = raw_scores_list
    else:
        # Legacy code path for direct numpy array or list input
        if isinstance(prediction_output, np.ndarray):
            logger.info(
                f"Prediction output numpy array shape: {prediction_output.shape}"
            )
            raw_scores_list = prediction_output.tolist()
        elif isinstance(prediction_output, list):
            raw_scores_list = prediction_output
        else:
            msg = f"Unsupported prediction output type: {type(prediction_output)}"
            logger.error(msg)
            raise ValueError(msg)

        # In legacy mode, calibrated scores are same as raw scores
        calibrated_scores_list = raw_scores_list

    if not raw_scores_list:
        raise ValueError("Empty prediction output")

    # Check if the predictions are already in list format
    if not isinstance(raw_scores_list[0], list):
        # Single probability output, convert to list of lists
        raw_scores_list = [[score] for score in raw_scores_list]
        if calibrated_scores_list == raw_scores_list:
            calibrated_scores_list = [[score] for score in calibrated_scores_list]

    # Get number of tasks (columns in prediction matrix)
    num_tasks = len(raw_scores_list[0])

    logger.debug(f"Number of tasks: {num_tasks}")
    return raw_scores_list, calibrated_scores_list, num_tasks


def format_json_record(
    raw_probs: List[float],
    calibrated_probs: List[float],
    task_names: List[str],
) -> Dict[str, Any]:
    """
    Format a single multi-task prediction record for JSON output.

    Output structure uses task-specific field names:
    - {task_name}_raw: Raw task probability
    - {task_name}_calibrated: Calibrated task probability

    Example output for tasks ["is_abuse_true", "is_spam_true"]:
    {
        "is_abuse_prob_raw": "0.7234",
        "is_abuse_prob_calibrated": "0.6891",
        "is_spam_prob_raw": "0.3156",
        "is_spam_prob_calibrated": "0.2943"
    }

    Args:
        raw_probs: List of raw task probabilities
        calibrated_probs: List of calibrated task probabilities
        task_names: List of task label names (e.g., ["is_abuse_true", "is_spam_true"])

    Returns:
        Dictionary containing formatted prediction record with task-specific names
    """
    if not raw_probs:
        raise ValueError("Empty probability list")

    # Ensure calibrated_probs exists, use raw_probs as fallback
    if calibrated_probs is None or len(calibrated_probs) != len(raw_probs):
        calibrated_probs = raw_probs

    # Use task-specific field names
    record = {}
    for i, task_label_name in enumerate(task_names):
        # Convert label name to score field name
        # "is_abuse" → "is_abuse_prob" or "is_abuse_true" → "is_abuse_prob"
        score_field_name = f"{task_label_name.replace('_true', '')}_prob"

        record[f"{score_field_name}_raw"] = str(raw_probs[i])
        record[f"{score_field_name}_calibrated"] = str(calibrated_probs[i])

    return record


def format_json_response(
    raw_scores_list: List[List[float]],
    calibrated_scores_list: List[List[float]],
    task_names: List[str],
) -> Tuple[str, str]:
    """
    Format multi-task predictions as JSON response with task-specific field names.

    Args:
        raw_scores_list: List of raw prediction scores
        calibrated_scores_list: List of calibrated prediction scores
        task_names: List of task label names

    Returns:
        Tuple of (JSON response string, content type)

    Example Output for tasks ["is_abuse_true", "is_spam_true"]:
        {
          "predictions": [
            {
              "is_abuse_prob_raw": "0.7234",
              "is_abuse_prob_calibrated": "0.6891",
              "is_spam_prob_raw": "0.3156",
              "is_spam_prob_calibrated": "0.2943"
            }
          ]
        }
    """
    output_records = [
        format_json_record(raw_probs, cal_probs, task_names)
        for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list)
    ]

    response = json.dumps({"predictions": output_records})
    return response, CONTENT_TYPE_JSON


def format_csv_response(
    raw_scores_list: List[List[float]],
    calibrated_scores_list: List[List[float]],
    task_names: List[str],
) -> Tuple[str, str]:
    """
    Format multi-task predictions as CSV response with task-specific header.

    Args:
        raw_scores_list: List of raw prediction scores
        calibrated_scores_list: List of calibrated prediction scores
        task_names: List of task label names

    Returns:
        Tuple of (CSV response string, content type)

    Example Output for tasks ["is_abuse_true", "is_spam_true"]:
        is_abuse_prob_raw,is_abuse_prob_calibrated,is_spam_prob_raw,is_spam_prob_calibrated
        0.7234,0.6891,0.3156,0.2943
        0.2156,0.1987,0.6789,0.6521
    """
    # Ensure calibrated scores exist, use raw scores as fallback
    if calibrated_scores_list is None or len(calibrated_scores_list) != len(
        raw_scores_list
    ):
        calibrated_scores_list = raw_scores_list

    # Generate header with task-specific names
    header_parts = []
    for task_label_name in task_names:
        score_field_name = f"{task_label_name.replace('_true', '')}_prob"
        header_parts.append(f"{score_field_name}_raw")
        header_parts.append(f"{score_field_name}_calibrated")

    csv_lines = [",".join(header_parts)]

    # Data rows
    for raw_probs, calibrated_probs in zip(raw_scores_list, calibrated_scores_list):
        line = []
        for i in range(len(raw_probs)):
            line.append(f"{raw_probs[i]:.4f}")
            line.append(f"{calibrated_probs[i]:.4f}")
        csv_lines.append(",".join(line))

    response_body = "\n".join(csv_lines) + "\n"
    return response_body, CONTENT_TYPE_CSV


def output_fn(
    prediction_output: Union[np.ndarray, List, Dict[str, np.ndarray]],
    accept: str = CONTENT_TYPE_JSON,
) -> Tuple[str, str]:
    """
    Serializes the multi-task prediction output with task-specific field names.

    SageMaker Handler Contract: Only receives prediction_output and accept parameters.
    Task names are extracted from the prediction_output dict (populated by predict_fn).

    Args:
        prediction_output: Model predictions dict with keys:
                          - 'raw_predictions': Raw probabilities (n_samples, n_tasks)
                          - 'calibrated_predictions': Calibrated probabilities
                          - 'task_names': List of task label names (REQUIRED)
        accept: The requested response MIME type

    Returns:
        Tuple[str, str]: (response_body, content_type)

    Raises:
        ValueError: If prediction output format is invalid, task_names is missing,
                   or content type is unsupported
    """
    logger.info(
        f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}"
    )

    try:
        # Extract task_names from prediction_output dict
        task_names = None
        if isinstance(prediction_output, dict) and "task_names" in prediction_output:
            task_names = prediction_output["task_names"]
            logger.info(f"Extracted task_names from prediction output: {task_names}")

        # Validate task_names available
        if task_names is None:
            raise ValueError(
                "task_names is required in prediction_output dict. "
                "Ensure predict_fn includes 'task_names' in its return dictionary."
            )

        # Normalize prediction format
        raw_scores_list, calibrated_scores_list, num_tasks = normalize_predictions(
            prediction_output
        )

        # Validate task_names count matches prediction output
        if len(task_names) != num_tasks:
            raise ValueError(
                f"Task name count mismatch: {len(task_names)} names provided "
                f"for {num_tasks} tasks in predictions"
            )

        # Format response based on accept type
        if accept.lower() == CONTENT_TYPE_JSON:
            return format_json_response(
                raw_scores_list, calibrated_scores_list, task_names
            )

        elif accept.lower() == CONTENT_TYPE_CSV:
            return format_csv_response(
                raw_scores_list, calibrated_scores_list, task_names
            )

        else:
            logger.error(f"Unsupported accept type: {accept}")
            error_msg = (
                f"Unsupported accept type: {accept}. "
                f"Supported types are {CONTENT_TYPE_JSON} and {CONTENT_TYPE_CSV}"
            )
            raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"Error during output formatting: {e}", exc_info=True)
        error_response = json.dumps({"error": f"Failed to format output: {e}"})
        return error_response, CONTENT_TYPE_JSON
