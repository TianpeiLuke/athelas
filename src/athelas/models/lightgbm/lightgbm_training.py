#!/usr/bin/env python3
import os
import sys

from subprocess import check_call
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI (default)
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
        USE_SECURE_PYPI: Set to "true" to use secure PyPI, "false" for public PyPI

    Example:
        # Install from public PyPI (default)
        install_packages(["pandas==1.5.0", "numpy"])

        # Install from secure PyPI
        os.environ["USE_SECURE_PYPI"] = "true"
        install_packages(["pandas==1.5.0", "numpy"])
    """
    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    logger.info(
        f"Environment Variable USE_SECURE_PYPI: {os.environ.get('USE_SECURE_PYPI', 'not set')}"
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
# Note: SageMaker PyTorch 2.1.2 containers pre-install: scikit-learn, pandas, matplotlib, seaborn
# Including them here is safe (pip will skip if already satisfied)
required_packages = [
    "lightgbm>=3.3.0,<4.0.0",  # LightGBM - NOT pre-installed
    "beautifulsoup4>=4.9.3",  # HTML parsing - NOT pre-installed
    "pyarrow>=4.0.0,<6.0.0",  # Parquet support - NOT pre-installed
    "pydantic>=2.0.0,<3.0.0",  # Config validation - NOT pre-installed
    "matplotlib>=3.3.0,<3.7.0",  # Plotting - pre-installed but version may differ
    "typing-extensions>=4.2.0",  # Type hints - pre-installed but safe to include
]

# Install packages using unified installation function
install_packages(required_packages)

print("***********************Package Installation Complete*********************")

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import pickle as pkl
import lightgbm as lgb  # Import LightGBM instead of XGBoost

import tarfile
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)


# -------------------------------------------------------------------------
# Feature Selection Integration Functions
# -------------------------------------------------------------------------
def detect_feature_selection_artifacts(
    model_artifacts_dir: Optional[str],
) -> Optional[str]:
    """
    Conservatively detect if feature selection was applied in model_artifacts_input.
    Returns path to selected_features.json if found, None otherwise.

    Args:
        model_artifacts_dir: Path to model artifacts directory

    Returns:
        Path to selected_features.json if found, None otherwise
    """
    if not model_artifacts_dir or not os.path.exists(model_artifacts_dir):
        logger.info("No model artifacts directory - no feature selection artifacts")
        return None

    # Check for selected_features.json in model_artifacts_input
    features_path = os.path.join(model_artifacts_dir, "selected_features.json")

    if os.path.exists(features_path):
        logger.info(f"Feature selection artifacts detected at: {features_path}")
        return features_path

    logger.info("No feature selection artifacts found in model_artifacts_input")
    return None


def load_selected_features(fs_artifacts_path: str) -> Optional[List[str]]:
    """
    Load selected features from feature selection artifacts.

    Args:
        fs_artifacts_path: Path to selected_features.json

    Returns:
        List of selected feature names, or None if loading fails
    """
    try:
        with open(fs_artifacts_path, "r") as f:
            fs_data = json.load(f)

        selected_features = fs_data.get("selected_features", [])
        if not selected_features:
            logger.warning("Empty selected_features list found")
            return None

        logger.info(f"Loaded {len(selected_features)} selected features from artifacts")
        logger.info(f"Selected features: {selected_features}")
        return selected_features

    except Exception as e:
        logger.warning(f"Error loading feature selection artifacts: {e}")
        return None


def get_effective_feature_columns(
    config: dict, model_artifacts_dir: Optional[str], train_df: pd.DataFrame
) -> Tuple[List[str], bool]:
    """
    Get feature columns with fallback-first approach.

    Args:
        config: Configuration dictionary
        model_artifacts_dir: Path to model artifacts directory
        train_df: Training dataframe for validation

    Returns:
        Tuple of (feature_columns, feature_selection_applied)
    """
    # STEP 1: Always start with original behavior
    original_features = config["tab_field_list"] + config["cat_field_list"]

    logger.info("=== FEATURE SELECTION DETECTION ===")
    logger.info(f"Original configuration features: {len(original_features)}")

    # STEP 2: Check if feature selection artifacts exist
    fs_artifacts_path = detect_feature_selection_artifacts(model_artifacts_dir)
    if fs_artifacts_path is None:
        # NO FEATURE SELECTION - Original behavior exactly
        logger.info(
            "Using original feature configuration (no feature selection detected)"
        )
        logger.info("=====================================")
        return original_features, False

    # STEP 3: Feature selection detected - try to load
    selected_features = load_selected_features(fs_artifacts_path)
    if selected_features is None:
        logger.warning(
            "Failed to load selected features - falling back to original behavior"
        )
        logger.info("=====================================")
        return original_features, False

    # STEP 4: Validate selected features exist in data
    available_columns = set(train_df.columns)
    missing_features = [f for f in selected_features if f not in available_columns]

    if missing_features:
        logger.warning(f"Selected features missing from data: {missing_features}")
        logger.warning("Falling back to original behavior")
        logger.info("=====================================")
        return original_features, False

    # STEP 5: Additional validation - ensure reasonable subset
    if len(selected_features) > len(original_features):
        logger.warning(
            f"Selected features ({len(selected_features)}) more than original ({len(original_features)}) - suspicious"
        )
        logger.warning("Falling back to original behavior")
        logger.info("=====================================")
        return original_features, False

    # STEP 6: Success - use selected features
    logger.info(f"Feature selection successfully applied!")
    logger.info(
        f"Features reduced from {len(original_features)} to {len(selected_features)}"
    )
    logger.info(
        f"Reduction ratio: {len(selected_features) / len(original_features):.2%}"
    )
    logger.info("=====================================")
    return selected_features, True


# -------------------------------------------------------------------------
# Assuming the processor is in a directory that can be imported
# -------------------------------------------------------------------------
from processing.categorical.risk_table_processor import RiskTableMappingProcessor
from processing.categorical.dictionary_encoding_processor import (
    DictionaryEncodingProcessor,
)
from processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)


# -------------------------------------------------------------------------
# Logging setup - Updated for CloudWatch compatibility
# -------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility"""
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            # StreamHandler with stdout for CloudWatch
            logging.StreamHandler(sys.stdout)
        ],
    )

    # Configure our module's logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Allow propagation to root logger

    # Force flush stdout
    sys.stdout.flush()

    return logger


# Initialize logger
logger = setup_logging()

# -------------------------------------------------------------------------
# Pydantic V2 model for all hyperparameters
# -------------------------------------------------------------------------
from pydantic import BaseModel, Field, model_validator
from hyperparams.hyperparameters_lightgbm import LightGBMModelHyperparameters


class LightGBMConfig(LightGBMModelHyperparameters):
    """
    Load everything from your pipeline's LightGBMModelHyperparameters,
    plus the two risk-table params this script needs.
    """

    smooth_factor: float = Field(
        default=0.0, description="Smoothing factor for risk table"
    )
    count_threshold: int = Field(
        default=0, description="Minimum count threshold for risk table"
    )


# -------------------------------------------------------------------------
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# -------------------------------------------------------------------------


def _detect_file_format(file_path: str) -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    from pathlib import Path

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
    """
    Load DataFrame and detect its format.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (DataFrame, format_string)
    """
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
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        output_path: Base output path (without extension)
        format_str: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
    from pathlib import Path

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


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_and_validate_config(hparam_path: str) -> dict:
    """Loads and validates the hyperparameters JSON file."""
    try:
        with open(hparam_path, "r") as f:
            config = json.load(f)

        required_keys = [
            "tab_field_list",
            "cat_field_list",
            "label_name",
            "multiclass_categories",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

        if "num_classes" not in config:
            config["num_classes"] = len(config["multiclass_categories"])

        if "is_binary" not in config:
            config["is_binary"] = config["num_classes"] == 2

        # Validate class_weights if present
        if "class_weights" in config:
            if len(config["class_weights"]) != config["num_classes"]:
                raise ValueError(
                    f"Number of class weights ({len(config['class_weights'])}) "
                    f"does not match number of classes ({config['num_classes']})"
                )

        return config
    except Exception as err:
        logger.error(f"Failed to load/validate hyperparameters: {err}")
        raise


def find_first_data_file(data_dir: str) -> str:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)
    return None


def load_datasets(
    input_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Loads the training, validation, and test datasets with format detection.

    Returns:
        Tuple of (train_df, val_df, test_df, detected_format)
    """
    train_file = find_first_data_file(os.path.join(input_path, "train"))
    val_file = find_first_data_file(os.path.join(input_path, "val"))
    test_file = find_first_data_file(os.path.join(input_path, "test"))

    if not train_file or not val_file or not test_file:
        raise FileNotFoundError(
            "Training, validation, or test data file not found in the expected subfolders."
        )

    # Load with format detection
    train_df, train_format = load_dataframe_with_format(train_file)
    val_df, val_format = load_dataframe_with_format(val_file)
    test_df, test_format = load_dataframe_with_format(test_file)

    # Use training data format as the primary format
    detected_format = train_format
    logger.info(f"Detected input format: {detected_format}")

    if val_format != detected_format or test_format != detected_format:
        logger.warning(
            f"Mixed formats detected - train:{train_format}, val:{val_format}, test:{test_format}. "
            f"Using train format ({detected_format}) for outputs."
        )

    logger.info(
        f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
    )
    return train_df, val_df, test_df, detected_format


def apply_numerical_imputation(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """
    Applies numerical imputation to the datasets using single-column architecture.

    Creates one processor per numerical column, similar to risk table mapping.
    """
    imputation_processors = {}
    train_df_imputed = train_df.copy()
    val_df_imputed = val_df.copy()
    test_df_imputed = test_df.copy()

    # Create one processor per numerical column (single-column architecture)
    for var in config["tab_field_list"]:
        proc = NumericalVariableImputationProcessor(column_name=var, strategy="mean")
        proc.fit(train_df[var])  # Fit on single column Series for consistency
        imputation_processors[var] = proc

        # Transform each split
        train_df_imputed[var] = proc.transform(train_df_imputed[var])
        val_df_imputed[var] = proc.transform(val_df_imputed[var])
        test_df_imputed[var] = proc.transform(test_df_imputed[var])

    # Build imputation dictionary for artifact saving
    impute_dict = {
        var: proc.get_imputation_value() for var, proc in imputation_processors.items()
    }

    return (
        train_df_imputed,
        val_df_imputed,
        test_df_imputed,
        impute_dict,
    )


def fit_and_apply_risk_tables(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Fits risk tables on training data and applies them to all splits."""
    risk_processors = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy()

    for var in config["cat_field_list"]:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config["label_name"],
            smooth_factor=config.get("smooth_factor", 0.0),
            count_threshold=config.get("count_threshold", 0),
        )
        proc.fit(train_df)
        risk_processors[var] = proc

        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
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


def prepare_datasets(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[lgb.Dataset, lgb.Dataset, List[str]]:
    """
    Prepares LightGBM Dataset objects from dataframes.

    Returns:
        Tuple containing:
        - Training Dataset
        - Validation Dataset
        - List of feature columns in the exact order used for the model
    """
    # Maintain exact ordering of features as they'll be used in the model
    feature_columns = config["tab_field_list"] + config["cat_field_list"]

    # Prepare data matrices with proper typing
    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    # Check if using native categorical features
    use_native_cat = config.get("use_native_categorical", True)

    if use_native_cat:
        # Ensure numerical features are float and categorical features are int
        for col in config["tab_field_list"]:
            X_train[col] = X_train[col].astype("float32")
            X_val[col] = X_val[col].astype("float32")

        for col in config["cat_field_list"]:
            X_train[col] = X_train[col].astype("int32")
            X_val[col] = X_val[col].astype("int32")
    else:
        # Risk table mode - all features are float
        X_train = X_train.astype("float32")
        X_val = X_val.astype("float32")

    # Check for any remaining NaN/inf values
    if X_train.isna().any().any() or np.isinf(X_train.values).any():
        raise ValueError("Training data contains NaN or inf values after preprocessing")
    if X_val.isna().any().any() or np.isinf(X_val.values).any():
        raise ValueError(
            "Validation data contains NaN or inf values after preprocessing"
        )

    # Get labels
    y_train = train_df[config["label_name"]].astype(int).values
    y_val = val_df[config["label_name"]].astype(int).values

    # Handle class weights for multiclass - create sample weights if needed
    sample_weights_train = None
    if not config.get("is_binary", True) and "class_weights" in config:
        sample_weights_train = np.ones(len(y_train))
        for i, weight in enumerate(config["class_weights"]):
            sample_weights_train[y_train == i] = weight

    # Specify categorical features for LightGBM if using native categorical
    categorical_feature = config["cat_field_list"] if use_native_cat else None

    if categorical_feature:
        logger.info(
            f"Specifying categorical features for LightGBM: {categorical_feature}"
        )

    # Create LightGBM Datasets
    train_set = lgb.Dataset(
        X_train.values,
        label=y_train,
        weight=sample_weights_train,  # Set weight during creation for multiclass
        feature_name=feature_columns,
        categorical_feature=categorical_feature,  # Specify categorical features
        free_raw_data=False,
    )

    val_set = lgb.Dataset(
        X_val.values,
        label=y_val,
        feature_name=feature_columns,
        categorical_feature=categorical_feature,  # Specify categorical features
        reference=train_set,  # Reference to training set for consistency
        free_raw_data=False,
    )

    return train_set, val_set, feature_columns


def train_model(
    config: dict, train_set: lgb.Dataset, val_set: lgb.Dataset
) -> lgb.Booster:
    """
    Trains the LightGBM model.

    Args:
        config: Configuration dictionary containing model parameters
        train_set: Training data as LightGBM Dataset
        val_set: Validation data as LightGBM Dataset

    Returns:
        Trained LightGBM model
    """
    # Map XGBoost parameters to LightGBM equivalents
    lgb_params = {
        "learning_rate": config.get("eta", 0.1),
        "min_split_gain": config.get("gamma", 0),
        "max_depth": config.get("max_depth", 6),
        "bagging_fraction": config.get("subsample", 1),
        "feature_fraction": config.get("colsample_bytree", 1),
        "lambda_l2": config.get("lambda_xgb", 1),
        "lambda_l1": config.get("alpha_xgb", 0),
        "bagging_freq": 1 if config.get("subsample", 1) < 1 else 0,
        "verbose": -1,
    }

    # Add categorical feature parameters if using native categorical
    use_native_cat = config.get("use_native_categorical", True)
    if use_native_cat:
        lgb_params["min_data_per_group"] = config.get("min_data_per_group", 100)
        lgb_params["cat_smooth"] = config.get("cat_smooth", 10.0)
        lgb_params["max_cat_threshold"] = config.get("max_cat_threshold", 32)
        logger.info(
            f"Added categorical parameters: min_data_per_group={lgb_params['min_data_per_group']}, "
            f"cat_smooth={lgb_params['cat_smooth']}, max_cat_threshold={lgb_params['max_cat_threshold']}"
        )

    # Set objective and handle class weights
    if config.get("is_binary", True):
        lgb_params["objective"] = "binary"
        if "class_weights" in config and len(config["class_weights"]) == 2:
            # For binary classification, use scale_pos_weight
            lgb_params["scale_pos_weight"] = (
                config["class_weights"][1] / config["class_weights"][0]
            )
    else:
        lgb_params["objective"] = "multiclass"
        lgb_params["num_class"] = config["num_classes"]

    logger.info(f"Starting LightGBM training with params: {lgb_params}")
    logger.info(f"Number of classes from config: {config.get('num_classes', 2)}")

    # Print label distribution for debugging
    y_train = train_set.get_label()
    y_val = val_set.get_label()
    logger.info(
        f"Label distribution in training data: {pd.Series(y_train).value_counts().sort_index()}"
    )
    logger.info(
        f"Label distribution in validation data: {pd.Series(y_val).value_counts().sort_index()}"
    )

    # Note: Class weights for multiclass are already handled in prepare_datasets via weight parameter

    # Create callbacks for training
    callbacks = [lgb.log_evaluation(period=1)]

    # Add early stopping if configured
    if config.get("early_stopping_rounds"):
        callbacks.append(
            lgb.early_stopping(stopping_rounds=config.get("early_stopping_rounds", 10))
        )

    return lgb.train(
        params=lgb_params,
        train_set=train_set,
        num_boost_round=config.get("num_round", 100),
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )


def save_artifacts(
    model: lgb.Booster,
    risk_tables: dict,
    impute_dict: dict,
    model_path: str,
    feature_columns: List[str],
    config: dict,
    categorical_mappings: dict = None,
):
    """
    Saves the trained model and preprocessing artifacts.

    Args:
        model: Trained LightGBM model
        risk_tables: Dictionary of risk tables
        impute_dict: Dictionary of imputation values
        model_path: Path to save model artifacts
        feature_columns: List of feature column names
        config: Configuration dictionary containing hyperparameters
        categorical_mappings: Dictionary of categorical feature encodings (optional)
    """
    os.makedirs(model_path, exist_ok=True)

    # Save LightGBM model
    model_file = os.path.join(model_path, "lightgbm_model.txt")
    model.save_model(model_file)
    logger.info(f"Saved LightGBM model to {model_file}")

    # Save preprocessing artifacts based on mode
    use_native_cat = config.get("use_native_categorical", True)

    if use_native_cat:
        # Save categorical mappings for native categorical mode
        if categorical_mappings:
            cat_mappings_file = os.path.join(model_path, "categorical_mappings.pkl")
            with open(cat_mappings_file, "wb") as f:
                pkl.dump(categorical_mappings, f)
            logger.info(f"Saved categorical mappings to {cat_mappings_file}")

            # Also save as JSON for readability
            cat_mappings_json = os.path.join(model_path, "categorical_mappings.json")
            with open(cat_mappings_json, "w") as f:
                json.dump(categorical_mappings, f, indent=2)
            logger.info(f"Saved categorical mappings (JSON) to {cat_mappings_json}")

        logger.info("Using native categorical features - risk tables not saved")
    else:
        # Save risk tables for XGBoost-style mode
        if risk_tables:
            risk_map_file = os.path.join(model_path, "risk_table_map.pkl")
            with open(risk_map_file, "wb") as f:
                pkl.dump(risk_tables, f)
            logger.info(f"Saved consolidated risk table map to {risk_map_file}")
        else:
            logger.warning("No risk tables to save in risk table mode")

    # Save imputation dictionary (used in both modes)
    impute_file = os.path.join(model_path, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")

    # Save categorical configuration
    cat_config = {
        "use_native_categorical": use_native_cat,
        "categorical_features": config["cat_field_list"],
        "min_data_per_group": config.get("min_data_per_group", 100),
        "cat_smooth": config.get("cat_smooth", 10.0),
        "max_cat_threshold": config.get("max_cat_threshold", 32),
    }
    cat_config_file = os.path.join(model_path, "categorical_config.json")
    with open(cat_config_file, "w") as f:
        json.dump(cat_config, f, indent=2)
    logger.info(f"Saved categorical configuration to {cat_config_file}")

    # Save feature importance
    fmap_json = os.path.join(model_path, "feature_importance.json")
    with open(fmap_json, "w") as f:
        # LightGBM returns numpy array, need to map to feature names
        importance_dict = dict(
            zip(feature_columns, model.feature_importance().tolist())
        )
        json.dump(importance_dict, f, indent=2)
    logger.info(f"Saved feature importance to {fmap_json}")

    # Save feature columns with ordering information
    feature_columns_file = os.path.join(model_path, "feature_columns.txt")
    with open(feature_columns_file, "w") as f:
        # Add a header comment to document the importance of ordering
        f.write(
            "# Feature columns in exact order required for LightGBM model inference\n"
        )
        f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
        f.write("# Each line contains: <column_index>,<column_name>\n")
        for idx, column in enumerate(feature_columns):
            f.write(f"{idx},{column}\n")
    logger.info(f"Saved ordered feature columns to {feature_columns_file}")

    # Save hyperparameters configuration
    hyperparameters_file = os.path.join(model_path, "hyperparameters.json")
    with open(hyperparameters_file, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    logger.info(f"Saved hyperparameters configuration to {hyperparameters_file}")


# -------------------------------------------------------------------------
# New: inference + evaluation helpers
# -------------------------------------------------------------------------
def save_preds_and_metrics(
    ids, y_true, y_prob, id_col, label_col, out_dir, is_binary, output_format="csv"
):
    """
    Save predictions and metrics with format preservation.

    Args:
        output_format: Format to save predictions in ('csv', 'tsv', or 'parquet')
    """
    os.makedirs(out_dir, exist_ok=True)
    # metrics
    metrics = {}
    if is_binary:
        score = y_prob[:, 1]
        metrics = {
            "auc_roc": roc_auc_score(y_true, score),
            "average_precision": average_precision_score(y_true, score),
            "f1_score": f1_score(y_true, score > 0.5),
        }
        logger.info(f"AUC-ROC: {metrics['auc_roc']}")
        logger.info(f"Average Precision: {metrics['average_precision']}")
        logger.info(f"F1-Score: {metrics['f1_score']}")
    else:
        n = y_prob.shape[1]
        for i in range(n):
            y_bin = (y_true == i).astype(int)
            metrics[f"auc_roc_class_{i}"] = roc_auc_score(y_bin, y_prob[:, i])
            metrics[f"average_precision_class_{i}"] = average_precision_score(
                y_bin, y_prob[:, i]
            )
            metrics[f"f1_score_class_{i}"] = f1_score(y_bin, y_prob[:, i] > 0.5)
        metrics["auc_roc_micro"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="micro"
        )
        metrics["auc_roc_macro"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
        metrics["average_precision_micro"] = average_precision_score(
            y_true, y_prob, average="micro"
        )
        metrics["average_precision_macro"] = average_precision_score(
            y_true, y_prob, average="macro"
        )
        y_pred = np.argmax(y_prob, axis=1)
        metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")
        metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")
        logger.info(f"AUC-ROC (micro): {metrics['auc_roc_micro']}")
        logger.info(f"AUC-ROC (macro): {metrics['auc_roc_macro']}")
        logger.info(f"Average Precision (micro): {metrics['average_precision_micro']}")
        logger.info(f"Average Precision (macro): {metrics['average_precision_macro']}")
        logger.info(f"F1-Score (micro): {metrics['f1_score_micro']}")
        logger.info(f"F1-Score (macro): {metrics['f1_score_macro']}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions with format preservation
    df = pd.DataFrame({id_col: ids, label_col: y_true})
    for i in range(y_prob.shape[1]):
        df[f"prob_class_{i}"] = y_prob[:, i]

    output_base = os.path.join(out_dir, "predictions")
    saved_path = save_dataframe_with_format(df, output_base, output_format)
    logger.info(f"Saved predictions (format={output_format}): {saved_path}")


def plot_curves(y_true, y_prob, out_dir, prefix, is_binary):
    os.makedirs(out_dir, exist_ok=True)
    if is_binary:
        score = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"{prefix} ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}roc.jpg"))
        plt.close()
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        plt.figure()
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.title(f"{prefix} PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}pr.jpg"))
        plt.close()
    else:
        n = y_prob.shape[1]
        for i in range(n):
            y_bin = (y_true == i).astype(int)
            if len(np.unique(y_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                auc = roc_auc_score(y_bin, y_prob[:, i])
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.title(f"{prefix} class {i} ROC")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.legend()
                plt.savefig(os.path.join(out_dir, f"{prefix}class_{i}_roc.jpg"))
                plt.close()
                precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, i])
                ap = average_precision_score(y_bin, y_prob[:, i])
                plt.figure()
                plt.plot(recall, precision, label=f"AP={ap:.3f}")
                plt.title(f"{prefix} class {i} PR")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend()
                plt.savefig(os.path.join(out_dir, f"{prefix}class_{i}_pr.jpg"))
                plt.close()


def evaluate_split(
    name, df, feats, model, cfg, output_format="csv", prefix="/opt/ml/output/data"
):
    """
    Evaluate a data split and save results with format preservation.

    Args:
        output_format: Format to save predictions in ('csv', 'tsv', or 'parquet')
    """
    is_bin = cfg.get("is_binary", True)
    label = cfg["label_name"]
    idi = cfg.get("id_name", "id")

    ids = df.get(idi, np.arange(len(df)))
    y_true = df[label].astype(int).values

    # LightGBM predicts directly from numpy array
    X = df[feats].values
    y_prob = model.predict(X)

    # Handle binary output format
    if y_prob.ndim == 1:
        y_prob = np.vstack([1 - y_prob, y_prob]).T

    # directories
    out_base = os.path.join(prefix, name)
    out_metrics = os.path.join(prefix, f"{name}_metrics")

    # save preds & metrics, then plots, then tar
    save_preds_and_metrics(
        ids, y_true, y_prob, idi, label, out_base, is_bin, output_format
    )
    plot_curves(y_true, y_prob, out_metrics, f"{name}_", is_bin)

    tar = os.path.join(prefix, f"{name}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(out_base, arcname=name)
        t.add(out_metrics, arcname=f"{name}_metrics")

    logger.info(f"{name} outputs packaged → {tar}")


# -------------------------------------------------------------------------
# PRE-COMPUTED ARTIFACT LOADING AND APPLICATION
# -------------------------------------------------------------------------
def load_precomputed_artifacts(
    model_artifacts_dir: Optional[str],
    use_imputation: bool,
    use_risk_tables: bool,
    use_features: bool,
) -> Dict[str, Any]:
    """
    Auto-detect and load pre-computed artifacts from model_artifacts_input directory.

    Args:
        model_artifacts_dir: Path to model artifacts directory
        use_imputation: Whether to use pre-computed imputation
        use_risk_tables: Whether to use pre-computed risk tables
        use_features: Whether to use pre-computed features

    Returns:
        Dictionary with loaded artifacts:
        {
            'impute_dict': dict or None,
            'risk_tables': dict or None,
            'selected_features': list or None,
            'loaded': {
                'imputation': bool,
                'risk_tables': bool,
                'features': bool
            }
        }
    """
    result = {
        "impute_dict": None,
        "risk_tables": None,
        "selected_features": None,
        "loaded": {"imputation": False, "risk_tables": False, "features": False},
    }

    if not model_artifacts_dir or not os.path.exists(model_artifacts_dir):
        logger.warning(
            f"Model artifacts directory not found or not provided: {model_artifacts_dir}"
        )
        return result

    logger.info(
        f"Attempting to load pre-computed artifacts from: {model_artifacts_dir}"
    )

    # 1. Try to load imputation dictionary
    if use_imputation:
        impute_path = os.path.join(model_artifacts_dir, "impute_dict.pkl")
        if os.path.exists(impute_path):
            try:
                with open(impute_path, "rb") as f:
                    result["impute_dict"] = pkl.load(f)
                result["loaded"]["imputation"] = True
                logger.info(
                    f"✓ Loaded pre-computed imputation dictionary from {impute_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load imputation dict: {e}. Will compute inline."
                )
        else:
            logger.warning(
                f"Imputation dict not found at {impute_path}. Will compute inline."
            )

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
                logger.warning(f"Failed to load risk tables: {e}. Will compute inline.")
        else:
            logger.warning(
                f"Risk tables not found at {risk_path}. Will compute inline."
            )

    # 3. Try to load selected features
    if use_features:
        features_path = os.path.join(model_artifacts_dir, "selected_features.json")
        if os.path.exists(features_path):
            try:
                with open(features_path, "r") as f:
                    result["selected_features"] = json.load(f)
                result["loaded"]["features"] = True
                logger.info(
                    f"✓ Loaded pre-computed feature selection from {features_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load features: {e}. Will compute inline.")
        else:
            logger.warning(
                f"Features not found at {features_path}. Will compute inline."
            )

    return result


def validate_precomputed_data_state(
    train_df: pd.DataFrame, config: dict, imputation_used: bool, risk_tables_used: bool
) -> None:
    """
    Validate that data state matches the pre-computed artifact flags.

    When using pre-computed artifacts, the incoming data should already be
    in the transformed state (imputed, risk-mapped, etc.).

    Args:
        train_df: Training DataFrame to validate
        config: Configuration dictionary
        imputation_used: Whether pre-computed imputation is being used
        risk_tables_used: Whether pre-computed risk tables are being used

    Raises:
        ValueError: If data state doesn't match the expected state
    """
    if imputation_used:
        # Verify data has no NaN in numerical columns
        tab_fields = config.get("tab_field_list", [])
        if tab_fields:
            nan_cols = (
                train_df[tab_fields].columns[train_df[tab_fields].isna().any()].tolist()
            )
            if nan_cols:
                raise ValueError(
                    f"USE_PRECOMPUTED_IMPUTATION=true but data contains NaN values in columns: {nan_cols}. "
                    "Data must be pre-imputed when using pre-computed imputation artifacts."
                )
            logger.info(
                "✓ Validated: Data has no NaN values (consistent with pre-computed imputation)"
            )

    if risk_tables_used:
        # Verify categorical columns are numeric (risk-mapped)
        cat_fields = config.get("cat_field_list", [])
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
# Main Orchestrator
# -------------------------------------------------------------------------
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function to execute the LightGBM training logic.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "input_path": Directory containing train/val/test data
            - "hyperparameters_s3_uri": Path to hyperparameters directory (now points to /opt/ml/code/hyperparams)
            - "model_artifacts_input": (Optional) Directory containing pre-computed model artifacts from previous steps
        output_paths: Dictionary of output paths with logical names
            - "model_output": Directory to save model artifacts
            - "evaluation_output": Directory to save evaluation outputs
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    try:
        logger.info("====== STARTING MAIN EXECUTION ======")

        # Extract paths from parameters using contract logical names
        data_dir = input_paths["input_path"]
        model_dir = output_paths["model_output"]
        output_dir = output_paths["evaluation_output"]

        # Optional: Get model artifacts input directory (for pre-computed parameters)
        model_artifacts_input_dir = input_paths.get("model_artifacts_input")
        if model_artifacts_input_dir:
            logger.info(f"Model artifacts input directory: {model_artifacts_input_dir}")
        else:
            logger.info(
                "No model_artifacts_input provided - will compute all parameters inline"
            )

        # Priority-based hyperparameters path resolution with region-specific support
        # Get region from environment variable
        region = environ_vars.get("REGION", "").upper()

        if region in ["NA", "EU", "FE"]:
            hparam_filename = f"hyperparameters_{region}.json"
            logger.info(f"Loading region-specific hyperparameters for region: {region}")
        else:
            hparam_filename = "hyperparameters.json"
            if region:
                logger.warning(
                    f"Unknown REGION '{region}', falling back to default hyperparameters.json"
                )
            else:
                logger.info("No REGION specified, using default hyperparameters.json")

        # Priority 1: Start with code directory (highest priority)
        hparam_path = f"/opt/ml/code/hyperparams/{hparam_filename}"

        # Priority 2: If code directory file doesn't exist, check input_paths
        if not os.path.exists(hparam_path):
            logger.info(f"Hyperparameters not found in code directory: {hparam_path}")

            if "hyperparameters_s3_uri" in input_paths:
                hparam_dir = input_paths["hyperparameters_s3_uri"]
                # If it's a directory path, append the filename
                if not hparam_dir.endswith(hparam_filename):
                    hparam_path = os.path.join(hparam_dir, hparam_filename)
                else:
                    hparam_path = hparam_dir
                logger.info(f"Using fallback hyperparameters path: {hparam_path}")
            else:
                logger.error("No hyperparameters_s3_uri provided in input_paths")
        else:
            logger.info(f"Found hyperparameters in code directory: {hparam_path}")

        logger.info(f"Loading hyperparameters from: {hparam_path}")

        logger.info("Starting LightGBM training process...")
        logger.info(f"Loading configuration from {hparam_path}")
        config = load_and_validate_config(hparam_path)
        logger.info("Configuration loaded successfully")

        logger.info("Loading datasets...")
        train_df, val_df, test_df, input_format = load_datasets(data_dir)
        logger.info("Datasets loaded successfully")

        # Store format in config for output preservation
        config["_input_format"] = input_format

        # Extract environment variables for preprocessing control
        use_precomputed_imputation = environ_vars.get(
            "USE_PRECOMPUTED_IMPUTATION", False
        )
        use_precomputed_risk_tables = environ_vars.get(
            "USE_PRECOMPUTED_RISK_TABLES", False
        )
        use_precomputed_features = environ_vars.get("USE_PRECOMPUTED_FEATURES", False)

        # Override use_native_categorical from environment variable if set
        if "USE_NATIVE_CATEGORICAL" in environ_vars:
            use_native_categorical_env = environ_vars.get("USE_NATIVE_CATEGORICAL")
            config["use_native_categorical"] = use_native_categorical_env
            logger.info(
                f"Overriding use_native_categorical from environment: {use_native_categorical_env}"
            )

        # ===== PREPROCESSING ARTIFACT CONTROL =====
        logger.info("=" * 70)
        logger.info("PREPROCESSING ARTIFACT CONTROL")
        logger.info("=" * 70)
        logger.info(f"USE_PRECOMPUTED_IMPUTATION: {use_precomputed_imputation}")
        logger.info(f"USE_PRECOMPUTED_RISK_TABLES: {use_precomputed_risk_tables}")
        logger.info(f"USE_PRECOMPUTED_FEATURES: {use_precomputed_features}")
        logger.info(
            f"USE_NATIVE_CATEGORICAL: {config.get('use_native_categorical', True)}"
        )
        logger.info(f"model_artifacts_input directory: {model_artifacts_input_dir}")
        logger.info("=" * 70)

        # Try to load pre-computed artifacts
        precomputed = load_precomputed_artifacts(
            model_artifacts_input_dir,
            use_precomputed_imputation,
            use_precomputed_risk_tables,
            use_precomputed_features,
        )

        # Validate data state matches pre-computed artifact flags
        validate_precomputed_data_state(
            train_df,
            config,
            precomputed["loaded"]["imputation"],
            precomputed["loaded"]["risk_tables"],
        )

        # ===== 1. Numerical Imputation =====
        if precomputed["loaded"]["imputation"]:
            # Data already imputed - just use the artifacts for model packaging
            impute_dict = precomputed["impute_dict"]
            logger.info(
                "✓ Using pre-computed imputation artifacts (data already transformed)"
            )
            logger.info("  → Skipping imputation transformation")
        else:
            # Compute inline AND transform data
            logger.info(
                "Computing numerical imputation inline and transforming data..."
            )
            train_df, val_df, test_df, impute_dict = apply_numerical_imputation(
                config, train_df, val_df, test_df
            )
            logger.info("✓ Numerical imputation completed")

        # ===== 2. Categorical Feature Encoding =====
        # Check if using native categorical features or risk table mapping
        use_native_cat = config.get("use_native_categorical", True)
        categorical_mappings = {}

        if use_native_cat:
            logger.info("=" * 70)
            logger.info("USING LIGHTGBM NATIVE CATEGORICAL FEATURE HANDLING")
            logger.info("=" * 70)
            logger.info(f"Categorical features: {config['cat_field_list']}")
            logger.info(
                "Encoding categorical features to integers using DictionaryEncodingProcessor..."
            )

            # Encode categorical features to integers for LightGBM
            encoding_processors = {}

            for cat_col in config["cat_field_list"]:
                if cat_col not in train_df.columns:
                    logger.warning(
                        f"Categorical column '{cat_col}' not found in data, skipping"
                    )
                    continue

                # Create processor for each categorical column
                processor = DictionaryEncodingProcessor(
                    columns=[cat_col],
                    unknown_strategy="default",  # Use default value for unseen categories
                    default_value=-1,  # -1 will be treated as missing by LightGBM
                )

                # Fit on training data
                processor.fit(train_df[[cat_col]])
                encoding_processors[cat_col] = processor

                # Get the mapping for artifacts
                categorical_mappings[cat_col] = processor.categorical_map.get(
                    cat_col, {}
                )

                # Transform all splits
                train_df = processor.process(train_df)
                val_df = processor.process(val_df)
                test_df = processor.process(test_df)

                # Ensure integer type
                train_df[cat_col] = train_df[cat_col].astype("int32")
                val_df[cat_col] = val_df[cat_col].astype("int32")
                test_df[cat_col] = test_df[cat_col].astype("int32")

                logger.info(
                    f"✓ Encoded '{cat_col}': {len(categorical_mappings[cat_col])} unique categories"
                )

            # Risk tables not needed when using native categorical
            risk_tables = {}
            logger.info("✓ Categorical encoding completed")
            logger.info("=" * 70)

        else:
            logger.info("=" * 70)
            logger.info("USING RISK TABLE MAPPING (XGBoost-STYLE)")
            logger.info("=" * 70)

            if precomputed["loaded"]["risk_tables"]:
                # Data already risk-mapped - just use the artifacts for model packaging
                risk_tables = precomputed["risk_tables"]
                logger.info(
                    "✓ Using pre-computed risk table artifacts (data already transformed)"
                )
                logger.info("  → Skipping risk table transformation")
            else:
                # Compute inline AND transform data
                logger.info("Computing risk tables inline and transforming data...")
                train_df, val_df, test_df, risk_tables = fit_and_apply_risk_tables(
                    config, train_df, val_df, test_df
                )
                logger.info("✓ Risk table mapping completed")

            logger.info("=" * 70)

        # ===== 3. Feature Selection =====
        if use_precomputed_features:
            logger.info(
                "Determining effective feature columns (USE_PRECOMPUTED_FEATURES=true)..."
            )
            feature_columns, fs_applied = get_effective_feature_columns(
                config, model_artifacts_input_dir, train_df
            )

            if fs_applied:
                logger.info("✓ Feature selection successfully applied")
                logger.info(f"  → Using {len(feature_columns)} selected features")

                # Filter DataFrames to only include selected features plus label
                label_col = config["label_name"]
                id_col = config.get("id_name", "id")

                # Keep only selected features, label, and ID (if present)
                cols_to_keep = feature_columns + [label_col]
                if id_col in train_df.columns:
                    cols_to_keep.append(id_col)

                train_df = train_df[cols_to_keep]
                val_df = val_df[cols_to_keep]
                test_df = test_df[cols_to_keep]

                logger.info(f"  → Filtered datasets to {len(cols_to_keep)} columns")
            else:
                logger.info(
                    "Feature selection artifacts not found - using original features"
                )
                feature_columns = config["tab_field_list"] + config["cat_field_list"]
        else:
            logger.info(
                "USE_PRECOMPUTED_FEATURES=false - using original feature configuration"
            )
            feature_columns = config["tab_field_list"] + config["cat_field_list"]
            logger.info(f"  → Using {len(feature_columns)} features from config")

        logger.info("Preparing Datasets for LightGBM...")
        # Update config with actual feature columns for Dataset preparation
        config["tab_field_list"] = [
            f for f in feature_columns if f in config.get("tab_field_list", [])
        ]
        config["cat_field_list"] = [
            f for f in feature_columns if f in config.get("cat_field_list", [])
        ]

        train_set, val_set, feature_columns = prepare_datasets(config, train_df, val_df)
        logger.info("Datasets prepared successfully")
        logger.info(
            f"Using {len(feature_columns)} features in order: {feature_columns}"
        )

        logger.info("Starting model training...")
        model = train_model(config, train_set, val_set)
        logger.info("Model training completed")

        logger.info("Saving model artifacts...")
        logger.info(f"Model path: {model_dir}, Output path: {output_dir}")
        logger.info(f"Output path exists: {os.path.exists(output_dir)}")

        save_artifacts(
            model=model,
            risk_tables=risk_tables,
            impute_dict=impute_dict,
            model_path=model_dir,
            feature_columns=feature_columns,
            config=config,
            categorical_mappings=categorical_mappings,
        )
        logger.info("✓ Model artifacts saved successfully")

        # --- inference + evaluation on val and test ---
        logger.info("====== STARTING EVALUATION PHASE ======")

        # Add explicit directory checks
        logger.info(f"Checking output directory: {output_dir}")
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory {output_dir} does not exist, creating...")
            os.makedirs(output_dir, exist_ok=True)

        # Get format for output preservation
        output_format = config.get("_input_format", "csv")
        logger.info(f"Using format {output_format} for evaluation outputs")

        # Validation evaluation with exception handling
        logger.info("Starting inference & evaluation on validation set")
        try:
            evaluate_split(
                "val", val_df, feature_columns, model, config, output_format, output_dir
            )
            logger.info("✓ Validation evaluation completed successfully")
        except Exception as e:
            logger.error(f"ERROR in validation evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        # Test evaluation with exception handling
        logger.info("Starting inference & evaluation on test set")
        try:
            evaluate_split(
                "test",
                test_df,
                feature_columns,
                model,
                config,
                output_format,
                output_dir,
            )
            logger.info("✓ Test evaluation completed successfully")
        except Exception as e:
            logger.error(f"ERROR in test evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        logger.info("All evaluation steps complete.")
        logger.info("====== MAIN EXECUTION COMPLETED SUCCESSFULLY ======")
        logger.info("Training script finished successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# -------------------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Script starting...")

    # Container path constants
    CONTAINER_PATHS = {
        "INPUT_DATA": "/opt/ml/input/data",
        "MODEL_DIR": "/opt/ml/model",
        "OUTPUT_DATA": "/opt/ml/output/data",
        "CONFIG_DIR": "/opt/ml/code/hyperparams",  # Source directory path
        "MODEL_ARTIFACTS_INPUT": "/opt/ml/input/data/model_artifacts_input",  # Optional pre-computed artifacts
    }

    # Define input and output paths using contract logical names
    # Use container defaults (no CLI arguments per contract)
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
        "REGION": os.environ.get("REGION", "NA"),
    }

    # Add USE_NATIVE_CATEGORICAL if set in environment
    if "USE_NATIVE_CATEGORICAL" in os.environ:
        environ_vars["USE_NATIVE_CATEGORICAL"] = (
            os.environ.get("USE_NATIVE_CATEGORICAL", "true").lower() == "true"
        )

    # Create empty args namespace to maintain function signature
    args = argparse.Namespace()

    try:
        logger.info(f"Starting main process with paths:")
        logger.info(f"  Data directory: {input_paths['input_path']}")
        logger.info(f"  Config directory: {input_paths['hyperparameters_s3_uri']}")
        logger.info(f"  Model directory: {output_paths['model_output']}")
        logger.info(f"  Output directory: {output_paths['evaluation_output']}")

        # Call the refactored main function
        main(input_paths, output_paths, environ_vars, args)

        logger.info("LightGBM training script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
