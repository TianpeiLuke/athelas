---
tags:
  - design
  - step_builders
  - preprocessing_steps
  - multi_label
  - multi_task_learning
  - sagemaker
keywords:
  - multilabel preprocessing
  - multi-label generation
  - payment method filtering
  - sparse labels
  - preprocessing step design
topics:
  - step builder design
  - multi-label preprocessing
  - preprocessing step implementation
  - data preprocessing
language: python
date of note: 2025-11-11
---

# Multilabel Preprocessing Step Design

## Overview

This document defines the design for a Multilabel Preprocessing step that extends standard tabular preprocessing to support multi-label dataset generation for multi-task learning scenarios. The step transforms single-label datasets into multi-label formats by creating task-specific label columns based on categorical features (e.g., payment methods), enabling training of multi-task models like MTGBM.

## Related Documents

### Design Documents
- **[Tabular Preprocessing Patterns](./tabular_preprocessing_patterns.md)** - Base preprocessing patterns
- **[LightGBM Multi-Task Training Step Design](./lightgbm_multi_task_training_step_design.md)** - Multi-task training consumer
- **[MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)** - Comprehensive MTGBM architecture
- **[Data Format Preservation Patterns](./data_format_preservation_patterns.md)** - Format handling patterns
- **[Label Ruleset Generation/Execution](../../src/cursus/steps/contracts/label_ruleset_generation_contract.py)** - Rule-based label generation (upstream option)

### Implementation References
- **[MTGBM Pipeline Reusability Analysis](../4_analysis/2025-11-11_mtgbm_pipeline_reusability_analysis.md)** - Reusability assessment
- **[LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)** - MTGBM implementation details
- **[Tabular Preprocessing Script](../../src/cursus/steps/scripts/tabular_preprocessing.py)** - Base preprocessing implementation
- **[Tabular Preprocessing Contract](../../src/cursus/steps/contracts/tabular_preprocess_contract.py)** - Standard preprocessing contract

## Design Principles

### Multi-Label Generation Philosophy
- **Task Decomposition**: Break single classification task into related subtasks based on categorical features
- **Sparse Representation**: Create sparse label matrices where each sample is labeled only for its relevant category
- **Label Consistency**: Maintain main task label while creating category-specific subtask labels
- **Feature Preservation**: Keep all original features intact while adding multi-label columns

### Integration Principles
- **Framework Consistency**: Follow established tabular preprocessing patterns
- **Specification-Driven**: Use specifications for input/output definitions
- **Reusability**: Design for use beyond MTGBM (any multi-label scenario)
- **Clean Separation**: Separate multi-label logic from standard preprocessing

## SageMaker Step Type Classification

The Multilabel Preprocessing step creates a **ProcessingStep** instance using the SKLearnProcessor:

- **Framework**: SKLearn container (for flexibility and library availability)
- **Processor**: SKLearnProcessor with custom multi-label preprocessing script
- **Step Type**: ProcessingStep
- **Data Handling**: Multi-label dataset generation with sparse labels

## Architecture Design

### Component Structure

```
src/cursus/steps/
├── builders/
│   └── builder_multilabel_preprocessing_step.py      # New: Multilabel preprocessing builder
├── contracts/
│   └── multilabel_preprocessing_contract.py          # New: Multi-label contract
├── configs/
│   └── config_multilabel_preprocessing_step.py       # New: Multi-label configuration
├── specs/
│   └── multilabel_preprocessing_spec.py              # New: Multi-label specification
└── scripts/
    └── multilabel_preprocessing.py                   # New: Multi-label preprocessing script
```

### Class Structure

```python
class MultilabelPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for Multilabel Preprocessing Step.
    
    Extends standard preprocessing to support:
    - Categorical feature-based multi-label generation
    - Payment method filtering (or other categorical filtering)
    - Sparse label matrix creation
    - Format preservation (CSV, TSV, Parquet)
    - Signature integration
    """
    
    def __init__(
        self,
        config: MultilabelPreprocessingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """Initialize with multilabel-specific configuration and specification."""
        if not isinstance(config, MultilabelPreprocessingConfig):
            raise ValueError(
                "MultilabelPreprocessingStepBuilder requires a MultilabelPreprocessingConfig instance."
            )
        
        # Load multilabel preprocessing specification
        if not SPEC_AVAILABLE or MULTILABEL_PREPROCESSING_SPEC is None:
            raise ValueError("Multilabel preprocessing specification not available")
        
        super().__init__(
            config=config,
            spec=MULTILABEL_PREPROCESSING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: MultilabelPreprocessingConfig = config
    
    def validate_configuration(self) -> None:
        """Validate multilabel-specific configuration."""
        
    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearnProcessor with multilabel configuration."""
        
    def _get_environment_variables(self) -> Dict[str, str]:
        """Build environment variables for multi-label preprocessing."""
        
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Create ProcessingInput objects for data and signature."""
        
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Return output paths for multi-label processed data."""
        
    def create_step(self, **kwargs) -> ProcessingStep:
        """Orchestrate multilabel preprocessing step creation."""
```

## Multi-Label Specific Patterns

### 1. Categorical Feature-Based Multi-Label Generation Pattern

```python
def create_multi_label_columns(
    df: pd.DataFrame,
    main_label: str,
    category_column: str,
    category_list: List[str],
    label_prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Create multi-label columns based on categorical feature.
    
    For each category in category_list, creates a new label column:
    - Column name: {label_prefix}_{category} or {main_label}_{category}
    - Column value:
        - 1 if (category_column == category AND main_label == 1)
        - 0 if (category_column == category AND main_label == 0)
        - NaN if category_column != category (sparse representation)
    
    Example:
        main_label = 'is_fraud'
        category_column = 'payment_method'
        category_list = ['CC', 'DC', 'ACH']
        
        Creates columns:
        - is_fraud_CC
        - is_fraud_DC
        - is_fraud_ACH
        
        For a credit card transaction (payment_method='CC'):
        - is_fraud_CC = 1 or 0 (depending on is_fraud)
        - is_fraud_DC = NaN
        - is_fraud_ACH = NaN
    
    Args:
        df: Input dataframe
        main_label: Main label column name
        category_column: Categorical feature column name
        category_list: List of categories to create labels for
        label_prefix: Optional prefix for new label columns
        
    Returns:
        DataFrame with added multi-label columns
    """
    if label_prefix is None:
        label_prefix = main_label
    
    # Create multi-label columns
    for category in category_list:
        new_label_col = f"{label_prefix}_{category}"
        
        # Initialize with NaN (sparse representation)
        df[new_label_col] = np.nan
        
        # Set values only for matching category
        mask = df[category_column] == category
        df.loc[mask, new_label_col] = df.loc[mask, main_label]
    
    return df
```

### 2. Category Filtering Pattern

```python
def filter_by_categories(
    df: pd.DataFrame,
    category_column: str,
    category_list: List[str],
    drop_column: bool = False
) -> pd.DataFrame:
    """
    Filter dataset to include only specified categories.
    
    This is useful when:
    - Training on a subset of categories
    - Excluding rare or problematic categories
    - Creating category-specific models
    
    Args:
        df: Input dataframe
        category_column: Column containing categories to filter
        category_list: List of categories to keep
        drop_column: Whether to drop the category column after filtering
        
    Returns:
        Filtered dataframe
        
    Example:
        # Keep only credit card, debit card, and ACH transactions
        df = filter_by_categories(
            df,
            category_column='payment_method',
            category_list=['CC', 'DC', 'ACH']
        )
    """
    # Filter to specified categories
    filtered_df = df[df[category_column].isin(category_list)].copy()
    
    # Optionally drop the category column
    if drop_column:
        filtered_df = filtered_df.drop(columns=[category_column])
    
    log_info = f"Filtered from {len(df)} to {len(filtered_df)} rows"
    log_info += f" ({len(filtered_df)/len(df)*100:.1f}%)"
    print(log_info)
    
    return filtered_df
```

### 3. Sparse Label Validation Pattern

```python
def validate_multi_label_sparsity(
    df: pd.DataFrame,
    label_columns: List[str],
    min_samples_per_task: int = 100
) -> Dict[str, Any]:
    """
    Validate multi-label sparsity and distribution.
    
    Checks:
    - Each label column has sufficient non-NaN samples
    - Label distribution is reasonable (not too imbalanced)
    - No label column is entirely NaN
    - Total coverage (% of samples with at least one label)
    
    Args:
        df: Dataframe with multi-label columns
        label_columns: List of label column names
        min_samples_per_task: Minimum required samples per task
        
    Returns:
        Validation report dictionary
        
    Raises:
        ValueError: If validation fails
    """
    report = {
        'total_samples': len(df),
        'label_columns': len(label_columns),
        'task_statistics': {},
        'warnings': [],
        'errors': []
    }
    
    for label_col in label_columns:
        # Count non-NaN values
        non_nan_count = df[label_col].notna().sum()
        if non_nan_count == 0:
            report['errors'].append(f"{label_col}: No non-NaN values")
            continue
        
        # Check minimum samples
        if non_nan_count < min_samples_per_task:
            report['warnings'].append(
                f"{label_col}: Only {non_nan_count} samples "
                f"(minimum {min_samples_per_task})"
            )
        
        # Calculate label distribution
        label_dist = df[label_col].value_counts(dropna=True)
        positive_ratio = label_dist.get(1, 0) / non_nan_count if non_nan_count > 0 else 0
        
        report['task_statistics'][label_col] = {
            'non_nan_samples': int(non_nan_count),
            'nan_samples': int(df[label_col].isna().sum()),
            'sparsity': float(df[label_col].isna().sum() / len(df)),
            'positive_ratio': float(positive_ratio),
            'label_distribution': label_dist.to_dict()
        }
    
    # Calculate coverage
    any_label_mask = df[label_columns].notna().any(axis=1)
    coverage = any_label_mask.sum() / len(df)
    report['coverage'] = float(coverage)
    
    # Check for errors
    if report['errors']:
        raise ValueError(f"Multi-label validation failed: {report['errors']}")
    
    return report
```

### 4. Format Preservation Pattern

```python
def save_multilabel_data_with_format(
    df: pd.DataFrame,
    output_path: str,
    format_type: str,
    job_type: str = 'training'
) -> str:
    """
    Save multi-label data preserving input format.
    
    Supports CSV, TSV, and Parquet formats.
    Creates split-specific subdirectories for training job type.
    
    Args:
        df: Multi-label dataframe to save
        output_path: Base output directory path
        format_type: Output format ('csv', 'tsv', or 'parquet')
        job_type: Job type ('training', 'validation', 'testing', 'calibration')
        
    Returns:
        Path to saved file(s)
    """
    from pathlib import Path
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file extension
    ext_map = {
        'csv': '.csv',
        'tsv': '.tsv',
        'parquet': '.parquet'
    }
    ext = ext_map.get(format_type, '.csv')
    
    if job_type == 'training':
        # Save to train/val/test subdirectories (already split)
        # This assumes df is a dict with 'train', 'val', 'test' keys
        for split_name, split_df in df.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            file_path = split_dir / f"{split_name}_multilabel{ext}"
            
            if format_type == 'csv':
                split_df.to_csv(file_path, index=False)
            elif format_type == 'tsv':
                split_df.to_csv(file_path, sep='\t', index=False)
            elif format_type == 'parquet':
                split_df.to_parquet(file_path, index=False)
    else:
        # Save to single file
        file_path = output_dir / f"{job_type}_multilabel{ext}"
        
        if format_type == 'csv':
            df.to_csv(file_path, index=False)
        elif format_type == 'tsv':
            df.to_csv(file_path, sep='\t', index=False)
        elif format_type == 'parquet':
            df.to_parquet(file_path, index=False)
    
    return str(output_dir)
```

## Preprocessing Script Integration Pattern

### Script Structure

```python
# multilabel_preprocessing.py
"""
Multilabel Preprocessing Script

Generates multi-label datasets from single-label data by creating
category-specific label columns. Supports categorical feature filtering
and sparse label representation for multi-task learning.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_from_shards(
    data_dir: str,
    signature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and combine data shards from directory.
    
    Supports CSV, TSV, Parquet, and JSON formats (plain and gzipped).
    
    Args:
        data_dir: Directory containing data shards
        signature_columns: Optional column names from signature file
        
    Returns:
        Combined dataframe
    """
    # Implementation reuses logic from tabular_preprocessing.py
    pass


def filter_samples_by_category(
    df: pd.DataFrame,
    category_column: str,
    category_list: List[str]
) -> pd.DataFrame:
    """
    Filter dataset to include only specified categories.
    
    Args:
        df: Input dataframe
        category_column: Column name containing categories
        category_list: List of categories to keep
        
    Returns:
        Filtered dataframe
    """
    logger.info(f"Filtering by {category_column} in {category_list}")
    logger.info(f"Original size: {len(df)} rows")
    
    filtered_df = df[df[category_column].isin(category_list)].copy()
    
    logger.info(f"Filtered size: {len(filtered_df)} rows")
    logger.info(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")
    
    return filtered_df


def create_multilabel_columns(
    df: pd.DataFrame,
    main_label: str,
    category_column: str,
    category_list: List[str],
    label_prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Create sparse multi-label columns based on categorical feature.
    
    Args:
        df: Input dataframe
        main_label: Main label column name
        category_column: Categorical feature column name
        category_list: List of categories to create labels for
        label_prefix: Optional prefix for new label columns
        
    Returns:
        DataFrame with added multi-label columns
    """
    if label_prefix is None:
        label_prefix = main_label
    
    logger.info(f"Creating multi-label columns for {len(category_list)} categories")
    
    for category in category_list:
        new_label_col = f"{label_prefix}_{category}"
        
        # Initialize with NaN (sparse representation)
        df[new_label_col] = np.nan
        
        # Set values only for matching category
        mask = df[category_column] == category
        df.loc[mask, new_label_col] = df.loc[mask, main_label]
        
        # Log statistics
        non_nan_count = df[new_label_col].notna().sum()
        positive_count = (df[new_label_col] == 1).sum()
        logger.info(
            f"  {new_label_col}: {non_nan_count} samples, "
            f"{positive_count} positive ({positive_count/non_nan_count*100:.1f}%)"
        )
    
    return df


def split_multilabel_data(
    df: pd.DataFrame,
    label_columns: List[str],
    train_ratio: float = 0.7,
    test_val_ratio: float = 0.5,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Split multi-label data into train/val/test sets.
    
    Uses random splits (not stratified) because:
    - Sparse labels make stratification complex
    - Each sample may belong to different categories
    - Random splits maintain overall distribution
    
    Args:
        df: Multi-label dataframe
        label_columns: List of label column names
        train_ratio: Training set ratio
        test_val_ratio: Test/(Test+Val) ratio
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataframes
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train vs holdout
    train_df, holdout_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state
    )
    
    # Second split: test vs val
    test_df, val_df = train_test_split(
        holdout_df,
        test_size=test_val_ratio,
        random_state=random_state
    )
    
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace
) -> Dict[str, pd.DataFrame]:
    """
    Main multilabel preprocessing logic.
    
    Args:
        input_paths: Dictionary of input paths
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        
    Returns:
        Dictionary of processed dataframes
    """
    logger.info("Starting multilabel preprocessing...")
    
    # Extract parameters
    job_type = job_args.job_type
    main_label = environ_vars.get('MAIN_LABEL_FIELD', 'label')
    category_column = environ_vars.get('CATEGORY_COLUMN')
    category_list_str = environ_vars.get('CATEGORY_LIST', '')
    category_list = [c.strip() for c in category_list_str.split(',') if c.strip()]
    label_prefix = environ_vars.get('LABEL_PREFIX')
    output_format = environ_vars.get('OUTPUT_FORMAT', 'csv').lower()
    
    # Validate required parameters
    if not category_column:
        raise ValueError("CATEGORY_COLUMN environment variable is required")
    if not category_list:
        raise ValueError("CATEGORY_LIST environment variable is required")
    
    logger.info(f"Main label: {main_label}")
    logger.info(f"Category column: {category_column}")
    logger.info(f"Category list: {category_list}")
    logger.info(f"Output format: {output_format}")
    
    # Load data
    data_dir = input_paths['DATA']
    df = load_data_from_shards(data_dir)
    logger.info(f"Loaded data: {df.shape}")
    
    # Validate required columns
    if main_label not in df.columns:
        raise ValueError(f"Main label '{main_label}' not found in data")
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found in data")
    
    # Filter by categories
    df = filter_samples_by_category(df, category_column, category_list)
    
    # Create multi-label columns
    df = create_multilabel_columns(
        df, main_label, category_column, category_list, label_prefix
    )
    
    # Get new label columns
    if label_prefix:
        new_label_cols = [f"{label_prefix}_{cat}" for cat in category_list]
    else:
        new_label_cols = [f"{main_label}_{cat}" for cat in category_list]
    
    # Validate multi-label data
    validation_report = validate_multi_label_sparsity(df, new_label_cols)
    logger.info(f"Multi-label validation: {validation_report['coverage']*100:.1f}% coverage")
    
    # Split or save based on job type
    output_dir = output_paths['processed_data']
    
    if job_type == 'training':
        # Split into train/val/test
        splits = split_multilabel_data(df, new_label_cols)
        
        # Save splits
        for split_name, split_df in splits.items():
            split_dir = Path(output_dir) / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"{split_name}_multilabel"
            if output_format == 'csv':
                split_df.to_csv(split_dir / f"{file_name}.csv", index=False)
            elif output_format == 'tsv':
                split_df.to_csv(split_dir / f"{file_name}.tsv", sep='\t', index=False)
            elif output_format == 'parquet':
                split_df.to_parquet(split_dir / f"{file_name}.parquet", index=False)
        
        logger.info(f"Saved training splits to {output_dir}")
        return splits
    else:
        # Save single file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{job_type}_multilabel"
        if output_format == 'csv':
            df.to_csv(output_path / f"{file_name}.csv", index=False)
        elif output_format == 'tsv':
            df.to_csv(output_path / f"{file_name}.tsv", sep='\t', index=False)
        elif output_format == 'parquet':
            df.to_parquet(output_path / f"{file_name}.parquet", index=False)
        
        logger.info(f"Saved {job_type} data to {output_dir}")
        return {job_type: df}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_type',
        type=str,
        required=True,
        choices=['training', 'validation', 'testing', 'calibration'],
        help="Job type"
    )
    args = parser.parse_args()
    
    # Standard SageMaker paths
    INPUT_DATA_DIR = '/opt/ml/processing/input/data'
    INPUT_SIGNATURE_DIR = '/opt/ml/processing/input/signature'
    OUTPUT_DIR = '/opt/ml/processing/output'
    
    # Read environment variables
    environ_vars = {
        'MAIN_LABEL_FIELD': os.environ.get('MAIN_LABEL_FIELD', 'label'),
        'CATEGORY_COLUMN': os.environ.get('CATEGORY_COLUMN'),
        'CATEGORY_LIST': os.environ.get('CATEGORY_LIST', ''),
        'LABEL_PREFIX': os.environ.get('LABEL_PREFIX'),
        'OUTPUT_FORMAT': os.environ.get('OUTPUT_FORMAT', 'csv'),
    }
    
    # Set up paths
    input_paths = {
        'DATA': INPUT_DATA_DIR,
        'SIGNATURE': INPUT_SIGNATURE_DIR
    }
    output_paths = {
        'processed_data': OUTPUT_DIR
    }
    
    # Execute main logic
    try:
        result = main(input_paths, output_paths, environ_vars, args)
        logger.info("Multilabel preprocessing completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in multilabel preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
```

## Configuration Design

### Configuration Class

The MultilabelPreprocessingConfig extends TabularPreprocessingConfig to add multi-label specific parameters while inheriting all tabular preprocessing functionality.

#### Inheritance Hierarchy

```
BasePipelineConfig (core/base/config_base.py)
    ↓
ProcessingStepConfigBase (steps/configs/config_processing_step_base.py)
    ↓
MultilabelPreprocessingConfig (steps/configs/config_multilabel_preprocessing_step.py)
```

**Note**: MultilabelPreprocessingConfig extends ProcessingStepConfigBase directly (not TabularPreprocessingConfig), as it represents a distinct preprocessing operation with different multi-label specific requirements.

#### Inherited Fields

From **BasePipelineConfig** (Tier 1 - Essential User Inputs):
- `author`: Author or owner of the pipeline
- `bucket`: S3 bucket name for artifacts
- `role`: IAM role for pipeline execution
- `region`: Custom region code (NA, EU, FE)
- `service_name`: Service name for the pipeline
- `pipeline_version`: Version string for the pipeline
- `project_root_folder`: Root folder for hybrid path resolution

From **BasePipelineConfig** (Tier 2 - System Inputs):
- `model_class`: Model class (default: "xgboost")
- `current_date`: Current date for versioning
- `framework_version`: Framework version (default: "2.1.0")
- `py_version`: Python version (default: "py310")
- `source_dir`: Common source directory for scripts
- `enable_caching`: Enable caching for pipeline steps
- `use_secure_pypi`: Use secure CodeArtifact PyPI

From **ProcessingStepConfigBase** (Tier 2 - Processing Specific):
- `processing_instance_count`: Instance count (default: 1)
- `processing_volume_size`: Volume size in GB (default: 500)
- `processing_instance_type_large`: Large instance type (default: "ml.m5.4xlarge")
- `processing_instance_type_small`: Small instance type (default: "ml.m5.2xlarge")
- `use_large_processing_instance`: Use large instance type flag (default: False)
- `processing_source_dir`: Source directory for processing scripts
- `processing_entry_point`: Script entry point (default: None, can be overridden)
- `processing_framework_version`: SKLearn framework version (default: "1.2-1")

#### New Multi-Label Preprocessing Fields

Since MultilabelPreprocessingConfig extends ProcessingStepConfigBase directly (not TabularPreprocessingConfig), all multi-label preprocessing fields are NEW and specific to this step type.

```python
"""
Multilabel Preprocessing Configuration

Extends ProcessingStepConfigBase with multi-label preprocessing specific parameters.
Includes all preprocessing fields plus multi-label generation parameters.
"""

from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator, PrivateAttr
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from ..contracts.multilabel_preprocessing_contract import MULTILABEL_PREPROCESSING_CONTRACT

logger = logging.getLogger(__name__)


class MultilabelPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for Multilabel Preprocessing Step.
    
    Extends ProcessingStepConfigBase with preprocessing and multi-label generation parameters
    following the three-tier design:
    - Tier 1: Essential User Inputs (required)
    - Tier 2: System Fields with Defaults (optional)
    - Tier 3: Derived Fields (private with read-only properties)
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    # Preprocessing-specific required fields
    
    job_type: str = Field(
        description="One of ['training','validation','testing','calibration']",
    )
    
    # Multi-label specific required fields
    
    main_label_field: str = Field(
        ...,  # Required
        description="Name of the main label column in the dataset (e.g., 'is_fraud')"
    )
    
    category_column: str = Field(
        ...,  # Required
        description="Name of the categorical feature column to create subtask labels from (e.g., 'payment_method')"
    )
    
    category_list: List[str] = Field(
        ...,  # Required
        description="List of category values to create subtask labels for (e.g., ['CC', 'DC', 'ACH'])"
    )
    
    # ===== System Fields with Defaults (Tier 2) =====
    # Preprocessing-specific optional fields
    
    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of data to allocate to the training set (only used if job_type=='training').",
    )
    
    test_val_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of the holdout to allocate to the test set vs. validation (only if job_type=='training').",
    )
    
    output_format: str = Field(
        default="CSV",
        description="Output format for processed data ('CSV', 'TSV', or 'Parquet'). Default: CSV",
    )
    
    # Multi-label specific optional fields
    
    label_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Prefix for new label columns. "
            "If None, uses main_label_field as prefix. "
            "Example: 'is_fraud' creates 'is_fraud_CC', 'is_fraud_DC', etc."
        )
    )
    
    filter_categories: bool = Field(
        default=True,
        description=(
            "Whether to filter dataset to only include rows with categories in category_list. "
            "Set to False to keep all rows (non-matching rows will have NaN for subtask labels)"
        )
    )
    
    min_samples_per_task: int = Field(
        default=100,
        ge=1,
        description="Minimum required samples per subtask for validation"
    )
    
    # Entry point for multilabel preprocessing script
    processing_entry_point: str = Field(
        default="multilabel_preprocessing.py",
        description="Entry point script for multilabel preprocessing"
    )
    
    # ===== Derived Fields (Tier 3) =====
    # Private fields with read-only property access
    # Note: _full_script_path and _preprocessing_environment_variables are inherited
    
    _multilabel_environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    
    # ===== Validators =====
    
    @field_validator('category_list')
    @classmethod
    def validate_category_list(cls, v: List[str]) -> List[str]:
        """Validate category_list is not empty."""
        if not v or len(v) == 0:
            raise ValueError("category_list must contain at least one category")
        # Check for duplicate categories
        if len(v) != len(set(v)):
            raise ValueError("category_list must not contain duplicates")
        return v
    
    @field_validator('min_samples_per_task')
    @classmethod
    def validate_min_samples(cls, v: int) -> int:
        """Validate min_samples_per_task is positive."""
        if v <= 0:
            raise ValueError("min_samples_per_task must be positive")
        return v
    
    @field_validator('main_label_field', 'category_column')
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Validate column names are not empty."""
        if not v or not v.strip():
            raise ValueError("Column name must be a non-empty string")
        return v.strip()
    
    # ===== Properties for Derived Fields =====
    
    @property
    def multilabel_environment_variables(self) -> Dict[str, str]:
        """
        Get multi-label specific environment variables.
        
        This extends the preprocessing_environment_variables from parent
        with multi-label specific variables.
        
        Returns:
            Dictionary mapping environment variable names to values
        """
        if self._multilabel_environment_variables is None:
            # Start with parent environment variables
            env_vars = self.preprocessing_environment_variables.copy()
            
            # Add multi-label specific variables
            env_vars["MAIN_LABEL_FIELD"] = self.main_label_field
            env_vars["CATEGORY_COLUMN"] = self.category_column
            env_vars["CATEGORY_LIST"] = ",".join(self.category_list)
            
            if self.label_prefix:
                env_vars["LABEL_PREFIX"] = self.label_prefix
            
            self._multilabel_environment_variables = env_vars
        
        return self._multilabel_environment_variables
    
    # ===== Script Contract =====
    
    def get_script_contract(self):
        """
        Get script contract for this configuration.
        
        Returns:
            The multilabel preprocessing script contract
        """
        return MULTILABEL_PREPROCESSING_CONTRACT
    
    # ===== Overrides for Inheritance =====
    
    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include multilabel preprocessing specific fields.
        
        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (TabularPreprocessingConfig)
        base_fields = super().get_public_init_fields()
        
        # Add multilabel preprocessing specific fields
        multilabel_fields = {
            "job_type": self.job_type,
            "main_label_field": self.main_label_field,
            "category_column": self.category_column,
            "category_list": self.category_list,
            "label_prefix": self.label_prefix,
            "filter_categories": self.filter_categories,
            "min_samples_per_task": self.min_samples_per_task,
            "processing_entry_point": self.processing_entry_point,
        }
        
        # Combine fields (multilabel fields take precedence if overlap)
        init_fields = {**base_fields, **multilabel_fields}
        
        return init_fields
    
    # ===== Serialization =====
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)
        
        # Add multilabel-specific derived properties
        data["multilabel_environment_variables"] = self.multilabel_environment_variables
        
        return data
```

### Configuration Usage Notes

1. **No Field Duplication**: All fields from parent classes are inherited automatically. Only NEW multi-label specific fields are defined.

2. **Three-Tier Design**: Follows the established pattern:
   - **Tier 1** (Essential): main_label, category_column, category_list
   - **Tier 2** (System): label_prefix, filter_categories, min_samples_per_task, processing_entry_point (override)
   - **Tier 3** (Derived): _multilabel_environment_variables

3. **Entry Point Override**: The processing_entry_point field is redefined to change its default from "tabular_preprocessing.py" to "multilabel_preprocessing.py"

4. **Environment Variables**: Extends parent's preprocessing_environment_variables with multi-label specific variables (MAIN_LABEL, CATEGORY_COLUMN, CATEGORY_LIST, LABEL_PREFIX)

5. **Contract Integration**: Overrides get_script_contract() to return MULTILABEL_PREPROCESSING_CONTRACT

6. **Validation**: Adds validators specific to multi-label parameters (category_list non-empty, no duplicates, min_samples positive)

## Contract Design

### Preprocessing Script Contract

```python
"""
Multilabel Preprocessing Script Contract

Defines the contract for the multilabel preprocessing script that transforms
single-label datasets into multi-label formats for multi-task learning.
"""

from ...core.base.contract_base import ScriptContract

MULTILABEL_PREPROCESSING_CONTRACT = ScriptContract(
    entry_point="multilabel_preprocessing.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "SIGNATURE": "/opt/ml/processing/input/signature",
    },
    expected_output_paths={"processed_data": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["TRAIN_RATIO", "TEST_VAL_RATIO"],
    optional_env_vars={
        "MAIN_LABEL_FIELD": "",  # Multi-label specific (required in practice)
        "CATEGORY_COLUMN": "",  # Multi-label specific (required in practice)
        "CATEGORY_LIST": "",  # Multi-label specific (required in practice)
        "LABEL_PREFIX": "",  # Optional prefix for label columns
        "OUTPUT_FORMAT": "CSV",  # Output format (CSV/TSV/Parquet)
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Multilabel preprocessing script that extends tabular preprocessing with multi-label generation:
    1. Combines data shards from input directory (inherits from tabular preprocessing)
    2. Loads column signature for CSV/TSV files if provided
    3. Filters samples by categorical feature (e.g., payment methods)
    4. Creates sparse multi-label columns for each category
    5. Validates multi-label sparsity and distribution
    6. Splits data into train/test/val for training jobs
    7. Outputs processed multi-label files in configurable format (CSV/TSV/Parquet)
    
    Contract aligned with tabular preprocessing contract for consistency:
    - Inputs (same as tabular preprocessing):
      * DATA (required) - reads from /opt/ml/processing/input/data
      * SIGNATURE (optional) - reads from /opt/ml/processing/input/signature
    - Outputs (same as tabular preprocessing):
      * processed_data (primary) - writes to /opt/ml/processing/output
    - Arguments: job_type (required) - defines processing mode (training/validation/testing/calibration)
    
    Multi-Label Generation Logic:
    - For each category in CATEGORY_LIST, creates a new label column
    - Column naming: {MAIN_LABEL}_{category} or {LABEL_PREFIX}_{category}
    - Sparse representation: NaN for non-matching categories
    - Example: payment_method='CC', is_fraud=1 → is_fraud_CC=1, is_fraud_DC=NaN, is_fraud_ACH=NaN
    
    Multi-Label Specific Environment Variables:
    - MAIN_LABEL_FIELD: Main task label column name (e.g., 'is_fraud')
    - CATEGORY_COLUMN: Categorical feature for task decomposition (e.g., 'payment_method')
    - CATEGORY_LIST: Comma-separated categories (e.g., 'CC,DC,ACH,AMEX,INVOICE')
    - LABEL_PREFIX: Optional prefix for subtask labels (defaults to MAIN_LABEL)
    
    Output Format Configuration (same as tabular preprocessing):
    - OUTPUT_FORMAT environment variable controls output format
    - Valid values: "CSV" (default), "TSV", "Parquet"
    - Case-insensitive, defaults to CSV if invalid value provided
    - Format applies to all output splits (train/val/test)
    - Parquet recommended for large datasets with many sparse columns
    
    Signature File Format (same as tabular preprocessing):
    - CSV format with comma-separated column names
    - Applied only to CSV/TSV files, ignored for JSON/Parquet formats
    - Backward compatible - works without signature file
    """,
)
```

## Specification Design

### Step Specification

```python
"""
Multilabel Preprocessing Step Specification.

This module defines the declarative specification for multilabel preprocessing steps,
including their dependencies and outputs based on the actual implementation.
Extends tabular preprocessing with multi-label generation capabilities.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


# Import the contract at runtime to avoid circular imports
def _get_multilabel_preprocess_contract():
    from ..contracts.multilabel_preprocessing_contract import (
        MULTILABEL_PREPROCESSING_CONTRACT,
    )

    return MULTILABEL_PREPROCESSING_CONTRACT


# Multilabel Preprocessing Step Specification
MULTILABEL_PREPROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("MultilabelPreprocessing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_multilabel_preprocess_contract(),
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "CradleDataLoading",
                "DummyDataLoading",
                "DataLoad",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "data",
                "input",
                "raw",
                "dataset",
                "source",
                "tabular",
                "training",
                "train",
                "model_training",
                "validation",
                "val",
                "model_validation",
                "holdout",
                "testing",
                "test",
                "model_testing",
                "calibration",
                "calib",
                "model_calibration",
                "multi_label",
                "multilabel",
                "multi_task",
                "multitask",
            ],
            data_type="S3Uri",
            description="Raw tabular data for multi-label preprocessing. Supports all job types: training, validation, testing, and calibration",
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading", "DummyDataLoading"],
            semantic_keywords=[
                "signature",
                "schema",
                "columns",
                "column_names",
                "metadata",
                "header",
            ],
            data_type="S3Uri",
            description="Column signature file for CSV/TSV data preprocessing",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "input_path",
                "training_data",
                "model_input_data",
                "input_data",
                "validation_data",
                "testing_data",
                "calibration_data",
                "processed_training_data",
                "processed_validation_data",
                "processed_testing_data",
                "processed_calibration_data",
                "multilabel_data",
                "multi_label_data",
                "multi_task_data",
                "multitask_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Multi-label processed tabular data with sparse label columns. Compatible with all job types (training, validation, testing, calibration)",
        )
    ],
)
```

### Key Specification Features

1. **Contract Integration**: Uses runtime contract import to avoid circular dependencies
2. **Step Type Discovery**: Leverages `get_spec_step_type("MultilabelPreprocessing")` for registry integration
3. **Node Type**: Classified as `NodeType.INTERNAL` (processing within pipeline)
4. **Dependency Types**: Uses `DependencyType.PROCESSING_OUTPUT` for both inputs and outputs
5. **Compatible Sources**: Same as tabular preprocessing (CradleDataLoading, DummyDataLoading, DataLoad, ProcessingStep)
6. **Semantic Keywords**: Extended with multi-label specific terms (multi_label, multilabel, multi_task, multitask)
7. **Output Aliases**: Includes standard aliases plus multi-label specific aliases
8. **Property Path**: Standard SageMaker property path for ProcessingStep outputs
9. **Data Type**: S3Uri for both inputs and outputs

### Specification Alignment

The multilabel preprocessing specification maintains alignment with tabular preprocessing while adding multi-label specific features:

**Shared Elements:**
- Same dependency structure (DATA required, SIGNATURE optional)
- Same compatible sources for data loading
- Same output structure (processed_data as primary output)
- Same property path format for SageMaker integration

**Multi-Label Extensions:**
- Additional semantic keywords for multi-label/multi-task discovery
- Extended output aliases for multi-label data
- Contract references multilabel preprocessing logic
- Description emphasizes sparse label columns

This alignment ensures seamless integration with existing preprocessing infrastructure while providing clear differentiation for multi-label use cases.

## Best Practices

### 1. Category Selection
- Choose categories with sufficient samples (minimum 100-1000 per category)
- Avoid rare categories that may cause imbalanced labels
- Consider business relevance of each category
- Limit to 3-10 categories for optimal performance

### 2. Label Field Selection
- Use binary labels (0/1) for main task
- Ensure label is well-distributed across categories
- Validate label quality before multi-label generation
- Consider label imbalance per category

### 3. Data Format
- Use Parquet for large datasets with sparse labels (better compression)
- Use CSV for small datasets or debugging (human-readable)
- Ensure consistent data types across all shards
- Validate column names match signature file

### 4. Validation
- Always review validation report after preprocessing
- Check coverage percentage (should be 100% for filtered data)
- Monitor per-task sample counts
- Watch for warnings about insufficient samples

### 5. Integration with Training
- Ensure category_list matches training step's sub_tasks_list
- Use label_prefix consistently across preprocessing and training
- Verify main_label name matches training configuration
- Test with small sample before full pipeline run

## Testing Implications

Multilabel Preprocessing step builders should be tested for:

1. **Configuration Validation**
   - Multi-label parameter validation (main_label, category_column, category_list)
   - Category list non-empty check
   - Min samples validation
   - Entry point configuration

2. **Processor Creation**
   - SKLearnProcessor with correct configuration
   - Framework version compatibility
   - Instance type validation
   - Volume size settings

3. **Multi-Label Generation**
   - Correct column naming ({prefix}_{category})
   - Sparse representation (NaN for non-matching categories)
   - Label value correctness (0/1 or NaN)
   - All categories processed

4. **Category Filtering**
   - Correct filtering by category list
   - Row count validation
   - Data integrity after filtering
   - Optional vs required filtering

5. **Data Splitting**
   - Train/val/test split ratios
   - Random state reproducibility
   - Split size validation
   - No data leakage between splits

6. **Format Preservation**
   - CSV output correctness
   - TSV separator handling
   - Parquet compression
   - Format auto-detection

7. **Sparse Label Validation**
   - Coverage calculation
   - Per-task statistics
   - Warning/error reporting
   - Minimum sample checks

8. **Specification Compliance**
   - Input dependency validation
   - Output specification adherence
   - Framework compatibility check
   - SageMaker step type verification

## Key Differences from Standard Tabular Preprocessing

### 1. Label Processing
- **Standard**: Single label column processing
- **Multilabel**: Multiple sparse label columns generation

### 2. Data Filtering
- **Standard**: No categorical filtering
- **Multilabel**: Category-based filtering (optional)

### 3. Output Structure
- **Standard**: Same columns as input (plus processed labels)
- **Multilabel**: Additional subtask label columns

### 4. Validation
- **Standard**: Single label distribution check
- **Multilabel**: Per-task sparsity and coverage validation

### 5. Splitting Strategy
- **Standard**: Stratified splits when possible
- **Multilabel**: Random splits (stratification complex with sparse labels)

### 6. Environment Variables
- **Standard**: LABEL_FIELD, OUTPUT_FORMAT
- **Multilabel**: MAIN_LABEL_FIELD, CATEGORY_COLUMN, CATEGORY_LIST, LABEL_PREFIX, OUTPUT_FORMAT

### 7. Use Case
- **Standard**: General tabular preprocessing
- **Multilabel**: Multi-task learning data preparation

## Registry Entry

### Step Registration

The Multilabel Preprocessing step must be registered in the step catalog for automatic discovery. Add the following entry to `src/cursus/registry/step_names_original.py`:

```python
"MultilabelPreprocessing": {
    "config_class": "MultilabelPreprocessingConfig",
    "builder_step_name": "MultilabelPreprocessingStepBuilder",
    "spec_type": "MultilabelPreprocessing",
    "sagemaker_step_type": "Processing",
    "description": "Multilabel preprocessing with categorical feature-based multi-label generation for multi-task learning",
}
```

### Registry Fields Explanation

- **Key (`MultilabelPreprocessing`)**: Canonical step name in PascalCase, matches auto-discovery output
- **config_class**: Configuration class name (must match class in `config_multilabel_preprocessing_step.py`)
- **builder_step_name**: Builder class name (must match class in `builder_multilabel_preprocessing_step.py`)
- **spec_type**: Specification type identifier (typically matches canonical step name)
- **sagemaker_step_type**: SageMaker step type (`Processing` for ProcessingStep)
- **description**: Human-readable description of the step's purpose

### Discovery Integration

With this registry entry, the step becomes discoverable through:

1. **Config Discovery**: Finds `MultilabelPreprocessingConfig` by scanning `config_multilabel_preprocessing_step.py`
2. **Builder Discovery**: Finds `MultilabelPreprocessingStepBuilder` by scanning `builder_multilabel_preprocessing_step.py`
3. **Spec Discovery**: Finds `MULTILABEL_PREPROCESSING_SPEC` by scanning `multilabel_preprocessing_spec.py`
4. **Contract Discovery**: Finds `MULTILABEL_PREPROCESSING_CONTRACT` by scanning `multilabel_preprocessing_contract.py`

### Verification

After adding the registry entry, verify registration:

```python
from cursus.registry.step_names import get_step_names, validate_step_name

# Check if step is registered
step_names = get_step_names()
assert "MultilabelPreprocessing" in step_names

# Validate step name
assert validate_step_name("MultilabelPreprocessing")

# Get step details
from cursus.registry.step_names import (
    get_config_class_name,
    get_builder_step_name,
    get_sagemaker_step_type
)

print(get_config_class_name("MultilabelPreprocessing"))  # MultilabelPreprocessingConfig
print(get_builder_step_name("MultilabelPreprocessing"))  # MultilabelPreprocessingStepBuilder
print(get_sagemaker_step_type("MultilabelPreprocessing"))  # Processing
```

## Implementation Checklist

Follow the standardized step creation process: script → script contract → step spec → register → config → step builder

### 1. Develop Processing Script
- [ ] Create `multilabel_preprocessing.py` in `src/cursus/steps/scripts/`
- [ ] Implement standardized main function with unified interface
- [ ] Implement data loading from shards (reuse from `tabular_preprocessing.py`)
- [ ] Implement category filtering logic
- [ ] Implement multi-label column generation with sparse representation
- [ ] Implement sparse label validation
- [ ] Implement train/val/test splitting (random, not stratified)
- [ ] Add comprehensive error handling and logging
- [ ] Test script independently with sample data

### 2. Create Script Contract
- [ ] Define `MULTILABEL_PREPROCESSING_CONTRACT` in `src/cursus/steps/contracts/multilabel_preprocessing_contract.py`
- [ ] Map logical names to SageMaker container paths (DATA, SIGNATURE, processed_data)
- [ ] Define entry_point="multilabel_preprocessing.py"
- [ ] Specify expected_input_paths (/opt/ml/processing/input/data, /opt/ml/processing/input/signature)
- [ ] Specify expected_output_paths (/opt/ml/processing/output)
- [ ] Define required environment variables (TRAIN_RATIO, TEST_VAL_RATIO)
- [ ] Define optional environment variables (MAIN_LABEL, CATEGORY_COLUMN, CATEGORY_LIST, etc.)
- [ ] Document framework_requirements (pandas, numpy, scikit-learn)

### 3. Define Step Specification
- [ ] Create `MULTILABEL_PREPROCESSING_SPEC` in `src/cursus/steps/specs/multilabel_preprocessing_spec.py`
- [ ] Define logical input dependencies (DATA, SIGNATURE)
- [ ] Define logical outputs (processed_data)
- [ ] Specify step_type="processing", sagemaker_step_type="ProcessingStep"
- [ ] Document validation rules for multi-label parameters

### 4. Register Step
- [ ] Add registry entry to `src/cursus/registry/step_names_original.py`:
  ```python
  "MultilabelPreprocessing": {
      "config_class": "MultilabelPreprocessingConfig",
      "builder_step_name": "MultilabelPreprocessingStepBuilder",
      "spec_type": "MultilabelPreprocessing",
      "sagemaker_step_type": "Processing",
      "description": "Multilabel preprocessing with categorical feature-based multi-label generation for multi-task learning",
  }
  ```

### 5. Create Configuration Classes
- [ ] Create `MultilabelPreprocessingConfig` in `src/cursus/steps/configs/config_multilabel_preprocessing_step.py`
- [ ] Extend `TabularPreprocessingConfig` for consistency
- [ ] Define Tier 1 fields (main_label, category_column, category_list - required user inputs)
- [ ] Define Tier 2 fields (label_prefix, filter_categories, min_samples_per_task - system defaults)
- [ ] Implement pydantic validators for category_list, min_samples_per_task
- [ ] Override processing_entry_point default to "multilabel_preprocessing.py"
- [ ] Use ConfigFieldManager for field categorization (if applicable)

### 6. Build Step Builder
- [ ] Implement `MultilabelPreprocessingStepBuilder` in `src/cursus/steps/builders/builder_multilabel_preprocessing_step.py`
- [ ] Extend `StepBuilderBase` (no decorator required for auto-discovery)
- [ ] Load MULTILABEL_PREPROCESSING_SPEC in __init__
- [ ] Implement `validate_configuration()` for multi-label parameter validation
- [ ] Implement `_create_processor()` using SKLearnProcessor
- [ ] Implement `_get_environment_variables()` for multi-label variables
- [ ] Implement `_get_inputs()` for data and signature channels
- [ ] Implement `_get_outputs()` for processed data
- [ ] Implement `create_step()` orchestration method

### 7. Validate and Test
- [ ] Verify auto-discovery finds all components:
  ```bash
  cursus list-steps --workspace main
  ```
- [ ] Run 4-tier alignment validation:
  ```bash
  cursus validate-alignment --step MultilabelPreprocessing --workspace main
  ```
- [ ] Run builder validation tests:
  ```bash
  cursus validate-builder --step MultilabelPreprocessing --workspace main
  ```
- [ ] Run script runtime testing:
  ```bash
  cursus runtime test-script multilabel_preprocessing --workspace-dir ./test_workspace --verbose
  ```
- [ ] Validate registry integration:
  ```bash
  cursus validate-registry --workspace main
  ```
- [ ] Write unit tests for configuration validation
- [ ] Write unit tests for builder methods
- [ ] Create integration tests for end-to-end preprocessing workflow
- [ ] Test with sample multi-label data
- [ ] Verify output format preservation
- [ ] Validate sparse label generation correctness
- [ ] Test integration with LightgbmmtTraining step
- [ ] Document usage examples and best practices
- [ ] Complete validation checklist from developer guide

## Usage Example

### Basic Configuration

```python
from cursus.steps.configs.config_multilabel_preprocessing_step import MultilabelPreprocessingConfig
from cursus.steps.builders.builder_multilabel_preprocessing_step import MultilabelPreprocessingStepBuilder

# Configure multi-label preprocessing
config = MultilabelPreprocessingConfig(
    # Multi-label parameters
    main_label="is_fraud",
    category_column="payment_method",
    category_list=["CC", "DC", "ACH", "AMEX", "INVOICE"],
    label_prefix=None,  # Will use main_label as prefix
    filter_categories=True,  # Only keep specified categories
    min_samples_per_task=100,  # Minimum samples per category
    
    # Preprocessing parameters (inherited from TabularPreprocessingConfig)
    train_ratio=0.7,
    test_val_ratio=0.5,
    output_format="parquet",  # Recommended for sparse data
    
    # Processing infrastructure
    processing_instance_type="ml.m5.xlarge",
    processing_instance_count=1,
    processing_volume_size=30,
    
    # Script configuration
    processing_entry_point="multilabel_preprocessing.py",
    source_dir="src/cursus/steps/scripts",
    framework_version="1.2-1",
    py_version="py3",
)

# Create builder
builder = MultilabelPreprocessingStepBuilder(
    config=config,
    role="arn:aws:iam::123456789:role/SageMakerRole"
)

# Create preprocessing step
preprocessing_step = builder.create_step(
    input_data="s3://bucket/prefix/raw_data/",
    enable_caching=True
)
```

### Advanced Configuration with Dependencies

```python
# Use in pipeline with dependencies
from cursus.mods.pipeline_assembler import PipelineAssembler

# Data loading step provides raw data
data_loading_step = create_data_loading_step(...)

# Multi-label preprocessing step consumes raw data
ml_preprocessing_step = builder.create_step(
    dependencies=[data_loading_step],
    enable_caching=True
)

# Multi-task training step consumes multi-label data
mt_training_step = create_mt_training_step(
    dependencies=[ml_preprocessing_step],
    ...
)

# Create pipeline
pipeline = PipelineAssembler.create_pipeline(
    steps=[data_loading_step, ml_preprocessing_step, mt_training_step],
    pipeline_name="mtgbm_fraud_detection_pipeline"
)
```

### Output Structure

```python
# After preprocessing, output structure:
/opt/ml/processing/output/
├── train/
│   └── train_multilabel.parquet  # Training data with multi-label columns
├── val/
│   └── val_multilabel.parquet    # Validation data with multi-label columns
└── test/
    └── test_multilabel.parquet   # Test data with multi-label columns

# Each file contains:
# - Original feature columns
# - Main label: is_fraud
# - Subtask labels: is_fraud_CC, is_fraud_DC, is_fraud_ACH, is_fraud_AMEX, is_fraud_INVOICE
# - Sparse representation: NaN for non-matching payment methods
```

## Troubleshooting Guide

### Common Issues

#### 1. Category Column Not Found
**Symptom**: `ValueError: Category column 'payment_method' not found in data`

**Solution**:
- Verify category_column name matches data column exactly (case-sensitive)
- Check if column exists in signature file
- Ensure data loading step preserved column names
- Use list_files to inspect raw data structure

#### 2. Empty Category List
**Symptom**: `ValueError: CATEGORY_LIST environment variable is required`

**Solution**:
- Ensure category_list is not empty in configuration
- Check environment variable serialization (comma-separated)
- Verify configuration passed correctly to step builder
- Review config validation logs

#### 3. Insufficient Samples Per Task
**Symptom**: Warnings about low sample counts per category

**Solution**:
- Review validation report for per-task statistics
- Consider removing rare categories from category_list
- Adjust min_samples_per_task threshold
- Check if filtering is too aggressive

#### 4. Low Coverage Percentage
**Symptom**: Coverage below expected (e.g., 50% instead of 100%)

**Solution**:
- Check if filter_categories=False (causes sparse coverage)
- Verify category values match data exactly
- Look for typos in category names
- Review data quality and null values in category_column

#### 5. Format Preservation Issues
**Symptom**: Output format doesn't match expected

**Solution**:
- Check OUTPUT_FORMAT environment variable value
- Verify format is valid (CSV, TSV, Parquet)
- Ensure case-insensitive handling works
- Review script logs for format detection

## Performance Considerations

### Computational Efficiency
- Category filtering reduces dataset size (faster processing)
- Sparse representation minimizes memory footprint
- Parquet format offers best compression for sparse data
- Multi-label generation is O(n * m) where n=rows, m=categories

### Scalability
- Scales well with number of samples (tested up to millions)
- Performance degrades with >20 categories
- Optimal range: 3-10 categories
- Consider category grouping for many related categories

### Resource Requirements
- Memory: ~1.5x standard preprocessing (due to multi-label columns)
- CPU: Similar to standard preprocessing
- Storage: Varies by format (Parquet < CSV for sparse data)
- Processing time: 1.1-1.3x standard preprocessing

### Optimization Tips
- Use Parquet for large sparse datasets
- Filter aggressively to reduce data size
- Limit category_list to essential categories
- Use appropriate instance types (ml.m5.xlarge for <1M rows)

## Future Enhancements

### Planned Improvements

1. **Hierarchical Categories**
   - Feature: Support category hierarchies (e.g., payment_type → payment_method)
   - Benefit: Better task relationships and multi-level labels

2. **Automatic Category Selection**
   - Feature: Automatically select top-K most frequent categories
   - Benefit: Reduces manual category list specification

3. **Stratified Splitting for Sparse Labels**
   - Feature: Advanced stratification that handles sparse labels
   - Benefit: Better train/val/test distribution

4. **Multi-Column Multi-Label**
   - Feature: Generate multi-labels from multiple categorical columns
   - Benefit: Broader applicability to complex scenarios

5. **Label Smoothing**
   - Feature: Apply label smoothing to reduce overconfidence
   - Benefit: Better generalization in multi-task learning

## Conclusion

The Multilabel Preprocessing step design provides a comprehensive framework for generating multi-label datasets for multi-task learning in SageMaker pipelines. Key benefits include:

- **Contract Consistency**: Same I/O structure as TabularPreprocessing for seamless integration
- **Sparse Representation**: Efficient memory usage with NaN-based sparse labels
- **Category Flexibility**: Generic design works for any categorical feature decomposition
- **Production Ready**: Full integration with Cursus framework and validation
- **Extensible**: Support for future enhancements (hierarchical categories, auto-selection)

This design establishes a solid foundation for multi-label preprocessing applications in fraud detection, content moderation, and other domains requiring multi-task classification with categorical task decomposition.

## References

### Related Design Documents
- [Tabular Preprocessing Patterns](./tabular_preprocessing_patterns.md)
- [LightGBM Multi-Task Training Step Design](./lightgbm_multi_task_training_step_design.md)
- [MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)
- [Data Format Preservation Patterns](./data_format_preservation_patterns.md)

### Implementation References
- [MTGBM Pipeline Reusability Analysis](../4_analysis/2025-11-11_mtgbm_pipeline_reusability_analysis.md)
- [LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)
- [Tabular Preprocessing Script](../../src/cursus/steps/scripts/tabular_preprocessing.py)
- [Tabular Preprocessing Contract](../../src/cursus/steps/contracts/tabular_preprocessing_contract.py)

### Research Papers
- "Multi-Task Learning Using Uncertainty to Weigh Losses" - Kendall et al., 2018
- "An Overview of Multi-Task Learning in Deep Neural Networks" - Ruder, 2017
- "Learning Multiple Tasks with Multilinear Relationship Networks" - Long et al., 2017

---

*This design document provides comprehensive specification for implementing a Multilabel Preprocessing step in the Cursus framework, covering architecture, patterns, configuration, and best practices for production multi-label data generation systems.*
