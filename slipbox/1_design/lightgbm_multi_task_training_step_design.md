---
tags:
  - design
  - step_builders
  - training_steps
  - multi_task_learning
  - lightgbm
  - sagemaker
keywords:
  - LightGBM multi-task
  - lightgbmmt
  - multi-label training
  - MTGBM
  - training step design
  - adaptive weighting
  - knowledge distillation
topics:
  - step builder design
  - multi-task learning
  - training step implementation
  - LightGBM training
language: python
date of note: 2025-11-11
---

# LightGBM Multi-Task Training Step Design

## Overview

This document defines the design for a LightGBM Multi-Task (MTGBM) training step that extends the standard LightGBM training capabilities to support multi-label/multi-task learning. The step uses a custom fork of LightGBM (`lightgbmmt`) with native multi-label support, advanced loss functions, and adaptive task weighting mechanisms.

## Related Documents

### Design Documents
- **[LightGBM Training Step Builder Patterns](./training_step_builder_patterns.md)** - Base training step patterns
- **[MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)** - Comprehensive MTGBM architecture and design
- **[Training Step Alignment Validation Patterns](./training_step_alignment_validation_patterns.md)** - Validation patterns

### Implementation References
- **[LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)** - Detailed analysis of lightgbmmt framework
- **[LightGBM Training Contract](../../src/cursus/steps/contracts/lightgbm_training_contract.py)** - Standard LightGBM training contract
- **[LightGBM Training Step Builder](../../src/cursus/steps/builders/builder_lightgbm_training_step.py)** - Existing LightGBM step implementation

## Design Principles

### Multi-Task Learning Philosophy
- **Shared Representations**: Learn common features across related tasks using single tree structure
- **Task Complementarity**: Leverage auxiliary tasks to improve main task performance
- **Dynamic Weighting**: Automatically adjust task importance during training
- **Knowledge Transfer**: Transfer knowledge from well-performing tasks to struggling ones

### Integration Principles
- **Framework Consistency**: Follow established LightGBM training step patterns
- **Specification-Driven**: Use specifications for input/output definitions
- **Backward Compatibility**: Maintain compatibility with single-task LightGBM patterns
- **Clean Separation**: Separate multi-task logic from single-task implementation

## SageMaker Step Type Classification

The LightGBM Multi-Task training step creates a **TrainingStep** instance using the SKLearn framework estimator with custom multi-task training logic:

- **Framework**: SKLearn container (for flexibility in package installation)
- **Custom Library**: lightgbmmt (packaged in source directory with compiled C library)
- **Step Type**: TrainingStep
- **Estimator**: SKLearn with multi-label data handling

## Architecture Design

### Component Structure

```
src/cursus/steps/
├── builders/
│   └── builder_lightgbmmt_training_step.py      # New: LightGBM MT builder
├── contracts/
│   └── lightgbmmt_training_contract.py          # New: Multi-task contract
├── configs/
│   └── config_lightgbmmt_training_step.py       # New: MT configuration
└── specs/
    └── lightgbmmt_training_spec.py              # New: MT specification

projects/cap_mtgbm/docker/
├── lightgbmmt_training.py                       # New: Entry point script (directly in docker/)
├── lightgbmmt/                                  # Packaged custom library
│   ├── __init__.py
│   ├── basic.py
│   ├── engine.py
│   ├── libpath.py
│   └── ...
├── compile/
│   └── lib_lightgbm.so                         # Compiled C library
├── models/
│   ├── __init__.py
│   ├── Mtgbm.py                                # MTGBM model class
│   ├── baseLoss.py                             # Fixed weight loss
│   ├── customLossNoKD.py                       # Adaptive weight loss
│   ├── customLossKDswap.py                     # Adaptive + KD loss
│   └── util.py                                 # Utility functions
└── hyperparams/
    └── hyperparameters.json                    # Default configuration
```

### Class Structure

```python
class LightgbmmtTrainingStepBuilder(StepBuilderBase):
    """
    Builder for LightGBM Multi-Task Training Step.
    
    Extends standard LightGBM training to support:
    - Multi-label/multi-task learning
    - Custom lightgbmmt library with compiled C extensions
    - Adaptive task weighting via Jensen-Shannon divergence
    - Knowledge distillation with label swapping
    - Three loss function modes: base, auto_weight, auto_weight_KD
    """
    
    def __init__(
        self,
        config: LightgbmmtTrainingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """Initialize with MTGBM-specific configuration and specification."""
        if not isinstance(config, LightgbmmtTrainingConfig):
            raise ValueError(
                "LightgbmmtTrainingStepBuilder requires a LightgbmmtTrainingConfig instance."
            )
        
        # Load MTGBM training specification
        if not SPEC_AVAILABLE or LIGHTGBMMT_TRAINING_SPEC is None:
            raise ValueError("LightGBM MT training specification not available")
        
        super().__init__(
            config=config,
            spec=LIGHTGBMMT_TRAINING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: LightgbmmtTrainingConfig = config
    
    def validate_configuration(self) -> None:
        """Validate MTGBM-specific configuration."""
        
    def _create_estimator(self, output_path=None) -> SKLearn:
        """Create SKLearn estimator with MTGBM configuration."""
        
    def _get_environment_variables(self) -> Dict[str, str]:
        """Build environment variables for multi-task training."""
        
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """Create TrainingInput objects for multi-label data."""
        
    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """Return output path for multi-task model artifacts."""
        
    def create_step(self, **kwargs) -> TrainingStep:
        """Orchestrate MTGBM training step creation."""
```

## Multi-Task Specific Patterns

### 1. Multi-Label Data Handling Pattern

```python
def _create_data_channels_from_source(self, base_path):
    """
    Create train, validation, and test channel inputs from a base path.
    
    Multi-task training expects the same channel structure as single-task,
    but the data files contain multiple label columns:
    - Main task label (e.g., 'is_abusive')
    - Subtask labels (e.g., 'harassment', 'hate_speech', 'spam', etc.)
    
    Args:
        base_path: Base S3 path containing train/val/test subdirectories
        
    Returns:
        Dictionary of channel name to TrainingInput
    """
    from sagemaker.workflow.functions import Join
    
    channels = {
        "train": TrainingInput(s3_data=Join(on="/", values=[base_path, "train/"])),
        "val": TrainingInput(s3_data=Join(on="/", values=[base_path, "val/"])),
        "test": TrainingInput(s3_data=Join(on="/", values=[base_path, "test/"])),
    }
    
    return channels
```

### 2. Custom Library Packaging Pattern

```python
def _create_estimator(self, output_path=None) -> SKLearn:
    """
    Creates SKLearn estimator with packaged lightgbmmt library.
    
    Key differences from standard LightGBM:
    - lightgbmmt packaged in source_dir (not installed via pip)
    - Compiled C library (lib_lightgbm.so) included in source_dir
    - Custom loss function modules packaged with source
    - Multi-task specific hyperparameters
    """
    # Use source_dir containing lightgbmmt package
    source_dir = self.config.effective_source_dir
    self.log_info("Using source directory with lightgbmmt: %s", source_dir)
    
    # SKLearn container allows custom package installation
    framework_version = getattr(self.config, "framework_version", "1.2-1")
    py_version = getattr(self.config, "py_version", "py3")
    
    self.log_info("Using Scikit-Learn framework version: %s", framework_version)
    self.log_info("lightgbmmt will be loaded from source directory")
    
    return SKLearn(
        entry_point=self.config.training_entry_point,  # lightgbmmt_training.py
        source_dir=source_dir,
        framework_version=framework_version,
        py_version=py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        max_run=86400,  # 24 hours default
        output_path=output_path,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        environment=self._get_environment_variables(),
    )
```

### 3. Multi-Task Hyperparameters Pattern

```python
def _validate_multi_task_hyperparameters(self) -> None:
    """
    Validate multi-task specific hyperparameters.
    
    Required Multi-Task Parameters:
    - num_labels: Number of tasks (main + subtasks)
    - tree_learner: Must be 'serial2' for multi-label
    - main_target: Name of main task column
    - sub_tasks_list: List of subtask column names
    - loss_type: None, 'auto_weight', or 'auto_weight_KD'
    
    Optional Multi-Task Parameters:
    - patience: KD patience threshold (default: 100)
    - weight_method: Weight update method (default: None)
    """
    # Validate main target
    if not hasattr(self.config, 'main_target') or not self.config.main_target:
        raise ValueError("Multi-task training requires 'main_target' parameter")
    
    # Validate subtasks list
    if not hasattr(self.config, 'sub_tasks_list') or not self.config.sub_tasks_list:
        raise ValueError("Multi-task training requires 'sub_tasks_list' parameter")
    
    # Calculate and validate num_labels
    num_labels = 1 + len(self.config.sub_tasks_list)
    if hasattr(self.config, 'num_labels') and self.config.num_labels != num_labels:
        raise ValueError(
            f"num_labels mismatch: config has {self.config.num_labels} "
            f"but should be {num_labels} (1 main + {len(self.config.sub_tasks_list)} subtasks)"
        )
    
    # Validate tree learner
    if hasattr(self.config, 'tree_learner'):
        if self.config.tree_learner != 'serial2':
            raise ValueError(
                f"Multi-task training requires tree_learner='serial2', "
                f"got '{self.config.tree_learner}'"
            )
    
    # Validate loss type
    valid_loss_types = [None, 'auto_weight', 'auto_weight_KD']
    if hasattr(self.config, 'loss_type'):
        if self.config.loss_type not in valid_loss_types:
            raise ValueError(
                f"Invalid loss_type '{self.config.loss_type}'. "
                f"Must be one of: {valid_loss_types}"
            )
    
    self.log_info("Multi-task hyperparameter validation succeeded")
    self.log_info("  Main task: %s", self.config.main_target)
    self.log_info("  Subtasks: %s", self.config.sub_tasks_list)
    self.log_info("  Total labels: %d", num_labels)
    self.log_info("  Loss type: %s", getattr(self.config, 'loss_type', None))
```

### 4. Environment Variables for Multi-Task Training

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Constructs environment variables for multi-task training.
    
    Multi-Task Specific Variables:
    - MAIN_TARGET: Main task column name
    - SUB_TASKS_LIST: Comma-separated subtask column names
    - LOSS_TYPE: Loss function type
    - NUM_LABELS: Total number of tasks
    - PATIENCE: KD patience threshold (optional)
    - WEIGHT_METHOD: Weight update method (optional)
    """
    # Get base environment variables
    env_vars = super()._get_environment_variables()
    
    # Add multi-task specific variables
    if hasattr(self.config, 'main_target'):
        env_vars['MAIN_TARGET'] = str(self.config.main_target)
    
    if hasattr(self.config, 'sub_tasks_list'):
        env_vars['SUB_TASKS_LIST'] = ','.join(self.config.sub_tasks_list)
    
    if hasattr(self.config, 'loss_type') and self.config.loss_type:
        env_vars['LOSS_TYPE'] = str(self.config.loss_type)
    
    if hasattr(self.config, 'num_labels'):
        env_vars['NUM_LABELS'] = str(self.config.num_labels)
    
    if hasattr(self.config, 'patience'):
        env_vars['PATIENCE'] = str(self.config.patience)
    
    if hasattr(self.config, 'weight_method') and self.config.weight_method:
        env_vars['WEIGHT_METHOD'] = str(self.config.weight_method)
    
    # Add environment variables from config if they exist
    if hasattr(self.config, 'env') and self.config.env:
        env_vars.update(self.config.env)
    
    self.log_info("Multi-task training environment variables: %s", env_vars)
    return env_vars
```

## Training Script Integration Pattern

### Script Structure

```python
# lightgbmmt_training.py
"""
LightGBM Multi-Task Training Script

Wraps the MTGBM implementation for SageMaker training job execution.
Handles data loading, model training, and artifact saving.
"""

import os
import sys
import json
import logging

# Add lightgbmmt and models to Python path
sys.path.insert(0, '/opt/ml/code/lightgbmmt')
sys.path.insert(0, '/opt/ml/code/models')

import pandas as pd
import numpy as np
from models.Mtgbm import MtGbm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_multi_label_data(data_dir, main_target, sub_tasks_list):
    """
    Load multi-label training data from SageMaker channels.
    
    Expected structure:
    /opt/ml/input/data/train/*.{csv,parquet,json}
    /opt/ml/input/data/val/*.{csv,parquet,json}
    /opt/ml/input/data/test/*.{csv,parquet,json}
    
    Each file must contain:
    - Feature columns
    - Main task column (main_target)
    - Subtask columns (sub_tasks_list)
    
    Returns:
        Tuple of (X_train, train_labels, X_val, val_labels, X_test, test_labels)
    """
    # Implementation details
    pass


def load_hyperparameters():
    """
    Load hyperparameters from multiple sources (priority order):
    1. /opt/ml/code/hyperparams/hyperparameters.json (embedded in source)
    2. Environment variables (MAIN_TARGET, SUB_TASKS_LIST, etc.)
    3. /opt/ml/input/config/hyperparameters.json (SageMaker provided)
    
    Returns:
        Dictionary of merged hyperparameters
    """
    # Implementation details
    pass


def main():
    """
    Main training orchestration function.
    
    Steps:
    1. Load hyperparameters from all sources
    2. Load multi-label training data
    3. Initialize MtGbm model with configuration
    4. Train model with selected loss function
    5. Save model artifacts to /opt/ml/model/
    6. Generate predictions and save to /opt/ml/output/data/
    """
    logger.info("Starting LightGBM Multi-Task training...")
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    main_target = hyperparams.get('main_target', 'is_abusive')
    sub_tasks_list = hyperparams.get('sub_tasks_list', [])
    loss_type = hyperparams.get('loss_type', None)
    
    logger.info(f"Main target: {main_target}")
    logger.info(f"Subtasks: {sub_tasks_list}")
    logger.info(f"Loss type: {loss_type}")
    
    # Load data
    X_train, train_labels, X_val, val_labels, X_test, test_labels = \
        load_multi_label_data(
            data_dir='/opt/ml/input/data',
            main_target=main_target,
            sub_tasks_list=sub_tasks_list
        )
    
    # Initialize model
    model = MtGbm(
        config=hyperparams,
        X_train=X_train,
        train_label=train_labels,
        sub_tasks_list=sub_tasks_list,
        main_target=main_target,
        loss_type=loss_type
    )
    
    # Train model
    logger.info("Training MTGBM model...")
    model.train()
    
    # Save model artifacts
    model_output_path = '/opt/ml/model'
    logger.info(f"Saving model to {model_output_path}")
    model.model.save_model(os.path.join(model_output_path, 'lightgbmmt_model.txt'))
    
    # Save metadata
    metadata = {
        'main_target': main_target,
        'sub_tasks_list': sub_tasks_list,
        'num_labels': 1 + len(sub_tasks_list),
        'loss_type': loss_type,
        'hyperparameters': hyperparams
    }
    with open(os.path.join(model_output_path, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate predictions on test set
    logger.info("Generating test predictions...")
    test_predictions = model.predict(X_test, test_labels)
    
    # Save evaluation results
    output_data_path = '/opt/ml/output/data'
    os.makedirs(output_data_path, exist_ok=True)
    test_predictions.to_csv(
        os.path.join(output_data_path, 'test_predictions.csv'),
        index=False
    )
    
    # Evaluate and save metrics
    model.evaluate(X_test, test_labels, test_predictions)
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()
```

## Configuration Design

### Configuration Class

```python
from typing import List, Optional, Dict, Any
from pydantic import Field, validator
from ..configs.config_lightgbm_training_step import LightGBMTrainingConfig


class LightgbmmtTrainingConfig(LightGBMTrainingConfig):
    """
    Configuration for LightGBM Multi-Task Training Step.
    
    Extends LightGBMTrainingConfig with multi-task specific parameters.
    """
    
    # Multi-task required parameters
    main_target: str = Field(
        default="is_abusive",
        description="Name of the main task column in the dataset"
    )
    
    sub_tasks_list: List[str] = Field(
        ...,  # Required
        description="List of subtask column names in the dataset"
    )
    
    # Multi-task training parameters
    loss_type: Optional[str] = Field(
        default=None,
        description=(
            "Loss function type: None (base loss with fixed weights), "
            "'auto_weight' (adaptive weighting without KD), "
            "'auto_weight_KD' (adaptive weighting with knowledge distillation)"
        )
    )
    
    num_labels: Optional[int] = Field(
        default=None,
        description="Total number of labels (main + subtasks). Auto-computed if not provided."
    )
    
    patience: int = Field(
        default=100,
        description="Patience threshold for knowledge distillation (KD patience parameter)"
    )
    
    weight_method: Optional[str] = Field(
        default=None,
        description=(
            "Weight update method: None (standard adaptive), "
            "'tenIters' (update every 50 iterations), "
            "'sqrt' (square root dampening), "
            "'delta' (incremental updates)"
        )
    )
    
    # Override entry point default for multi-task script
    training_entry_point: str = Field(
        default="lightgbmmt_training.py",
        description="Entry point script for multi-task training"
    )
    
    @validator('num_labels', always=True)
    def compute_num_labels(cls, v, values):
        """Auto-compute num_labels if not provided."""
        if v is None and 'sub_tasks_list' in values:
            return 1 + len(values['sub_tasks_list'])
        return v
    
    @validator('loss_type')
    def validate_loss_type(cls, v):
        """Validate loss_type parameter."""
        valid_types = [None, 'auto_weight', 'auto_weight_KD']
        if v not in valid_types:
            raise ValueError(
                f"Invalid loss_type '{v}'. Must be one of: {valid_types}"
            )
        return v
    
    @validator('weight_method')
    def validate_weight_method(cls, v):
        """Validate weight_method parameter."""
        valid_methods = [None, 'tenIters', 'sqrt', 'delta']
        if v not in valid_methods:
            raise ValueError(
                f"Invalid weight_method '{v}'. Must be one of: {valid_methods}"
            )
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional hyperparameters
        validate_assignment = True
```

## Contract Design

### Training Script Contract

```python
"""
LightGBM Multi-Task Training Script Contract

Defines the contract for the LightGBM multi-task training script that handles
multi-label training with adaptive weighting and knowledge distillation.
"""

from .training_script_contract import TrainingScriptContract

LIGHTGBMMT_TRAINING_CONTRACT = TrainingScriptContract(
    entry_point="lightgbmmt_training.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams/hyperparameters.json",
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "evaluation_output": "/opt/ml/output/data",
    },
    expected_arguments={
        # No command-line arguments - using paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables - script uses hyperparameters.json
    ],
    optional_env_vars={
        "MAIN_TARGET": "is_abusive",  # Main task column name
        "SUB_TASKS_LIST": "",  # Comma-separated subtask column names
        "LOSS_TYPE": "None",  # Loss function type
        "NUM_LABELS": "0",  # Total number of labels (auto-computed)
        "PATIENCE": "100",  # KD patience threshold
        "WEIGHT_METHOD": "None",  # Weight update method
    },
    framework_requirements={
        # lightgbmmt is packaged in source_dir, not installed via pip
        "scipy": ">=1.7.0",
        "scikit-learn": ">=1.3.0",
        "pandas": "==2.1.4",
        "numpy": ">=1.21.0",
        "matplotlib": "==3.8.2",
        "seaborn": ">=0.11.0",
        "pyarrow": "==14.0.2",
    },
    description="""
    LightGBM multi-task training script for multi-label classification that:
    1. Loads multi-label training data from split directories (train/val/test)
    2. Supports multiple related classification tasks (main task + subtasks)
    3. Uses custom lightgbmmt library (packaged locally with compiled C extensions)
    4. Trains with adaptive task weighting based on Jensen-Shannon divergence
    5. Optionally applies knowledge distillation with label swapping
    6. Supports three loss function modes: base, auto_weight, auto_weight_KD
    7. Generates per-task evaluation metrics and visualizations
    8. Saves multi-task model artifacts and metadata
    
    Multi-Task Training Features:
    - Shared tree structures across related tasks
    - Dynamic task importance adjustment during training
    - Knowledge transfer from well-performing to struggling tasks
    - Patience-based label replacement mechanism
    - Comprehensive per-task evaluation and monitoring
    
    Input Structure:
    - /opt/ml/input/data/train: Training data with all task labels
    - /opt/ml/input/data/val: Validation data with all task labels
    - /opt/ml/input/data/test: Test data with all task labels
    - /opt/ml/code/hyperparams/hyperparameters.json: Model configuration
    
    Data Format Requirements:
    - Each file must contain feature columns plus all task label columns
    - Main task column specified by main_target parameter
    - Subtask columns specified by sub_tasks_list parameter
    - All tasks must be binary classification (0/1 labels)
    
    Output Structure:
    - /opt/ml/model/lightgbmmt_model.txt: Trained multi-task model (LightGBM text format)
    - /opt/ml/model/model_metadata.json: Model metadata (tasks, loss type, hyperparameters)
    - /opt/ml/output/data/test_predictions.csv: Multi-task predictions on test set
    - /opt/ml/output/data/evaluation_metrics.json: Per-task evaluation metrics
    - /opt/ml/output/data/mtg_eval.png: Per-task training curves
    - /opt/ml/output/data/weight_change.png: Task weight evolution (if adaptive)
    - /opt/ml/output/data/roc_curves.png: Per-task ROC curves
    - /opt/ml/output/data/feature_importance.png: Feature importance plot
    
    Hyperparameters (via JSON config):
    - Multi-task: main_target, sub_tasks_list, loss_type, num_labels, patience, weight_method
    - Data fields: tab_field_list, cat_field_list (if preprocessing needed)
    - LightGBM: learning_rate, num_leaves, max_depth, num_rounds, tree_learner='serial2'
    - Training: bagging_fraction, feature_fraction, lambda_l1, lambda_l2
    
    Loss Function Types:
    - None (default): Fixed weight vector, simple baseline
    - 'auto_weight': Dynamic weighting based on task similarity (JS divergence)
    - 'auto_weight_KD': Adaptive weighting + knowledge distillation with label swapping
    
    Model Output Format:
    - lightgbm_mt_model.txt: Standard LightGBM text format (can be loaded with lightgbm.Booster)
    - Multi-label predictions: Flattened array [N_samples * N_labels] requiring reshaping
    - Per-task predictions: Reshape to [N_samples, N_labels] matrix for evaluation
    
    Custom Library Requirements:
    - lightgbmmt package (packaged in source_dir)
    - lib_lightgbm.so (compiled C library in compile/ subdirectory)
    - Custom loss function modules (baseLoss, customLossNoKD, customLossKDswap)
    - MtGbm model class wrapper
    """,
)
```

## Specification Design

### Step Specification

```python
"""
LightGBM Multi-Task Training Step Specification
"""

from ...core.specs.step_specification import StepSpecification, DependencySpec, OutputSpec

LIGHTGBMMT_TRAINING_SPEC = StepSpecification(
    step_type="training",
    step_name="lightgbm_mt_training",
    description="LightGBM multi-task training step with adaptive weighting and knowledge distillation",
    sagemaker_step_type="TrainingStep",
    framework="sklearn",  # Using SKLearn container for flexibility
    
    dependencies={
        "input_data": DependencySpec(
            logical_name="input_path",
            description="Multi-label training data (train/val/test splits)",
            required=True,
            supported_step_types=["processing", "transform"],
            path_type="directory",
        ),
        "hyperparameters": DependencySpec(
            logical_name="hyperparameters_s3_uri",
            description="Multi-task hyperparameters JSON file",
            required=False,
            supported_step_types=["processing"],
            path_type="file",
        ),
    },
    
    outputs={
        "model": OutputSpec(
            logical_name="model_output",
            description="Multi-task model artifacts (model.tar.gz)",
            path_type="directory",
            is_primary=True,
        ),
        "evaluation": OutputSpec(
            logical_name="evaluation_output",
            description="Per-task evaluation results and metrics (output.tar.gz)",
            path_type="directory",
            is_primary=False,
        ),
    },
    
    validation_rules={
        "multi_task_params": "Must specify main_target and sub_tasks_list",
        "tree_learner": "Must use tree_learner='serial2' for multi-label training",
        "loss_type": "Must be None, 'auto_weight', or 'auto_weight_KD'",
    },
)
```

## Best Practices

### 1. Data Preparation
- Ensure all task label columns are present in training data
- Use consistent data formats across train/val/test splits
- Validate label distributions for each task
- Handle missing labels appropriately (multi-task supports partial labels)

### 2. Task Selection
- Choose related tasks that benefit from shared representations
- Limit number of subtasks to 3-6 for optimal performance
- Ensure subtasks provide complementary information
- Consider task correlation when selecting subtasks

### 3. Loss Function Selection
- Start with base loss (None) to establish baseline
- Use auto_weight for automatic task importance learning
- Apply auto_weight_KD when dealing with performance plateaus
- Monitor weight evolution to understand task relationships

### 4. Hyperparameter Tuning
- Tune tree structure parameters (max_depth, num_leaves) first
- Adjust learning_rate and num_rounds for convergence
- Use patience parameter to control KD aggressiveness
- Experiment with weight_method for stability

### 5. Model Evaluation
- Monitor per-task metrics throughout training
- Compare multi-task vs single-task baselines
- Analyze task weight evolution for insights
- Validate on hold-out test set for final assessment

## Testing Implications

LightGBM Multi-Task training step builders should be tested for:

1. **Configuration Validation**
   - Multi-task parameter validation (main_target, sub_tasks_list)
   - Loss type validation
   - Tree learner validation (must be 'serial2')
   - Num_labels auto-computation

2. **Estimator Creation**
   - SKLearn estimator with correct configuration
   - Source directory packaging validation
   - lightgbmmt library availability check
   - Compiled C library path verification

3. **Multi-Label Data Handling**
   - Data channel creation for multi-label data
   - Label matrix formation validation
   - Missing label handling
   - Data split consistency

4. **Environment Variables**
   - Multi-task specific variable construction
   - Subtask list serialization (comma-separated)
   - Loss type propagation
   - Optional parameter handling

5. **Training Script Integration**
   - Entry point script execution
   - Hyperparameter loading from multiple sources
   - MtGbm class instantiation
   - Model artifact generation

6. **Loss Function Selection**
   - Base loss instantiation
   - Adaptive weighting setup
   - Knowledge distillation configuration
   - Weight method application

7. **Output Artifacts**
   - Multi-task model file (lightgbm_mt_model.txt)
   - Model metadata file (model_metadata.json)
   - Per-task predictions (test_predictions.csv)
   - Evaluation metrics and visualizations

8. **Specification Compliance**
   - Input dependency validation
   - Output specification adherence
   - Framework compatibility check
   - SageMaker step type verification

## Key Differences from Standard LightGBM Training

### 1. Library Packaging
- **Standard**: lightgbm installed via pip
- **Multi-Task**: lightgbmmt packaged in source_dir with compiled C library

### 2. Data Format
- **Standard**: Single label column
- **Multi-Task**: Multiple label columns (main + subtasks)

### 3. Hyperparameters
- **Standard**: Standard LightGBM parameters
- **Multi-Task**: Additional multi-task parameters (loss_type, num_labels, patience, etc.)

### 4. Tree Learner
- **Standard**: Any tree learner
- **Multi-Task**: Must use 'serial2' for multi-label support

### 5. Loss Function
- **Standard**: Built-in objectives
- **Multi-Task**: Custom loss functions with adaptive weighting

### 6. Model Output
- **Standard**: Single prediction per sample
- **Multi-Task**: Multiple predictions per sample (one per task)

### 7. Evaluation
- **Standard**: Single-task metrics
- **Multi-Task**: Per-task metrics plus weight evolution

## Registry Entry

### Step Registration

The LightGBM Multi-Task training step must be registered in the step catalog for automatic discovery. Add the following entry to `src/cursus/registry/step_names_original.py`:

```python
"LightgbmmtTraining": {
    "config_class": "LightgbmmtTrainingConfig",
    "builder_step_name": "LightgbmmtTrainingStepBuilder",
    "spec_type": "LightgbmmtTraining",
    "sagemaker_step_type": "Training",
    "description": "LightGBM multi-task training with adaptive weighting and knowledge distillation for multi-label classification",
}
```

### Registry Fields Explanation

- **Key (`LightgbmmtTraining`)**: Canonical step name in PascalCase, matches the auto-discovery output from file name `builder_lightgbmmt_training_step.py`
- **config_class**: Configuration class name (must match class in `config_lightgbmmt_training_step.py`)
- **builder_step_name**: Builder class name (must match class in `builder_lightgbmmt_training_step.py`)
- **spec_type**: Specification type identifier (typically matches the canonical step name)
- **sagemaker_step_type**: SageMaker step type (`Training` for TrainingStep)
- **description**: Human-readable description of the step's purpose

### Discovery Integration

With this registry entry, the step becomes discoverable through:

1. **Config Discovery**: Finds `LightgbmmtTrainingConfig` by scanning `config_lightgbmmt_training_step.py`
2. **Builder Discovery**: Finds `LightgbmmtTrainingStepBuilder` by scanning `builder_lightgbmmt_training_step.py`
3. **Spec Discovery**: Finds `LIGHTGBM_MT_TRAINING_SPEC` by scanning `lightgbmmt_training_spec.py`
4. **Contract Discovery**: Finds `LIGHTGBM_MT_TRAINING_CONTRACT` by scanning `lightgbmmt_training_contract.py`

### Verification

After adding the registry entry, verify registration:

```python
from cursus.registry.step_names import get_step_names, validate_step_name

# Check if step is registered
step_names = get_step_names()
assert "LightgbmmtTraining" in step_names

# Validate step name
assert validate_step_name("LightgbmmtTraining")

# Get step details
from cursus.registry.step_names import (
    get_config_class_name,
    get_builder_step_name,
    get_sagemaker_step_type
)

print(get_config_class_name("LightgbmmtTraining"))  # LightgbmmtTrainingConfig
print(get_builder_step_name("LightgbmmtTraining"))  # LightgbmmtTrainingStepBuilder
print(get_sagemaker_step_type("LightgbmmtTraining"))  # Training
```

## Implementation Checklist

Follow the standardized step creation process: script → script contract → step spec → register → config → step builder

### 1. Develop Processing Script
- [ ] Create `lightgbmmt_training.py` in `projects/cap_mtgbm/docker/`
- [ ] Implement standardized main function with unified interface
- [ ] Implement multi-label data loading from SageMaker channels
- [ ] Implement hyperparameter merging from multiple sources (embedded JSON, env vars, SageMaker config)
- [ ] Integrate MtGbm model wrapper from `projects/cap_mtgbm/docker/models/Mtgbm.py`
- [ ] Implement model training with loss function selection (base, auto_weight, auto_weight_KD)
- [ ] Create model artifact saving logic (lightgbmmt_model.txt, model_metadata.json)
- [ ] Implement per-task evaluation and visualization generation
- [ ] Add comprehensive error handling and logging
- [ ] Package lightgbmmt library in `projects/cap_mtgbm/docker/lightgbmmt/`
- [ ] Include compiled `lib_lightgbm.so` in `projects/cap_mtgbm/docker/compile/`
- [ ] Package custom loss modules in `projects/cap_mtgbm/docker/models/` (baseLoss, customLossNoKD, customLossKDswap)

### 2. Create Script Contract
- [ ] Define `LIGHTGBMMT_TRAINING_CONTRACT` in `src/cursus/steps/contracts/lightgbmmt_training_contract.py`
- [ ] Map logical names to SageMaker container paths
- [ ] Define entry_point="lightgbmmt_training.py"
- [ ] Specify expected_input_paths (/opt/ml/input/data, /opt/ml/code/hyperparams)
- [ ] Specify expected_output_paths (/opt/ml/model, /opt/ml/output/data)
- [ ] Define optional environment variables (MAIN_TARGET, SUB_TASKS_LIST, LOSS_TYPE, etc.)
- [ ] Document framework_requirements (scipy, scikit-learn, pandas, numpy, etc.)

### 3. Define Step Specification
- [ ] Create `LIGHTGBMMT_TRAINING_SPEC` in `src/cursus/steps/specs/lightgbmmt_training_spec.py`
- [ ] Define logical input dependencies (input_path, hyperparameters_s3_uri)
- [ ] Define logical outputs (model_output, evaluation_output)
- [ ] Specify step_type="training", sagemaker_step_type="Training"
- [ ] Document validation rules for multi-task parameters

### 4. Register Step
- [ ] Add registry entry to `src/cursus/registry/step_names_original.py`:
  ```python
  "LightgbmmtTraining": {
      "config_class": "LightgbmmtTrainingConfig",
      "builder_step_name": "LightgbmmtTrainingStepBuilder",
      "spec_type": "LightgbmmtTraining",
      "sagemaker_step_type": "Training",
      "description": "LightGBM multi-task training with adaptive weighting and knowledge distillation for multi-label classification",
  }
  ```

### 5. Create Configuration Classes
- [ ] Create `LightgbmmtTrainingConfig` in `src/cursus/steps/configs/config_lightgbmmt_training_step.py`
- [ ] Extend `LightGBMTrainingConfig` for inheritance
- [ ] Define Tier 1 fields (main_target, sub_tasks_list - required user inputs)
- [ ] Define Tier 2 fields (loss_type, patience, weight_method - system defaults)
- [ ] Define Tier 3 fields (num_labels - auto-computed from sub_tasks_list)
- [ ] Implement pydantic validators for loss_type, weight_method, num_labels
- [ ] Override training_entry_point default to "lightgbmmt_training.py"
- [ ] Use ConfigFieldManager for field categorization (if applicable)

### 6. Build Step Builder
- [ ] Implement `LightgbmmtTrainingStepBuilder` in `src/cursus/steps/builders/builder_lightgbmmt_training_step.py`
- [ ] Extend `StepBuilderBase` (no decorator required for auto-discovery)
- [ ] Load LIGHTGBM_MT_TRAINING_SPEC in __init__
- [ ] Implement `validate_configuration()` for multi-task parameter validation
- [ ] Implement `_create_estimator()` using SKLearn framework with source_dir packaging
- [ ] Implement `_get_environment_variables()` for multi-task variables
- [ ] Implement `_get_inputs()` for multi-label data channels
- [ ] Implement `_get_outputs()` for model artifacts
- [ ] Implement `create_step()` orchestration method

### 7. Add Hyperparameters and Validate
- [ ] Multi-task hyperparameters handled via config class (no separate hyperparameter class needed)
- [ ] Verify hyperparameter propagation through environment variables and JSON config
- [ ] Ensure tree_learner='serial2' requirement is documented and validated

- [ ] Verify auto-discovery finds all components:
  ```bash
  cursus list-steps --workspace main
  ```
- [ ] Run 4-tier alignment validation:
  ```bash
  cursus validate-alignment --step LightgbmmtTraining --workspace main
  ```
- [ ] Run builder validation tests:
  ```bash
  cursus validate-builder --step LightgbmmtTraining --workspace main
  ```
- [ ] Run script runtime testing:
  ```bash
  cursus runtime test-script lightgbmmt_training --workspace-dir ./test_workspace --verbose
  ```
- [ ] Validate registry integration:
  ```bash
  cursus validate-registry --workspace main
  ```
- [ ] Write unit tests for configuration validation
- [ ] Write unit tests for builder methods
- [ ] Create integration tests for end-to-end training workflow
- [ ] Document usage examples and best practices
- [ ] Complete validation checklist from developer guide

## Usage Example

### Basic Configuration

```python
from cursus.steps.configs.config_lightgbmmt_training_step import LightgbmmtTrainingConfig
from cursus.steps.builders.builder_lightgbmmt_training_step import LightgbmmtTrainingStepBuilder

# Configure multi-task training
config = LightgbmmtTrainingConfig(
    # Multi-task parameters
    main_target="is_abusive",
    sub_tasks_list=["harassment", "hate_speech", "spam", "violence", "nsfw"],
    loss_type="auto_weight_KD",
    patience=100,
    weight_method=None,
    
    # Training infrastructure
    training_instance_type="ml.m5.2xlarge",
    training_instance_count=1,
    training_volume_size=30,
    
    # Source directory with lightgbmmt (training script in docker/, not scripts/)
    source_dir="projects/cap_mtgbm/docker",
    training_entry_point="lightgbmmt_training.py",
    
    # Framework
    framework_version="1.2-1",
    py_version="py3",
    
    # LightGBM hyperparameters
    learning_rate=0.05,
    num_leaves=750,
    max_depth=16,
    num_rounds=100,
    bagging_fraction=0.9,
    feature_fraction=0.9,
    lambda_l1=0.5,
    lambda_l2=0.05,
)

# Create builder
builder = LightgbmmtTrainingStepBuilder(
    config=config,
    role="arn:aws:iam::123456789:role/SageMakerRole"
)

# Create training step
training_step = builder.create_step(
    input_path="s3://bucket/prefix/multi_label_data/",
    enable_caching=True
)
```

### Advanced Configuration with Dependencies

```python
# Use in pipeline with dependencies
from cursus.mods.pipeline_assembler import PipelineAssembler

# Preprocessing step provides multi-label data
preprocessing_step = create_preprocessing_step(...)

# Multi-task training step consumes preprocessed data
mt_training_step = builder.create_step(
    dependencies=[preprocessing_step],
    enable_caching=True
)

# Create pipeline
pipeline = PipelineAssembler.create_pipeline(
    steps=[preprocessing_step, mt_training_step],
    pipeline_name="mtgbm_fraud_detection_pipeline"
)
```

## Troubleshooting Guide

### Common Issues

#### 1. Library Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'lightgbmmt'`

**Solution**:
- Verify lightgbmmt is packaged in source_dir
- Check Python path additions in training script
- Ensure __init__.py files are present

#### 2. Compiled Library Not Found
**Symptom**: `OSError: lib_lightgbm.so not found`

**Solution**:
- Verify lib_lightgbm.so is in compile/ subdirectory
- Check libpath.py configuration
- Ensure proper permissions on compiled library

#### 3. Tree Learner Error
**Symptom**: `ValueError: serial2 tree learner not supported`

**Solution**:
- Verify using lightgbmmt (not standard lightgbm)
- Check tree_learner parameter is set to 'serial2'
- Ensure num_labels parameter is specified

#### 4. Label Column Missing
**Symptom**: `KeyError: 'harassment'` (or other subtask)

**Solution**:
- Verify all subtask columns exist in data
- Check main_target and sub_tasks_list configuration
- Validate data format before training

#### 5. Loss Function Instantiation Error
**Symptom**: `AttributeError: custom_loss_KDswap object has no attribute 'self_obj'`

**Solution**:
- Verify loss function modules are imported correctly
- Check loss_type parameter value
- Ensure custom loss classes are properly defined

## Performance Considerations

### Computational Efficiency
- Multi-task training is only marginally slower than single-task
- Shared tree structures reduce memory footprint
- Adaptive weighting adds ~5-10% overhead
- Knowledge distillation adds ~10-15% overhead

### Scalability
- Scales well with number of samples (up to millions)
- Performance degrades with >10 subtasks
- Optimal range: 3-6 subtasks
- Consider task grouping for many related tasks

### Resource Requirements
- Memory: ~2x single-task (due to label matrix)
- CPU: Similar to single-task
- Storage: Slightly larger model files (multi-label outputs)
- Training time: 1.1-1.3x single-task depending on loss function

## Future Enhancements

### Planned Improvements

1. **Automatic Task Selection**
   - Feature: Automatically select most beneficial subtasks
   - Benefit: Reduces manual task engineering

2. **Hierarchical Task Structures**
   - Feature: Support task hierarchies and dependencies
   - Benefit: Better modeling of related task relationships

3. **Online Learning Support**
   - Feature: Incremental model updates with new data
   - Benefit: Continuous learning without full retraining

4. **Multi-Class Multi-Task**
   - Feature: Support multi-class tasks (not just binary)
   - Benefit: Broader applicability to diverse problems

5. **Automated Hyperparameter Tuning**
   - Feature: Integrate with SageMaker Automatic Model Tuning
   - Benefit: Optimal hyperparameter selection

## Conclusion

The LightGBM Multi-Task training step design provides a comprehensive framework for implementing multi-label/multi-task gradient boosting in SageMaker pipelines. Key benefits include:

- **Shared Representations**: Improved generalization through multi-task learning
- **Adaptive Weighting**: Automatic task importance learning
- **Knowledge Distillation**: Enhanced convergence and performance
- **Production Ready**: Full integration with Cursus framework
- **Extensible**: Support for custom loss functions and weighting strategies

This design establishes a solid foundation for multi-task learning applications in fraud detection, content moderation, and other domains with related classification tasks.

## References

### Related Design Documents
- [Training Step Builder Patterns](./training_step_builder_patterns.md)
- [MTGBM Multi-Task Learning Design](./mtgbm_multi_task_learning_design.md)
- [LightGBM Training Step Builder](../../src/cursus/steps/builders/builder_lightgbm_training_step.py)

### Implementation References
- [LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)
- [MtGbm Model Class](../../projects/cap_mtgbm/docker/models/Mtgbm.py)

### Research Papers
- "Multi-Task Learning Using Uncertainty to Weigh Losses" - Kendall et al., 2018
- "An Overview of Multi-Task Learning in Deep Neural Networks" - Ruder, 2017
- "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" - Ke et al., 2017

---

*This design document provides comprehensive specification for implementing a LightGBM Multi-Task training step in the Cursus framework, covering architecture, patterns, configuration, and best practices for production multi-task learning systems.*
