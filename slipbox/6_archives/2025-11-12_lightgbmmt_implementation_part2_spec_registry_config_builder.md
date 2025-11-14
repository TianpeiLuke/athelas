---
tags:
  - project
  - implementation
  - lightgbmmt
  - multi_task_learning
  - step_specification
  - step_builder
  - cursus_integration
keywords:
  - lightgbmmt step specification
  - step builder
  - registry integration
  - configuration class
  - cursus framework integration
topics:
  - step specification design
  - step builder implementation
  - cursus integration patterns
language: python
date of note: 2025-11-12
---

# LightGBMMT Implementation Part 2: Spec, Registry, Config & Builder

## Overview

This document covers the second phase of LightGBMMT Training Step implementation, focusing on complete Cursus framework integration:
1. **Step Specification**: Define dependencies and outputs (Week 3, Days 1-2)
2. **Registry Integration**: Register step in Cursus registry (Week 3, Day 3)
3. **Configuration Class**: Three-tier config design (Week 3, Days 4-5)
4. **Step Builder**: Builder with specification-driven architecture (Week 3, Days 6-7)

**Timeline**: 1 week
**Prerequisites**: Part 1 completed (script, contract, hyperparameters)

## Executive Summary

### Objectives
- **Step Specification**: Enable automatic dependency resolution
- **Registry Integration**: Make step discoverable in Cursus framework
- **Configuration Class**: Three-tier config following established patterns
- **Step Builder**: Complete integration with refactored components

### Success Metrics
- ✅ Specification-driven dependency resolution
- ✅ Seamless pipeline integration
- ✅ Follows all Cursus patterns (three-tier config, builder base, etc.)
- ✅ >90% test coverage with validation framework

## Phase 1: Step Specification ✅ COMPLETED (2025-11-13)

### 1.1 Create Step Specification ✅

**File**: `src/cursus/steps/specs/lightgbmmt_training_spec.py` ✅ CREATED

```python
"""
LightGBMMT Training Step Specification.

Defines the declarative specification for LightGBMMT multi-task training steps,
following specification-driven architecture patterns.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


def _get_lightgbmmt_train_contract():
    """Import contract at runtime to avoid circular imports."""
    from ..contracts.lightgbmmt_training_contract import LIGHTGBMMT_TRAIN_CONTRACT
    return LIGHTGBMMT_TRAIN_CONTRACT


LIGHTGBMMT_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMMTTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_lightgbmmt_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "ProcessingStep",
                "DataLoad",
            ],
            semantic_keywords=[
                "data",
                "input",
                "training",
                "dataset",
                "multi_label",
                "multi_task",
                "processed",
                "train",
                "tabular",
            ],
            data_type="S3Uri",
            description="Multi-label training dataset S3 location with train/val/test subdirectories",
        ),
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=[
                "config",
                "params",
                "hyperparameters",
                "settings",
                "hyperparams",
                "multi_task_config",
            ],
            data_type="S3Uri",
            description="S3 URI containing hyperparameters configuration file (optional - falls back to source directory)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained LightGBMMT multi-task model artifacts",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
                "lightgbmmt_model",
                "multi_task_model",
                "mtgbm_model",
            ],
        ),
        OutputSpec(
            logical_name="evaluation_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.OutputDataConfig.S3OutputPath",
            data_type="S3Uri",
            description="Multi-task evaluation results, predictions, and metrics (per-task and aggregate)",
            aliases=[
                "evaluation_data",
                "eval_data",
                "validation_output",
                "test_output",
                "prediction_results",
                "multi_task_results",
                "task_predictions",
            ],
        ),
    ],
)
```

**Success Criteria**:
- ✅ Follows specification-driven architecture
- ✅ Clear dependency specification with semantic keywords
- ✅ Multi-task specific outputs with comprehensive aliases
- ✅ Compatible with existing preprocessing steps

### 1.2 Update Specs Module ✅

**File**: `src/cursus/steps/specs/__init__.py` ✅ UPDATED

Added import and export for LIGHTGBMMT_TRAINING_SPEC

## Phase 2: Registry Integration ✅ COMPLETED (2025-11-13)

### 2.1 Update Step Registry ✅

**File**: `src/cursus/registry/step_names_original.py` ✅ UPDATED

Add entry to `STEP_NAMES` dictionary:

```python
STEP_NAMES = {
    # ... existing entries ...
    
    "LightgbmmtTraining": {
        "config_class": "LightgbmmtTrainingConfig",
        "builder_step_name": "LightgbmmtTrainingStepBuilder",
        "spec_type": "LightgbmmtTraining",
        "sagemaker_step_type": "Training",
        "description": "LightGBM multi-task training with adaptive weighting and knowledge distillation",
    },
    
    # ... rest of entries ...
}
```

**Success Criteria**:
- ✅ Registry entry added with all required fields
- ✅ All mappings auto-generated from STEP_NAMES
- ✅ Step discoverable by name in Cursus framework

**Automatic Mappings Generated**:
- `CONFIG_CLASSES`: Maps step name → config class name
- `BUILDER_NAMES`: Maps step name → builder name
- `SPEC_TYPES`: Maps step name → specification type
- `SAGEMAKER_STEP_TYPES`: Maps step name → SageMaker step type

### 2.2 Naming Convention: Using `LightGBMMT` for Readability

**✅ Final Decision**: Use `LightGBMMT` (mixed case) throughout for better readability and consistency with the LightGBM naming convention.

#### Multi-Layer Discovery Strategy

The discovery system in `src/cursus/step_catalog/builder_discovery.py` uses a **multi-layer strategy** where registry matching is the PRIMARY method:

```python
def _extract_step_name_from_builder_file(self, file_name, class_name):
    """Extract step name using multi-layer strategy."""
    
    # Layer 1: Registry + Class Name Match (PRIMARY - MOST ROBUST)
    for step_name, step_info in self._registry_info.items():
        builder_step_name = step_info.get("builder_step_name")
        if builder_step_name and builder_step_name == class_name:
            return step_name  # ← Direct match! No file parsing needed
    
    # Layer 2: File Name Pattern Matching (FALLBACK)
    if file_name.startswith("builder_") and file_name.endswith("_step"):
        step_name_parts = file_name[8:-5].split("_")
        step_name = self._convert_parts_to_pascal_case_with_special_cases(step_name_parts)
    
    # Layer 3: Class Name Extraction (FALLBACK)
    if class_name.endswith("StepBuilder"):
        step_name = class_name[:-11]
    
    # Layer 4: Registry Validation
    if step_name in self._registry_info:
        return step_name
```

#### Why `LightGBMMT` Works Perfectly

**For `LightGBMMTTrainingStepBuilder`**:

1. **Layer 1 (Registry Match)**: ✅ PRIMARY METHOD
   - Class name: `LightGBMMTTrainingStepBuilder`
   - Registry entry: `"builder_step_name": "LightGBMMTTrainingStepBuilder"`
   - **Match found!** Returns `"LightGBMMTTraining"` immediately
   - **No need for file name parsing!**

2. **Layer 2-4**: Never reached because Layer 1 succeeds

#### Discovery Strategy Insights

| Layer | Method | When Used | For LightGBMMT |
|-------|--------|-----------|----------------|
| 1 | Registry + Class Match | **Primary** | ✅ **Used** - Direct match |
| 2 | File Name Parsing | Fallback | ❌ Not reached |
| 3 | Class Name Extract | Fallback | ❌ Not reached |
| 4 | Registry Validation | Final check | ❌ Not reached |

**Key Insight**: The special case for `lightgbm` we added helps **future base LightGBM steps** (not multi-task), while `LightGBMMT` uses the standard registry matching via Layer 1!

#### Framework Comparison

| Framework | Class Name | Registry Match | Result |
|-----------|------------|----------------|---------|
| XGBoost | `XGBoostTrainingStepBuilder` | Via registry | ✅ `XGBoostTraining` |
| PyTorch | `PyTorchTrainingStepBuilder` | Via registry | ✅ `PyTorchTraining` |
| LightGBM | `LightGBMTrainingStepBuilder` | Via registry | ✅ `LightGBMTraining` |
| **LightGBMMT** | `LightGBMMTTrainingStepBuilder` | **Via registry** | ✅ **`LightGBMMTTraining`** |

#### Updated Registry Entry

```python
"LightGBMMTTraining": {  # ← Note: LightGBMMT for readability!
    "config_class": "LightGBMMTTrainingConfig",
    "builder_step_name": "LightGBMMTTrainingStepBuilder",
    "spec_type": "LightGBMMTTraining",
    "sagemaker_step_type": "Training",
    "description": "LightGBM multi-task training with adaptive weighting and knowledge distillation",
}
```

**All class names use readable convention**:
- ✅ `LightGBMMTTrainingConfig` (readable, follows LightGBM pattern)
- ✅ `LightGBMMTTrainingStepBuilder` (readable, follows LightGBM pattern)
- ✅ File: `builder_lightgbmmt_training_step.py` (lowercase in filename, standard)

#### Special Case Addition (Defensive)

We added a special case for `lightgbm` in builder discovery:

```python
elif part.lower() == "lightgbm":
    result_parts.append("LightGBM")
```

This helps **Layer 2 (file name fallback)** for future base LightGBM steps, but our `LightGBMMT` step doesn't need it - it works via **Layer 1 (registry match)**! 🎯

This naming ensures the step is **readable, maintainable, and fully integrated** with the Cursus framework's multi-layer discovery architecture.

## Phase 3: Configuration Class ✅ COMPLETED (2025-11-13)

### 3.1 Create LightGBMMT Configuration ✅

**File**: `src/cursus/steps/configs/config_lightgbmmt_training_step.py` ✅ CREATED

```python
from pydantic import Field, field_validator
from typing import Optional

from ...core.base.config_base import BasePipelineConfig


class LightGBMMTTrainingConfig(BasePipelineConfig):
    """
    Configuration for LightGBMMT Training Step.
    
    Three-tier field classification:
    - Tier 1: Essential User Inputs (training_entry_point)
    - Tier 2: System Inputs with Defaults (instance types, model_class, etc.)
    - Tier 3: Derived Fields (from pipeline location via properties)
    
    Hyperparameters are managed separately via LightGBMMtModelHyperparameters
    and saved as hyperparameters.json in the source_dir.
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    training_entry_point: str = Field(
        description="Entry point script for LightGBMMT multi-task training."
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    training_instance_type: str = Field(
        default="ml.m5.4xlarge",
        description="Instance type for LightGBMMT training (CPU-optimized for LightGBM)."
    )
    
    training_instance_count: int = Field(
        default=1,
        ge=1,
        description="Number of instances for training job."
    )
    
    training_volume_size: int = Field(
        default=30,
        ge=1,
        description="Volume size (GB) for training instances."
    )
    
    model_class: str = Field(
        default="lightgbmmt",
        description="Model class identifier for multi-task LightGBM."
    )
    
    max_run_seconds: int = Field(
        default=86400,
        ge=1,
        description="Maximum runtime for training job (seconds). Default: 24 hours."
    )
    
    @field_validator("training_instance_type")
    @classmethod
    def validate_lightgbmmt_instance_type(cls, v: str) -> str:
        """
        Validate instance types suitable for LightGBMMT.
        
        LightGBM works efficiently on CPU instances, especially for multi-task learning
        where memory and compute balance is important.
        """
        # CPU-optimized instances for LightGBM
        valid_cpu_instances = [
            # General purpose (balanced)
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
            "ml.m5.8xlarge", "ml.m5.12xlarge",
            # Compute optimized (faster training)
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge",
            "ml.c5.9xlarge", "ml.c5.18xlarge",
            # Memory optimized (large datasets)
            "ml.r5.large", "ml.r5.xlarge", "ml.r5.2xlarge", "ml.r5.4xlarge",
            "ml.r5.8xlarge", "ml.r5.12xlarge",
        ]
        
        if v not in valid_cpu_instances:
            raise ValueError(
                f"Invalid instance type for LightGBMMT: {v}. "
                f"LightGBM requires CPU instances. "
                f"Valid options: {', '.join(valid_cpu_instances[:6])}..."
            )
        
        return v
```

**Success Criteria**:
- ✅ Three-tier field classification
- ✅ CPU-optimized instance validation
- ✅ Aligned with LightGBM config patterns
- ✅ Multi-task specific model_class
- ✅ Clear docstrings for all fields

### 3.2 Update Configs Module ✅

**File**: `src/cursus/steps/configs/__init__.py` ✅ UPDATED

Added import and export for LightGBMMTTrainingConfig

## Phase 4: Step Builder ✅ COMPLETED (2025-11-13)

### 4.1 Create LightGBMMT Step Builder ✅

**File**: `src/cursus/steps/builders/builder_lightgbmmt_training_step.py` ✅ CREATED

```python
"""
LightGBMMT Training Step Builder

Builds SageMaker Training steps for multi-task LightGBM models using
custom LightGBMMT framework with refactored loss functions and model architecture.
"""

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from typing import Dict, Optional
import logging

from ...core.base.builder_base import StepBuilderBase
from ..configs.config_lightgbmmt_training_step import LightGBMMTTrainingConfig
from ..specs.lightgbmmt_training_spec import LIGHTGBMMT_TRAINING_SPEC


class LightGBMMTTrainingStepBuilder(StepBuilderBase):
    """
    Step builder for LightGBMMT multi-task training.
    
    Uses custom LightGBMMT framework (fork of LightGBM with multi-label support)
    integrated with refactored loss functions and model architecture.
    
    Follows established patterns from LightGBM/XGBoost builders while
    accommodating multi-task specific requirements.
    """
    
    def __init__(self, config: LightgbmmtTrainingConfig, **kwargs):
        """
        Initialize Lightgbmmt training step builder.
        
        Parameters
        ----------
        config : LightgbmmtTrainingConfig
            Configuration for the training step
        **kwargs
            Additional arguments passed to StepBuilderBase
        """
        super().__init__(config, **kwargs)
        
        # Set specification for automatic dependency resolution
        self.spec = LIGHTGBMMT_TRAINING_SPEC
        
        self.log_info("Initialized LightGBMMT Training Step Builder")
        self.log_info(f"Using instance type: {config.training_instance_type}")
        self.log_info(f"Using instance count: {config.training_instance_count}")
    
    def _get_lightgbmmt_image_uri(self) -> str:
        """
        Get custom LightGBMMT Docker image URI.
        
        Note: Uses custom-built image with lightgbmmt framework and
        refactored loss functions/model architecture.
        
        For production deployment, this image should be in ECR.
        For development, may use local build.
        
        Returns
        -------
        image_uri : str
            Full Docker image URI
        """
        # TODO: Replace with actual ECR image URI
        # Example: 123456789.dkr.ecr.us-west-2.amazonaws.com/lightgbmmt:latest
        
        image_uri = (
            f"{self.account_id}.dkr.ecr.{self.aws_region}.amazonaws.com/"
            f"lightgbmmt:latest"
        )
        
        self.log_info(f"Using LightGBMMT image: {image_uri}")
        return image_uri
    
    def _create_estimator(self, output_path: Optional[str] = None) -> Estimator:
        """
        Creates and configures the LightGBMMT estimator.
        
        Uses generic Estimator class with custom LightGBMMT image URI.
        This defines the execution environment for the multi-task training job.
        
        Parameters
        ----------
        output_path : str, optional
            S3 path for model artifacts output
        
        Returns
        -------
        estimator : Estimator
            Configured SageMaker estimator
        """
        # Get custom LightGBMMT image URI
        image_uri = self._get_lightgbmmt_image_uri()
        
        # Use modernized effective_source_dir with hybrid resolution
        source_dir = self.config.effective_source_dir
        self.log_info(f"Using source directory: {source_dir}")
        
        # Create generic estimator with custom image
        estimator = Estimator(
            image_uri=image_uri,
            entry_point=self.config.training_entry_point,
            source_dir=source_dir,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            max_run=self.config.max_run_seconds,
            input_mode='File',
            output_path=output_path,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            environment=self._get_environment_variables(),
        )
        
        self.log_info("Created LightGBMMT estimator successfully")
        return estimator
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Construct environment variables for the training job.
        
        Returns minimal environment variables as the contract specifies
        no required env vars - configuration comes from hyperparameters.json.
        
        Returns
        -------
        env_vars : dict
            Environment variables for training container
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Add any LightGBMMT-specific environment variables from config
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)
        
        self.log_info(f"Training environment variables: {list(env_vars.keys())}")
        return env_vars
    
    def _get_inputs(self, inputs: Dict) -> Dict[str, TrainingInput]:
        """
        Process input dependencies into SageMaker training inputs.
        
        Creates train/val/test channels from the input_path dependency.
        
        Parameters
        ----------
        inputs : dict
            Input dependencies resolved from specification
            Expected keys: 'input_path', 'hyperparameters_s3_uri' (optional)
        
        Returns
        -------
        training_inputs : dict
            Dictionary mapping channel names to TrainingInput objects
        """
        training_inputs = {}
        
        if "input_path" in inputs:
            input_path = inputs["input_path"]
            self.log_info(f"Creating training channels from: {input_path}")
            
            # Create channels for train/val/test subdirectories
            training_inputs["train"] = TrainingInput(
                s3_data=f"{input_path}/train",
                content_type="text/csv"
            )
            training_inputs["val"] = TrainingInput(
                s3_data=f"{input_path}/val",
                content_type="text/csv"
            )
            training_inputs["test"] = TrainingInput(
                s3_data=f"{input_path}/test",
                content_type="text/csv"
            )
            
            self.log_info("Created train/val/test channels")
        
        # Note: hyperparameters_s3_uri is optional and handled via source_dir
        # if provided, training script will load from /opt/ml/code/hyperparams/
        
        return training_inputs
    
    def _get_outputs(self, inputs: Dict) -> str:
        """
        Generate output path for model artifacts.
        
        Parameters
        ----------
        inputs : dict
            Input dependencies (not used for output path generation)
        
        Returns
        -------
        output_path : str
            S3 path for model artifacts
        """
        # Generate default output path based on pipeline location
        output_path = f"{self.config.pipeline_s3_loc}/lightgbmmt_training"
        
        self.log_info(f"Model artifacts will be saved to: {output_path}")
        return output_path
    
    def create_step(
        self,
        inputs: Optional[Dict] = None,
        dependencies: Optional[list] = None
    ) -> TrainingStep:
        """
        Create a SageMaker TrainingStep for LightGBMMT multi-task training.
        
        This method:
        1. Resolves inputs using specification-driven dependency resolution
        2. Creates estimator with custom LightGBMMT image
        3. Configures training inputs (train/val/test channels)
        4. Creates TrainingStep with proper dependencies
        
        Parameters
        ----------
        inputs : dict, optional
            Input dependencies (if None, uses specification to resolve)
        dependencies : list, optional
            List of step dependencies
        
        Returns
        -------
        step : TrainingStep
            Configured SageMaker training step
        """
        self.log_info("Creating LightGBMMT training step")
        
        # Resolve inputs if not provided
        if inputs is None:
            inputs = {}
        
        # Get training inputs (train/val/test channels)
        training_inputs = self._get_inputs(inputs)
        
        # Get output path
        output_path = self._get_outputs(inputs)
        
        # Create estimator
        estimator = self._create_estimator(output_path=output_path)
        
        # Create training step
        step = TrainingStep(
            name=self.step_name,
            estimator=estimator,
            inputs=training_inputs,
            depends_on=dependencies or [],
        )
        
        # Attach specification for output resolution
        step._spec = self.spec
        
        self.log_info(f"Created training step: {self.step_name}")
        return step
```

**Success Criteria**:
- ✅ Extends StepBuilderBase
- ✅ Uses LIGHTGBMMT_TRAINING_SPEC for dependency resolution
- ✅ Custom LightGBMMT image URI
- ✅ Proper input/output handling
- ✅ Comprehensive logging
- ✅ Aligned with LightGBM builder patterns

### 4.2 Update Builders Module ✅

**File**: `src/cursus/steps/builders/__init__.py` ✅ UPDATED

Added import and export for LightGBMMTTrainingStepBuilder

## Phase 5: Testing & Validation (Week 3, Day 8)

### 5.1 Validation Framework Testing

Run comprehensive validation tests:

```bash
# Test step specification alignment
cursus validate-alignment \
  --step LightGBMMTTraining \
  --workspace main \
  --verbose

# Test step builder functionality
cursus validate-builder \
  --step LightGBMMTTraining \
  --workspace main \
  --verbose

# Test registry integration
python -c "
from cursus.registry.step_names import get_spec_step_type, get_config_class
print('Step type:', get_spec_step_type('LightGBMMTTraining'))
print('Config class:', get_config_class('LightGBMMTTraining'))
"
```

### 5.2 Integration Testing

Test complete pipeline integration:

```python
# test_lightgbmmt_pipeline_integration.py

from cursus.steps.configs import LightGBMMTTrainingConfig
from cursus.steps.builders import LightGBMMTTrainingStepBuilder
from cursus.steps.hyperparams import LightGBMMtModelHyperparameters

def test_complete_integration():
    """Test complete LightGBMMT integration."""
    
    # 1. Create hyperparameters
    hyperparams = LightGBMMtModelHyperparameters(
        full_field_list=full_fields,
        cat_field_list=cat_fields,
        tab_field_list=tab_fields,
        id_name='transaction_id',
        label_name='is_fraud',
        multiclass_categories=[0, 1],
        num_leaves=750,
        learning_rate=0.05,
        num_tasks=6,
        loss_type='adaptive_kd',
        loss_beta=0.3,
        loss_patience=50,
        loss_weight_method='sqrt',
    )
    
    # 2. Save hyperparameters for source_dir
    hyperparams_dict = hyperparams.model_dump()
    # Save to source_dir/hyperparams/hyperparameters.json
    
    # 3. Create config
    config = LightGBMMTTrainingConfig(
        region="us-west-2",
        pipeline_s3_loc="s3://bucket/prefix",
        training_entry_point="lightgbmmt_training.py",
    )
    
    # 4. Create builder
    builder = LightGBMMTTrainingStepBuilder(config)
    
    # 5. Verify specification attached
    assert builder.spec is not None
    assert builder.spec.step_type == "LightGBMMTTraining"
    
    # 6. Create step with dependencies
    step = builder.create_step(
        inputs={"input_path": "s3://bucket/preprocessed-data"}
    )
    
    # 7. Verify step created
    assert step is not None
    assert hasattr(step, '_spec')
    
    print("✅ Integration test passed!")

if __name__ == "__main__":
    test_complete_integration()
```

### 5.3 Unit Tests

Create comprehensive unit tests:

```python
# tests/steps/builders/test_builder_lightgbmmt_training_step.py

import unittest
from unittest.mock import patch, MagicMock

from cursus.steps.configs import LightGBMMTTrainingConfig
from cursus.steps.builders import LightGBMMTTrainingStepBuilder
from cursus.steps.specs import LIGHTGBMMT_TRAINING_SPEC


class TestLightGBMMTTrainingStepBuilder(unittest.TestCase):
    """Test suite for LightGBMMT training step builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LightGBMMTTrainingConfig(
            region="us-west-2",
            pipeline_s3_loc="s3://test-bucket/test-prefix",
            training_entry_point="lightgbmmt_training.py",
        )
        self.builder = LightGBMMTTrainingStepBuilder(self.config)
    
    def test_initialization(self):
        """Test that builder initializes correctly with specification."""
        self.assertIsNotNone(self.builder.spec)
        self.assertEqual(self.builder.spec, LIGHTGBMMT_TRAINING_SPEC)
        self.assertEqual(self.builder.spec.step_type, "LightGBMMTTraining")
    
    def test_create_estimator(self):
        """Test that LightGBMMT estimator is created correctly."""
        estimator = self.builder._create_estimator()
        
        # Verify it's using Estimator class
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.instance_type, "ml.m5.4xlarge")
        self.assertEqual(estimator.instance_count, 1)
    
    def test_get_lightgbmmt_image_uri(self):
        """Test that LightGBMMT image URI is generated correctly."""
        image_uri = self.builder._get_lightgbmmt_image_uri()
        
        # Should contain lightgbmmt in the URI
        self.assertIsNotNone(image_uri)
        self.assertIn("lightgbmmt", image_uri.lower())
        self.assertIn(self.builder.aws_region, image_uri)
    
    def test_get_inputs(self):
        """Test that inputs are correctly derived from dependencies."""
        inputs = {"input_path": "s3://bucket/input/data"}
        training_inputs = self.builder._get_inputs(inputs)
        
        # Should create train/val/test channels
        self.assertIn("train", training_inputs)
        self.assertIn("val", training_inputs)
        self.assertIn("test", training_inputs)
        
        # Verify S3 paths
        self.assertIn("/train", training_inputs["train"].config["DataSource"]["S3DataSource"]["S3Uri"])
    
    def test_get_outputs(self):
        """Test that outputs are correctly configured."""
        output_path = self.builder._get_outputs({})
        
        # Should generate output path
        self.assertIsNotNone(output_path)
        self.assertIn("lightgbmmt_training", output_path)
        self.assertIn("s3://", output_path)
    
    def test_create_step(self):
        """Test step creation with dependencies."""
        inputs = {"input_path": "s3://bucket/input/data"}
        step = self.builder.create_step(inputs=inputs)
        
        # Verify step was created
        self.assertIsNotNone(step)
        self.assertTrue(hasattr(step, '_spec'))
        self.assertEqual(step._spec, LIGHTGBMMT_TRAINING_SPEC)

if __name__ == '__main__':
    unittest.main()
```

## Summary

### Timeline
- **Week 3, Days 1-2**: Step specification
- **Week 3, Day 3**: Registry integration
- **Week 3, Days 4-5**: Configuration class
- **Week 3, Days 6-7**: Step builder
- **Week 3, Day 8**: Testing & validation (optional)

**Total**: 1 week (can overlap with Part 1 testing)

### Deliverables
1. ✅ Step specification for dependency resolution
2. ✅ Registry integration (step discoverable)
3. ✅ Configuration class (three-tier design)
4. ✅ Step builder (specification-driven)
5. ✅ Complete Cursus integration
6. ✅ Comprehensive testing
