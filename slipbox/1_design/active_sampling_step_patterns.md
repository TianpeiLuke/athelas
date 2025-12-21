---
tags:
  - design
  - step_builders
  - active_sampling
  - semi_supervised_learning
  - active_learning
  - sagemaker
  - processing_steps
keywords:
  - active sample selection
  - confidence-based sampling
  - uncertainty sampling
  - diversity sampling
  - BADGE sampling
  - SSL pipeline
  - active learning pipeline
topics:
  - step builder patterns
  - active sampling strategies
  - SageMaker processing
  - semi-supervised learning
  - model-agnostic selection
language: python
date of note: 2025-11-17
---

# Active Sampling Step Builder Patterns

## Overview

This document defines the design patterns for Active Sampling step builder implementations in the cursus framework. The Active Sampling step creates **ProcessingStep** instances that intelligently select high-value samples from model predictions for either:
1. **Semi-Supervised Learning (SSL)**: Selecting high-confidence predictions for pseudo-labeling
2. **Active Learning**: Selecting uncertain/diverse predictions for human labeling

The step is **model-agnostic** and integrates seamlessly with XGBoost, LightGBM, and PyTorch model inference outputs.

## Key Architectural Approach: Model-Agnostic Processing

The Active Sampling step uses a **model-agnostic approach** where sample selection is based solely on prediction probabilities, regardless of the model type that generated them.

**Input Requirements:**
```
predictions.csv/parquet with columns:
├── id (or configured ID field)
├── feature columns (preserved in output)
├── prob_class_0, prob_class_1, ... (prediction probabilities)
└── optional: label column (for validation mode)
```

**Supported Model Types:**
- XGBoost (via XGBoostModelInference)
- LightGBM (via LightGBMModelInference)
- PyTorch (via PyTorchModelInference)
- Bedrock/LLM (via BedrockBatchProcessing with probability extraction)

## Integration with ML Pipelines

### Semi-Supervised Learning Integration

```
Labeled Data → XGBoostTraining (pretrain) → Pretrained Model
                                                   ↓
Unlabeled Data → TabularPreprocessing → XGBoostModelInference → Predictions
                                                                      ↓
                                         ActiveSampleSelection (SSL) → High-confidence samples
                                                                      ↓
                                            PseudoLabelMerge → Combined labeled + pseudo-labeled
                                                                      ↓
                                         XGBoostTraining (finetune) → Fine-tuned Model
```

### Active Learning Integration

```
Unlabeled Pool → TabularPreprocessing → XGBoostModelInference → Predictions
                                                                      ↓
                                    ActiveSampleSelection (AL) → Uncertain/diverse samples
                                                                      ↓
                                           Human Labeling → New labeled samples
                                                                      ↓
                                    Merge with existing labeled → Expanded training set
                                                                      ↓
                                         XGBoostTraining → Improved Model
```

## Core Implementation Components

### 1. Script Contract Design

**File**: `src/cursus/steps/contracts/active_sample_selection_contract.py`

```python
from ...core.base.contract_base import ScriptContract

ACTIVE_SAMPLE_SELECTION_CONTRACT = ScriptContract(
    entry_point="active_sample_selection.py",
    expected_input_paths={
        "evaluation_data": "/opt/ml/processing/input/evaluation_data",
    },
    expected_output_paths={
        "selected_samples": "/opt/ml/processing/output/selected_samples",
        "selection_metadata": "/opt/ml/processing/output/selection_metadata",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["SELECTION_STRATEGY"],
    optional_env_vars={
        "USE_CASE": "auto",  # ssl, active_learning, auto
        "CONFIDENCE_THRESHOLD": "0.9",
        "MAX_SAMPLES": "0",
        "K_PER_CLASS": "100",
        "UNCERTAINTY_MODE": "margin",
        "BATCH_SIZE": "32",
        "METRIC": "euclidean",
        "ID_FIELD": "id",
        "LABEL_FIELD": "",
        "OUTPUT_FORMAT": "csv",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Active sample selection script that intelligently selects high-value samples
    from model predictions for Semi-Supervised Learning or Active Learning workflows.
    
    Selection Strategies:
    
    1. SSL Strategies (confidence-based):
       - confidence_threshold: Selects samples with prob >= threshold
       - top_k_per_class: Selects top-k most confident per predicted class
    
    2. Active Learning Strategies (uncertainty/diversity-based):
       - uncertainty: Selects uncertain samples (margin, entropy, least_confidence)
       - diversity: Selects diverse samples via k-center algorithm
       - badge: Combines uncertainty + diversity via gradient embeddings
    
    Input Structure:
    - /opt/ml/processing/input/predictions: Model predictions directory containing:
      - predictions.csv/parquet: Predictions with probability columns
        Required columns: id (or configured ID_FIELD), prob_class_0, prob_class_1, ...
        Optional columns: label (for validation), feature columns (preserved)
    
    Output Structure:
    - /opt/ml/processing/output/selected: Selected samples directory containing:
      - selected_samples.csv/parquet: Selected samples with metadata
        Columns: Original columns + pseudo_label + confidence + data_source
    
    Environment Variables:
    - SELECTION_STRATEGY (required): Strategy name
    - USE_CASE (optional): ssl, active_learning, or auto (default: auto)
    - ID_FIELD (optional): ID column name (default: "id")
    - LABEL_FIELD (optional): Label column name (default: "")
    - OUTPUT_FORMAT (optional): csv or parquet (default: "csv")
    
    SSL-specific variables:
    - CONFIDENCE_THRESHOLD (optional): Min confidence (default: 0.9)
    - MAX_SAMPLES (optional): Max samples to select (default: 0 = no limit)
    - K_PER_CLASS (optional): Samples per class (default: 100)
    
    Active Learning-specific variables:
    - UNCERTAINTY_MODE (optional): margin, entropy, least_confidence (default: margin)
    - BATCH_SIZE (optional): Number of samples to select (default: 32)
    - METRIC (optional): euclidean or cosine (default: euclidean)
    
    Arguments:
    - job_type: Type of selection job (e.g., "ssl_selection", "active_learning_selection")
    
    Use Case Validation:
    When USE_CASE is set to "ssl" or "active_learning", the script validates that
    the selected strategy is appropriate:
    - SSL: Only allows confidence_threshold, top_k_per_class
    - Active Learning: Only allows uncertainty, diversity, badge
    - Auto: No validation (advanced users)
    
    Downstream Integration:
    - SSL output → PseudoLabelMerge → Combined training data
    - Active Learning output → Human labeling interface → Labeled data
    
    The output preserves all original columns and adds selection metadata for
    tracking provenance and confidence scores.
    """,
)
```

### 2. Step Specification Design

**File**: `src/cursus/steps/specs/active_sample_selection_spec.py`

```python
from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.active_sample_selection_contract import (
        ACTIVE_SAMPLE_SELECTION_CONTRACT,
    )


def _get_active_sample_selection_contract():
    from ..contracts.active_sample_selection_contract import (
        ACTIVE_SAMPLE_SELECTION_CONTRACT,
    )
    return ACTIVE_SAMPLE_SELECTION_CONTRACT


ACTIVE_SAMPLE_SELECTION_SPEC = StepSpecification(
    step_type=get_spec_step_type("ActiveSampleSelection"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_active_sample_selection_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                # Model inference outputs
                "XGBoostModelInference",
                "LightGBMModelInference",
                "PyTorchModelInference",
                # Model evaluation outputs (includes predictions)
                "XGBoostModelEval",
                "LightGBMModelEval",
                "PyTorchModelEval",
                # Bedrock/LLM outputs (with probability extraction)
                "BedrockBatchProcessing",
                "BedrockProcessing",
                # Label ruleset execution (classification outputs)
                "LabelRulesetExecution",
            ],
            semantic_keywords=[
                "evaluation",
                "predictions",
                "inference",
                "model_predictions",
                "inference_results",
                "eval_output",
                "prediction_data",
                "processed_data",
                "classification_output",
                "probabilities",
            ],
            data_type="S3Uri",
            description="Model predictions with probability columns for sample selection",
        ),
    },
    outputs={
        "selected_samples": OutputSpec(
            logical_name="selected_samples",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selected_samples'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "active_samples",
                "high_confidence_samples",
                "pseudo_labeled_samples",
                "uncertain_samples",
                "selection_output",
                # Training data aliases for XGBoost/LightGBM/PyTorch Training compatibility
                "input_path",  # Exact match for XGBoost/LightGBM/PyTorch Training input
                "input_data",
                "processed_data",
            ],
            description="Selected samples with confidence scores and metadata",
        ),
        "selection_metadata": OutputSpec(
            logical_name="selection_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selection_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "metadata",
                "selection_info",
                "sampling_metadata",
            ],
            description="Selection metadata including strategy config, counts, and timestamp",
        ),
    },
)
```

**Key Design Decisions for Step Spec:**

1. **Compatible Sources**: Includes ALL model types that produce predictions:
   - Tree-based: XGBoost, LightGBM
   - Neural: PyTorch
   - LLM: Bedrock (with prob extraction)
   - Rule-based: Label Ruleset Execution

2. **Semantic Keywords**: Broad coverage for different prediction output names across model types

3. **Output Aliases**: Multiple aliases to support both SSL and Active Learning terminology

### 3. Configuration Design

**File**: `src/cursus/steps/configs/config_active_sample_selection_step.py`

```python
from pydantic import Field, field_validator, model_validator
from typing import Literal, Dict, Any, Optional
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class ActiveSampleSelectionConfig(ProcessingStepConfigBase):
    """
    Configuration for Active Sample Selection step.
    
    Supports both Semi-Supervised Learning (SSL) and Active Learning workflows
    with Pydantic validation to prevent strategy misuse.
    
    Three-Tier Configuration:
    - Tier 1: Essential User Inputs (none - all have defaults)
    - Tier 2: System Fields with Defaults (all selection parameters)
    - Tier 3: Derived Fields (inherited from ProcessingStepConfigBase)
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    # None for this step - all have reasonable defaults
    
    # ===== System Fields with Defaults (Tier 2) =====
    
    # Core selection parameters
    selection_strategy: Literal[
        "confidence_threshold",
        "top_k_per_class", 
        "uncertainty",
        "diversity",
        "badge"
    ] = Field(
        default="confidence_threshold",
        description=(
            "Selection strategy. "
            "SSL: confidence_threshold, top_k_per_class. "
            "Active Learning: uncertainty, diversity, badge."
        )
    )
    
    use_case: Literal["ssl", "active_learning", "auto"] = Field(
        default="auto",
        description=(
            "Use case for validation: ssl, active_learning, or auto. "
            "When auto, no validation. When specified, validates strategy compatibility."
        )
    )
    
    # Data field configuration
    id_field: str = Field(
        default="id",
        description="ID column name in predictions data"
    )
    
    label_field: str = Field(
        default="",
        description="Label column name (optional, for validation mode)"
    )
    
    output_format: Literal["csv", "parquet"] = Field(
        default="csv",
        description="Output format for selected samples"
    )
    
    # SSL-specific parameters
    confidence_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="For SSL: minimum confidence threshold (0.5-1.0)"
    )
    
    k_per_class: int = Field(
        default=100,
        ge=1,
        description="For SSL: top-k samples per class"
    )
    
    max_samples: int = Field(
        default=0,
        ge=0,
        description="For SSL: max samples to select (0=no limit)"
    )
    
    # Active Learning-specific parameters
    uncertainty_mode: Literal["margin", "entropy", "least_confidence"] = Field(
        default="margin",
        description="For Active Learning: uncertainty sampling mode"
    )
    
    batch_size: int = Field(
        default=32,
        ge=1,
        description="For Active Learning: number of samples to select"
    )
    
    metric: Literal["euclidean", "cosine"] = Field(
        default="euclidean",
        description="For Active Learning diversity/BADGE: distance metric"
    )
    
    # Processing configuration
    processing_entry_point: str = Field(
        default="active_sample_selection.py",
        description="Entry point script for active sample selection"
    )
    
    processing_framework_version: str = Field(
        default="1.2-1",
        description="SKLearn framework version for processing"
    )
    
    # ===== Validators =====
    
    @field_validator("selection_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate selection strategy is one of the allowed values."""
        allowed = {
            "confidence_threshold",
            "top_k_per_class",
            "uncertainty", 
            "diversity",
            "badge"
        }
        if v not in allowed:
            raise ValueError(
                f"selection_strategy must be one of {allowed}, got '{v}'"
            )
        return v
    
    @field_validator("use_case")
    @classmethod
    def validate_use_case(cls, v: str) -> str:
        """Validate use case is one of the allowed values."""
        allowed = {"ssl", "active_learning", "auto"}
        if v not in allowed:
            raise ValueError(
                f"use_case must be one of {allowed}, got '{v}'"
            )
        return v
    
    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"csv", "parquet"}
        if v not in allowed:
            raise ValueError(
                f"output_format must be one of {allowed}, got '{v}'"
            )
        return v
    
    @model_validator(mode="after")
    def validate_strategy_use_case_compatibility(
        self
    ) -> "ActiveSampleSelectionConfig":
        """
        ⚠️ CRITICAL: Validate strategy is compatible with use case.
        
        This cross-field validation prevents:
        - Using uncertainty strategies for SSL (creates noisy pseudo-labels)
        - Using confidence strategies for Active Learning (wastes human effort)
        
        Raises:
            ValueError: If strategy is incompatible with use_case
        """
        # Define strategy categories
        SSL_STRATEGIES = {"confidence_threshold", "top_k_per_class"}
        ACTIVE_LEARNING_STRATEGIES = {"uncertainty", "diversity", "badge"}
        
        # Skip validation if use_case is "auto"
        if self.use_case == "auto":
            logger.debug(
                f"use_case='auto', skipping validation for strategy '{self.selection_strategy}'"
            )
            return self
        
        # Validate SSL use case
        if self.use_case == "ssl":
            if self.selection_strategy not in SSL_STRATEGIES:
                raise ValueError(
                    f"❌ Strategy '{self.selection_strategy}' is NOT valid for SSL! "
                    f"SSL requires confidence-based strategies: {SSL_STRATEGIES}. "
                    f"Strategy '{self.selection_strategy}' selects UNCERTAIN samples, "
                    f"which would create noisy pseudo-labels and degrade model performance. "
                    f"Use 'confidence_threshold' or 'top_k_per_class' instead."
                )
            logger.info(
                f"✓ Strategy '{self.selection_strategy}' validated for SSL use case"
            )
        
        # Validate Active Learning use case
        elif self.use_case == "active_learning":
            if self.selection_strategy not in ACTIVE_LEARNING_STRATEGIES:
                raise ValueError(
                    f"⚠️ Strategy '{self.selection_strategy}' is NOT recommended for Active Learning! "
                    f"Active Learning typically uses: {ACTIVE_LEARNING_STRATEGIES}. "
                    f"Strategy '{self.selection_strategy}' selects CONFIDENT samples, "
                    f"which wastes human labeling effort on easy samples. "
                    f"Use 'uncertainty', 'diversity', or 'badge' instead."
                )
            logger.info(
                f"✓ Strategy '{self.selection_strategy}' validated for Active Learning use case"
            )
        
        return self
    
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ActiveSampleSelectionConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        
        # Add any active-sampling-specific initialization here
        
        return self
    
    # ===== Helper Methods =====
    
    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the processing job.
        
        Returns:
            Dictionary of environment variables
        """
        env_vars = {
            "SELECTION_STRATEGY": self.selection_strategy,
            "USE_CASE": self.use_case,
            "ID_FIELD": self.id_field,
            "LABEL_FIELD": self.label_field,
            "OUTPUT_FORMAT": self.output_format,
        }
        
        # Add SSL-specific variables
        if self.selection_strategy in {"confidence_threshold", "top_k_per_class"}:
            env_vars["CONFIDENCE_THRESHOLD"] = str(self.confidence_threshold)
            env_vars["MAX_SAMPLES"] = str(self.max_samples)
            env_vars["K_PER_CLASS"] = str(self.k_per_class)
        
        # Add Active Learning-specific variables
        if self.selection_strategy in {"uncertainty", "diversity", "badge"}:
            env_vars["UNCERTAINTY_MODE"] = self.uncertainty_mode
            env_vars["BATCH_SIZE"] = str(self.batch_size)
            env_vars["METRIC"] = self.metric
        
        return env_vars
```

### 4. Step Builder Design

**File**: `src/cursus/steps/builders/builder_active_sample_selection_step.py`

```python
from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_active_sample_selection_step import ActiveSampleSelectionConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the specification
try:
    from ..specs.active_sample_selection_spec import ACTIVE_SAMPLE_SELECTION_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    ACTIVE_SAMPLE_SELECTION_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ActiveSampleSelectionStepBuilder(StepBuilderBase):
    """
    Builder for Active Sample Selection ProcessingStep.
    
    This step intelligently selects samples from model predictions for:
    - Semi-Supervised Learning: High-confidence samples for pseudo-labeling
    - Active Learning: Uncertain/diverse samples for human labeling
    
    Supports multiple selection strategies:
    - confidence_threshold: Samples with confidence >= threshold
    - top_k_per_class: Top-k confident samples per class
    - uncertainty: Uncertain samples (margin, entropy, least_confidence)
    - diversity: Diverse samples via k-center algorithm
    - badge: Combined uncertainty + diversity via gradient embeddings
    """

    def __init__(
        self,
        config: ActiveSampleSelectionConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initialize the builder with configuration.

        Args:
            config: ActiveSampleSelectionConfig instance
            sagemaker_session: SageMaker session for AWS interactions
            role: IAM role ARN for SageMaker Processing Job
            registry_manager: Optional registry manager
            dependency_resolver: Optional dependency resolver
        """
        if not isinstance(config, ActiveSampleSelectionConfig):
            raise ValueError(
                "ActiveSampleSelectionStepBuilder requires ActiveSampleSelectionConfig instance"
            )

        # Use specification if available
        spec = ACTIVE_SAMPLE_SELECTION_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: ActiveSampleSelectionConfig = config

    def validate_configuration(self) -> None:
        """
        Validate configuration for active sample selection.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        self.log_info("Validating ActiveSampleSelectionConfig...")

        # Validate required attributes
        required_attrs = [
            "processing_entry_point",
            "processing_source_dir",
            "processing_instance_count",
            "processing_volume_size",
            "pipeline_name",
            "selection_strategy",
            "use_case",
            "id_field",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(
                    f"ActiveSampleSelectionConfig missing required attribute: {attr}"
                )

        # Validate instance type settings
        if not hasattr(self.config, "processing_instance_type_large"):
            raise ValueError("Missing required attribute: processing_instance_type_large")
        if not hasattr(self.config, "processing_instance_type_small"):
            raise ValueError("Missing required attribute: processing_instance_type_small")
        if not hasattr(self.config, "use_large_processing_instance"):
            raise ValueError("Missing required attribute: use_large_processing_instance")

        # Strategy-use case validation is handled by config Pydantic validators
        
        self.log_info("ActiveSampleSelectionConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Create SKLearnProcessor for sample selection.

        Returns:
            Configured SKLearnProcessor instance
        """
        # Get appropriate instance type
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the processing job.

        Returns:
            Dictionary of environment variables
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add selection-specific environment variables from config
        selection_env_vars = self.config.get_environment_variables()
        env_vars.update(selection_env_vars)

        self.log_info("Active sampling environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects

        Raises:
            ValueError: If specification or contract is unavailable
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        processing_inputs = []

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue

            # Ensure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")

            # Get container path from contract
            if logical_name not in self.contract.expected_input_paths:
                raise ValueError(f"No container path found for input: {logical_name}")
            
            container_path = self.contract.expected_input_paths[logical_name]

            # Create ProcessingInput
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                )
            )

        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the step using specification and contract.

        Args:
            outputs: Output destinations keyed by logical name

        Returns:
            List of ProcessingOutput objects

        Raises:
            ValueError: If specification or contract is unavailable
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for output mapping")

        processing_outputs = []

        # Process each output in the specification
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name

            # Get container path from contract
            if logical_name not in self.contract.expected_output_paths:
                raise ValueError(f"No container path found for output: {logical_name}")
            
            container_path = self.contract.expected_output_paths[logical_name]

            # Get destination
            destination = None
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from base path
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                destination = Join(
                    on="/", values=[base_output_path, "active_sampling", logical_name]
                )
                self.log_info(
                    "Using generated destination for '%s': %s",
                    logical_name,
                    destination,
                )

            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination,
                )
            )

        return processing_outputs

    def _get_job_arguments(self) -> List[str]:
        """
        Get command-line arguments for the processing script.

        Returns:
            List of command-line arguments
        """
        # Get job_type from configuration or use default
        job_type = getattr(self.config, "job_type", "ssl_selection")
        self.log_info("Setting job_type argument to: %s", job_type)

        return ["--job_type", job_type]

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create the fully configured SageMaker ProcessingStep.

        Args:
            **kwargs: Configuration parameters including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of dependent steps
                - enable_caching: Boolean for caching results

        Returns:
            Configured ProcessingStep instance
        """
        self.log_info("Creating ActiveSampleSelection ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get("inputs", {})
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        # Handle inputs
        inputs = {}

        # Extract inputs from dependencies if provided
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Add explicitly provided inputs (overriding extracted ones)
        inputs.update(inputs_raw)

        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name
        step_name = self._get_step_name()

        # Get script path
        script_path = self.config.get_script_path()
        self.log_info("Using script path: %s", script_path)

        # Create step arguments
        step_args = processor.run(
            code=script_path,
            inputs=proc_inputs,
            outputs=proc_outputs,
            arguments=job_args,
        )

        # Create and return the step
        processing_step = ProcessingStep(
            name=step_name,
            step_args=step_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

        # Attach specification for future reference
        if hasattr(self, "spec") and self.spec:
            setattr(processing_step, "_spec", self.spec)

        self.log_info("Created ProcessingStep with name: %s", processing_step.name)
        return processing_step
```

## Key Design Principles

### 1. Model-Agnostic Architecture

The Active Sampling step works with any model that produces prediction probabilities:

```python
# Works with XGBoost predictions
predictions_xgb.csv:
id, feature1, feature2, prob_class_0, prob_class_1

# Works with LightGBM predictions  
predictions_lgbm.csv:
id, feature1, feature2, prob_class_0, prob_class_1

# Works with PyTorch predictions
predictions_pytorch.csv:
id, feature1, feature2, prob_class_0, prob_class_1, prob_class_2

# Works with Bedrock/LLM predictions (after probability extraction)
predictions_bedrock.csv:
id, feature1, feature2, prob_class_0, prob_class_1
```

The step only requires `prob_class_*` columns and doesn't need model-specific artifacts.

### 2. Use Case Validation via Pydantic

**Critical Feature**: Pydantic validators prevent strategy misuse at configuration time:

```python
# ✅ VALID: SSL with confidence strategy
config = ActiveSampleSelectionConfig(
    selection_strategy="confidence_threshold",
    use_case="ssl",  # ← Validates compatibility
    confidence_threshold=0.9
)

# ❌ INVALID: SSL with uncertainty strategy - Raises ValueError
config = ActiveSampleSelectionConfig(
    selection_strategy="uncertainty",  # ← Wrong for SSL!
    use_case="ssl",
)
# ValueError: Strategy 'uncertainty' is NOT valid for SSL!
```

This validation happens at **config creation time**, not runtime, providing immediate feedback.

### 3. Flexible Output Compatibility

The selected samples output is designed to be compatible with multiple downstream steps:

**For SSL (Semi-Supervised Learning):**
```python
# Output includes pseudo-labels
selected_samples.csv:
id, features..., prob_class_0, prob_class_1, pseudo_label, confidence, data_source

# Compatible with PseudoLabelMerge input
# Compatible with XGBoostTraining input (after merge)
```

**For Active Learning:**
```python
# Output includes uncertainty scores
selected_samples.csv:
id, features..., prob_class_0, prob_class_1, uncertainty_score, data_source

# Compatible with human labeling interface
# Compatible with training after human labels added
```

### 4. Provenance Tracking

All selected samples include `data_source` column for tracking:

```python
data_source values:
- "pseudo_labeled" (for SSL-selected samples)
- "active_learning" (for AL-selected samples)  
- "original" (for human-labeled samples, added by PseudoLabelMerge)
```

This enables:
- Quality monitoring of pseudo-labels vs ground truth
- Ablation studies on pseudo-label impact
- Iterative refinement of selection strategies

### 5. Strategy-Specific Environment Variables

The builder only includes relevant environment variables based on strategy:

```python
def get_environment_variables(self) -> Dict[str, str]:
    env_vars = {"SELECTION_STRATEGY": self.selection_strategy, ...}
    
    # SSL-specific
    if self.selection_strategy in {"confidence_threshold", "top_k_per_class"}:
        env_vars["CONFIDENCE_THRESHOLD"] = str(self.confidence_threshold)
        env_vars["K_PER_CLASS"] = str(self.k_per_class)
    
    # Active Learning-specific
    if self.selection_strategy in {"uncertainty", "diversity", "badge"}:
        env_vars["UNCERTAINTY_MODE"] = self.uncertainty_mode
        env_vars["BATCH_SIZE"] = str(self.batch_size)
    
    return env_vars
```

This keeps the processing job environment clean and focused.

### 6. Specification-Driven Dependency Resolution

The step specification defines broad compatibility with model inference outputs:

```python
compatible_sources=[
    "XGBoostModelInference",
    "LightGBMModelInference",
    "PyTorchModelInference",
    "XGBoostModelEval",
    "LightGBMModelEval",
    "PyTorchModelEval",
    "BedrockBatchProcessing",
    "BedrockProcessing",
    "LabelRulesetExecution",
]
```

This enables automatic dependency resolution in the pipeline assembly:

```python
# Automatic dependency resolution via spec
dag = PipelineDAG()
dag.add_node("XGBoostModelInference")
dag.add_node("ActiveSampleSelection")
dag.add_edge("XGBoostModelInference", "ActiveSampleSelection")  # Auto-resolved!
```

## Usage Examples

### Example 1: SSL Pipeline with Confidence Threshold

```python
from cursus.steps.configs.config_active_sample_selection_step import ActiveSampleSelectionConfig
from cursus.steps.builders.builder_active_sample_selection_step import ActiveSampleSelectionStepBuilder

# Create SSL configuration
ssl_config = ActiveSampleSelectionConfig(
    # Essential fields from base config
    author="data-scientist",
    bucket="my-ml-bucket",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    region="NA",
    service_name="recommendation",
    pipeline_version="v1.0",
    project_root_folder="projects/recommendation",
    
    # SSL-specific configuration
    selection_strategy="confidence_threshold",
    use_case="ssl",  # ← Enables validation
    confidence_threshold=0.9,
    max_samples=10000,
    
    # Data configuration
    id_field="user_id",
    output_format="parquet",
    
    # Processing settings
    processing_source_dir="projects/recommendation/scripts",
    use_large_processing_instance=False,
)

# Create step builder
builder = ActiveSampleSelectionStepBuilder(
    config=ssl_config,
    role=ssl_config.role,
)

# Create step with automatic dependency resolution
ssl_selection_step = builder.create_step(
    inputs={
        "evaluation_data": inference_step.properties
            .ProcessingOutputConfig
            .Outputs['eval_output']
            .S3Output.S3Uri
    },
    outputs={
        "selected_samples": "s3://my-ml-bucket/ssl/selected/"
    },
    dependencies=[inference_step]
)
```

### Example 2: Active Learning with Uncertainty Sampling

```python
# Create Active Learning configuration
al_config = ActiveSampleSelectionConfig(
    # Base fields...
    author="ml-engineer",
    bucket="my-ml-bucket",
    # ... other base fields
    
    # Active Learning-specific
    selection_strategy="uncertainty",
    use_case="active_learning",  # ← Enables validation
    uncertainty_mode="margin",
    batch_size=50,
    
    # Data configuration
    id_field="sample_id",
    output_format="csv",
)

# Create step
al_builder = ActiveSampleSelectionStepBuilder(config=al_config)

al_selection_step = al_builder.create_step(
    inputs={
        "evaluation_data": model_inference_output
    },
    outputs={
        "selected_samples": "s3://my-ml-bucket/active_learning/batch_1/"
    }
)
```

### Example 3: Top-K Per Class for Balanced SSL

```python
# Top-k ensures balanced pseudo-labeling
balanced_ssl_config = ActiveSampleSelectionConfig(
    # Base fields...
    selection_strategy="top_k_per_class",
    use_case="ssl",
    k_per_class=500,  # 500 samples per class
    id_field="record_id",
)

builder = ActiveSampleSelectionStepBuilder(config=balanced_ssl_config)

balanced_step = builder.create_step(
    inputs={"evaluation_data": predictions_s3_uri},
    outputs={"selected_samples": output_s3_uri},
)
```

## Integration Patterns

### Pattern 1: SSL with XGBoost

```python
# Phase 1: Pretrain on small labeled data
pretrain_step = XGBoostTrainingStepBuilder(pretrain_config).create_step(...)

# Phase 2: Inference on large unlabeled data
inference_step = XGBoostModelInferenceStepBuilder(inference_config).create_step(
    inputs={
        "model_input": pretrain_step.properties.ModelArtifacts.S3ModelArtifacts,
        "processed_data": unlabeled_data_s3
    }
)

# Phase 3: Select high-confidence samples
selection_step = ActiveSampleSelectionStepBuilder(ssl_config).create_step(
    inputs={
        "evaluation_data": inference_step.properties
            .ProcessingOutputConfig
            .Outputs['eval_output']
            .S3Output.S3Uri
    },
    dependencies=[inference_step]
)

# Phase 4: Merge labeled + pseudo-labeled
merge_step = PseudoLabelMergeStepBuilder(merge_config).create_step(
    inputs={
        "original_labeled": labeled_data_s3,
        "pseudo_labeled": selection_step.properties
            .ProcessingOutputConfig
            .Outputs['selected_samples']
            .S3Output.S3Uri
    },
    dependencies=[selection_step]
)

# Phase 5: Fine-tune on combined data
finetune_step = XGBoostTrainingStepBuilder(finetune_config).create_step(
    inputs={
        "train": merge_step.properties
            .ProcessingOutputConfig
            .Outputs['combined_data']
            .S3Output.S3Uri
    },
    dependencies=[merge_step]
)
```

### Pattern 2: Active Learning Loop

```python
# Iteration 1: Train on initial labeled data
initial_model = XGBoostTrainingStepBuilder(config).create_step(...)

# Generate predictions on unlabeled pool
predictions = XGBoostModelInferenceStepBuilder(config).create_step(...)

# Select uncertain samples for labeling
uncertain_samples = ActiveSampleSelectionStepBuilder(al_config).create_step(...)

# Human labels uncertain samples → new_labeled_data

# Iteration 2: Train on expanded dataset (initial + new)
improved_model = XGBoostTrainingStepBuilder(config).create_step(
    inputs={"train": combined_labeled_data}
)

# Repeat until performance plateau or budget exhausted
```

### Pattern 3: Multi-Model Ensemble SSL

```python
# Train multiple models
xgb_inference = XGBoostModelInferenceStepBuilder(config).create_step(...)
lgbm_inference = LightGBMModelInferenceStepBuilder(config).create_step(...)

# Select samples with agreement from both models
# (requires custom merge logic before ActiveSampleSelection)
ensemble_predictions = merge_predictions(xgb_inference, lgbm_inference)

# Select high-confidence samples where models agree
selection = ActiveSampleSelectionStepBuilder(ssl_config).create_step(
    inputs={"evaluation_data": ensemble_predictions}
)
```

## Error Handling and Validation

### Configuration Validation Errors

```python
# ❌ Example 1: Wrong strategy for SSL
try:
    config = ActiveSampleSelectionConfig(
        selection_strategy="uncertainty",
        use_case="ssl",
        # ...
    )
except ValueError as e:
    print(e)
    # "Strategy 'uncertainty' is NOT valid for SSL! ..."

# ❌ Example 2: Invalid confidence threshold
try:
    config = ActiveSampleSelectionConfig(
        confidence_threshold=1.5,  # > 1.0
        # ...
    )
except ValidationError as e:
    print(e)
    # "confidence_threshold must be between 0.5 and 1.0"

# ❌ Example 3: Missing required base fields
try:
    config = ActiveSampleSelectionConfig(
        selection_strategy="confidence_threshold",
        # Missing: author, bucket, role, region, service_name, pipeline_version
    )
except ValidationError as e:
    print(e)
    # "Field required: author ..."
```

### Runtime Validation

The script performs additional validation at runtime:

```python
# Runtime checks:
# 1. Prediction file has required prob_class_* columns
# 2. ID field exists in data
# 3. At least 2 probability columns (binary classification minimum)
# 4. Probability columns sum to ~1.0 for each sample
# 5. No NaN values in probability columns
```

## Performance Considerations

### Memory Management

- **Batch Processing**: For large datasets, process in chunks
- **Format Selection**: Use Parquet for datasets >1GB
- **Instance Sizing**: Use `use_large_processing_instance=True` for >10M samples

### Computational Efficiency

- **Confidence Threshold**: O(n) complexity - fastest
- **Top-K Per Class**: O(n log k) complexity - efficient
- **Uncertainty**: O(n) complexity - fast
- **Diversity (k-center)**: O(n²) complexity - slowest, use for small batches
- **BADGE**: O(n²) complexity - slowest, use for small batches

### Cost Optimization

```python
# For small datasets (<1M samples)
config = ActiveSampleSelectionConfig(
    use_large_processing_instance=False,  # Use ml.m5.2xlarge
    processing_instance_count=1,
)

# For large datasets (>10M samples)
config = ActiveSampleSelectionConfig(
    use_large_processing_instance=True,  # Use ml.m5.4xlarge
    processing_instance_count=2,  # Parallel processing
)
```

## Testing Strategy

### Unit Testing

```python
def test_ssl_strategy_validation():
    """Test SSL-specific strategy validation."""
    # Valid SSL strategies
    for strategy in ["confidence_threshold", "top_k_per_class"]:
        config = ActiveSampleSelectionConfig(
            selection_strategy=strategy,
            use_case="ssl",
            # ... other fields
        )
        assert config.selection_strategy == strategy
    
    # Invalid SSL strategies
    for strategy in ["uncertainty", "diversity", "badge"]:
        with pytest.raises(ValueError, match="NOT valid for SSL"):
            ActiveSampleSelectionConfig(
                selection_strategy=strategy,
                use_case="ssl",
                # ... other fields
            )

def test_active_learning_strategy_validation():
    """Test Active Learning-specific strategy validation."""
    # Valid AL strategies
    for strategy in ["uncertainty", "diversity", "badge"]:
        config = ActiveSampleSelectionConfig(
            selection_strategy=strategy,
            use_case="active_learning",
            # ... other fields
        )
        assert config.selection_strategy == strategy
    
    # Invalid AL strategies (warning, not error)
    for strategy in ["confidence_threshold", "top_k_per_class"]:
        with pytest.raises(ValueError, match="NOT recommended for Active Learning"):
            ActiveSampleSelectionConfig(
                selection_strategy=strategy,
                use_case="active_learning",
                # ... other fields
            )
```

### Integration Testing

```python
def test_end_to_end_ssl_pipeline():
    """Test complete SSL pipeline."""
    # 1. Create mock inference predictions
    predictions = create_mock_predictions(n_samples=1000)
    
    # 2. Run selection
    config = ActiveSampleSelectionConfig(
        selection_strategy="confidence_threshold",
        use_case="ssl",
        confidence_threshold=0.9,
    )
    builder = ActiveSampleSelectionStepBuilder(config)
    step = builder.create_step(...)
    
    # 3. Verify output format
    output = load_output(step)
    assert "pseudo_label" in output.columns
    assert "confidence" in output.columns
    assert "data_source" in output.columns
    assert all(output["confidence"] >= 0.9)
```

## Summary

The Active Sampling step provides a production-ready implementation that:

1. **Supports both SSL and Active Learning** workflows with appropriate strategies
2. **Validates strategy-use case compatibility** via Pydantic at config time
3. **Works with any model type** that produces prediction probabilities
4. **Integrates seamlessly** with Cursus pipeline infrastructure
5. **Provides flexible configuration** with sensible defaults
6. **Tracks provenance** of selected samples for quality monitoring
7. **Optimizes performance** based on dataset size and strategy choice
8. **Follows Cursus patterns** for contracts, specs, configs, and builders

This design ensures reliable, maintainable, and optimized sample selection for semi-supervised and active learning pipelines.

---

## Related Documentation

### Active Sampling Implementation Details

**Core Algorithm Designs:**
- [Active Sampling Script Design](./active_sampling_script_design.md) - Main script implementation with uncertainty, diversity, and BADGE strategies
- [Active Sampling BADGE Design](./active_sampling_badge.md) - BADGE (Batch Active learning by Diverse Gradient Embeddings) algorithm for hybrid sampling
- [Active Sampling Uncertainty Design](./active_sampling_uncertainty_margin_entropy.md) - Uncertainty-based strategies (margin, entropy, least confidence)
- [Active Sampling Core-Set Design](./active_sampling_core_set_leaf_core_set.md) - Diversity sampling with core-set and k-center algorithms

### Semi-Supervised Learning Integration

**XGBoost SSL Pipeline:**
- [XGBoost Semi-Supervised Learning Pipeline Design](./xgboost_semi_supervised_learning_pipeline_design.md) - Complete SSL pipeline architecture with pseudo-labeling workflow
- [XGBoost Semi-Supervised Learning Training Design](./xgboost_semi_supervised_learning_training_design.md) - Training step extension with job_type support for SSL pretraining and fine-tuning

### Model Inference Integration

**Inference Step Designs:**
- [XGBoost Model Inference Design](./xgboost_model_inference_design.md) - XGBoost model inference step producing evaluation_data output
- [PyTorch Model Inference Design](./pytorch_model_inference_design.md) - PyTorch model inference step with compatible output format
- [LightGBMMT Model Inference Design](./lightgbmmt_model_inference_design.md) - Multi-task LightGBM inference with per-task predictions

### Core Framework Patterns

**Step Builder Patterns:**
- [Processing Step Builder Patterns](./processing_step_builder_patterns.md) - General processing step builder patterns and best practices
- [Step Builder Patterns Summary](./step_builder_patterns_summary.md) - Comprehensive summary of all step builder patterns
- [Processing Step Alignment Validation Patterns](./processing_step_alignment_validation_patterns.md) - Validation patterns for processing steps

**Configuration & Architecture:**
- [Three-Tier Config Design](../0_developer_guide/three_tier_config_design.md) - Three-tier configuration architecture (Essential, System, Derived)
- [Config Manager Three-Tier Implementation](./config_manager_three_tier_implementation.md) - Implementation details for three-tier configs
- [Step Specification](./step_specification.md) - Step specification format and requirements
- [Script Contract](./script_contract.md) - Contract definitions for pipeline steps

**Dependency Resolution:**
- [Dependency Resolution System](./dependency_resolution_system.md) - Specification-driven dependency resolution architecture
- [Enhanced Property Reference](./enhanced_property_reference.md) - Property reference system for step outputs
- [Semantic Matcher](../core/deps/semantic_matcher.md) - Semantic name matching with aliases support

### Developer Guides

**Step Development:**
- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Complete guide for adding new steps
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Step builder implementation guide
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Script development best practices
- [Script Contract Guide](../0_developer_guide/script_contract.md) - Understanding and implementing script contracts

**Testing & Validation:**
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Validation system usage
- [Script Testability Implementation](../0_developer_guide/script_testability_implementation.md) - Making scripts testable

### Entry Point Documentation

**Index Documents:**
- [Processing Steps Index](../00_entry_points/processing_steps_index.md) - Complete catalog of processing steps
- [Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md) - All step-related documentation
- [XGBoost Pipelines Index](../00_entry_points/xgboost_pipelines_index.md) - XGBoost pipeline variants and integration
- [Cursus Package Overview](../00_entry_points/cursus_package_overview.md) - System architecture overview

### Implementation Examples

**Pipeline Examples:**
- [XGBoost End-to-End Pipeline](../examples/mods_pipeline_xgboost_end_to_end.md) - Complete production workflow example
- [XGBoost Semi-Supervised Learning Example](../examples/xgboost_ssl_example.md) - SSL pipeline with active sampling (if available)
