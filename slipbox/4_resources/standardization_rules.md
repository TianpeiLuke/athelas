---
tags:
  - resource
  - standardization
  - architecture
  - validation
  - rules
keywords:
  - standardization rules
  - naming conventions
  - interface standards
  - documentation standards
  - error handling
  - testing standards
  - script testability
  - validation framework
  - architectural constraints
  - quality gates
topics:
  - standardization framework
  - architectural constraints
  - validation tools
  - development standards
language: python
date of note: 2025-08-12
---

# Standardization Rules

This document outlines the standardization rules that govern the development of pipeline components. These rules serve as enhanced architectural constraints that enforce universal patterns and consistency across all pipeline components.

## Purpose of Standardization Rules

Standardization Rules provide the enhanced constraint enforcement layer that:

1. **Universal Pattern Enforcement** - Ensure consistent patterns across all pipeline components
2. **Quality Gate Implementation** - Establish mandatory quality standards and validation rules
3. **Architectural Constraint Definition** - Define and enforce architectural boundaries and limitations
4. **Consistency Validation** - Provide automated checking of standardization compliance
5. **Evolution Governance** - Control how the system can evolve while maintaining standards

## Key Standardization Rules

### 1. Naming Conventions

All components must follow consistent naming conventions. These conventions are centrally defined and enforced through the `step_names.py` registry, which serves as the single source of truth for step naming across the system.

#### Core Naming Patterns (Based on STEP_NAMES Registry)

The `STEP_NAMES` dictionary defines the canonical relationships between all component names. Here are the actual patterns used:

| Component | Pattern | Registry Examples | Counter-Examples |
|-----------|---------|-------------------|-----------------|
| **Canonical Step Names** | PascalCase (registry keys) | `CradleDataLoading`, `XGBoostTraining`, `PyTorchModel`, `TabularPreprocessing` | `cradle_data_loading`, `xgboost_training`, `PytorchTraining` |
| **Config Classes** | PascalCase + `Config` suffix | `CradleDataLoadConfig`, `XGBoostTrainingConfig`, `PyTorchModelConfig` | `CradleDataLoadingConfiguration`, `XGBoostConfig` |
| **Builder Classes** | PascalCase + `StepBuilder` suffix | `CradleDataLoadingStepBuilder`, `XGBoostTrainingStepBuilder`, `PyTorchModelStepBuilder` | `DataLoadingBuilder`, `XGBoostStepBuilder` |
| **Spec Types** | Same as canonical step name | `CradleDataLoading`, `XGBoostTraining`, `PyTorchModel` | `cradle_data_loading_spec`, `XGBoostTrainingSpec` |
| **SageMaker Step Types** | Step class name minus "Step" suffix | `Processing`, `Training`, `Transform`, `CreateModel`, `MimsModelRegistrationProcessing`, `CradleDataLoading` | `ProcessingStep`, `TrainingStep`, `processing` |
| **Logical Names** | snake_case | `input_data`, `model_artifacts`, `training_data` | `InputData`, `model-artifacts` |

#### Real Examples from STEP_NAMES Registry

Here are actual examples showing the naming relationships:

```python
# From STEP_NAMES registry - these are the authoritative patterns:

"CradleDataLoading": {
    "config_class": "CradleDataLoadConfig",           # Config: Remove "ing" + "Config"
    "builder_step_name": "CradleDataLoadingStepBuilder", # Builder: Keep full name + "StepBuilder"
    "spec_type": "CradleDataLoading",                 # Spec: Same as canonical name
    "sagemaker_step_type": "Processing",              # SageMaker: SDK type name
},

"XGBoostTraining": {
    "config_class": "XGBoostTrainingConfig",          # Config: Full name + "Config"
    "builder_step_name": "XGBoostTrainingStepBuilder", # Builder: Full name + "StepBuilder"
    "spec_type": "XGBoostTraining",                   # Spec: Same as canonical name
    "sagemaker_step_type": "Training",                # SageMaker: SDK type name
},

"PyTorchModel": {
    "config_class": "PyTorchModelConfig",             # Config: Full name + "Config"
    "builder_step_name": "PyTorchModelStepBuilder",   # Builder: Full name + "StepBuilder"
    "spec_type": "PyTorchModel",                      # Spec: Same as canonical name
    "sagemaker_step_type": "CreateModel",             # SageMaker: SDK type name
}
```

#### Config Class Naming Pattern Analysis

From the registry, config class names follow these patterns:

| Canonical Name | Config Class | Pattern |
|----------------|--------------|---------|
| `CradleDataLoading` | `CradleDataLoadConfig` | Remove "ing" suffix, add "Config" |
| `TabularPreprocessing` | `TabularPreprocessingConfig` | Keep full name, add "Config" |
| `XGBoostTraining` | `XGBoostTrainingConfig` | Keep full name, add "Config" |
| `PyTorchModel` | `PyTorchModelConfig` | Keep full name, add "Config" |
| `ModelCalibration` | `ModelCalibrationConfig` | Keep full name, add "Config" |
| `HyperparameterPrep` | `HyperparameterPrepConfig` | Keep full name, add "Config" |

**Rule**: Most config classes use the full canonical name + "Config", except for some "-ing" ending names where the "ing" is dropped.

#### Builder Class Naming Pattern

All builder classes consistently follow: `{CanonicalName}StepBuilder`

| Canonical Name | Builder Class |
|----------------|---------------|
| `CradleDataLoading` | `CradleDataLoadingStepBuilder` |
| `XGBoostTraining` | `XGBoostTrainingStepBuilder` |
| `PyTorchModel` | `PyTorchModelStepBuilder` |
| `TabularPreprocessing` | `TabularPreprocessingStepBuilder` |

**Rule**: Always use the full canonical name + "StepBuilder" suffix.

#### SageMaker Step Type Classification

The registry defines SageMaker step types following the rule: **Step class name minus "Step" suffix**

| SageMaker Type | Step Count | Examples | Derived From |
|----------------|------------|----------|--------------|
| `Processing` | 8 steps | `TabularPreprocessing`, `ModelCalibration`, `Package`, `Payload` | `ProcessingStep` → `Processing` |
| `Training` | 2 steps | `XGBoostTraining`, `PyTorchTraining` | `TrainingStep` → `Training` |
| `CreateModel` | 2 steps | `XGBoostModel`, `PyTorchModel` | `CreateModelStep` → `CreateModel` |
| `Transform` | 1 step | `BatchTransform` | `TransformStep` → `Transform` |
| `Lambda` | 1 step | `HyperparameterPrep` | `LambdaStep` → `Lambda` |
| `MimsModelRegistrationProcessing` | 1 step | `Registration` | `MimsModelRegistrationProcessingStep` → `MimsModelRegistrationProcessing` |
| `CradleDataLoading` | 1 step | `CradleDataLoading` | `CradleDataLoadingStep` → `CradleDataLoading` |
| `Base` | 1 step | `Base` | Special case for base configurations |

**Naming Rule**: Take the actual step class name returned by `create_step()` and remove the "Step" suffix:
- `ProcessingStep` → `Processing`
- `TrainingStep` → `Training`
- `MimsModelRegistrationProcessingStep` → `MimsModelRegistrationProcessing`
- `CradleDataLoadingStep` → `CradleDataLoading`

Additionally, all files must follow consistent naming patterns:

| File Type | Pattern | Examples | Counter-Examples |
|-----------|---------|----------|-----------------|
| Step Builder Files | `builder_xxx_step.py` | `builder_data_loading_step.py`, `builder_xgboost_training_step.py` | `DataLoadingStepBuilder.py`, `xgboost_step_builder.py` |
| Config Files | `config_xxx_step.py` | `config_data_loading_step.py`, `config_xgboost_training_step.py` | `DataLoadingConfig.py`, `xgboost_config.py` |
| Step Specification Files | `xxx_spec.py` | `data_loading_spec.py`, `xgboost_training_spec.py` | `DataLoadingSpecification.py`, `spec_xgboost.py` |
| Script Contract Files | `xxx_contract.py` | `data_loading_contract.py`, `xgboost_training_contract.py` | `DataLoadingContract.py`, `contract_xgboost.py` |

This consistency helps with:
- Auto-discovery of components
- Code navigation
- Understanding component relationships
- Automated validation

**Rule**: All components must follow consistent naming conventions.

**Enforcement**: Automated validation during component registration.

```python
class NamingStandardValidator:
    """Enforce naming conventions across all components"""
    
    STEP_TYPE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')  # PascalCase
    LOGICAL_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')  # snake_case
    CONFIG_CLASS_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*Config$')  # PascalCase + Config suffix
    BUILDER_CLASS_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*StepBuilder$')  # PascalCase + StepBuilder suffix
    
    def validate_step_specification(self, spec: StepSpecification):
        """Validate step specification naming"""
        errors = []
        
        # Validate step type naming
        if not self.STEP_TYPE_PATTERN.match(spec.step_type):
            errors.append(f"Step type '{spec.step_type}' must be PascalCase")
        
        # Validate dependency logical names
        for dep_name in spec.dependencies.keys():
            if not self.LOGICAL_NAME_PATTERN.match(dep_name):
                errors.append(f"Dependency name '{dep_name}' must be snake_case")
        
        # Validate output logical names
        for output_name in spec.outputs.keys():
            if not self.LOGICAL_NAME_PATTERN.match(output_name):
                errors.append(f"Output name '{output_name}' must be snake_case")
        
        return errors
    
    def validate_config_class(self, config_class: Type):
        """Validate configuration class naming"""
        class_name = config_class.__name__
        if not self.CONFIG_CLASS_PATTERN.match(class_name):
            raise StandardizationError(
                f"Config class '{class_name}' must follow pattern: PascalCaseConfig"
            )
    
    def validate_builder_class(self, builder_class: Type):
        """Validate builder class naming"""
        class_name = builder_class.__name__
        if not self.BUILDER_CLASS_PATTERN.match(class_name):
            raise StandardizationError(
                f"Builder class '{class_name}' must follow pattern: PascalCaseStepBuilder"
            )

# Centralized Registry
All naming standards are now enforced through the centralized `step_names.py` registry:

```python
# src/pipeline_registry/step_names.py
STEP_NAMES = {
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",           # For config registry
        "builder_step_name": "PyTorchTrainingStepBuilder", # For builder registry
        "spec_type": "PyTorchTraining",                    # For StepSpecification.step_type
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    # ... other steps
}
```

# Standard naming examples
```python
GOOD_NAMES = {
    "step_types": ["CradleDataLoading", "XGBoostTraining", "PyTorchTraining"],
    "logical_names": ["input_data", "model_artifacts", "processed_features"],
    "config_classes": ["CradleDataLoadConfig", "XGBoostTrainingConfig"],
    "builder_classes": ["CradleDataLoadingStepBuilder", "XGBoostTrainingStepBuilder"]
}

BAD_NAMES = {
    "step_types": ["cradle_data_loading", "xgboost_training", "PytorchTraining"],
    "logical_names": ["InputData", "model-artifacts", "processedFeatures"],
    "config_classes": ["CradleDataLoadingConfig", "XGBoostTrainingConfiguration"],
    "builder_classes": ["DataLoadingBuilder", "XGBoostTrainer"]
}
```

# Job Type Variants
Job type variants follow a consistent naming pattern:
```
{base_step_name}_{job_type}
```

Examples:
- `CradleDataLoading_training`
- `CradleDataLoading_calibration`
- `TabularPreprocessing_training`

### 2. Interface Standardization

All components must implement standardized interfaces:

#### Step Builders

All step builders must:
- Inherit from `StepBuilderBase`
- Use the `@register_builder` decorator to register with the registry (or have their naming follow the standard pattern to be auto-discovered)
- Follow the strict naming convention `XXXStepBuilder` where XXX is the step type
- Implement the required methods:
  - `validate_configuration()`
  - `_get_inputs()`
  - `_get_outputs()`
  - `create_step()`

Example:

```python
from cursus.steps.registry.builder_registry import register_builder

@register_builder() # Step type will be auto-derived from class name (YourStepBuilder -> YourStep)
class YourStepBuilder(StepBuilderBase):
    """Builder for your processing step."""
    
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        super().__init__(
            config=config,
            spec=YOUR_STEP_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config = config
    
    def validate_configuration(self):
        """Validate the configuration."""
        # Validation logic
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor."""
        # Input generation logic
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor."""
        # Output generation logic
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step."""
        # Step creation logic
```

#### Config Classes (Three-Tier Design)

All configuration classes must follow the three-tier field classification design:

1. **Tier 1 (Essential Fields)**:
   - Required inputs explicitly provided by users
   - No default values
   - Subject to validation
   - Public access
   - Example: `region: str = Field(..., description="Region code")`

2. **Tier 2 (System Fields)**:
   - Default values that can be overridden
   - Subject to validation
   - Public access
   - Example: `instance_type: str = Field(default="ml.m5.xlarge", description="Instance type")`

3. **Tier 3 (Derived Fields)**:
   - Private fields with leading underscores
   - Values calculated from other fields
   - Accessed through read-only properties
   - Example:
     ```python
     _pipeline_name: Optional[str] = Field(default=None, exclude=True)
     
     @property
     def pipeline_name(self) -> str:
         """Get derived pipeline name."""
         if self._pipeline_name is None:
             self._pipeline_name = f"{self.service_name}_{self.region}"
         return self._pipeline_name
     ```

All config classes must:
- Inherit from a base config class (e.g., `BasePipelineConfig`, `ProcessingStepConfigBase`)
- Use Pydantic for field declarations and validation
- Override `model_dump()` to include derived properties
- Implement required methods:
  - `get_script_contract()`
  - `get_script_path()` (for processing steps)
  - Additional getters as needed

Example:

```python
class YourStepConfig(BasePipelineConfig):
    """Configuration for your step."""
    
    # Tier 1: Essential fields
    region: str = Field(..., description="AWS region code")
    input_path: str = Field(..., description="Input data path")
    
    # Tier 2: System fields
    instance_type: str = Field(default="ml.m5.xlarge", description="Instance type")
    instance_count: int = Field(default=1, description="Number of instances")
    
    # Tier 3: Derived fields
    _output_path: Optional[str] = Field(default=None, exclude=True)
    
    @property
    def output_path(self) -> str:
        """Get output path based on input path."""
        if self._output_path is None:
            self._output_path = f"{self.input_path}/output"
        return self._output_path
    
    # Include derived fields in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        data["output_path"] = self.output_path
        return data
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from cursus.steps.contracts.your_script_contract import YOUR_SCRIPT_CONTRACT
        return YOUR_SCRIPT_CONTRACT
    
    def get_script_path(self):
        """Return the path to the script."""
        return "your_script.py"
```

**Rule**: All components must implement standardized interfaces.

**Enforcement**: Abstract base classes and interface validation.

```python
class InterfaceStandardValidator:
    """Enforce interface standardization"""
    
    REQUIRED_STEP_BUILDER_METHODS = [
        "get_specification",
        "build_step", 
        "validate_inputs",
        "get_output_reference"
    ]
    
    REQUIRED_CONFIG_METHODS = [
        "validate_configuration",
        "merge_with",
        "to_dict",
        "from_dict"
    ]
    
    def validate_step_builder_interface(self, builder_class: Type[BuilderStepBase]):
        """Validate step builder implements required interface"""
        errors = []
        
        for method_name in self.REQUIRED_STEP_BUILDER_METHODS:
            if not hasattr(builder_class, method_name):
                errors.append(f"Builder {builder_class.__name__} missing required method: {method_name}")
            else:
                method = getattr(builder_class, method_name)
                if not callable(method):
                    errors.append(f"Builder {builder_class.__name__}.{method_name} is not callable")
        
        # Validate get_specification returns StepSpecification
        if hasattr(builder_class, 'get_specification'):
            try:
                spec = builder_class.get_specification()
                if not isinstance(spec, StepSpecification):
                    errors.append(f"Builder {builder_class.__name__}.get_specification() must return StepSpecification")
            except Exception as e:
                errors.append(f"Builder {builder_class.__name__}.get_specification() failed: {e}")
        
        return errors
    
    def validate_config_interface(self, config_class: Type[ConfigBase]):
        """Validate config implements required interface"""
        errors = []
        
        for method_name in self.REQUIRED_CONFIG_METHODS:
            if not hasattr(config_class, method_name):
                errors.append(f"Config {config_class.__name__} missing required method: {method_name}")
        
        # Validate inheritance from ConfigBase
        if not issubclass(config_class, ConfigBase):
            errors.append(f"Config {config_class.__name__} must inherit from ConfigBase")
        
        return errors

# Standard interface enforcement
@dataclass
class StandardizedStepBuilder(BuilderStepBase):
    """Standardized base class enforcing interface compliance"""
    
    @classmethod
    @abstractmethod
    def get_specification(cls) -> StepSpecification:
        """REQUIRED: Return step specification"""
        pass
    
    @abstractmethod
    def build_step(self, inputs: Dict[str, Any]) -> Any:
        """REQUIRED: Build SageMaker step"""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """REQUIRED: Validate inputs against specification"""
        spec = self.get_specification()
        errors = []
        
        # Standard validation logic
        for dep_name, dep_spec in spec.dependencies.items():
            if dep_spec.required and dep_name not in inputs:
                errors.append(f"Required input '{dep_name}' missing")
        
        return errors
    
    def get_output_reference(self, logical_name: str) -> Any:
        """REQUIRED: Get output reference by logical name"""
        spec = self.get_specification()
        if logical_name not in spec.outputs:
            raise ValueError(f"Output '{logical_name}' not found in specification")
        
        output_spec = spec.outputs[logical_name]
        return self._resolve_property_path(output_spec.property_path)
```

### 3. Documentation Standards

**Rule**: All components must have comprehensive, standardized documentation.

**Enforcement**: Documentation validation and auto-generation.

```python
class DocumentationStandardValidator:
    """Enforce documentation standards"""
    
    REQUIRED_DOCSTRING_SECTIONS = [
        "Purpose",
        "Parameters", 
        "Returns",
        "Raises",
        "Example"
    ]
    
    def validate_class_documentation(self, cls: Type):
        """Validate class has required documentation"""
        errors = []
        
        if not cls.__doc__:
            errors.append(f"Class {cls.__name__} missing docstring")
            return errors
        
        docstring = cls.__doc__.strip()
        
        # Check for purpose section
        if "Purpose:" not in docstring and "What is the purpose" not in docstring:
            errors.append(f"Class {cls.__name__} docstring missing Purpose section")
        
        # Check for example section
        if "Example:" not in docstring and "Usage:" not in docstring:
            errors.append(f"Class {cls.__name__} docstring missing Example section")
        
        return errors
    
    def validate_method_documentation(self, method: callable):
        """Validate method has required documentation"""
        errors = []
        
        if not method.__doc__:
            errors.append(f"Method {method.__name__} missing docstring")
            return errors
        
        docstring = method.__doc__.strip()
        
        # Check for parameter documentation
        if "Parameters:" not in docstring and "Args:" not in docstring:
            errors.append(f"Method {method.__name__} missing parameter documentation")
        
        # Check for return documentation
        if "Returns:" not in docstring and "Return:" not in docstring:
            errors.append(f"Method {method.__name__} missing return documentation")
        
        return errors

# Standard documentation template
STANDARD_CLASS_DOCSTRING_TEMPLATE = '''
"""
Purpose: {purpose}

This class {detailed_description}

Key Features:
- {feature_1}
- {feature_2}
- {feature_3}

Integration:
- Works with: {integration_points}
- Depends on: {dependencies}

Example:
    ```python
    {example_code}
    ```

See Also:
    {related_components}
"""
'''

STANDARD_METHOD_DOCSTRING_TEMPLATE = '''
"""
{brief_description}

Parameters:
    {parameter_name} ({parameter_type}): {parameter_description}
    {parameter_name} ({parameter_type}, optional): {parameter_description}

Returns:
    {return_type}: {return_description}

Raises:
    {exception_type}: {exception_description}

Example:
    ```python
    {example_code}
    ```
"""
'''
```

### 4. Error Handling Standards

**Rule**: All components must implement standardized error handling.

**Enforcement**: Error handling validation and standard exception hierarchy.

```python
# Standard exception hierarchy
class PipelineError(Exception):
    """Base exception for all pipeline errors"""
    pass

class ValidationError(PipelineError):
    """Raised when validation fails"""
    pass

class ConnectionError(PipelineError):
    """Raised when step connection fails"""
    pass

class ConfigurationError(PipelineError):
    """Raised when configuration is invalid"""
    pass

class SpecificationError(PipelineError):
    """Raised when specification is invalid"""
    pass

class StandardizationError(PipelineError):
    """Raised when standardization rules are violated"""
    pass

class ErrorHandlingStandardValidator:
    """Enforce error handling standards"""
    
    REQUIRED_ERROR_ATTRIBUTES = ["message", "error_code", "suggestions"]
    
    def validate_error_handling(self, method: callable):
        """Validate method implements standard error handling"""
        errors = []
        
        # Check if method has proper exception documentation
        if method.__doc__:
            docstring = method.__doc__
            if "Raises:" not in docstring:
                errors.append(f"Method {method.__name__} missing exception documentation")
        
        return errors
    
    def validate_exception_class(self, exception_class: Type[Exception]):
        """Validate exception class follows standards"""
        errors = []
        
        # Check inheritance from PipelineError
        if not issubclass(exception_class, PipelineError):
            errors.append(f"Exception {exception_class.__name__} must inherit from PipelineError")
        
        # Check for required attributes
        for attr in self.REQUIRED_ERROR_ATTRIBUTES:
            if not hasattr(exception_class, attr):
                errors.append(f"Exception {exception_class.__name__} missing attribute: {attr}")
        
        return errors

# Standard error handling pattern
class StandardizedError(PipelineError):
    """Standardized error with required attributes"""
    
    def __init__(self, message: str, error_code: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.suggestions = suggestions or []
    
    def _generate_error_code(self) -> str:
        """Generate standard error code"""
        class_name = self.__class__.__name__
        return f"{class_name.upper()}_{hash(self.message) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "suggestions": self.suggestions
        }

# Usage example with standard error handling
def connect_steps_with_standard_errors(source_step, target_step):
    """Connect steps with standardized error handling"""
    try:
        # Validation logic
        if not source_step:
            raise ValidationError(
                message="Source step cannot be None",
                error_code="CONN_001",
                suggestions=["Provide a valid source step instance"]
            )
        
        # Connection logic
        return create_connection(source_step, target_step)
        
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        # Wrap unexpected errors in standard format
        raise ConnectionError(
            message=f"Unexpected error during step connection: {str(e)}",
            error_code="CONN_999",
            suggestions=["Check step compatibility", "Verify step specifications"]
        ) from e
```

### 5. Testing Standards

**Rule**: All components must have comprehensive, standardized tests.

**Enforcement**: Test coverage validation and standard test patterns.

```python
class TestingStandardValidator:
    """Enforce testing standards"""
    
    REQUIRED_TEST_TYPES = [
        "unit_tests",
        "integration_tests", 
        "validation_tests",
        "error_handling_tests"
    ]
    
    MINIMUM_COVERAGE_THRESHOLD = 0.85  # 85% coverage required
    
    def validate_test_coverage(self, component_class: Type):
        """Validate component has required test coverage"""
        errors = []
        
        # Check for test file existence
        test_file_path = self._get_test_file_path(component_class)
        if not os.path.exists(test_file_path):
            errors.append(f"Missing test file for {component_class.__name__}: {test_file_path}")
        
        # Check coverage
        coverage = self._calculate_coverage(component_class)
        if coverage < self.MINIMUM_COVERAGE_THRESHOLD:
            errors.append(
                f"Test coverage for {component_class.__name__} is {coverage:.1%}, "
                f"minimum required: {self.MINIMUM_COVERAGE_THRESHOLD:.1%}"
            )
        
        return errors
    
    def validate_test_structure(self, test_class: Type):
        """Validate test class follows standard structure"""
        errors = []
        
        # Check for required test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        required_patterns = [
            r'test_.*_success',      # Success case tests
            r'test_.*_validation',   # Validation tests
            r'test_.*_error',        # Error handling tests
        ]
        
        for pattern in required_patterns:
            if not any(re.match(pattern, method) for method in test_methods):
                errors.append(f"Test class {test_class.__name__} missing tests matching pattern: {pattern}")
        
        return errors

# Standard test base class
class StandardizedTestCase(unittest.TestCase):
    """Base class for standardized tests"""
    
    def setUp(self):
        """Standard test setup"""
        self.test_config = self._create_test_config()
        self.mock_dependencies = self._create_mock_dependencies()
    
    def tearDown(self):
        """Standard test cleanup"""
        self._cleanup_test_resources()
    
    def assert_validation_error(self, callable_obj, *args, **kwargs):
        """Standard assertion for validation errors"""
        with self.assertRaises(ValidationError) as context:
            callable_obj(*args, **kwargs)
        
        # Validate error has required attributes
        error = context.exception
        self.assertIsNotNone(error.message)
        self.assertIsNotNone(error.error_code)
        self.assertIsInstance(error.suggestions, list)
    
    def assert_specification_compliance(self, component, specification):
        """Standard assertion for specification compliance"""
        # Validate component implements specification
        errors = specification.validate_implementation(component)
        self.assertEqual([], errors, f"Specification compliance errors: {errors}")

# Standard test patterns
class TestXGBoostTrainingStepBuilder(StandardizedTestCase):
    """Example of standardized test structure"""
    
    def test_build_step_success(self):
        """Test successful step building"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        inputs = {"input_data": "s3://bucket/data"}
        
        step = builder.build_step(inputs)
        
        self.assertIsNotNone(step)
        self.assertEqual(step.name, self.test_config.step_name)
    
    def test_build_step_validation_missing_input(self):
        """Test validation error for missing required input"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        inputs = {}  # Missing required input
        
        self.assert_validation_error(builder.build_step, inputs)
    
    def test_build_step_error_invalid_config(self):
        """Test error handling for invalid configuration"""
        invalid_config = XGBoostTrainingStepConfig(instance_type="invalid")
        builder = XGBoostTrainingStepBuilder(invalid_config)
        
        with self.assertRaises(ConfigurationError):
            builder.build_step({"input_data": "s3://bucket/data"})
    
    def test_specification_compliance(self):
        """Test builder complies with specification"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        specification = builder.get_specification()
        
        self.assert_specification_compliance(builder, specification)
```

### 6. Script Testability Standards

**Rule**: All processing scripts must implement testable patterns for efficient development and testing.

**Enforcement**: Script testability validation and refactoring checklist compliance.

```python
class ScriptTestabilityStandardValidator:
    """Enforce script testability standards"""
    
    REQUIRED_TESTABILITY_PATTERNS = [
        "parameterized_main_function",
        "environment_collection_entry_point", 
        "helper_function_parameterization",
        "container_path_handling",
        "unit_testing_standards",
        "error_handling_with_markers",
        "script_contract_integration",
        "hybrid_execution_mode"
    ]
    
    SCRIPT_REFACTORING_CHECKLIST = [
        "main_function_parameterized",
        "environment_collection_implemented",
        "helper_functions_parameterized", 
        "container_paths_defined",
        "unit_tests_comprehensive",
        "error_handling_robust",
        "contract_integration_aligned",
        "success_failure_markers",
        "hybrid_mode_supported",
        "path_constants_defined",
        "argument_parsing_standardized",
        "logging_implemented"
    ]
    
    def validate_script_testability(self, script_path: str, contract_path: str = None):
        """Validate script implements testability patterns"""
        errors = []
        
        # Read script content
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Check for parameterized main function
        if not self._has_parameterized_main(script_content):
            errors.append(f"Script {script_path} missing parameterized main function")
        
        # Check for environment collection entry point
        if not self._has_environment_collection_entry(script_content):
            errors.append(f"Script {script_path} missing environment collection entry point")
        
        # Check for helper function parameterization
        if not self._has_parameterized_helpers(script_content):
            errors.append(f"Script {script_path} helpers not properly parameterized")
        
        # Check for container path handling
        if not self._has_container_path_handling(script_content):
            errors.append(f"Script {script_path} missing container path handling")
        
        # Check for error handling with markers
        if not self._has_error_handling_markers(script_content):
            errors.append(f"Script {script_path} missing success/failure markers")
        
        # Validate contract integration if contract provided
        if contract_path:
            contract_errors = self._validate_contract_integration(script_path, contract_path)
            errors.extend(contract_errors)
        
        return errors
    
    def _has_parameterized_main(self, script_content: str) -> bool:
        """Check if script has parameterized main function"""
        # Look for main function with required parameters
        main_pattern = r'def\s+main\s*\(\s*input_paths\s*,\s*output_paths\s*,\s*environ_vars\s*,\s*job_args\s*\)'
        return bool(re.search(main_pattern, script_content))
    
    def _has_environment_collection_entry(self, script_content: str) -> bool:
        """Check if script has environment collection entry point"""
        # Look for if __name__ == "__main__" with environment collection
        entry_pattern = r'if\s+__name__\s*==\s*["\']__main__["\']:'
        env_collection_pattern = r'collect_environment_variables|get_environment_config'
        return (bool(re.search(entry_pattern, script_content)) and 
                bool(re.search(env_collection_pattern, script_content)))
    
    def _has_parameterized_helpers(self, script_content: str) -> bool:
        """Check if helper functions accept parameters instead of accessing environment directly"""
        # Look for functions that don't use os.environ directly
        helper_functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', script_content)
        
        # Check that helper functions (excluding main and entry point) don't access os.environ
        for func_name in helper_functions:
            if func_name in ['main', 'collect_environment_variables', 'get_environment_config']:
                continue
            
            # Extract function body
            func_pattern = rf'def\s+{func_name}\s*\([^)]*\):.*?(?=def\s+\w+|$)'
            func_match = re.search(func_pattern, script_content, re.DOTALL)
            if func_match:
                func_body = func_match.group(0)
                if 'os.environ' in func_body:
                    return False
        
        return True
    
    def _has_container_path_handling(self, script_content: str) -> bool:
        """Check if script defines container path constants"""
        container_path_patterns = [
            r'CONTAINER_INPUT_PATH\s*=',
            r'CONTAINER_OUTPUT_PATH\s*=',
            r'LOCAL_INPUT_PATH\s*=',
            r'LOCAL_OUTPUT_PATH\s*='
        ]
        
        return any(bool(re.search(pattern, script_content)) for pattern in container_path_patterns)
    
    def _has_error_handling_markers(self, script_content: str) -> bool:
        """Check if script has success/failure markers"""
        marker_patterns = [
            r'SUCCESS_MARKER\s*=',
            r'FAILURE_MARKER\s*=',
            r'write_success_marker|create_success_marker',
            r'write_failure_marker|create_failure_marker'
        ]
        
        return any(bool(re.search(pattern, script_content)) for pattern in marker_patterns)
    
    def _validate_contract_integration(self, script_path: str, contract_path: str) -> List[str]:
        """Validate script integration with contract"""
        errors = []
        
        # Read contract to get expected paths and environment variables
        with open(contract_path, 'r') as f:
            contract_content = f.read()
        
        # Extract contract paths and environment variables
        contract_paths = self._extract_contract_paths(contract_content)
        contract_env_vars = self._extract_contract_env_vars(contract_content)
        
        # Read script content
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Validate script uses contract paths
        for path_name in contract_paths:
            if path_name not in script_content:
                errors.append(f"Script {script_path} doesn't use contract path: {path_name}")
        
        # Validate script handles contract environment variables
        for env_var in contract_env_vars:
            if env_var not in script_content:
                errors.append(f"Script {script_path} doesn't handle contract env var: {env_var}")
        
        return errors

# Standard testable script pattern
TESTABLE_SCRIPT_TEMPLATE = '''
"""
Testable script following standardization patterns.

This script implements the 12-point testability checklist:
1. Parameterized main function
2. Environment collection entry point
3. Helper function parameterization
4. Container path constants
5. Unit testing structure
6. Error handling with markers
7. Contract integration
8. Success/failure markers
9. Hybrid execution mode
10. Path constants definition
11. Argument parsing standardization
12. Comprehensive logging
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Container path constants for hybrid execution
CONTAINER_INPUT_PATH = "/opt/ml/processing/input"
CONTAINER_OUTPUT_PATH = "/opt/ml/processing/output"
LOCAL_INPUT_PATH = "./local_input"
LOCAL_OUTPUT_PATH = "./local_output"

# Success/failure markers
SUCCESS_MARKER = "SUCCESS"
FAILURE_MARKER = "FAILURE"

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup standardized logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_execution_paths(is_container: bool = None) -> Dict[str, str]:
    """Get input/output paths based on execution environment"""
    if is_container is None:
        # Auto-detect container environment
        is_container = os.path.exists(CONTAINER_INPUT_PATH)
    
    if is_container:
        return {
            "input_path": CONTAINER_INPUT_PATH,
            "output_path": CONTAINER_OUTPUT_PATH
        }
    else:
        return {
            "input_path": LOCAL_INPUT_PATH,
            "output_path": LOCAL_OUTPUT_PATH
        }

def create_success_marker(output_path: str, message: str = "Processing completed successfully"):
    """Create success marker file"""
    marker_path = Path(output_path) / f"{SUCCESS_MARKER}.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, 'w') as f:
        f.write(f"{message}\\n")

def create_failure_marker(output_path: str, error_message: str):
    """Create failure marker file"""
    marker_path = Path(output_path) / f"{FAILURE_MARKER}.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, 'w') as f:
        f.write(f"Error: {error_message}\\n")

def process_data(input_path: str, output_path: str, config: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Parameterized helper function for data processing.
    
    Parameters:
        input_path: Path to input data
        output_path: Path for output data
        config: Processing configuration
        logger: Logger instance
    
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        logger.info(f"Processing data from {input_path} to {output_path}")
        
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Processing logic here
        # ... implementation ...
        
        logger.info("Data processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return False

def validate_inputs(input_paths: Dict[str, str], logger: logging.Logger) -> bool:
    """
    Validate input paths and data.
    
    Parameters:
        input_paths: Dictionary of input paths
        logger: Logger instance
    
    Returns:
        bool: True if validation successful, False otherwise
    """
    try:
        for name, path in input_paths.items():
            if not os.path.exists(path):
                logger.error(f"Input path does not exist: {name} = {path}")
                return False
            logger.info(f"Validated input: {name} = {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        return False

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: Dict[str, Any]) -> bool:
    """
    Parameterized main function following testability standards.
    
    Parameters:
        input_paths: Dictionary of input paths from contract
        output_paths: Dictionary of output paths from contract  
        environ_vars: Dictionary of environment variables from contract
        job_args: Dictionary of job arguments
    
    Returns:
        bool: True if processing successful, False otherwise
    """
    # Setup logging
    log_level = environ_vars.get('LOG_LEVEL', 'INFO')
    logger = setup_logging(log_level)
    
    try:
        logger.info("Starting script execution")
        logger.info(f"Input paths: {input_paths}")
        logger.info(f"Output paths: {output_paths}")
        
        # Validate inputs
        if not validate_inputs(input_paths, logger):
            create_failure_marker(output_paths.get('output_data', './'), "Input validation failed")
            return False
        
        # Process data using parameterized helper
        processing_config = {
            'param1': environ_vars.get('PARAM1', 'default_value'),
            'param2': environ_vars.get('PARAM2', 'default_value'),
            **job_args
        }
        
        success = process_data(
            input_path=input_paths.get('input_data'),
            output_path=output_paths.get('output_data'),
            config=processing_config,
            logger=logger
        )
        
        if success:
            create_success_marker(output_paths.get('output_data', './'))
            logger.info("Script execution completed successfully")
            return True
        else:
            create_failure_marker(output_paths.get('output_data', './'), "Data processing failed")
            return False
            
    except Exception as e:
        error_msg = f"Script execution failed: {str(e)}"
        logger.error(error_msg)
        create_failure_marker(output_paths.get('output_data', './'), error_msg)
        return False

def collect_environment_variables() -> Dict[str, str]:
    """
    Environment collection entry point.
    
    Collects environment variables defined in the script contract.
    """
    return {
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        'PARAM1': os.environ.get('PARAM1', ''),
        'PARAM2': os.environ.get('PARAM2', ''),
        # Add other contract-defined environment variables
    }

def parse_job_arguments() -> Dict[str, Any]:
    """Parse job arguments from command line or environment"""
    # Standard argument parsing logic
    return {
        'job_id': os.environ.get('JOB_ID', 'default_job'),
        'batch_size': int(os.environ.get('BATCH_SIZE', '1000')),
        # Add other job-specific arguments
    }

if __name__ == "__main__":
    """Environment collection entry point for container execution"""
    
    # Collect environment variables from contract
    environ_vars = collect_environment_variables()
    
    # Parse job arguments
    job_args = parse_job_arguments()
    
    # Get execution paths (auto-detect container vs local)
    paths = get_execution_paths()
    
    # Define input/output paths based on contract
    input_paths = {
        'input_data': os.path.join(paths['input_path'], 'data')
    }
    
    output_paths = {
        'output_data': os.path.join(paths['output_path'], 'processed_data')
    }
    
    # Execute main function
    success = main(input_paths, output_paths, environ_vars, job_args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
'''

# Unit testing standards for testable scripts
TESTABLE_SCRIPT_TEST_TEMPLATE = '''
"""
Unit tests for testable script following standardization patterns.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Import the script module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts import testable_script

class TestTestableScript(unittest.TestCase):
    """Comprehensive unit tests for testable script"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        # Create test directories
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test input data
        self.test_input_file = os.path.join(self.input_dir, 'test_data.txt')
        with open(self.test_input_file, 'w') as f:
            f.write('test data content')
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_main_function_success(self):
        """Test main function with valid inputs"""
        input_paths = {'input_data': self.input_dir}
        output_paths = {'output_data': self.output_dir}
        environ_vars = {'LOG_LEVEL': 'INFO', 'PARAM1': 'test_value'}
        job_args = {'job_id': 'test_job'}
        
        result = testable_script.main(input_paths, output_paths, environ_vars, job_args)
        
        self.assertTrue(result)
        # Verify success marker created
        success_marker = Path(self.output_dir) / 'SUCCESS.txt'
        self.assertTrue(success_marker.exists())
    
    def test_main_function_validation_failure(self):
        """Test main function with invalid inputs"""
        input_paths = {'input_data': '/nonexistent/path'}
        output_paths = {'output_data': self.output_dir}
        environ_vars = {'LOG_LEVEL': 'INFO'}
        job_args = {}
        
        result = testable_script.main(input_paths, output_paths, environ_vars, job_args)
        
        self.assertFalse(result)
        # Verify failure marker created
        failure_marker = Path(self.output_dir) / 'FAILURE.txt'
        self.assertTrue(failure_marker.exists())
    
    def test_process_data_parameterized(self):
        """Test process_data helper function with parameters"""
        config = {'param1': 'test_value', 'param2': 'another_value'}
        logger = MagicMock()
        
        result = testable_script.process_data(
            input_path=self.input_dir,
            output_path=self.output_dir,
            config=config,
            logger=logger
        )
        
        self.assertTrue(result)
        logger.info.assert_called()
    
    def test_validate_inputs_success(self):
        """Test input validation with valid paths"""
        input_paths = {'input_data': self.input_dir}
        logger = MagicMock()
        
        result = testable_script.validate_inputs(input_paths, logger)
        
        self.assertTrue(result)
        logger.info.assert_called()
    
    def test_validate_inputs_failure(self):
        """Test input validation with invalid paths"""
        input_paths = {'input_data': '/nonexistent/path'}
        logger = MagicMock()
        
        result = testable_script.validate_inputs(input_paths, logger)
        
        self.assertFalse(result)
        logger.error.assert_called()
    
    def test_get_execution_paths_container(self):
        """Test execution path detection for container environment"""
        paths = testable_script.get_execution_paths(is_container=True)
        
        self.assertEqual(paths['input_path'], testable_script.CONTAINER_INPUT_PATH)
        self.assertEqual(paths['output_path'], testable_script.CONTAINER_OUTPUT_PATH)
    
    def test_get_execution_paths_local(self):
        """Test execution path detection for local environment"""
        paths = testable_script.get_execution_paths(is_container=False)
        
        self.assertEqual(paths['input_path'], testable_script.LOCAL_INPUT_PATH)
        self.assertEqual(paths['output_path'], testable_script.LOCAL_OUTPUT_PATH)
    
    def test_success_marker_creation(self):
        """Test success marker file creation"""
        testable_script.create_success_marker(self.output_dir, "Test success")
        
        marker_path = Path(self.output_dir) / 'SUCCESS.txt'
        self.assertTrue(marker_path.exists())
        
        with open(marker_path, 'r') as f:
            content = f.read()
            self.assertIn("Test success", content)
    
    def test_failure_marker_creation(self):
        """Test failure marker file creation"""
        testable_script.create_failure_marker(self.output_dir, "Test error")
        
        marker_path = Path(self.output_dir) / 'FAILURE.txt'
        self.assertTrue(marker_path.exists())
        
        with open(marker_path, 'r') as f:
            content = f.read()
            self.assertIn("Test error", content)
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG', 'PARAM1': 'env_value'})
    def test_collect_environment_variables(self):
        """Test environment variable collection"""
        env_vars = testable_script.collect_environment_variables()
        
        self.assertEqual(env_vars['LOG_LEVEL'], 'DEBUG')
        self.assertEqual(env_vars['PARAM1'], 'env_value')
    
    @patch.dict(os.environ, {'JOB_ID': 'test_job_123', 'BATCH_SIZE': '2000'})
    def test_parse_job_arguments(self):
        """Test job argument parsing"""
        job_args = testable_script.parse_job_arguments()
        
        self.assertEqual(job_args['job_id'], 'test_job_123')
        self.assertEqual(job_args['batch_size'], 2000)

if __name__ == '__main__':
    unittest.main()
'''
```

### 7. SageMaker Step Type Classification Standards

All step builders must be properly classified according to their actual SageMaker step type. This classification is mandatory for the Universal Builder Test framework and step-type-specific validation.

#### Step Registry Requirements

All steps must be registered in `src/cursus/registry/step_names.py` with the correct `sagemaker_step_type` field:

```python
STEP_NAMES = {
    "YourNewStep": {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder", 
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",  # MANDATORY: Must match create_step() return type
        "description": "Description of your new step"
    },
}
```

#### Valid SageMaker Step Types

The `sagemaker_step_type` field follows the rule: **Step class name minus "Step" suffix**

| SageMaker Step Type | When to Use | create_step() Return Type | Examples |
|-------------------|-------------|---------------------------|----------|
| `Processing` | Steps that create ProcessingStep instances | `ProcessingStep` | TabularPreprocessing, ModelCalibration, Package, Payload |
| `Training` | Steps that create TrainingStep instances | `TrainingStep` | XGBoostTraining, PyTorchTraining |
| `Transform` | Steps that create TransformStep instances | `TransformStep` | BatchTransform |
| `CreateModel` | Steps that create CreateModelStep instances | `CreateModelStep` | XGBoostModel, PyTorchModel |
| `Lambda` | Steps that create LambdaStep instances | `LambdaStep` | HyperparameterPrep |
| `MimsModelRegistrationProcessing` | Steps that create MimsModelRegistrationProcessingStep | `MimsModelRegistrationProcessingStep` | Registration |
| `CradleDataLoading` | Steps that create CradleDataLoadingStep | `CradleDataLoadingStep` | CradleDataLoading |
| `Base` | Base/utility steps | N/A | Base configuration steps |

**Naming Rule Examples**:
- `ProcessingStep` → `Processing`
- `TrainingStep` → `Training`
- `MimsModelRegistrationProcessingStep` → `MimsModelRegistrationProcessing`
- `CradleDataLoadingStep` → `CradleDataLoading`
- `LambdaStep` → `Lambda`

#### Verification Requirements

**CRITICAL**: The `sagemaker_step_type` field must be verified against the actual implementation:

1. **Source Code Analysis**: Examine the `create_step()` method in your step builder
2. **Return Type Verification**: Ensure the return type annotation matches the classification
3. **Implementation Verification**: Confirm the actual step creation logic matches the classification

Example verification process:

```python
# In your step builder
def create_step(self, **kwargs) -> ProcessingStep:  # Return type must match registry
    """Create the processing step."""
    # Implementation must create ProcessingStep
    return ProcessingStep(...)  # Must match both return type and registry classification
```

#### Step-Type-Specific Validation

Each SageMaker step type has specific validation requirements enforced by the Universal Builder Test framework:

**Processing Steps**:
- Must define ProcessingInputs and ProcessingOutputs correctly
- Must use proper container paths and S3 URIs
- Must handle environment variables appropriately

**Training Steps**:
- Must define training inputs (training data, validation data)
- Must specify model output location
- Must configure hyperparameters correctly

**Transform Steps**:
- Must define transform input and output
- Must specify model name or model data
- Must configure instance types appropriately

**CreateModel Steps**:
- Must reference model artifacts from training
- Must define inference code and dependencies
- Must specify model name and role

**RegisterModel Steps**:
- Must handle model registration with external systems
- Must validate model artifacts and metadata
- Must follow custom registration protocols

#### Universal Builder Test Integration

All step builders are automatically tested using the Universal Builder Test framework, which provides:

1. **Step-Type-Specific Validation**: Tests tailored to each SageMaker step type
2. **Interface Compliance Testing**: Validates standard interfaces and methods
3. **Specification Alignment Testing**: Ensures specs and contracts are aligned
4. **Path Mapping Testing**: Validates input/output path configurations
5. **Integration Testing**: Tests step interactions and dependencies

Example test execution:

```python
from src.cursus.validation.builders.universal_test import UniversalBuilderTester

# Test your step builder with step-type-specific validation
tester = UniversalBuilderTester(YourStepBuilder, config)
results = tester.run_all_tests()

# Results include step-type-specific validation
print(f"SageMaker Step Type: {results.sagemaker_step_type}")
print(f"Step-Type-Specific Tests: {results.sagemaker_validation_results}")
```

## Validation Tools

We provide comprehensive tools to validate compliance with these standardization rules:

### Interface Standard Validation

The `InterfaceStandardValidator` provides comprehensive validation for step builder interface compliance according to standardization rules.

**Implementation Location**: `src/cursus/validation/interface/interface_standard_validator.py`

```python
# Example interface validator usage
from cursus.validation.interface.interface_standard_validator import InterfaceStandardValidator

validator = InterfaceStandardValidator()

# Validate complete step builder interface
violations = validator.validate_step_builder_interface(YourStepBuilder)

if violations:
    print("Interface compliance violations:")
    for violation in violations:
        print(f"  - {violation.violation_type}: {violation.message}")
        if violation.suggestions:
            print(f"    Suggestions: {', '.join(violation.suggestions)}")
else:
    print("✅ Step builder passes all interface compliance checks")

# Individual validation methods available:
# - validate_inheritance_compliance(step_builder_class)
# - validate_required_methods(step_builder_class)
# - validate_method_signatures(step_builder_class)
# - validate_method_documentation(step_builder_class)
# - validate_class_documentation(step_builder_class)
# - validate_builder_registry_compliance(step_builder_class)
```

**Validation Categories**:

1. **Inheritance Compliance**
   - Validates inheritance from `StepBuilderBase`
   - Checks method resolution order (MRO)

2. **Required Methods Validation**
   - Ensures all required methods are implemented: `validate_configuration`, `_get_inputs`, `_get_outputs`, `create_step`
   - Validates method callability

3. **Method Signature Validation**
   - Validates parameter signatures for required methods
   - Checks for required parameters like `inputs` for `_get_inputs`, `**kwargs` for `create_step`
   - Validates return type annotations

4. **Documentation Validation**
   - **Method Documentation**: Validates docstring presence and quality for required methods
   - **Class Documentation**: Validates class-level documentation including purpose and examples
   - Checks for missing return documentation when methods have return types

5. **Registry Compliance**
   - Validates naming conventions (classes should end with "StepBuilder")
   - Ensures compatibility with step builder registry requirements

**Test Coverage**: The interface validator is thoroughly tested with 24 comprehensive tests split across multiple test files:
- `test/validation/interface/test_interface_violation.py` - Tests violation data structure (4 tests)
- `test/validation/interface/test_validator_core.py` - Tests core validator functionality (17 tests)
- `test/validation/interface/test_validator_integration.py` - Integration tests (3 tests)

### Naming Convention Validation

The `NamingStandardValidator` provides comprehensive validation for naming conventions as defined in the standardization rules document.

```python
# Example validator usage
from cursus.validation.naming import NamingStandardValidator

validator = NamingStandardValidator()

# Validate step specification naming
errors = validator.validate_step_specification(YOUR_STEP_SPEC)
if errors:
    print("Naming convention violations:")
    for error in errors:
        print(f"  - {error}")

# Validate step builder class naming
errors = validator.validate_step_builder_class(YourStepBuilder)
if errors:
    print("Builder naming violations:")
    for error in errors:
        print(f"  - {error}")

# Validate config class naming
errors = validator.validate_config_class(YourConfigClass)
if errors:
    print("Config naming violations:")
    for error in errors:
        print(f"  - {error}")

# Validate file naming patterns
errors = validator.validate_file_naming("builder_your_step.py", "builder")
if errors:
    print("File naming violations:")
    for error in errors:
        print(f"  - {error}")

# Validate all registry entries
errors = validator.validate_all_registry_entries()
if errors:
    print("Registry naming violations:")
    for error in errors:
        print(f"  - {error}")
```

### Universal Builder Test Framework

The `UniversalStepBuilderTest` provides comprehensive validation across all architectural levels, including interface compliance, specification alignment, path mapping, and integration testing.

```python
# Example comprehensive builder testing
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

# Test a specific builder with comprehensive validation
tester = UniversalStepBuilderTest(YourStepBuilder)
results = tester.run_all_tests()

# Check results
total_tests = len(results)
passed_tests = sum(1 for result in results.values() if result["passed"])
pass_rate = (passed_tests / total_tests) * 100

print(f"Builder validation: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")

# Check for failed tests
failed_tests = {k: v for k, v in results.items() if not v["passed"]}
if failed_tests:
    print("Failed tests:")
    for test_name, result in failed_tests.items():
        print(f"  ❌ {test_name}: {result['error']}")
```

### SageMaker Step Type Validation

The `SageMakerStepTypeValidator` provides specialized validation for SageMaker step type compliance and step-type-specific requirements.

```python
# Example SageMaker step type validation
from cursus.validation.builders.sagemaker_step_type_validator import SageMakerStepTypeValidator

validator = SageMakerStepTypeValidator(YourStepBuilder)

# Get step type information
step_type_info = validator.get_step_type_info()
print(f"Detected step type: {step_type_info['sagemaker_step_type']}")
print(f"Step name: {step_type_info['detected_step_name']}")

# Validate step type compliance
violations = validator.validate_step_type_compliance()
if violations:
    print("Step type violations:")
    for violation in violations:
        print(f"  {violation.level.name}: {violation.message}")
```

### Builder Registry Validation

```python
# Example registry validator usage
from cursus.steps.registry.builder_registry import get_global_registry

registry = get_global_registry()
validation = registry.validate_registry()

# Check validation results
print(f"Valid entries: {len(validation['valid'])}")
if validation['invalid']:
    print("Invalid entries:")
    for entry in validation['invalid']:
        print(f"  - {entry}")

if validation['missing']:
    print("Missing entries:")
    for entry in validation['missing']:
        print(f"  - {entry}")

# Get registry statistics
stats = registry.get_registry_stats()
print(f"Registry stats: {stats}")
```

### Command-Line Interface Validation

For convenient validation during development, use the CLI validation tools:

```bash
# Validate all registry entries
python -m cursus.cli.validation_cli validate-registry --verbose

# Validate specific file naming
python -m cursus.cli.validation_cli validate-file-name "builder_your_step.py" "builder" --verbose

# Validate step names
python -m cursus.cli.validation_cli validate-step-name "YourStepName" --verbose

# Validate logical names
python -m cursus.cli.validation_cli validate-logical-name "your_logical_name" --verbose
```

### Job Type Handling

When working with step types that need to handle different job types (e.g., training, calibration), follow these patterns:

1. **Node Naming**: Use underscore suffix for job type variants:
   ```
   CradleDataLoading_training
   CradleDataLoading_calibration
   TabularPreprocessing_training
   ```

2. **Configuration Classes**: Job type should be a field in the config:
   ```python
   class CradleDataLoadConfig(BasePipelineConfig):
       job_type: str = Field(default="training", description="Job type (training, calibration)")
   ```

3. **Builder Resolution**: The builder registry will automatically resolve job types:
   ```python
   # This will resolve to CradleDataLoadingStepBuilder even though node name has _training suffix
   builder = registry.get_builder_for_config(config, node_name="CradleDataLoading_training") 
   ```

## Integration with Development Process

These standardization rules should be integrated into your development process:

1. **Initial Development**: Use as a reference when creating new components
2. **Pre-Commit Validation**: Run validation tools before committing code
3. **Code Review**: Include rule compliance in code review checklist
4. **Continuous Integration**: Add rule validation to CI/CD pipelines
5. **Documentation**: Include rule compliance in your documentation

By following these standardization rules, you'll contribute to a cohesive, maintainable pipeline architecture that is easier to understand, extend, and troubleshoot.

## See Also

- [Design Principles](design_principles.md)
- [Best Practices](best_practices.md)
- [Alignment Rules](alignment_rules.md)
- [Validation Checklist](validation_checklist.md)

## Standardization Enforcement

### 1. Automated Validation Pipeline

```python
class StandardizationEnforcer:
    """Automated enforcement of standardization rules"""
    
    def __init__(self):
        self.validators = [
            NamingStandardValidator(),
            InterfaceStandardValidator(),
            DocumentationStandardValidator(),
            ErrorHandlingStandardValidator(),
            TestingStandardValidator()
        ]
    
    def validate_component(self, component_class: Type) -> ValidationResult:
        """Validate component against all standardization rules"""
        all_errors = []
        all_warnings = []
        
        for validator in self.validators:
            try:
                errors = validator.validate(component_class)
                all_errors.extend(errors)
            except Exception as e:
                all_warnings.append(f"Validator {validator.__class__.__name__} failed: {e}")
        
        return ValidationResult(
            component=component_class.__name__,
            errors=all_errors,
            warnings=all_warnings,
            is_compliant=len(all_errors) == 0
        )
    
    def enforce_standards_on_registration(self, registry: ComponentRegistry):
        """Enforce standards when components are registered"""
        original_register = registry.register_builder
        
        def validated_register(step_type: str, builder_class: Type[BuilderStepBase]):
            # Validate before registration
            result = self.validate_component(builder_class)
            if not result.is_compliant:
                raise StandardizationError(
                    f"Component {builder_class.__name__} fails standardization: {result.errors}"
                )
            
            # Proceed with registration
            return original_register(step_type, builder_class)
        
        registry.register_builder = validated_register

# Integration with CI/CD pipeline
class ContinuousStandardizationValidator:
    """Validate standardization in CI/CD pipeline"""
    
    def validate_pull_request(self, changed_files: List[str]) -> bool:
        """Validate changed components meet standards"""
        enforcer = StandardizationEnforcer()
        
        for file_path in changed_files:
            if self._is_component_file(file_path):
                component_classes = self._extract_component_classes(file_path)
                
                for component_class in component_classes:
                    result = enforcer.validate_component(component_class)
                    if not result.is_compliant:
                        print(f"❌ {component_class.__name__} fails standardization:")
                        for error in result.errors:
                            print(f"  - {error}")
                        return False
        
        print("✅ All components meet standardization requirements")
        return True
```

## Strategic Value

Standardization Rules provide:

1. **System-Wide Consistency**: Ensure uniform patterns across all components
2. **Quality Assurance**: Enforce mandatory quality standards automatically
3. **Maintainability**: Reduce cognitive load through consistent interfaces
4. **Onboarding Efficiency**: New developers can quickly understand standardized patterns
5. **Evolution Control**: Govern how the system can evolve while maintaining quality
6. **Automated Compliance**: Reduce manual review burden through automated validation

## Example Usage

```python
# Component development with standardization enforcement
@standardized_component
class DataLoadingStepBuilder(StandardizedStepBuilder):
    """
    Purpose: Build SageMaker processing steps for data loading operations.
    
    This builder creates ProcessingStep instances configured for data loading
    from various sources (S3, databases, APIs) with standardized output formats.
    
    Key Features:
    - Supports multiple data source types
    - Automatic schema validation
    - Standardized output formatting
    
    Integration:
    - Works with: PreprocessingStepBuilder, ValidationStepBuilder
    - Depends on: DataLoadingStepConfig, ProcessingStepFactory
    
    Example:
        ```python
        config = DataLoadingStepConfig(
            data_source="s3://bucket/data/",
            output_format="parquet"
        )
        builder = DataLoadingStepBuilder(config)
        step = builder.build_step({})
        ```
    
    See Also:
        PreprocessingStepBuilder, DataLoadingStepConfig
    """
    
    @classmethod
    def get_specification(cls) -> StepSpecification:
        """Return the step specification for data loading."""
        return DATA_LOADING_SPEC
    
    def build_step(self, inputs: Dict[str, Any]) -> ProcessingStep:
        """
        Build a SageMaker ProcessingStep for data loading.
        
        Parameters:
            inputs (Dict[str, Any]): Input parameters (typically empty for SOURCE steps)
        
        Returns:
            ProcessingStep: Configured SageMaker processing step
        
        Raises:
            ValidationError: If inputs don't meet specification requirements
            ConfigurationError: If configuration is invalid
        
        Example:
            ```python
            step = builder.build_step({})
            ```
        """
        # Standard validation
        validation_errors = self.validate_inputs(inputs)
        if validation_errors:
            raise ValidationError(
                message=f"Input validation failed: {validation_errors}",
                error_code="DATA_LOAD_001",
                suggestions=["Check input requirements in specification"]
            )
        
        # Build step with standard error handling
        try:
            return self._create_processing_step(inputs)
        except Exception as e:
            raise ConfigurationError(
                message=f"Failed to create data loading step: {str(e)}",
                error_code="DATA_LOAD_002",
                suggestions=["Verify configuration parameters", "Check AWS permissions"]
            ) from e

# Automatic validation during registration
registry = ComponentRegistry()
enforcer = StandardizationEnforcer()
enforcer.enforce_standards_on_registration(registry)

# This will automatically validate standardization compliance
registry.register_builder("DataLoading", DataLoadingStepBuilder)
```

Standardization Rules represent the **maturation of the architectural system** from flexible guidelines to enforceable standards that ensure system-wide quality, consistency, and maintainability while enabling controlled evolution and growth.
