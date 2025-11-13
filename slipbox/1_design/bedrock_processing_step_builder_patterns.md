---
tags:
  - design
  - step_builders
  - bedrock_steps
  - patterns
  - sagemaker
  - llm_processing
  - aws_bedrock
keywords:
  - bedrock processing step patterns
  - AWS Bedrock integration
  - LLM processing
  - inference profile management
  - programmable response models
  - configurable model lists
topics:
  - step builder patterns
  - bedrock processing implementation
  - SageMaker LLM processing
  - AWS Bedrock step architecture
language: python
date of note: 2025-10-26
---

# Bedrock Processing Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock processing step builder implementations in the cursus framework. Bedrock processing steps create **ProcessingStep** instances that invoke AWS Bedrock models for Large Language Model (LLM) processing tasks. These steps integrate seamlessly with **Bedrock Prompt Template Generation** steps to provide automated categorization workflows with configurable model management, programmable response models, and intelligent fallback strategies for production LLM workflows.

## Integration with Bedrock Prompt Template Generation

Bedrock processing steps are designed to work in tandem with Bedrock Prompt Template Generation steps:

1. **Template Generation Step**: Generates structured prompt templates from category definitions
2. **Processing Step**: Consumes generated templates to categorize input data
3. **Seamless Integration**: Templates override configuration for dynamic prompt management

**Integration Flow:**
```
Category Definitions → Prompt Template Generation → Prompt Templates → Bedrock Processing → Categorized Results
```

## SageMaker Step Type Classification

Bedrock processing steps create **ProcessingStep** instances using **SKLearnProcessor** with specialized Bedrock integration:
- **SKLearnProcessor**: Standard processing framework with Bedrock client integration
- **Bedrock Runtime**: AWS Bedrock service integration for LLM inference
- **Model Management**: Intelligent switching between inference profiles and on-demand models
- **Response Processing**: Structured response parsing with Pydantic model validation

## Key Differences from Standard Processing Steps

### 1. Model Management Pattern
```python
# Standard Processing Step: Fixed processor configuration
processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=config.processing_instance_type
)

# Bedrock Processing Step: Dynamic model selection with fallback
class BedrockModelStrategy:
    def __init__(self, config):
        self.primary_model = config.primary_model_id
        self.inference_profile_arn = config.inference_profile_arn
        self.fallback_model = config.fallback_model_id
        self.strategy = self._determine_model_strategy()
    
    def _determine_model_strategy(self):
        if self.primary_model in config.inference_profile_required_models:
            return {
                'use_inference_profile': True,
                'effective_model': self.inference_profile_arn or self._get_global_profile(),
                'fallback_model': self.fallback_model
            }
        return {
            'use_inference_profile': False,
            'effective_model': self.primary_model,
            'fallback_model': None
        }
```

### 2. Configurable Model Lists Pattern
```python
# User-configurable model compatibility lists
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    # Configurable Model Lists - User can override defaults
    inference_profile_required_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        ],
        description="Models that require inference profiles (user-configurable)"
    )
    
    on_demand_compatible_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        ],
        description="Models compatible with on-demand throughput (user-configurable)"
    )
```

### 3. Programmable Response Models Pattern
```python
# Standard Processing: Fixed output format
outputs = [
    ProcessingOutput(
        output_name="processed_data",
        source="/opt/ml/processing/output/data"
    )
]

# Bedrock Processing: Programmable response models
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    response_model_class: Optional[str] = Field(
        default=None,
        description="Fully qualified class name for response model (e.g., 'mymodule.MyResponseModel')"
    )
    
    response_format: str = Field(
        default="json",
        description="Expected response format: 'json', 'text', or 'structured'"
    )

# Example usage with custom response model
from pydantic import BaseModel

class CustomAnalysisResponse(BaseModel):
    category: str
    confidence_score: float
    reasoning: List[str]
    evidence: Dict[str, List[str]]

# Configuration
config.response_model_class = "myproject.models.CustomAnalysisResponse"
config.response_format = "structured"
```

### 4. Prompt Template Integration Pattern
```python
# Standard Processing: Fixed script logic
job_arguments = ["--job_type", config.job_type]

# Bedrock Processing: Dynamic prompt template loading from Template Generation step
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    # Default prompts (overridden by template generation input)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Default system prompt (overridden by prompt_templates input)"
    )
    
    user_prompt_template: str = Field(
        default="Analyze the following data: {input_data}",
        description="Default user prompt template (overridden by prompt_templates input)"
    )
    
    additional_input_columns: List[str] = Field(
        default_factory=list,
        description="Additional columns to include in prompt template"
    )

# Template loading and configuration override in processing script
def load_prompt_templates(prompt_templates_path: str) -> Dict[str, str]:
    """Load prompt templates from Bedrock Prompt Template Generation output."""
    templates_dir = Path(prompt_templates_path)
    prompts_file = templates_dir / "prompts.json"
    
    if prompts_file.exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return {
            'system_prompt': templates.get('system_prompt'),
            'user_prompt_template': templates.get('user_prompt_template')
        }
    
    raise ValueError(f"No prompts.json found in {prompt_templates_path}")

# Configuration priority: Templates > Environment Variables > Defaults
def configure_from_templates(templates: Dict[str, str]) -> None:
    """Override configuration with template-provided values."""
    if templates.get('system_prompt'):
        os.environ['BEDROCK_SYSTEM_PROMPT'] = templates['system_prompt']
    if templates.get('user_prompt_template'):
        os.environ['BEDROCK_USER_PROMPT_TEMPLATE'] = templates['user_prompt_template']
```

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Bedrock processing step builders follow this architecture:

```python
@register_builder()
class BedrockProcessingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load Bedrock processing specification
        spec = BEDROCK_PROCESSING_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate Bedrock-specific configuration
        
    def _create_processor(self) -> SKLearnProcessor:
        # Create SKLearnProcessor with Bedrock environment
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build Bedrock-specific environment variables
        
    def _get_model_strategy_config(self) -> Dict[str, Any]:
        # Determine model usage strategy
        
    def _get_inputs(self, inputs) -> List[ProcessingInput]:
        # Create ProcessingInput objects using specification
        
    def _get_outputs(self, outputs) -> List[ProcessingOutput]:
        # Create ProcessingOutput objects for Bedrock results
        
    def _get_job_arguments(self) -> List[str]:
        # Build command-line arguments for Bedrock processing
        
    def create_step(self, **kwargs) -> ProcessingStep:
        # Orchestrate Bedrock processing step creation
```

### 2. Model Strategy Determination Pattern

```python
def _get_model_strategy_config(self) -> Dict[str, Any]:
    """Determine model usage strategy based on configuration."""
    model_id = self.config.primary_model_id
    
    strategy = {
        'primary_model': model_id,
        'use_inference_profile': False,
        'fallback_model': self.config.fallback_model_id,
        'inference_profile_arn': self.config.inference_profile_arn
    }
    
    # Check if model requires inference profile
    if model_id in self.config.inference_profile_required_models:
        strategy['use_inference_profile'] = True
        
        if self.config.inference_profile_arn:
            strategy['effective_model'] = self.config.inference_profile_arn
        else:
            # Try to use global profile ID if available
            if model_id.startswith('anthropic.claude-4') or 'claude-sonnet-4' in model_id:
                global_profile = model_id.replace('anthropic.', 'global.anthropic.')
                strategy['effective_model'] = global_profile
            else:
                strategy['effective_model'] = model_id
    else:
        strategy['effective_model'] = model_id
        
    return strategy
```

### 3. Environment Variables Pattern for Bedrock

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Build Bedrock-specific environment variables."""
    env_vars = super()._get_environment_variables()
    
    # Model configuration
    env_vars["BEDROCK_PRIMARY_MODEL_ID"] = self.config.primary_model_id
    env_vars["BEDROCK_FALLBACK_MODEL_ID"] = self.config.fallback_model_id
    
    # Inference profile configuration
    if self.config.inference_profile_arn:
        env_vars["BEDROCK_INFERENCE_PROFILE_ARN"] = self.config.inference_profile_arn
    
    # Model lists as JSON
    env_vars["BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS"] = json.dumps(
        self.config.inference_profile_required_models
    )
    env_vars["BEDROCK_ON_DEMAND_COMPATIBLE_MODELS"] = json.dumps(
        self.config.on_demand_compatible_models
    )
    
    # Note: Prompt configuration now comes from prompt_templates input
    # BEDROCK_SYSTEM_PROMPT and BEDROCK_USER_PROMPT_TEMPLATE are set dynamically
    # by the processing script after loading templates from the input
    
    # Response configuration
    env_vars["BEDROCK_RESPONSE_FORMAT"] = self.config.response_format
    if self.config.response_model_class:
        env_vars["BEDROCK_RESPONSE_MODEL_CLASS"] = self.config.response_model_class
    
    # API configuration
    env_vars["BEDROCK_MAX_TOKENS"] = str(self.config.max_tokens)
    env_vars["BEDROCK_TEMPERATURE"] = str(self.config.temperature)
    env_vars["BEDROCK_TOP_P"] = str(self.config.top_p)
    env_vars["BEDROCK_MAX_RETRIES"] = str(self.config.max_retries)
    
    # Processing configuration
    env_vars["BEDROCK_BATCH_SIZE"] = str(self.config.batch_size)
    env_vars["BEDROCK_INPUT_DATA_COLUMN"] = self.config.input_data_column
    env_vars["BEDROCK_OUTPUT_COLUMN_PREFIX"] = self.config.output_column_prefix
    
    # Additional input columns as JSON
    if self.config.additional_input_columns:
        env_vars["BEDROCK_ADDITIONAL_INPUT_COLUMNS"] = json.dumps(
            self.config.additional_input_columns
        )
    
    return env_vars
```

### 4. Job Arguments Pattern for Bedrock

```python
def _get_job_arguments(self) -> List[str]:
    """Build command-line arguments for Bedrock processing script."""
    # Following cursus pattern: minimal job arguments, most config in environment variables
    args = []
    
    # Only essential arguments that might vary per execution
    # Most configuration should be in environment variables
    return args
```

### 5. Specification-Driven Input/Output Pattern for Bedrock

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Create ProcessingInput objects for Bedrock processing."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_inputs = []
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        # Skip optional inputs not provided
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        # Validate required inputs
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        # Get container path from contract
        container_path = self.contract.expected_input_paths[logical_name]
        
        processing_inputs.append(ProcessingInput(
            input_name=logical_name,
            source=inputs[logical_name],
            destination=container_path
        ))
        
    return processing_inputs

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Create ProcessingOutput objects for Bedrock results."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_outputs = []
    
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name
        container_path = self.contract.expected_output_paths[logical_name]
        
        # Use provided destination or generate default
        destination = outputs.get(logical_name) or self._generate_output_path(logical_name)
        
        processing_outputs.append(ProcessingOutput(
            output_name=logical_name,
            source=container_path,
            destination=destination
        ))
        
    return processing_outputs
```

### 6. Processor Creation Pattern for Bedrock

```python
def _create_processor(self) -> SKLearnProcessor:
    """Create SKLearnProcessor with Bedrock-specific configuration."""
    instance_type = (self.config.processing_instance_type_large 
                    if self.config.use_large_processing_instance 
                    else self.config.processing_instance_type_small)
    
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
```

### 7. Step Creation Pattern for Bedrock

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create Bedrock ProcessingStep."""
    # Extract parameters
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Create components
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    job_args = self._get_job_arguments()
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create step directly (Pattern A - same as standard processing steps)
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=self.config.get_script_path(),
        job_arguments=job_args,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(step, '_spec', self.spec)
    
    return step
```

## Configuration Validation Patterns

### Standard Bedrock Configuration Validation
```python
def validate_configuration(self) -> None:
    """Validate Bedrock processing configuration."""
    # Validate base processing configuration
    required_processing_attrs = [
        'processing_instance_count', 'processing_volume_size',
        'processing_instance_type_large', 'processing_instance_type_small',
        'processing_framework_version', 'use_large_processing_instance'
    ]
    
    for attr in required_processing_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required processing attribute: {attr}")
    
    # Validate Bedrock-specific configuration
    required_bedrock_attrs = [
        'primary_model_id', 'fallback_model_id', 'max_tokens', 
        'temperature', 'top_p', 'batch_size', 'max_retries'
    ]
    
    for attr in required_bedrock_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required Bedrock attribute: {attr}")
    
    # Validate model lists
    if not self.config.inference_profile_required_models:
        raise ValueError("inference_profile_required_models cannot be empty")
    
    if not self.config.on_demand_compatible_models:
        raise ValueError("on_demand_compatible_models cannot be empty")
    
    # Validate response format
    valid_formats = ['json', 'text', 'structured']
    if self.config.response_format not in valid_formats:
        raise ValueError(f"Invalid response_format: {self.config.response_format}")
    
    # Validate response model class if provided
    if self.config.response_model_class:
        try:
            self._validate_response_model_class(self.config.response_model_class)
        except Exception as e:
            raise ValueError(f"Invalid response_model_class: {e}")
```

### Response Model Validation Pattern
```python
def _validate_response_model_class(self, class_path: str) -> None:
    """Validate that response model class exists and is a Pydantic model."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Check if it's a Pydantic model
        from pydantic import BaseModel
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Response model class must inherit from pydantic.BaseModel")
            
    except ImportError as e:
        raise ValueError(f"Cannot import response model module: {e}")
    except AttributeError as e:
        raise ValueError(f"Response model class not found: {e}")
```

## Design Components Integration

### 1. Step Specification Pattern (Updated for Template Integration)

```python
# specs/bedrock_processing_spec.py
BEDROCK_PROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockProcessing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_bedrock_processing_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["ProcessingStep", "DataLoad", "TabularPreprocessing"],
            semantic_keywords=["data", "input", "text", "dataset", "processed", "analyze"],
            data_type="S3Uri",
            description="Input data for Bedrock LLM processing"
        ),
        DependencySpec(
            logical_name="prompt_templates",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,  # Now REQUIRED for template integration
            compatible_sources=["BedrockPromptTemplateGeneration"],
            semantic_keywords=["templates", "prompts", "prompt_config", "generated_templates"],
            data_type="S3Uri",
            description="Generated prompt templates from Bedrock Prompt Template Generation step (prompts.json)"
        ),
        DependencySpec(
            logical_name="validation_schema",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,  # REQUIRED for response format configuration
            compatible_sources=["BedrockPromptTemplateGeneration"],
            semantic_keywords=["schema", "validation", "response_format", "output_schema"],
            data_type="S3Uri",
            description="Validation schema from Bedrock Prompt Template Generation step (validation_schema_*.json) - contains response format configuration and processing metadata"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Data with Bedrock LLM analysis results",
            aliases=["bedrock_output", "llm_results", "analyzed_data"]
        ),
        OutputSpec(
            logical_name="analysis_summary",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['analysis_summary'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Summary of Bedrock processing results and statistics",
            aliases=["summary", "stats", "processing_report"]
        )
    ]
)
```

### 2. Script Contract Pattern (Updated for Template Integration)

```python
# contracts/bedrock_processing_contract.py
BEDROCK_PROCESSING_CONTRACT = ProcessingScriptContract(
    entry_point="bedrock_processing.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "prompt_templates": "/opt/ml/processing/input/templates",  # Updated for template integration
        "validation_schema": "/opt/ml/processing/input/schema"  # NEW: Validation schema from Template Generation step
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data",
        "analysis_summary": "/opt/ml/processing/output/summary"
    },
    expected_arguments={
        # Minimal job arguments - most configuration in environment variables
    },
    required_env_vars=[
        "BEDROCK_PRIMARY_MODEL_ID",
        "BEDROCK_FALLBACK_MODEL_ID",
        "BEDROCK_RESPONSE_FORMAT",
        "BEDROCK_BATCH_SIZE",
        "BEDROCK_MAX_RETRIES",
        "BEDROCK_INPUT_DATA_COLUMN",
        "BEDROCK_OUTPUT_COLUMN_PREFIX"
    ],
    optional_env_vars={
        "BEDROCK_INFERENCE_PROFILE_ARN": "Inference profile ARN for provisioned throughput",
        "BEDROCK_RESPONSE_MODEL_CLASS": "Pydantic model class for response validation",
        "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": "JSON list of models requiring inference profiles",
        "BEDROCK_ON_DEMAND_COMPATIBLE_MODELS": "JSON list of on-demand compatible models",
        "BEDROCK_ADDITIONAL_INPUT_COLUMNS": "JSON list of additional input columns",
        "BEDROCK_MAX_TOKENS": "Maximum tokens for model response",
        "BEDROCK_TEMPERATURE": "Temperature parameter for model",
        "BEDROCK_TOP_P": "Top-p parameter for model"
    },
    framework_requirements={
        "boto3": ">=1.26.0",
        "pydantic": ">=2.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "tenacity": ">=8.0.0",
        "numpy": ">=1.19.0"
    },
    description="""
    Bedrock processing script that integrates with Bedrock Prompt Template Generation:
    1. Loads input data from CSV/Parquet files
    2. Loads generated prompt templates from Template Generation step
    3. Configures AWS Bedrock client with model strategy (inference profile vs on-demand)
    4. Processes data in batches through Bedrock LLM models using generated templates
    5. Handles intelligent fallback between inference profiles and on-demand models
    6. Parses and validates responses using configurable Pydantic models
    7. Saves processed results and analysis summary
    
    Template Integration Features:
    - Loads structured prompt templates from Bedrock Prompt Template Generation step
    - Dynamic configuration override based on generated templates
    - Seamless integration with 5-component template architecture
    - Support for category-driven classification workflows
    
    Model Management Features:
    - Automatic detection of models requiring inference profiles
    - Intelligent fallback to on-demand models when inference profiles fail
    - User-configurable model compatibility lists
    - Support for both ARN-based and global profile ID-based inference profiles
    
    Prompt System Features:
    - Generated prompt templates override default configuration
    - Support for additional input columns in prompt templates
    - Dynamic prompt formatting based on input data structure
    - Category-specific prompt generation for classification tasks
    
    Response Processing Features:
    - Configurable response formats (JSON, text, structured)
    - Pydantic model validation for structured responses
    - Automatic error handling and retry logic with exponential backoff
    - Comprehensive logging and monitoring
    
    Input Structure:
    - /opt/ml/processing/input/data: CSV/Parquet files with text data
    - /opt/ml/processing/input/templates: Generated prompt templates (prompts.json)
    
    Output Structure:
    - /opt/ml/processing/output/data: Processed data with LLM results
    - /opt/ml/processing/output/summary: Processing statistics and summary
    """
)
```

### 3. Configuration Class Pattern

```python
# configs/config_bedrock_processing_step.py
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    """Configuration for Bedrock processing step with configurable model management."""
    
    def __init__(self):
        super().__init__()
        
        # Model Configuration
        self.primary_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        self.inference_profile_arn: Optional[str] = None
        self.fallback_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        # Configurable Model Lists - User can override these
        self.inference_profile_required_models: List[str] = [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "anthropic.claude-4-opus-20250101-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        ]
        
        self.on_demand_compatible_models: List[str] = [
            "anthropic.claude-v2",
            "anthropic.claude-v2:1",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0", 
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-instant-v1"
        ]
        
        # Prompt Configuration
        self.system_prompt: Optional[str] = None
        self.user_prompt_template: str = "Analyze the following data: {input_data}"
        
        # Response Model Configuration
        self.response_model_class: Optional[str] = None
        self.response_format: str = "json"  # "json", "text", "structured"
        
        # Bedrock API Configuration
        self.max_tokens: int = 4000
        self.temperature: float = 0.1
        self.top_p: float = 0.9
        
        # Processing Configuration
        self.batch_size: int = 10
        self.max_retries: int = 3
        
        # Input/Output Configuration
        self.input_data_column: str = "input_text"
        self.additional_input_columns: List[str] = []
        self.output_column_prefix: str = "bedrock_"
    
    def validate_bedrock_configuration(self) -> None:
        """Validate Bedrock-specific configuration."""
        # Validate model configuration
        if not self.primary_model_id:
            raise ValueError("primary_model_id is required")
        
        if not self.fallback_model_id:
            raise ValueError("fallback_model_id is required")
        
        # Validate model lists
        if not self.inference_profile_required_models:
            raise ValueError("inference_profile_required_models cannot be empty")
        
        if not self.on_demand_compatible_models:
            raise ValueError("on_demand_compatible_models cannot be empty")
        
        # Validate API parameters
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        # Validate processing parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        # Validate response format
        valid_formats = ['json', 'text', 'structured']
        if self.response_format not in valid_formats:
            raise ValueError(f"response_format must be one of: {valid_formats}")
        
        # Validate response model class if provided
        if self.response_model_class and self.response_format != 'structured':
            raise ValueError("response_model_class requires response_format='structured'")
    
    def get_model_strategy(self) -> Dict[str, Any]:
        """Get model usage strategy based on configuration."""
        strategy = {
            'primary_model': self.primary_model_id,
            'use_inference_profile': False,
            'fallback_model': self.fallback_model_id,
            'inference_profile_arn': self.inference_profile_arn
        }
        
        # Check if model requires inference profile
        if self.primary_model_id in self.inference_profile_required_models:
            strategy['use_inference_profile'] = True
            
            if self.inference_profile_arn:
                strategy['effective_model'] = self.inference_profile_arn
            else:
                # Try to use global profile ID if available
                if 'claude-4' in self.primary_model_id or 'claude-sonnet-4' in self.primary_model_id:
                    global_profile = self.primary_model_id.replace('anthropic.', 'global.anthropic.')
                    strategy['effective_model'] = global_profile
                else:
                    strategy['effective_model'] = self.primary_model_id
        else:
            strategy['effective_model'] = self.primary_model_id
            
        return strategy
```

### 4. Processing Script Pattern (Revised Based on Real Implementation)

```python
# scripts/bedrock_processing.py
"""
Bedrock processing script following nlp-pipeline patterns with Pydantic model integration.
Based on real implementation from nlp-pipeline/src/nlp_pipeline/bedrock/rnr_bedrock_main.py
"""

import os
import json
import argparse
import pandas as pd
import boto3
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BedrockProcessor:
    """
    Bedrock processor following nlp-pipeline patterns with Pydantic model integration.
    Supports inference profiles and structured output validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock_client = None
        self.response_model_class = None
        self.effective_model_id = config['primary_model_id']
        self.inference_profile_info = {}
        
        self._initialize_bedrock_client()
        self._configure_inference_profile()
        self._load_response_model()
    
    def _initialize_bedrock_client(self):
        """Initialize Bedrock client."""
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.config.get('region_name', 'us-east-1'))
        logger.info("Initialized Bedrock client")
    
    def _configure_inference_profile(self):
        """Configure inference profile settings based on model and environment."""
        model_id = self.config['primary_model_id']
        inference_profile_arn = self.config.get('inference_profile_arn')
        
        # Check if model requires inference profile
        inference_profile_required = json.loads(
            self.config.get('inference_profile_required_models', '[]')
        )
        
        if inference_profile_arn:
            # Use provided ARN
            os.environ['BEDROCK_INFERENCE_PROFILE_ARN'] = inference_profile_arn
            self.inference_profile_info = {
                'arn': inference_profile_arn,
                'method': 'arn'
            }
            logger.info(f"Using inference profile ARN: {inference_profile_arn}")
            
        elif model_id in inference_profile_required:
            # Auto-configure for known models
            if model_id == "anthropic.claude-sonnet-4-20250514-v1:0":
                # Use global profile ID for Claude 4
                self.effective_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
                self.inference_profile_info = {
                    'profile_id': 'global.anthropic.claude-sonnet-4-20250514-v1:0',
                    'original_model_id': model_id,
                    'method': 'profile_id'
                }
                logger.info(f"Auto-configured to use inference profile ID: {self.effective_model_id}")
            
            elif 'claude-4' in model_id or 'claude-sonnet-4' in model_id:
                logger.warning(f"Model {model_id} may require an inference profile. Consider setting BEDROCK_INFERENCE_PROFILE_ARN.")
        
        # If model already starts with 'global.', it's already a profile ID
        if model_id.startswith('global.'):
            self.inference_profile_info = {
                'profile_id': model_id,
                'method': 'profile_id'
            }
            logger.info(f"Using provided inference profile ID: {model_id}")
    
    def _load_response_model(self):
        """Load Pydantic response model class if specified."""
        if self.config.get('response_model_class'):
            try:
                module_path, class_name = self.config['response_model_class'].rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                self.response_model_class = getattr(module, class_name)
                logger.info(f"Loaded Pydantic response model: {self.config['response_model_class']}")
            except Exception as e:
                logger.warning(f"Failed to load response model: {e}")
    
    def _format_prompt(self, input_data: str, additional_data: Dict[str, Any] = None) -> str:
        """Format prompt with input data and additional columns."""
        template_vars = {'input_data': input_data}
        
        if additional_data:
            template_vars.update(additional_data)
            
        return self.config['user_prompt_template'].format(**template_vars)
    
    def _invoke_bedrock(self, prompt: str) -> Dict[str, Any]:
        """Invoke Bedrock with intelligent fallback strategy."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config['max_tokens']),
            "temperature": float(self.config['temperature']),
            "top_p": float(self.config['top_p']),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.config.get('system_prompt'):
            request_body["system"] = self.config['system_prompt']
        
        # Try primary model/profile first
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.effective_model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            return json.loads(response['body'].read())
            
        except Exception as e:
            # Fallback to on-demand model if inference profile fails
            fallback_model = self.config.get('fallback_model_id')
            if fallback_model and 'ValidationException' in str(e):
                logger.warning(f"Inference profile failed, falling back to: {fallback_model}")
                try:
                    response = self.bedrock_client.invoke_model(
                        modelId=fallback_model,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json"
                    )
                    return json.loads(response['body'].read())
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e
    
    def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Bedrock response using Pydantic model validation."""
        if 'content' in response and len(response['content']) > 0:
            response_text = response['content'][0].get('text', '')
        else:
            raise ValueError("No content in Bedrock response")
        
        try:
            if self.response_model_class:
                # Use Pydantic model for structured parsing (like nlp-pipeline)
                validated_response = self.response_model_class.model_validate_json(response_text)
                
                # Flatten Pydantic model to dictionary (following nlp-pipeline pattern)
                result = self._flatten_pydantic_response(validated_response)
                
                # Add formatted output if available
                if hasattr(validated_response, 'to_formatted_output'):
                    result['formatted_output'] = validated_response.to_formatted_output()
                
                return result
            else:
                # Fallback to JSON parsing
                return json.loads(response_text)
                
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            return {
                'raw_response': response_text,
                'validation_error': str(e),
                'parse_status': 'validation_failed'
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                'raw_response': response_text,
                'json_error': str(e),
                'parse_status': 'json_failed'
            }
    
    def _flatten_pydantic_response(self, validated_response: BaseModel) -> Dict[str, Any]:
        """Flatten Pydantic model response following nlp-pipeline patterns."""
        result = {}
        
        # Get model dump
        model_data = validated_response.model_dump()
        
        # Flatten nested structures (following nlp-pipeline pattern)
        for key, value in model_data.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries with underscore separation
                for nested_key, nested_value in value.items():
                    result[f"{key}_{nested_key}"] = nested_value
            else:
                result[key] = value
        
        return result
    
    def process_single_case(
        self,
        input_data: str,
        additional_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single case through Bedrock (following nlp-pipeline pattern).
        
        Args:
            input_data: Main input text to process
            additional_data: Additional data for prompt template variables
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Format prompt
            prompt = self._format_prompt(input_data, additional_data)
            
            # Invoke Bedrock
            response = self._invoke_bedrock(prompt)
            
            # Parse response with Pydantic validation
            parsed_result = self._parse_response_with_pydantic(response)
            
            # Add processing metadata (following nlp-pipeline pattern)
            result = {
                **parsed_result,
                'processing_status': 'success',
                'error_message': None,
                'model_info': {
                    'effective_model_id': self.effective_model_id,
                    'inference_profile_info': self.inference_profile_info
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing case: {str(e)}")
            
            # Return structured error response (following nlp-pipeline pattern)
            error_result = {
                'processing_status': 'error',
                'error_message': str(e),
                'raw_response': None,
                'model_info': {
                    'effective_model_id': self.effective_model_id,
                    'inference_profile_info': self.inference_profile_info
                }
            }
            
            # Add default values for expected fields if Pydantic model is available
            if self.response_model_class:
                try:
                    # Create default instance to get field names
                    default_fields = self.response_model_class.model_fields.keys()
                    for field in default_fields:
                        if field not in error_result:
                            error_result[field] = None
                except Exception:
                    pass
            
            return error_result
    
    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of data through Bedrock (following nlp-pipeline pattern).
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with analysis results
        """
        batch_size = batch_size or self.config.get('batch_size', 10)
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        input_column = self.config['input_data_column']
        additional_columns = json.loads(self.config.get('additional_input_columns', '[]'))
        output_prefix = self.config['output_column_prefix']
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records)")
            
            batch_results = []
            for idx, row in batch_df.iterrows():
                # Prepare input data
                input_data = row[input_column]
                additional_data = {col: row[col] for col in additional_columns if col in row}
                
                # Process single case
                result = self.process_single_case(input_data, additional_data)
                
                # Add original row data
                result_row = row.to_dict()
                
                # Add Bedrock results with prefix (following nlp-pipeline pattern)
                for key, value in result.items():
                    if key not in ['processing_status', 'error_message', 'model_info']:
                        result_row[f"{output_prefix}{key}"] = value
                
                # Add processing metadata
                result_row[f"{output_prefix}status"] = result['processing_status']
                if result.get('error_message'):
                    result_row[f"{output_prefix}error"] = result['error_message']
                
                batch_results.append(result_row)
            
            results.extend(batch_results)
            
            # Save intermediate results (following nlp-pipeline pattern)
            if save_intermediate:
                intermediate_df = pd.DataFrame(batch_results)
                output_dir = Path("/opt/ml/processing/output/data")
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_file = output_dir / f"batch_{batch_num:04d}_results.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed processing {len(results_df)} records")
        
        return results_df


def load_prompt_templates(prompt_templates_path: str, log: Callable[[str], None]) -> Dict[str, str]:
    """
    Load prompt templates from Bedrock Prompt Template Generation step output.
    
    Expected file structure from Template Generation step:
    - prompts.json: JSON file containing system_prompt and user_prompt_template
    
    JSON format (generated by Template Generation step):
    {
        "system_prompt": "You are an expert analyst with extensive knowledge in data analysis, classification...",
        "user_prompt_template": "Categories and their criteria:\n\n1. Category1\n    - Description...\n\n## Required Output Format\n..."
    }
    
    Args:
        prompt_templates_path: Path to prompt templates directory from Template Generation step
        log: Logger function
        
    Returns:
        Dictionary with 'system_prompt' and 'user_prompt_template' keys
    """
    templates = {}
    templates_path = Path(prompt_templates_path)
    
    if not templates_path.exists():
        raise ValueError(f"Prompt templates directory not found: {prompt_templates_path}")
    
    # Load prompts.json (standard output from Template Generation step)
    prompts_file = templates_path / "prompts.json"
    if prompts_file.exists():
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                json_templates = json.load(f)
            
            if 'system_prompt' in json_templates:
                templates['system_prompt'] = json_templates['system_prompt']
                log(f"Loaded system prompt from {prompts_file}")
            
            if 'user_prompt_template' in json_templates:
                templates['user_prompt_template'] = json_templates['user_prompt_template']
                log(f"Loaded user prompt template from {prompts_file}")
                
        except Exception as e:
            raise ValueError(f"Failed to load templates from {prompts_file}: {e}")
    else:
        raise ValueError(f"Required prompts.json not found in {prompt_templates_path}")
    
    return templates


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for Bedrock processing, refactored for testability.
    Following nlp-pipeline patterns with Pydantic model integration.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary containing processing results and statistics
    """
    # Use print function if no logger is provided
    log = logger or print
    
    # Load prompt templates from Template Generation step (REQUIRED)
    if 'prompt_templates' not in input_paths:
        raise ValueError("prompt_templates input is required for Bedrock Processing")
    
    templates = load_prompt_templates(input_paths['prompt_templates'], log)
    log(f"Loaded templates: system_prompt={bool(templates.get('system_prompt'))}, user_prompt_template={bool(templates.get('user_prompt_template'))}")
    
    # Build configuration with template integration
    # Priority: Templates (highest) > Environment Variables > Defaults (lowest)
    config = {
        'primary_model_id': environ_vars.get('BEDROCK_PRIMARY_MODEL_ID'),
        'fallback_model_id': environ_vars.get('BEDROCK_FALLBACK_MODEL_ID', ''),
        'inference_profile_arn': environ_vars.get('BEDROCK_INFERENCE_PROFILE_ARN'),
        'inference_profile_required_models': environ_vars.get('BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS', '[]'),
        'on_demand_compatible_models': environ_vars.get('BEDROCK_ON_DEMAND_COMPATIBLE_MODELS', '[]'),
        'region_name': environ_vars.get('AWS_DEFAULT_REGION', 'us-east-1'),
        # Templates override environment variables and defaults
        'system_prompt': (
            templates.get('system_prompt') or 
            environ_vars.get('BEDROCK_SYSTEM_PROMPT')
        ),
        'user_prompt_template': (
            templates.get('user_prompt_template') or 
            environ_vars.get('BEDROCK_USER_PROMPT_TEMPLATE', 'Analyze: {input_data}')
        ),
        'response_format': environ_vars.get('BEDROCK_RESPONSE_FORMAT', 'structured'),
        'response_model_class': environ_vars.get('BEDROCK_RESPONSE_MODEL_CLASS'),
        'max_tokens': int(environ_vars.get('BEDROCK_MAX_TOKENS', '4000')),
        'temperature': float(environ_vars.get('BEDROCK_TEMPERATURE', '0.1')),
        'top_p': float(environ_vars.get('BEDROCK_TOP_P', '0.9')),
        'batch_size': int(environ_vars.get('BEDROCK_BATCH_SIZE', '10')),
        'max_retries': int(environ_vars.get('BEDROCK_MAX_RETRIES', '3')),
        'input_data_column': environ_vars.get('BEDROCK_INPUT_DATA_COLUMN', 'input_text'),
        'output_column_prefix': environ_vars.get('BEDROCK_OUTPUT_COLUMN_PREFIX', 'bedrock_'),
        'additional_input_columns': environ_vars.get('BEDROCK_ADDITIONAL_INPUT_COLUMNS', '[]')
    }
    
    try:
        # Initialize processor (following nlp-pipeline pattern)
        processor = BedrockProcessor(config)
        
        # Load input data
        input_path = Path("/opt/ml/processing/input/data")
        output_path = Path("/opt/ml/processing/output/data")
        summary_path = Path("/opt/ml/processing/output/summary")
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.mkdir(parents=True, exist_ok=True)
        
        # Process all CSV/Parquet files in input directory
        input_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.parquet"))
        
        if not input_files:
            raise ValueError("No input files found in /opt/ml/processing/input/data")
        
        all_results = []
        processing_stats = {
            'total_files': len(input_files),
            'total_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'files_processed': [],
            'model_info': processor.inference_profile_info,
            'effective_model_id': processor.effective_model_id
        }
        
        for input_file in input_files:
            log(f"Processing file: {input_file}")
            
            # Load data
            if input_file.suffix == '.csv':
                df = pd.read_csv(input_file)
            else:
                df = pd.read_parquet(input_file)
            
            # Process batch (following nlp-pipeline pattern)
            result_df = processor.process_batch(df, save_intermediate=True)
            
            # Update statistics
            processing_stats['total_records'] += len(df)
            success_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"])
            failed_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "error"])
            
            processing_stats['successful_records'] += success_count
            processing_stats['failed_records'] += failed_count
            processing_stats['files_processed'].append({
                'filename': input_file.name,
                'records': len(df),
                'successful': success_count,
                'failed': failed_count,
                'success_rate': success_count / len(df) if len(df) > 0 else 0
            })
            
            # Save results in multiple formats (following nlp-pipeline pattern)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"processed_{input_file.stem}_{timestamp}"
            
            # Save as Parquet (efficient for large datasets)
            parquet_file = output_path / f"{base_filename}.parquet"
            result_df.to_parquet(parquet_file, index=False)
            
            # Save as CSV (human-readable)
            csv_file = output_path / f"{base_filename}.csv"
            result_df.to_csv(csv_file, index=False)
            
            all_results.append(result_df)
            log(f"Saved results to: {parquet_file} and {csv_file}")
        
        # Combine all results if multiple files (following nlp-pipeline pattern)
        if len(all_results) > 1:
            combined_df = pd.concat(all_results, ignore_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_parquet = output_path / f"combined_results_{timestamp}.parquet"
            combined_csv = output_path / f"combined_results_{timestamp}.csv"
            
            combined_df.to_parquet(combined_parquet, index=False)
            combined_df.to_csv(combined_csv, index=False)
            log(f"Saved combined results to: {combined_parquet} and {combined_csv}")
        
        # Save processing summary (following nlp-pipeline pattern)
        processing_stats['overall_success_rate'] = (
            processing_stats['successful_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        processing_stats['processing_timestamp'] = datetime.now().isoformat()
        
        summary_file = summary_path / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        
        log(f"Processing completed successfully")
        log(f"Total records: {processing_stats['total_records']}")
        log(f"Success rate: {processing_stats['overall_success_rate']:.2%}")
        log(f"Model used: {processing_stats['effective_model_id']}")
        
        return processing_stats
        
    except Exception as e:
        log(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Minimal argument parser - most configuration comes from environment variables
        parser = argparse.ArgumentParser(description="Bedrock processing script")
        args = parser.parse_args()

        # Define standard SageMaker paths as constants
        INPUT_DATA_DIR = "/opt/ml/processing/input/data"
        INPUT_CONFIG_DIR = "/opt/ml/code/prompts"  # Same folder structure as script
        OUTPUT_DATA_DIR = "/opt/ml/processing/output/data"
        OUTPUT_SUMMARY_DIR = "/opt/ml/processing/output/summary"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters from environment variables
        logger.info(f"Starting Bedrock processing with parameters:")
        logger.info(f"  Primary Model ID: {os.environ.get('BEDROCK_PRIMARY_MODEL_ID')}")
        logger.info(f"  Response Format: {os.environ.get('BEDROCK_RESPONSE_FORMAT', 'structured')}")
        logger.info(f"  Response Model Class: {os.environ.get('BEDROCK_RESPONSE_MODEL_CLASS')}")
        logger.info(f"  Batch Size: {os.environ.get('BEDROCK_BATCH_SIZE', '10')}")
        logger.info(f"  Input Column: {os.environ.get('BEDROCK_INPUT_DATA_COLUMN', 'input_text')}")
        logger.info(f"  Output Prefix: {os.environ.get('BEDROCK_OUTPUT_COLUMN_PREFIX', 'bedrock_')}")
        logger.info(f"  Input Data Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Output Data Directory: {OUTPUT_DATA_DIR}")

        # Set up path dictionaries
        input_paths = {
            "input_data": INPUT_DATA_DIR,
            "prompt_templates": "/opt/ml/processing/input/templates",  # From Template Generation step
            "validation_schema": "/opt/ml/processing/input/schema"  # NEW: Validation schema from Template Generation step
        }

        output_paths = {
            "processed_data": OUTPUT_DATA_DIR,
            "analysis_summary": OUTPUT_SUMMARY_DIR
        }

        # Environment variables dictionary - all configuration from environment
        environ_vars = {
            "BEDROCK_PRIMARY_MODEL_ID": os.environ.get("BEDROCK_PRIMARY_MODEL_ID"),
            "BEDROCK_FALLBACK_MODEL_ID": os.environ.get("BEDROCK_FALLBACK_MODEL_ID", ""),
            "BEDROCK_INFERENCE_PROFILE_ARN": os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN"),
            "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": os.environ.get("BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS", "[]"),
            "BEDROCK_ON_DEMAND_COMPATIBLE_MODELS": os.environ.get("BEDROCK_ON_DEMAND_COMPATIBLE_MODELS", "[]"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "BEDROCK_SYSTEM_PROMPT": os.environ.get("BEDROCK_SYSTEM_PROMPT"),
            "BEDROCK_USER_PROMPT_TEMPLATE": os.environ.get("BEDROCK_USER_PROMPT_TEMPLATE", "Analyze: {input_data}"),
            "BEDROCK_RESPONSE_FORMAT": os.environ.get("BEDROCK_RESPONSE_FORMAT", "structured"),
            "BEDROCK_RESPONSE_MODEL_CLASS": os.environ.get("BEDROCK_RESPONSE_MODEL_CLASS"),
            "BEDROCK_MAX_TOKENS": os.environ.get("BEDROCK_MAX_TOKENS", "4000"),
            "BEDROCK_TEMPERATURE": os.environ.get("BEDROCK_TEMPERATURE", "0.1"),
            "BEDROCK_TOP_P": os.environ.get("BEDROCK_TOP_P", "0.9"),
            "BEDROCK_BATCH_SIZE": os.environ.get("BEDROCK_BATCH_SIZE", "10"),
            "BEDROCK_MAX_RETRIES": os.environ.get("BEDROCK_MAX_RETRIES", "3"),
            "BEDROCK_INPUT_DATA_COLUMN": os.environ.get("BEDROCK_INPUT_DATA_COLUMN", "input_text"),
            "BEDROCK_OUTPUT_COLUMN_PREFIX": os.environ.get("BEDROCK_OUTPUT_COLUMN_PREFIX", "bedrock_"),
            "BEDROCK_ADDITIONAL_INPUT_COLUMNS": os.environ.get("BEDROCK_ADDITIONAL_INPUT_COLUMNS", "[]")
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        logger.info(f"Bedrock processing completed successfully. Results: {result}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in Bedrock processing script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
```

## Key Differences Between Bedrock Step Types

### 1. By Model Management Strategy
- **Inference Profile Required**: Models requiring provisioned throughput (Claude 4, latest Claude 3.5)
- **On-Demand Compatible**: Models supporting on-demand throughput (Claude 3, older Claude 3.5)
- **Hybrid Strategy**: Intelligent fallback between inference profiles and on-demand models

### 2. By Response Processing
- **JSON Response**: Structured JSON output with optional Pydantic validation
- **Text Response**: Raw text output for simple use cases
- **Structured Response**: Custom parsing with mandatory Pydantic model validation

### 3. By Prompt Complexity
- **Simple Prompts**: Single input column with basic template
- **Complex Prompts**: Multiple input columns with advanced templating
- **Dynamic Prompts**: Runtime prompt generation based on data characteristics

### 4. By Use Case
- **Text Classification**: Categorizing text data with confidence scores
- **Content Analysis**: Extracting insights and structured information
- **Data Enrichment**: Adding LLM-generated features to existing datasets
- **Quality Assessment**: Evaluating content quality and compliance

## Best Practices Identified

1. **Configurable Model Lists**: Allow users to override default model compatibility lists
2. **Intelligent Fallback**: Implement robust fallback strategies for model failures
3. **Programmable Responses**: Support custom Pydantic models for structured outputs
4. **Template-Driven Prompts**: Use configurable prompt templates with variable substitution
5. **Batch Processing**: Process data in configurable batches for efficiency
6. **Comprehensive Logging**: Detailed logging for debugging and monitoring
7. **Error Handling**: Graceful handling of API failures and invalid responses
8. **Retry Logic**: Exponential backoff retry strategies for transient failures
9. **Resource Optimization**: Appropriate instance sizing for LLM processing workloads
10. **Cost Management**: Intelligent model selection to optimize costs

## Testing Implications

Bedrock processing step builders should be tested for:

1. **Model Strategy Determination**: Correct model selection based on configuration
2. **Inference Profile Handling**: Proper ARN and global profile ID usage
3. **Fallback Logic**: Correct fallback to on-demand models when profiles fail
4. **Environment Variable Processing**: Proper handling of all Bedrock-specific env vars
5. **Prompt Template Formatting**: Correct variable substitution in templates
6. **Response Model Validation**: Pydantic model loading and validation
7. **Batch Processing**: Correct handling of different batch sizes
8. **Error Recovery**: Proper error handling and retry logic
9. **Output Generation**: Correct ProcessingOutput creation for all formats
10. **Configuration Validation**: Comprehensive validation of all Bedrock parameters
11. **API Integration**: Mock Bedrock API calls for testing
12. **File Format Support**: Support for CSV and Parquet input/output formats

### Recommended Test Categories

#### Model Management Tests
- Model compatibility list validation
- Inference profile ARN handling
- Global profile ID generation
- Fallback model selection

#### Response Processing Tests
- JSON response parsing and validation
- Pydantic model integration
- Structured response extraction
- Error response handling

#### Prompt System Tests
- Template variable substitution
- Additional input column handling
- System prompt integration
- Dynamic prompt generation

#### Integration Tests
- End-to-end processing workflow
- Multiple file processing
- Batch size optimization
- Resource utilization

## Implementation Examples

### Complete Bedrock Processing Step Builder

```python
from typing import Dict, Optional, Any, List
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_processing_step import BedrockProcessingStepConfig
from ...core.base.builder_base import StepBuilderBase

# Import Bedrock processing specification
try:
    from ..specs.bedrock_processing_spec import BEDROCK_PROCESSING_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROCESSING_SPEC = None
    SPEC_AVAILABLE = False


class BedrockProcessingStepBuilder(StepBuilderBase):
    """Builder for Bedrock Processing Step with configurable model management."""
    
    def __init__(self, config: BedrockProcessingStepConfig, sagemaker_session=None, 
                 role: Optional[str] = None, registry_manager=None, 
                 dependency_resolver=None):
        if not isinstance(config, BedrockProcessingStepConfig):
            raise ValueError("BedrockProcessingStepBuilder requires BedrockProcessingStepConfig")
            
        if not SPEC_AVAILABLE or BEDROCK_PROCESSING_SPEC is None:
            raise ValueError("Bedrock processing specification not available")
            
        super().__init__(
            config=config,
            spec=BEDROCK_PROCESSING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockProcessingStepConfig = config
    
    def validate_configuration(self) -> None:
        """Validate Bedrock processing configuration."""
        # Validate base processing configuration
        required_processing_attrs = [
            'processing_instance_count', 'processing_volume_size',
            'processing_instance_type_large', 'processing_instance_type_small',
            'processing_framework_version', 'use_large_processing_instance'
        ]
        
        for attr in required_processing_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required processing attribute: {attr}")
        
        # Validate Bedrock-specific configuration
        self.config.validate_bedrock_configuration()
        
        self.log_info("BedrockProcessingStepConfig validation succeeded")
    
    def _get_model_strategy_config(self) -> Dict[str, Any]:
        """Get model usage strategy based on configuration."""
        return self.config.get_model_strategy()
    
    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearnProcessor with Bedrock-specific configuration."""
        instance_type = (self.config.processing_instance_type_large 
                        if self.config.use_large_processing_instance 
                        else self.config.processing_instance_type_small)
        
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
        """Build Bedrock-specific environment variables."""
        env_vars = super()._get_environment_variables()
        
        # Model configuration
        env_vars["BEDROCK_PRIMARY_MODEL_ID"] = self.config.primary_model_id
        env_vars["BEDROCK_FALLBACK_MODEL_ID"] = self.config.fallback_model_id
        
        # Inference profile configuration
        if self.config.inference_profile_arn:
            env_vars["BEDROCK_INFERENCE_PROFILE_ARN"] = self.config.inference_profile_arn
        
        # Model lists as JSON
        env_vars["BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS"] = json.dumps(
            self.config.inference_profile_required_models
        )
        env_vars["BEDROCK_ON_DEMAND_COMPATIBLE_MODELS"] = json.dumps(
            self.config.on_demand_compatible_models
        )
        
        # Prompt configuration
        if self.config.system_prompt:
            env_vars["BEDROCK_SYSTEM_PROMPT"] = self.config.system_prompt
        env_vars["BEDROCK_USER_PROMPT_TEMPLATE"] = self.config.user_prompt_template
        
        # Response configuration
        env_vars["BEDROCK_RESPONSE_FORMAT"] = self.config.response_format
        if self.config.response_model_class:
            env_vars["BEDROCK_RESPONSE_MODEL_CLASS"] = self.config.response_model_class
        
        # API configuration
        env_vars["BEDROCK_MAX_TOKENS"] = str(self.config.max_tokens)
        env_vars["BEDROCK_TEMPERATURE"] = str(self.config.temperature)
        env_vars["BEDROCK_TOP_P"] = str(self.config.top_p)
        env_vars["BEDROCK_MAX_RETRIES"] = str(self.config.max_retries)
        
        # Processing configuration
        env_vars["BEDROCK_BATCH_SIZE"] = str(self.config.batch_size)
        env_vars["BEDROCK_INPUT_DATA_COLUMN"] = self.config.input_data_column
        env_vars["BEDROCK_OUTPUT_COLUMN_PREFIX"] = self.config.output_column_prefix
        
        # Additional input columns as JSON
        if self.config.additional_input_columns:
            env_vars["BEDROCK_ADDITIONAL_INPUT_COLUMNS"] = json.dumps(
                self.config.additional_input_columns
            )
        
        return env_vars
    
    def _get_job_arguments(self) -> List[str]:
        """Build command-line arguments for Bedrock processing script."""
        args = [
            "--primary-model-id", self.config.primary_model_id,
            "--batch-size", str(self.config.batch_size),
            "--max-retries", str(self.config.max_retries),
            "--input-column", self.config.input_data_column,
            "--output-prefix", self.config.output_column_prefix,
            "--response-format", self.config.response_format
        ]
        
        # Add optional arguments
        if self.config.system_prompt:
            args.extend(["--system-prompt", self.config.system_prompt])
            
        if self.config.response_model_class:
            args.extend(["--response-model-class", self.config.response_model_class])
            
        if self.config.inference_profile_arn:
            args.extend(["--inference-profile-arn", self.config.inference_profile_arn])
        
        return args
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create Bedrock ProcessingStep."""
        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs from dependencies
        inputs = {}
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
        
        inputs.update(inputs_raw)
        
        # Create components
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()
        
        # Get standardized step name
        step_name = self._get_step_name()
        
        # Create step directly (Pattern A - same as standard processing steps)
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification for future reference
        setattr(step, '_spec', self.spec)
        
        self.log_info("Created Bedrock ProcessingStep: %s", step_name)
        return step
```

## Step Registry Integration

### Registry Pattern for Bedrock Steps

Following the cursus framework registry pattern, Bedrock steps must be registered in `src/cursus/registry/step_names_original.py`:

```python
# Registry entries for Bedrock processing steps
"BedrockProcessing": {
    "config_class": "BedrockProcessingStepConfig",
    "builder_step_name": "BedrockProcessingStepBuilder", 
    "spec_type": "BedrockProcessing",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "AWS Bedrock LLM processing step with configurable model management",
},
"BedrockTextClassification": {
    "config_class": "BedrockTextClassificationConfig",
    "builder_step_name": "BedrockTextClassificationStepBuilder",
    "spec_type": "BedrockTextClassification", 
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock text classification with structured response models",
},
"BedrockContentAnalysis": {
    "config_class": "BedrockContentAnalysisConfig",
    "builder_step_name": "BedrockContentAnalysisStepBuilder",
    "spec_type": "BedrockContentAnalysis",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock content analysis and insight extraction",
},
```

**Key Registry Pattern Notes:**
- **sagemaker_step_type**: Must be `"Processing"` for all Bedrock processing steps
- **spec_type**: Use case-specific specification type (e.g., "BedrockProcessing")
- **config_class**: Use case-specific configuration class name
- **builder_step_name**: Use case-specific builder class name

## Integration with Bedrock Prompt Template Generation

The `prompt_templates` input is a **required S3 directory** that contains generated prompt templates from the Bedrock Prompt Template Generation step:

### **File Contents and Structure:**

#### **Generated Format: `prompts.json`**
```json
{
    "system_prompt": "You are an expert analyst with extensive knowledge in data analysis, classification, pattern recognition. Your task is to analyze data accurately, classify content systematically, provide clear reasoning. Always be precise, be objective, be thorough, be consistent in your analysis.",
    "user_prompt_template": "Categories and their criteria:\n\n1. TrueDNR\n    - Delivered Not Received - Package marked as delivered but buyer claims non-receipt\n    - Key elements:\n        * delivered but not received\n        * tracking shows delivered\n        * missing package investigation\n    - Conditions:\n        * Package marked as delivered (EVENT_301)\n        * Buyer claims non-receipt\n        * Tracking shows delivery\n    - Must NOT include:\n        * Buyer received wrong item\n        * Package damaged on delivery\n\n2. FalsePositive\n    - Cases incorrectly flagged as DNR\n    - Key elements:\n        * incorrect classification\n        * false alarm\n        * misidentified case\n\nAnalysis Instructions:\n\nPlease analyze:\nInput_data: {input_data}\n\nProvide your analysis in the following structured format:\n\n1. Carefully review all provided data\n2. Identify key patterns and indicators\n3. Match against category criteria\n4. Select the most appropriate category\n5. Validate evidence against conditions and exceptions\n6. Provide confidence assessment and reasoning\n\n## Required Output Format\n\n**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**\n\n```json\n{\n    \"category\": \"The classified category name (must be exactly one of the defined categories)\",\n    \"confidence\": \"Confidence score between 0.0 and 1.0 indicating certainty of classification\",\n    \"key_evidence\": \"Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions\",\n    \"reasoning\": \"Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected\"\n}\n```\n\nDo not include any text before or after the JSON object. Only return valid JSON."
}
```

### **Template Generation Integration:**

#### **5-Component Architecture:**
The generated templates follow a structured 5-component architecture:
1. **System Prompt**: Role definition and expertise areas
2. **Category Definitions**: Structured category descriptions with conditions/exceptions
3. **Input Placeholders**: Variable placeholders for data injection
4. **Instructions**: Processing rules and analysis steps
5. **Output Format**: Structured JSON schema with field definitions

#### **Category-Driven Generation:**
Templates are automatically generated based on category definitions:
```python
# Input to Template Generation step
categories = [
    {
        "name": "TrueDNR",
        "description": "Delivered Not Received - Package marked as delivered but buyer claims non-receipt",
        "conditions": ["Package marked as delivered (EVENT_301)", "Buyer claims non-receipt"],
        "exceptions": ["Buyer received wrong item", "Package damaged on delivery"],
        "key_indicators": ["delivered but not received", "tracking shows delivered"]
    }
]

# Output: Structured prompt template optimized for classification
```

### **Configuration Priority:**
1. **Generated Templates** (highest priority): From Bedrock Prompt Template Generation step
2. **Environment Variables** (fallback): Use `BEDROCK_SYSTEM_PROMPT` and `BEDROCK_USER_PROMPT_TEMPLATE`
3. **Defaults** (lowest priority): Built-in default templates

### **Integration Benefits:**
- **Automated Prompt Engineering**: No manual prompt creation needed
- **Category-Specific Optimization**: Templates tailored to specific classification tasks
- **Quality Validation**: Templates validated for completeness and structure
- **Version Control**: Templates versioned with category definitions
- **Consistency**: Standardized prompt structure across all classification tasks

### **Pipeline Integration Example:**
```python
# Step 1: Generate templates from categories
template_step = BedrockPromptTemplateGenerationStepBuilder(config).create_step(
    inputs={'category_definitions': 's3://bucket/categories.json'},
    outputs={'prompt_templates': 's3://bucket/templates/'}
)

# Step 2: Process data using generated templates
processing_step = BedrockProcessingStepBuilder(config).create_step(
    inputs={
        'input_data': 's3://bucket/data/',
        'prompt_templates': template_step.properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri
    },
    outputs={'processed_data': 's3://bucket/results/'},
    dependencies=[template_step]
)
```

This integration provides a complete automated categorization workflow from category definitions to categorized results, with optimal prompt engineering handled automatically by the Template Generation step.
