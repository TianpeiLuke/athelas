---
tags:
  - project_planning
  - bedrock_steps
  - prompt_generation
  - implementation_plan
  - step_creation_process
keywords:
  - bedrock prompt template generation
  - step implementation plan
  - script development
  - contract alignment
  - specification design
topics:
  - step creation process
  - bedrock integration
  - prompt template automation
  - cursus framework extension
language: python
date of note: 2025-11-02
---

# Bedrock Prompt Template Generation Step Implementation Plan

## Overview

This document provides a comprehensive implementation plan for creating the Bedrock Prompt Template Generation step in the cursus framework. The implementation follows the standardized step creation process outlined in the developer guide and ensures full alignment with existing framework patterns.

## Implementation Roadmap

### Phase 1: Script Development
- [ ] Create testable processing script with main signature
- [ ] Implement 5-component prompt template generation
- [ ] Add comprehensive validation and quality scoring
- [ ] Ensure alignment with testability requirements

### Phase 2: Contract Definition
- [ ] Create script contract with aligned paths and arguments
- [ ] Define input/output specifications
- [ ] Specify environment variables and requirements

### Phase 3: Specification Design
- [ ] Create step specification with dependency definitions
- [ ] Define output specifications with property paths
- [ ] Ensure alignment with script contract

### Phase 4: Registry Integration
- [ ] Register new step in step_names_original.py
- [ ] Follow standardization rules for naming

### Phase 5: Configuration Class
- [ ] Create config class inheriting from ProcessingStepConfigBase
- [ ] Implement category-driven configuration
- [ ] Add validation and auto-configuration logic

### Phase 6: Builder Implementation
- [ ] Create step builder using SKLearnProcessor
- [ ] Implement specification-driven approach
- [ ] Add comprehensive input/output handling

## Detailed Implementation Specifications

### 1. Script Implementation

#### File Location
```
src/cursus/steps/scripts/bedrock_prompt_template_generation.py
```

#### Main Function Signature (Following Testability Pattern)
```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for prompt template generation, refactored for testability.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary containing generation results and statistics
    """
```

#### Core Implementation Components

**1. Template Generator Class**
```python
class PromptTemplateGenerator:
    """
    Generates structured prompt templates for classification tasks using
    the 5-component architecture pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = self._load_categories()
        self.template_env = Environment(loader=BaseLoader())
        
    def generate_template(self) -> Dict[str, Any]:
        """Generate complete prompt template with 5-component structure."""
        return {
            'system_prompt': self._generate_system_prompt(),
            'user_prompt_template': self._generate_user_prompt_template(),
            'metadata': self._generate_template_metadata()
        }
```

**2. Template Validator Class**
```python
class TemplateValidator:
    """Validates generated prompt templates for quality and completeness."""
    
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template and return validation results."""
        # Implementation details in design document
```

**3. Category Definition Processing**
```python
def load_category_definitions(categories_path: str, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Load category definitions from input files (JSON/CSV)."""
    # Support both JSON and CSV formats
    # Validate category structure
    # Return standardized category list
```

#### Container Path Constants
```python
# Container path constants
CONTAINER_PATHS = {
    "INPUT_CATEGORIES_DIR": "/opt/ml/processing/input/categories",
    "INPUT_REQUIREMENTS_DIR": "/opt/ml/processing/input/requirements", 
    "INPUT_SCHEMA_DIR": "/opt/ml/processing/input/schema",
    "OUTPUT_TEMPLATES_DIR": "/opt/ml/processing/output/templates",
    "OUTPUT_METADATA_DIR": "/opt/ml/processing/output/metadata",
    "OUTPUT_SCHEMA_DIR": "/opt/ml/processing/output/schema"
}
```

#### Entry Point Implementation
```python
if __name__ == "__main__":
    try:
        # Argument parser
        parser = argparse.ArgumentParser(description="Bedrock prompt template generation script")
        parser.add_argument("--include-examples", action="store_true", help="Include examples in template")
        parser.add_argument("--generate-validation-schema", action="store_true", help="Generate validation schema")
        parser.add_argument("--template-version", default="1.0", help="Template version identifier")
        
        args = parser.parse_args()

        # Set up path dictionaries
        input_paths = {
            "category_definitions": CONTAINER_PATHS["INPUT_CATEGORIES_DIR"],
            "task_requirements": CONTAINER_PATHS["INPUT_REQUIREMENTS_DIR"],
            "output_schema_template": CONTAINER_PATHS["INPUT_SCHEMA_DIR"]
        }

        output_paths = {
            "prompt_templates": CONTAINER_PATHS["OUTPUT_TEMPLATES_DIR"],
            "template_metadata": CONTAINER_PATHS["OUTPUT_METADATA_DIR"],
            "validation_schema": CONTAINER_PATHS["OUTPUT_SCHEMA_DIR"]
        }

        # Environment variables dictionary
        environ_vars = {
            "TEMPLATE_TASK_TYPE": os.environ.get("TEMPLATE_TASK_TYPE", "classification"),
            "TEMPLATE_STYLE": os.environ.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": os.environ.get("VALIDATION_LEVEL", "standard"),
            # ... other environment variables
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        logger.info(f"Prompt template generation completed successfully. Results: {result}")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Error in prompt template generation script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
```

### 2. Script Contract Implementation

#### File Location
```
src/cursus/steps/contracts/bedrock_prompt_template_generation_contract.py
```

#### Contract Definition
```python
"""
Bedrock Prompt Template Generation Script Contract

Defines the contract for the bedrock prompt template generation script that creates
structured, reusable prompt templates for categorization and classification tasks.
"""

from ...core.base.contract_base import ScriptContract

BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT = ScriptContract(
    entry_point="bedrock_prompt_template_generation.py",
    expected_input_paths={
        "category_definitions": "/opt/ml/processing/input/categories",
        "task_requirements": "/opt/ml/processing/input/requirements",
        "output_schema_template": "/opt/ml/processing/input/schema"
    },
    expected_output_paths={
        "prompt_templates": "/opt/ml/processing/output/templates",
        "template_metadata": "/opt/ml/processing/output/metadata",
        "validation_schema": "/opt/ml/processing/output/schema"
    },
    expected_arguments={
        "--include-examples": "Include examples in generated templates (flag)",
        "--generate-validation-schema": "Generate validation schema (flag)",
        "--template-version": "Version identifier for generated templates"
    },
    required_env_vars=[
        "TEMPLATE_TASK_TYPE",
        "TEMPLATE_STYLE", 
        "VALIDATION_LEVEL"
    ],
    optional_env_vars={
        "SYSTEM_PROMPT_CONFIG": "JSON configuration for system prompt generation",
        "OUTPUT_FORMAT_CONFIG": "JSON configuration for output format generation",
        "INSTRUCTION_CONFIG": "JSON configuration for instruction generation",
        "INPUT_PLACEHOLDERS": "JSON list of input placeholder configurations",
        "ADDITIONAL_CONTEXT_FIELDS": "JSON list of additional context field names",
        "REQUIRED_OUTPUT_FIELDS": "JSON list of required output field names",
        "INCLUDE_EXAMPLES": "Whether to include examples in generated templates",
        "GENERATE_VALIDATION_SCHEMA": "Whether to generate validation schema",
        "TEMPLATE_VERSION": "Version identifier for generated templates"
    },
    framework_requirements={
        "pydantic": ">=2.0.0",
        "jinja2": ">=3.0.0",
        "jsonschema": ">=4.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.19.0"
    },
    description="""
    Bedrock prompt template generation script that:
    1. Loads category definitions and task requirements from input files
    2. Generates structured prompt templates with 5-component architecture:
       - System prompt with role assignment and expertise definition
       - Category definitions with conditions, exceptions, and key indicators
       - Input placeholders for data and context variables
       - Instructions and rules for LLM inference guidance
       - Output format schema with field definitions and validation rules
    3. Validates generated templates for completeness and quality
    4. Outputs templates in JSON format compatible with Bedrock processing steps
    5. Generates validation schemas and metadata for template quality assurance
    
    Template Generation Features:
    - Configurable template styles (structured, conversational, technical, detailed)
    - Dynamic category definition processing with priority handling
    - Automated output schema generation with field validation
    - Template quality scoring and validation reporting
    - Integration-ready output format for seamless Bedrock processing
    
    Input Structure:
    - /opt/ml/processing/input/categories: Category definitions (JSON/CSV)
    - /opt/ml/processing/input/requirements: Task requirements (JSON)
    - /opt/ml/processing/input/schema: Output schema template (JSON)
    
    Output Structure:
    - /opt/ml/processing/output/templates: Generated prompt templates (prompts.json)
    - /opt/ml/processing/output/metadata: Template metadata and validation results
    - /opt/ml/processing/output/schema: Generated validation schemas
    
    Template Structure (5-Component Architecture):
    1. System Prompt: Role definition, expertise areas, behavioral guidelines
    2. Category Definitions: Structured category descriptions with conditions/exceptions
    3. Input Placeholders: Variable placeholders for data and context injection
    4. Instructions: Processing rules, guidelines, and inference directions
    5. Output Format: Structured schema with field definitions and validation rules
    """
)
```

#### Alignment Verification
- **Script ↔ Contract**: All paths in script match contract `expected_input_paths` and `expected_output_paths`
- **Arguments**: Contract uses CLI-style hyphens, script uses Python-style underscores (argparse standard)
- **Environment Variables**: All required and optional env vars are documented and used consistently

### 3. Step Specification Implementation

#### File Location
```
src/cursus/steps/specs/bedrock_prompt_template_generation_spec.py
```

#### Specification Definition
```python
"""
Bedrock Prompt Template Generation Step Specification.

This module defines the declarative specification for Bedrock prompt template generation steps,
including their dependencies and outputs based on the actual implementation.
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
def _get_bedrock_prompt_template_generation_contract():
    from ..contracts.bedrock_prompt_template_generation_contract import BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT
    return BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT


# Bedrock Prompt Template Generation Step Specification
BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockPromptTemplateGeneration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_bedrock_prompt_template_generation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="category_definitions",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["ProcessingStep", "DataLoad", "ConfigPrep"],
            semantic_keywords=["categories", "definitions", "classification", "config", "schema", "taxonomy"],
            data_type="S3Uri",
            description="Category definitions with conditions, exceptions, and metadata for template generation"
        ),
        DependencySpec(
            logical_name="task_requirements",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "ConfigPrep", "DataLoad"],
            semantic_keywords=["requirements", "task", "config", "specification", "parameters", "settings"],
            data_type="S3Uri",
            description="Optional task requirements and configuration parameters for template customization"
        ),
        DependencySpec(
            logical_name="output_schema_template",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "SchemaPrep", "ConfigPrep"],
            semantic_keywords=["schema", "template", "format", "structure", "output", "validation"],
            data_type="S3Uri",
            description="Optional output schema template for customizing generated output format"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="prompt_templates",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated prompt templates in JSON format ready for Bedrock processing",
            aliases=["templates", "prompts", "prompt_config", "generated_templates"]
        ),
        OutputSpec(
            logical_name="template_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['template_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata about generated templates including validation results and quality metrics",
            aliases=["metadata", "validation_report", "template_info", "quality_metrics"]
        ),
        OutputSpec(
            logical_name="validation_schema",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['validation_schema'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated validation schema for output format validation",
            aliases=["schema", "validation", "output_schema", "format_schema"]
        )
    ]
)
```

#### Alignment Verification
- **Contract ↔ Specification**: Logical names match between contract `expected_input_paths`/`expected_output_paths` and specification dependencies/outputs
- **Property Paths**: Use valid ProcessingStep property path pattern for SageMaker
- **Dependencies**: Compatible sources align with actual step types that produce required outputs

### 4. Registry Integration

#### File Location
```
src/cursus/registry/step_names_original.py
```

#### Registry Entry Addition
```python
# Add to STEP_NAMES dictionary
"BedrockPromptTemplateGeneration": {
    "config_class": "BedrockPromptTemplateGenerationConfig",
    "builder_step_name": "BedrockPromptTemplateGenerationStepBuilder",
    "spec_type": "BedrockPromptTemplateGeneration",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock prompt template generation step for classification tasks",
},
```

#### Standardization Rules Compliance
- **Step Name**: PascalCase, descriptive, follows existing naming patterns
- **Config Class**: Matches step name + "Config" suffix
- **Builder Name**: Matches step name + "StepBuilder" suffix
- **Spec Type**: Matches step name exactly
- **SageMaker Step Type**: "Processing" for SKLearnProcessor-based steps

### 5. Configuration Class Implementation

#### File Location
```
src/cursus/steps/configs/config_bedrock_prompt_template_generation_step.py
```

#### Configuration Class Definition
```python
from pydantic import Field, model_validator, PrivateAttr
from typing import TYPE_CHECKING, Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.bedrock_prompt_template_generation_contract import BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


@dataclass
class CategoryDefinition:
    """Definition of a single category for classification tasks."""
    name: str
    description: str
    conditions: List[str]
    exceptions: List[str]
    key_indicators: List[str]
    examples: Optional[List[str]] = None
    priority: int = 1
    validation_rules: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate category definition after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Category name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Category description cannot be empty")
        if not self.conditions:
            raise ValueError("At least one condition is required")
        if not self.key_indicators:
            raise ValueError("At least one key indicator is required")


@dataclass
class TemplateComponents:
    """Configuration for template component generation."""
    system_prompt_config: 'SystemPromptConfig' = field(default_factory=lambda: SystemPromptConfig())
    category_section_config: 'CategorySectionConfig' = field(default_factory=lambda: CategorySectionConfig())
    input_placeholder_config: 'InputPlaceholderConfig' = field(default_factory=lambda: InputPlaceholderConfig())
    instruction_config: 'InstructionConfig' = field(default_factory=lambda: InstructionConfig())
    output_format_config: 'OutputFormatConfig' = field(default_factory=lambda: OutputFormatConfig())


@dataclass
class SystemPromptConfig:
    """Configuration for system prompt generation."""
    role_definition: str = "expert analyst"
    expertise_areas: List[str] = field(default_factory=lambda: ["data analysis", "classification"])
    responsibilities: List[str] = field(default_factory=lambda: ["analyze data", "classify content", "provide insights"])
    behavioral_guidelines: List[str] = field(default_factory=lambda: ["be precise", "be objective", "be thorough"])
    tone: str = "professional"
    include_expertise_statement: bool = True
    include_task_context: bool = True


@dataclass
class CategorySectionConfig:
    """Configuration for category section generation."""
    include_priority_ordering: bool = True
    include_examples: bool = True
    include_validation_rules: bool = True
    detailed_conditions: bool = True
    exception_handling: bool = True
    cross_category_guidance: bool = True


@dataclass
class InputPlaceholderConfig:
    """Configuration for input placeholder generation."""
    placeholder_format: str = "curly_braces"  # curly_braces, angle_brackets, custom
    include_descriptions: bool = True
    include_data_types: bool = True
    include_examples: bool = False
    custom_format_template: Optional[str] = None


@dataclass
class InstructionConfig:
    """Configuration for instruction section generation."""
    include_analysis_steps: bool = True
    include_decision_criteria: bool = True
    include_edge_case_handling: bool = True
    include_confidence_guidance: bool = True
    include_reasoning_requirements: bool = True
    step_by_step_format: bool = True


@dataclass
class OutputFormatConfig:
    """Configuration for output format generation."""
    format_type: str = "structured_json"  # structured_json, formatted_text, hybrid
    required_fields: List[str] = field(default_factory=lambda: ["category", "confidence", "reasoning"])
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    validation_requirements: List[str] = field(default_factory=list)
    example_output: Optional[str] = None
    include_field_constraints: bool = True
    include_formatting_rules: bool = True
    
    def __post_init__(self):
        """Set default field descriptions if not provided."""
        if not self.field_descriptions:
            self.field_descriptions = {
                "category": "The classified category name",
                "confidence": "Confidence score between 0.0 and 1.0",
                "reasoning": "Explanation of the classification decision"
            }


class BedrockPromptTemplateGenerationConfig(ProcessingStepConfigBase):
    """
    Configuration for a Bedrock prompt template generation step.

    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """

    # ===== Essential User Inputs (Tier 1) =====
    # Core Required Configuration - Only thing user needs to provide
    category_definitions: List[CategoryDefinition] = Field(
        default_factory=list,
        description="List of category definitions with conditions, exceptions, and metadata for template generation"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="bedrock_prompt_template_generation.py", 
        description="Entry point script for prompt template generation."
    )

    # Optimal Defaults - No user configuration needed
    task_type: str = Field(
        default="classification",
        description="Type of task: classification, categorization, analysis"
    )
    
    template_style: str = Field(
        default="structured",
        description="Template style: structured, conversational, technical, detailed"
    )
    
    validation_level: str = Field(
        default="standard",
        description="Validation level: basic, standard, comprehensive"
    )

    # Auto-configured based on category complexity
    template_components: TemplateComponents = Field(
        default_factory=TemplateComponents,
        description="Configuration for each template component"
    )

    # Smart defaults for common use cases
    input_placeholders: List[str] = Field(
        default_factory=lambda: ["dialogue", "shiptrack", "max_estimated_arrival_date"],
        description="List of input placeholder field names"
    )
    
    additional_context_fields: List[str] = Field(
        default_factory=list,
        description="Additional context fields to include in templates"
    )

    # Auto-enabled features
    include_examples: bool = Field(
        default=True,
        description="Whether to include examples in generated templates"
    )
    
    generate_validation_schema: bool = Field(
        default=True,
        description="Whether to generate validation schema for output format"
    )
    
    template_version: str = Field(
        default="1.0",
        description="Version identifier for generated templates"
    )

    # Quality control with optimal settings
    min_quality_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum quality score required for template validation"
    )
    
    enable_quality_checks: bool = Field(
        default=True,
        description="Whether to enable comprehensive quality checks"
    )

    # Update to Pydantic V2 style model_config
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",  # Allow extra fields like __model_type__ and __model_module__ for type-aware serialization
    }

    @model_validator(mode="after")
    def validate_config(self) -> "BedrockPromptTemplateGenerationConfig":
        """
        Validate configuration and ensure defaults are set.

        This validator ensures that:
        1. Entry point is provided
        2. Script contract is available and valid
        3. Required category definitions are provided
        4. Auto-configuration is applied based on category complexity
        """
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("Bedrock prompt template generation step requires a processing_entry_point")

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # Validate category definitions
        if not self.category_definitions:
            raise ValueError("At least one category definition is required")

        # Validate each category definition
        category_names = set()
        for i, category in enumerate(self.category_definitions):
            if category.name in category_names:
                raise ValueError(f"Duplicate category name: {category.name}")
            category_names.add(category.name)

        # Auto-configure based on categories after validation
        self._auto_configure_from_categories()

        return self

    def _auto_configure_from_categories(self):
        """Automatically configure optimal settings based on category complexity."""
        if not self.category_definitions:
            return
            
        # Auto-detect complexity indicators
        category_count = len(self.category_definitions)
        has_examples = any(cat.examples for cat in self.category_definitions if cat.examples)
        has_exceptions = any(cat.exceptions for cat in self.category_definitions if cat.exceptions)
        avg_conditions = sum(len(cat.conditions) for cat in self.category_definitions) / category_count
        
        # Auto-configure template components based on complexity
        if category_count > 10 or has_examples or has_exceptions or avg_conditions > 3:
            # Complex categorization detected
            self.template_components.category_section_config.detailed_conditions = True
            self.template_components.category_section_config.exception_handling = True
            self.template_components.instruction_config.include_edge_case_handling = True
        else:
            # Simple classification detected
            self.template_components.category_section_config.detailed_conditions = False
            self.template_components.category_section_config.exception_handling = False
            self.template_components.instruction_config.include_edge_case_handling = False
        
        # Auto-configure output format based on category names and structure
        category_names = [cat.name for cat in self.category_definitions]
        self.template_components.output_format_config.field_descriptions["category"] = f"Exactly one of: {', '.join(category_names)}"

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The bedrock prompt template generation script contract
        """
        return BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT

    def get_template_generation_summary(self) -> Dict[str, Any]:
        """Get summary of template generation configuration."""
        return {
            'task_type': self.task_type,
            'template_style': self.template_style,
            'validation_level': self.validation_level,
            'category_count': len(self.category_definitions),
            'category_names': [cat.name for cat in self.category_definitions],
            'output_format': self.template_components.output_format_config.format_type,
            'required_fields': self.template_components.output_format_config.required_fields,
            'input_placeholders': self.input_placeholders,
            'auto_configured': True,
            'complexity_detected': self._detect_complexity(),
            'template_version': self.template_version,
            'quality_checks_enabled': self.enable_quality_checks,
            'min_quality_score': self.min_quality_score
        }
    
    def _detect_complexity(self) -> str:
        """Detect complexity level based on category definitions."""
        if not self.category_definitions:
            return "simple"
            
        category_count = len(self.category_definitions)
        has_examples = any(cat.examples for cat in self.category_definitions if cat.examples)
        has_exceptions = any(cat.exceptions for cat in self.category_definitions if cat.exceptions)
        avg_conditions = sum(len(cat.conditions) for cat in self.category_definitions) / category_count
        
        if category_count > 10 or has_examples or has_exceptions or avg_conditions > 3:
            return "complex"
        else:
            return "simple"

```

#### Configuration Features
- **Simplified User Experience**: Users only need to provide category definitions
- **Auto-Configuration**: System automatically detects complexity and configures optimal settings
- **Category-Driven Design**: All template generation driven by category structure
- **Quality Control**: Built-in validation and quality scoring

### 6. Builder Implementation

#### File Location
```
src/cursus/steps/builders/builder_bedrock_prompt_template_generation_step.py
```

#### Builder Class Definition
```python
from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_prompt_template_generation_step import BedrockPromptTemplateGenerationConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the specification
try:
    from ..specs.bedrock_prompt_template_generation_spec import BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class BedrockPromptTemplateGenerationStepBuilder(StepBuilderBase):
    """
    Builder for a Bedrock Prompt Template Generation ProcessingStep.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the specification.
    """

    def __init__(
        self,
        config: BedrockPromptTemplateGenerationConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the prompt template generation step.

        Args:
            config: A BedrockPromptTemplateGenerationConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, BedrockPromptTemplateGenerationConfig):
            raise ValueError("BedrockPromptTemplateGenerationStepBuilder requires a BedrockPromptTemplateGenerationConfig instance.")

        # Use the specification if available
        spec = BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockPromptTemplateGenerationConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating BedrockPromptTemplateGenerationConfig...")

        # Validate processing script settings
        if (
            not hasattr(self.config, "processing_entry_point")
            or not self.config.processing_entry_point
        ):
            raise ValueError("Bedrock prompt template generation step requires a processing_entry_point")

        # Validate category definitions
        if not self.config.category_definitions:
            raise ValueError("At least one category definition is required")

        # Validate other required attributes
        required_attrs = [
            "processing_instance_count",
            "processing_volume_size",
            "processing_instance_type_large",
            "processing_instance_type_small",
            "pipeline_name",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(f"BedrockPromptTemplateGenerationConfig missing required attribute: {attr}")

        self.log_info("BedrockPromptTemplateGenerationConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the appropriate instance type based on use_large_processing_instance
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        # Get framework version
        framework_version = getattr(
            self.config, "processing_framework_version", "1.2-1"
        )

        return SKLearnProcessor(
            framework_version=framework_version,
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
        Constructs a dictionary of environment variables to be passed to the processing job.

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add template generation specific environment variables
        env_vars["TEMPLATE_TASK_TYPE"] = self.config.task_type
        env_vars["TEMPLATE_STYLE"] = self.config.template_style
        env_vars["VALIDATION_LEVEL"] = self.config.validation_level

        # Add template component configurations as JSON
        import json
        from dataclasses import asdict
        
        env_vars["SYSTEM_PROMPT_CONFIG"] = json.dumps(
            asdict(self.config.template_components.system_prompt_config)
        )
        env_vars["OUTPUT_FORMAT_CONFIG"] = json.dumps(
            asdict(self.config.template_components.output_format_config)
        )
        env_vars["INSTRUCTION_CONFIG"] = json.dumps(
            asdict(self.config.template_components.instruction_config)
        )

        # Add input placeholder configuration
        env_vars["INPUT_PLACEHOLDERS"] = json.dumps(self.config.input_placeholders)
        env_vars["ADDITIONAL_CONTEXT_FIELDS"] = json.dumps(self.config.additional_context_fields)

        # Add output configuration
        env_vars["OUTPUT_FORMAT_TYPE"] = self.config.template_components.output_format_config.format_type
        env_vars["REQUIRED_OUTPUT_FIELDS"] = json.dumps(
            self.config.template_components.output_format_config.required_fields
        )

        # Add feature flags
        env_vars["INCLUDE_EXAMPLES"] = str(self.config.include_examples).lower()
        env_vars["GENERATE_VALIDATION_SCHEMA"] = str(self.config.generate_validation_schema).lower()
        env_vars["TEMPLATE_VERSION"] = self.config.template_version

        # Add quality control settings
        env_vars["MIN_QUALITY_SCORE"] = str(self.config.min_quality_score)
        env_vars["ENABLE_QUALITY_CHECKS"] = str(self.config.enable_quality_checks).lower()

        # Add pipeline information
        if hasattr(self.config, "pipeline_name"):
            env_vars["PIPELINE_NAME"] = self.config.pipeline_name

        if hasattr(self.config, "region"):
            env_vars["REGION"] = self.config.region

        self.log_info("Bedrock prompt template generation environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects

        Raises:
            ValueError: If no specification or contract is available
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

            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")

            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")

            # Use the input value directly - property references are handled by PipelineAssembler
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

        This method creates ProcessingOutput objects for each output defined in the specification.

        Args:
            outputs: Output destinations keyed by logical name

        Returns:
            List of ProcessingOutput objects

        Raises:
            ValueError: If no specification or contract is available
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
            container_path = None
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
            else:
                raise ValueError(f"No container path found for output: {logical_name}")

            # Try to find destination in outputs
            destination = None

            # Look in outputs by logical name
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from base path using Join instead of f-string
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                destination = Join(on="/", values=[base_output_path, "bedrock_templates", logical_name])
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
        Build command-line arguments for template generation script.

        Returns:
            List of command-line arguments
        """
        args = []
        
        # Only include arguments that provide functionality not covered by environment variables
        if self.config.include_examples:
            args.append("--include-examples")
            
        if self.config.generate_validation_schema:
            args.append("--generate-validation-schema")
        
        if self.config.template_version != "1.0":
            args.extend(["--template-version", self.config.template_version])
        
        return args

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        self.log_info("Creating Bedrock Prompt Template Generation ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get("inputs", {})
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        # Handle inputs
        inputs = {}

        # If dependencies are provided, extract inputs from them
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)

        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()

        # Get script path using modernized method with comprehensive fallbacks
        script_path = self.config.get_script_path()
        self.log_info("Using script path: %s", script_path)

        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

        # Attach specification to the step for future reference
        if hasattr(self, "spec") and self.spec:
            setattr(step, "_spec", self.spec)

        self.log_info("Created ProcessingStep with name: %s", step.name)
        return step
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Implement script with testable main signature
- [ ] Create script contract with proper alignment
- [ ] Set up basic template generation framework

### Week 2: Core Logic
- [ ] Implement 5-component template generation
- [ ] Add category definition processing
- [ ] Create template validation system

### Week 3: Integration
- [ ] Create step specification
- [ ] Register step in framework
- [ ] Implement configuration class

### Week 4: Builder and Testing
- [ ] Create step builder
- [ ] Add comprehensive unit tests
- [ ] Integration testing with existing steps

### Week 5: Documentation and Polish
- [ ] Complete documentation
- [ ] Performance optimization
- [ ] Final validation and alignment checks

## Testing Strategy

### Unit Tests
- [ ] Script main function with various input scenarios
- [ ] Template generator with different category complexities
- [ ] Template validator with quality scoring
- [ ] Configuration validation and auto-configuration
- [ ] Builder input/output handling

### Integration Tests
- [ ] End-to-end template generation workflow
- [ ] Integration with Bedrock processing steps
- [ ] Pipeline assembly and execution
- [ ] Error handling and recovery

### Validation Tests
- [ ] Alignment between script, contract, and specification
- [ ] Property path validation for SageMaker
- [ ] Registry integration verification
- [ ] Configuration inheritance and validation

## Quality Assurance Checklist

### Alignment Verification
- [ ] Script paths match contract `expected_input_paths`/`expected_output_paths`
- [ ] Contract arguments use CLI-style hyphens, script uses Python underscores
- [ ] Specification logical names match contract paths
- [ ] Property paths follow SageMaker ProcessingStep patterns
- [ ] Builder environment variables cover all required contract vars

### Framework Integration
- [ ] Registry entry follows standardization rules
- [ ] Configuration inherits from ProcessingStepConfigBase
- [ ] Builder inherits from StepBuilderBase and uses SKLearnProcessor
- [ ] Specification follows existing patterns and conventions

### Code Quality
- [ ] Comprehensive error handling and logging
- [ ] Type hints and documentation
- [ ] Pydantic validation and model configuration
- [ ] Testable design with dependency injection

## Success Criteria

### Functional Requirements
- [ ] Generate high-quality prompt templates from category definitions
- [ ] Support both simple and complex categorization scenarios
- [ ] Validate template quality and provide scoring
- [ ] Integrate seamlessly with existing Bedrock processing steps

### Technical Requirements
- [ ] Follow cursus framework patterns and conventions
- [ ] Maintain full alignment between all components
- [ ] Support testable design with proper separation of concerns
- [ ] Provide comprehensive error handling and logging

### User Experience
- [ ] Simplified configuration requiring only category definitions
- [ ] Automatic complexity detection and optimal configuration
- [ ] Clear validation messages and quality feedback
- [ ] Integration-ready outputs for downstream processing

## Risk Mitigation

### Technical Risks
- **Alignment Issues**: Use comprehensive validation tests and alignment checkers
- **Integration Problems**: Follow existing patterns and test with real pipelines
- **Performance Issues**: Optimize template generation and validation algorithms

### Implementation Risks
- **Complexity Underestimation**: Break down into smaller, testable components
- **Framework Changes**: Stay aligned with latest framework patterns and updates
- **Testing Gaps**: Implement comprehensive test coverage at all levels

## Conclusion

This implementation plan provides a comprehensive roadmap for creating the Bedrock Prompt Template Generation step in the cursus framework. The plan follows the established step creation process and ensures full alignment with existing framework patterns and conventions.

### Key Implementation Highlights

1. **Testable Design**: The script follows the testability pattern with a main function that accepts parameters instead of directly accessing the environment, enabling comprehensive unit testing.

2. **5-Component Architecture**: The template generation implements the structured 5-component pattern (system prompt, category definitions, input placeholders, instructions, output format) for optimal LLM performance.

3. **Simplified User Experience**: Users only need to provide category definitions; the system automatically configures all other settings based on complexity detection.

4. **Full Framework Integration**: All components (script, contract, specification, config, builder) are properly aligned and follow cursus framework standards.

5. **Quality Assurance**: Built-in template validation, quality scoring, and comprehensive error handling ensure reliable operation.

### Implementation Benefits

- **Reduced Development Time**: Automated prompt template generation eliminates manual template creation
- **Consistent Quality**: Standardized 5-component architecture ensures optimal template structure
- **Easy Integration**: Seamless compatibility with existing Bedrock processing steps
- **Maintainable Code**: Follows framework patterns and includes comprehensive testing
- **User-Friendly**: Simplified configuration with intelligent auto-configuration

### Next Steps

1. **Phase 1**: Begin with script implementation and basic template generation
2. **Phase 2**: Create contract and specification with proper alignment
3. **Phase 3**: Implement configuration class with auto-configuration logic
4. **Phase 4**: Build step builder with comprehensive input/output handling
5. **Phase 5**: Add comprehensive testing and documentation

This plan ensures that the Bedrock Prompt Template Generation step will be a valuable addition to the cursus framework, providing users with powerful prompt template automation capabilities while maintaining the high standards of code quality and framework integration that cursus is known for.

### References

- [Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)
- [Script Testability Implementation Guide](../0_developer_guide/script_testability_implementation.md)
- [Alignment Rules](../0_developer_guide/alignment_rules.md)
- [Step Creation Process](../0_developer_guide/creation_process.md)
- [Standardization Rules](../0_developer_guide/standardization_rules.md)
