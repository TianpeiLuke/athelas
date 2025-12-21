from pydantic import BaseModel, Field, model_validator, PrivateAttr, ConfigDict
from typing import List, Union, Dict, Any, Optional, ClassVar
import json
from io import StringIO


class ModelHyperparameters(BaseModel):
    """
    Base model hyperparameters for training tasks.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # --- Field lists ---
    full_field_list: List[str] = Field(description="Full list of original field names.")

    cat_field_list: List[str] = Field(
        description="Categorical fields using original names."
    )

    tab_field_list: List[str] = Field(
        description="Tabular/numeric fields using original names."
    )

    # --- Identifier and label fields ---
    id_name: str = Field(description="ID field name.")

    label_name: str = Field(description="Label field name.")

    # --- Classification parameters ---
    multiclass_categories: List[Union[int, str]] = Field(
        description="List of unique category labels."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    categorical_features_to_encode: List[str] = Field(
        default_factory=list,
        description="List of categorical fields that require specific encoding.",
    )

    # --- Model and Training Parameters ---
    model_class: str = Field(default="base_model", description="Model class name.")

    device: int = Field(default=-1, description="Device ID for training (-1 for CPU).")

    header: int = Field(default=0, description="Header row for CSV files.")

    lr: float = Field(default=3e-05, description="Learning rate.")

    batch_size: int = Field(
        default=2, gt=0, le=256, description="Batch size for training."
    )

    max_epochs: int = Field(
        default=3, gt=0, le=10, description="Maximum epochs for training."
    )

    metric_choices: List[str] = Field(
        default=["f1_score", "auroc"], description="Metric choices for evaluation."
    )

    optimizer: str = Field(default="SGD", description="Optimizer type.")

    # --- Will be derived from multiclass_categories but can be overridden ---
    class_weights: Optional[List[float]] = Field(
        default=None,
        description="Class weights for loss function. Defaults to [1.0] * num_classes.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _input_tab_dim: Optional[int] = PrivateAttr(default=None)
    _is_binary: Optional[bool] = PrivateAttr(default=None)
    _num_classes: Optional[int] = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Changed from "forbid" to "allow" to fix circular reference handling
        protected_namespaces=(),
    )

    # Public read-only properties for derived fields

    @property
    def input_tab_dim(self) -> int:
        """Get input tabular dimension derived from tab_field_list."""
        if self._input_tab_dim is None:
            self._input_tab_dim = len(self.tab_field_list)
        return self._input_tab_dim

    @property
    def num_classes(self) -> int:
        """Get number of classes derived from multiclass_categories."""
        if self._num_classes is None:
            self._num_classes = len(self.multiclass_categories)
        return self._num_classes

    @property
    def is_binary(self) -> bool:
        """Determine if this is a binary classification task based on num_classes."""
        if self._is_binary is None:
            self._is_binary = self.num_classes == 2
        return self._is_binary

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ModelHyperparameters":
        """Validate model dimensions and configurations"""
        # Initialize derived fields
        self._input_tab_dim = len(self.tab_field_list)
        self._num_classes = len(self.multiclass_categories)
        self._is_binary = self._num_classes == 2

        # Set default class_weights if not provided
        if self.class_weights is None:
            self.class_weights = [1.0] * self._num_classes

        # Validate class weights length
        if len(self.class_weights) != self._num_classes:
            raise ValueError(
                f"class_weights length ({len(self.class_weights)}) must match multiclass_categories length ({self._num_classes})."
            )

        # Validate binary classification consistency
        if self._is_binary and self._num_classes != 2:
            raise ValueError(
                "For binary classification, multiclass_categories length must be 2."
            )

        return self

    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - fields with no defaults (required)
        2. Tier 2: System Inputs - fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes

        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Initialize categories
        categories = {
            "essential": [],  # Tier 1: Required, public
            "system": [],  # Tier 2: Optional (has default), public
            "derived": [],  # Tier 3: Public properties
        }

        # Get model fields from the class (not instance) to avoid deprecation warnings
        model_fields = self.__class__.model_fields

        # Categorize public fields into essential (required) or system (with defaults)
        for field_name, field_info in model_fields.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Use is_required() to determine if a field is essential
            if field_info.is_required():
                categories["essential"].append(field_name)
            else:
                categories["system"].append(field_name)

        # Find derived properties (public properties that aren't in model_fields)
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and attr_name not in model_fields
                and isinstance(getattr(type(self), attr_name, None), property)
            ):
                categories["derived"].append(attr_name)

        return categories

    def __str__(self) -> str:
        """
        Custom string representation that shows fields by category.
        This overrides the default __str__ method so that print(hyperparam) shows
        a nicely formatted representation with fields organized by tier.

        Returns:
            A formatted string with fields organized by tier
        """
        # Use StringIO to build the string
        output = StringIO()

        # Get class name
        print(f"=== {self.__class__.__name__} ===", file=output)

        # Get fields categorized by tier
        categories = self.categorize_fields()

        # Print Tier 1 fields (essential user inputs)
        if categories["essential"]:
            print("\n- Essential User Inputs -", file=output)
            for field_name in sorted(categories["essential"]):
                print(f"{field_name}: {getattr(self, field_name)}", file=output)

        # Print Tier 2 fields (system inputs with defaults)
        if categories["system"]:
            print("\n- System Inputs -", file=output)
            for field_name in sorted(categories["system"]):
                value = getattr(self, field_name)
                if value is not None:  # Skip None values for cleaner output
                    print(f"{field_name}: {value}", file=output)

        # Print Tier 3 fields (derived properties)
        if categories["derived"]:
            print("\n- Derived Fields -", file=output)
            for field_name in sorted(categories["derived"]):
                try:
                    value = getattr(self, field_name)
                    if not callable(value):  # Skip methods
                        print(f"{field_name}: {value}", file=output)
                except Exception:
                    # Skip properties that cause errors
                    pass

        return output.getvalue()

    def print_hyperparam(self) -> None:
        """
        Print complete hyperparameter information organized by tiers.
        This method automatically categorizes fields by examining their characteristics.
        """
        print("\n===== HYPERPARAMETERS =====")
        print(f"Class: {self.__class__.__name__}")

        # Get fields categorized by tier
        categories = self.categorize_fields()

        # Print Tier 1 fields (essential user inputs)
        print("\n----- Essential User Inputs (Tier 1) -----")
        for field_name in sorted(categories["essential"]):
            print(f"{field_name.title()}: {getattr(self, field_name)}")

        # Print Tier 2 fields (system inputs with defaults)
        print("\n----- System Inputs with Defaults (Tier 2) -----")
        for field_name in sorted(categories["system"]):
            value = getattr(self, field_name)
            if value is not None:  # Skip None values for cleaner output
                print(f"{field_name.title()}: {value}")

        # Print Tier 3 fields (derived properties)
        print("\n----- Derived Fields (Tier 3) -----")
        for field_name in sorted(categories["derived"]):
            try:
                value = getattr(self, field_name)
                if not callable(value):  # Skip methods
                    print(f"{field_name.title()}: {value}")
            except Exception as e:
                print(f"{field_name.title()}: <Error: {e}>")

        print("\n===================================\n")

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Get a dictionary of public fields suitable for initializing a child hyperparameter.
        Only includes fields that should be passed to child class constructors.
        Both essential user inputs and system inputs with defaults or user-overridden values
        are included to ensure all user customizations are properly propagated.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Use categorize_fields to get essential and system fields
        categories = self.categorize_fields()

        # Initialize result dict
        init_fields = {}

        # Add all essential fields (Tier 1)
        for field_name in categories["essential"]:
            init_fields[field_name] = getattr(self, field_name)

        # Add all system fields (Tier 2) that aren't None
        for field_name in categories["system"]:
            value = getattr(self, field_name)
            if value is not None:  # Only include non-None values
                init_fields[field_name] = value

        return init_fields

    @classmethod
    def from_base_hyperparam(
        cls, base_hyperparam: "ModelHyperparameters", **kwargs
    ) -> "ModelHyperparameters":
        """
        Create a new hyperparameter instance from a base hyperparameter.
        This is a virtual method that all derived classes can use to inherit from a parent config.

        Args:
            base_hyperparam: Parent ModelHyperparameters instance
            **kwargs: Additional arguments specific to the derived class

        Returns:
            A new instance of the derived class initialized with parent fields and additional kwargs
        """
        # Get public fields from parent
        parent_fields = base_hyperparam.get_public_init_fields()

        # Combine with additional fields (kwargs take precedence)
        config_dict = {**parent_fields, **kwargs}

        # Create new instance of the derived class (cls refers to the actual derived class)
        return cls(**config_dict)

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.model_dump()

    def serialize_config(self) -> Dict[str, str]:
        """
        Serialize configuration for SageMaker.
        """
        # Start with the full model configuration
        config = self.get_config()

        # Add derived fields (these won't be in model_dump)
        config["input_tab_dim"] = self.input_tab_dim
        config["is_binary"] = self.is_binary
        config["num_classes"] = self.num_classes

        # Serialize all values to strings for SageMaker
        return {
            k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
            for k, v in config.items()
        }
