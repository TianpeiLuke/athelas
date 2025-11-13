"""
Categorical Validation Processor for Data Quality Checks

This module provides atomic validation of categorical data quality.
Extracted from TSA data validation requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Set
import logging

from ..processors import Processor

logger = logging.getLogger(__name__)


class CategoricalValidationProcessor(Processor):
    """
    Validates categorical data quality and consistency.

    Extracted from TSA data validation requirements.

    Args:
        allowed_values: Dictionary of field -> allowed values mappings
        validation_rules: Custom validation rules
        validation_strategy: 'strict', 'warn', 'filter'
        report_violations: Whether to report validation violations
        max_violations: Maximum allowed violations before error
    """

    def __init__(
        self,
        allowed_values: Optional[Dict[str, Set[Any]]] = None,
        validation_rules: Optional[Dict[str, callable]] = None,
        validation_strategy: str = "warn",
        report_violations: bool = True,
        max_violations: Optional[int] = None,
    ):
        super().__init__()
        self.allowed_values = allowed_values or {}
        self.validation_rules = validation_rules or {}
        self.validation_strategy = validation_strategy
        self.report_violations = report_violations
        self.max_violations = max_violations
        self.learned_values = {}
        self.violation_counts = {}
        self.is_fitted = False

        if validation_strategy not in ["strict", "warn", "filter"]:
            raise ValueError(
                f"validation_strategy must be one of ['strict', 'warn', 'filter'], got {validation_strategy}"
            )

    def fit(self, data: Union[Dict, pd.DataFrame]) -> "CategoricalValidationProcessor":
        """Learn allowed values from training data if not provided"""
        if not self.allowed_values:
            if isinstance(data, pd.DataFrame):
                for col in data.select_dtypes(include=["object"]).columns:
                    unique_values = set(data[col].dropna().unique())
                    self.learned_values[col] = unique_values
            elif isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        unique_values = set(v for v in values if pd.notna(v))
                        self.learned_values[key] = unique_values

        self.is_fitted = True
        logger.info(
            f"CategoricalValidationProcessor fitted with {len(self.allowed_values)} predefined and {len(self.learned_values)} learned value sets"
        )
        return self

    def process(
        self, input_data: Union[Dict, pd.DataFrame]
    ) -> Union[Dict, pd.DataFrame]:
        """Apply categorical validation"""
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before processing")

        if isinstance(input_data, pd.DataFrame):
            return self._process_dataframe(input_data)
        elif isinstance(input_data, dict):
            return self._process_dict(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _process_dataframe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame input"""
        result = input_data.copy()
        violations = {}

        # Check allowed values
        for col in result.select_dtypes(include=["object"]).columns:
            allowed_set = self.allowed_values.get(col) or self.learned_values.get(col)
            if allowed_set:
                invalid_mask = ~result[col].isin(allowed_set) & result[col].notna()
                invalid_values = result.loc[invalid_mask, col].unique()

                if len(invalid_values) > 0:
                    violations[col] = {
                        "invalid_values": list(invalid_values),
                        "count": invalid_mask.sum(),
                        "percentage": (invalid_mask.sum() / len(result)) * 100,
                    }

                    if self.validation_strategy == "strict":
                        raise ValueError(
                            f"Invalid values found in column {col}: {invalid_values}"
                        )
                    elif self.validation_strategy == "warn":
                        logger.warning(
                            f"Invalid values found in column {col}: {invalid_values} (count: {invalid_mask.sum()})"
                        )
                    elif self.validation_strategy == "filter":
                        result = result[~invalid_mask]
                        logger.info(
                            f"Filtered {invalid_mask.sum()} rows with invalid values in column {col}"
                        )

        # Apply custom validation rules
        for col, rule_func in self.validation_rules.items():
            if col in result.columns:
                try:
                    rule_violations = result[col].apply(
                        lambda x: not rule_func(x) if pd.notna(x) else False
                    )
                    if rule_violations.any():
                        violation_count = rule_violations.sum()
                        violations[f"{col}_custom_rule"] = {
                            "count": violation_count,
                            "percentage": (violation_count / len(result)) * 100,
                        }

                        if self.validation_strategy == "strict":
                            raise ValueError(
                                f"Custom validation rule failed for column {col}: {violation_count} violations"
                            )
                        elif self.validation_strategy == "warn":
                            logger.warning(
                                f"Custom validation rule failed for column {col}: {violation_count} violations"
                            )
                        elif self.validation_strategy == "filter":
                            result = result[~rule_violations]
                            logger.info(
                                f"Filtered {violation_count} rows failing custom rule for column {col}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error applying custom validation rule for column {col}: {e}"
                    )

        # Store violation counts
        self.violation_counts = violations

        # Check maximum violations threshold
        if self.max_violations is not None:
            total_violations = sum(v["count"] for v in violations.values())
            if total_violations > self.max_violations:
                raise ValueError(
                    f"Total violations ({total_violations}) exceed maximum allowed ({self.max_violations})"
                )

        # Report violations if requested
        if self.report_violations and violations:
            self._report_violations(violations)

        return result

    def _process_dict(self, input_data: Dict) -> Dict:
        """Process dictionary input"""
        result = input_data.copy()
        violations = {}

        # Check allowed values
        for key, values in result.items():
            allowed_set = self.allowed_values.get(key) or self.learned_values.get(key)
            if allowed_set and isinstance(values, list):
                invalid_values = [
                    v for v in values if v not in allowed_set and pd.notna(v)
                ]

                if invalid_values:
                    violations[key] = {
                        "invalid_values": list(set(invalid_values)),
                        "count": len(invalid_values),
                        "percentage": (len(invalid_values) / len(values)) * 100,
                    }

                    if self.validation_strategy == "strict":
                        raise ValueError(
                            f"Invalid values found in key {key}: {set(invalid_values)}"
                        )
                    elif self.validation_strategy == "warn":
                        logger.warning(
                            f"Invalid values found in key {key}: {set(invalid_values)} (count: {len(invalid_values)})"
                        )
                    elif self.validation_strategy == "filter":
                        result[key] = [
                            v for v in values if v in allowed_set or pd.isna(v)
                        ]
                        logger.info(
                            f"Filtered {len(invalid_values)} invalid values from key {key}"
                        )

        # Apply custom validation rules
        for key, rule_func in self.validation_rules.items():
            if key in result and isinstance(result[key], list):
                try:
                    invalid_indices = [
                        i
                        for i, v in enumerate(result[key])
                        if pd.notna(v) and not rule_func(v)
                    ]
                    if invalid_indices:
                        violation_count = len(invalid_indices)
                        violations[f"{key}_custom_rule"] = {
                            "count": violation_count,
                            "percentage": (violation_count / len(result[key])) * 100,
                        }

                        if self.validation_strategy == "strict":
                            raise ValueError(
                                f"Custom validation rule failed for key {key}: {violation_count} violations"
                            )
                        elif self.validation_strategy == "warn":
                            logger.warning(
                                f"Custom validation rule failed for key {key}: {violation_count} violations"
                            )
                        elif self.validation_strategy == "filter":
                            result[key] = [
                                v
                                for i, v in enumerate(result[key])
                                if i not in invalid_indices
                            ]
                            logger.info(
                                f"Filtered {violation_count} values failing custom rule for key {key}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error applying custom validation rule for key {key}: {e}"
                    )

        # Store violation counts
        self.violation_counts = violations

        # Check maximum violations threshold
        if self.max_violations is not None:
            total_violations = sum(v["count"] for v in violations.values())
            if total_violations > self.max_violations:
                raise ValueError(
                    f"Total violations ({total_violations}) exceed maximum allowed ({self.max_violations})"
                )

        # Report violations if requested
        if self.report_violations and violations:
            self._report_violations(violations)

        return result

    def _report_violations(self, violations: Dict[str, Dict[str, Any]]) -> None:
        """Report validation violations"""
        logger.info("=== Categorical Validation Report ===")
        for field, violation_info in violations.items():
            logger.info(f"Field: {field}")
            logger.info(f"  Violations: {violation_info['count']}")
            logger.info(f"  Percentage: {violation_info['percentage']:.2f}%")
            if "invalid_values" in violation_info:
                logger.info(f"  Invalid values: {violation_info['invalid_values']}")
        logger.info("=====================================")

    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        if not self.violation_counts:
            return {"status": "no_violations", "total_violations": 0}

        total_violations = sum(v["count"] for v in self.violation_counts.values())
        return {
            "status": "violations_found" if total_violations > 0 else "no_violations",
            "total_violations": total_violations,
            "violations_by_field": self.violation_counts,
            "validation_strategy": self.validation_strategy,
        }

    def add_allowed_values(self, field: str, values: Set[Any]) -> None:
        """Add allowed values for a field"""
        if field in self.allowed_values:
            self.allowed_values[field].update(values)
        else:
            self.allowed_values[field] = set(values)
        logger.info(f"Added {len(values)} allowed values for field {field}")

    def remove_allowed_values(self, field: str, values: Set[Any]) -> None:
        """Remove allowed values for a field"""
        if field in self.allowed_values:
            self.allowed_values[field] -= values
            logger.info(f"Removed {len(values)} allowed values for field {field}")

    def add_validation_rule(
        self, field: str, rule_func: callable, rule_name: Optional[str] = None
    ) -> None:
        """Add custom validation rule for a field"""
        rule_key = rule_name or f"{field}_custom"
        self.validation_rules[rule_key] = rule_func
        logger.info(f"Added validation rule '{rule_key}' for field {field}")

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            "allowed_values": {k: list(v) for k, v in self.allowed_values.items()},
            "validation_rules": list(
                self.validation_rules.keys()
            ),  # Can't serialize functions
            "validation_strategy": self.validation_strategy,
            "report_violations": self.report_violations,
            "max_violations": self.max_violations,
            "learned_values": {k: list(v) for k, v in self.learned_values.items()},
        }

    def __repr__(self) -> str:
        return (
            f"CategoricalValidationProcessor(strategy='{self.validation_strategy}', "
            f"n_allowed_value_sets={len(self.allowed_values)}, "
            f"n_validation_rules={len(self.validation_rules)})"
        )
