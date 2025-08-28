from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

from .processors import Processor, ComposedProcessor

# Setup logger
logger = logging.getLogger(__name__)


class NumericalVariableImputationProcessor:
    """
    A processor that performs imputation on numerical variables using predefined or computed values.
    Supports mean, median, and mode imputation strategies.
    """
    
    def __init__(
        self,
        variables: Optional[List[str]] = None,
        imputation_dict: Optional[Dict[str, Union[int, float]]] = None,
        strategy: str = 'mean'
    ):
        self.processor_name = 'numerical_variable_imputation_processor'
        self.function_name_list = ['fit', 'process', 'transform']
        
        self.variables = variables
        self.strategy = strategy
        self.is_fitted = False
        
        if imputation_dict:
            self._validate_imputation_dict(imputation_dict)
            self.imputation_dict = imputation_dict
            self.is_fitted = True
        else:
            self.imputation_dict = None

    def get_name(self) -> str:
        return self.processor_name

    def __call__(self, input_data):
        return self.process(input_data)

    def __rshift__(self, other):
        if isinstance(self, ComposedProcessor):
            return ComposedProcessor(self.processors + [other])
        return ComposedProcessor([self, other])

    def _validate_imputation_dict(self, imputation_dict: Dict[str, Any]) -> None:
        if not isinstance(imputation_dict, dict):
            raise ValueError("imputation_dict must be a dictionary")
        if not imputation_dict:
            raise ValueError("imputation_dict cannot be empty")
        for k, v in imputation_dict.items():
            if not isinstance(k, str):
                raise ValueError(f"All keys must be strings, got {type(k)} for key {k}")
            if not isinstance(v, (int, float, np.number)):
                raise ValueError(f"All values must be numeric, got {type(v)} for key {k}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericalVariableImputationProcessor':
        if self.imputation_dict is None:
            self.imputation_dict = {}
            
            if self.variables is None:
                self.variables = X.select_dtypes(include=np.number).columns.tolist()
            
            for var in self.variables:
                if var not in X.columns:
                    raise ValueError(f"Variable {var} not found in the input data")
                
                if X[var].isna().all():
                    self.imputation_dict[var] = np.nan
                    continue
                    
                if self.strategy == 'mean':
                    self.imputation_dict[var] = float(X[var].mean())
                elif self.strategy == 'median':
                    self.imputation_dict[var] = float(X[var].median())
                elif self.strategy == 'mode':
                    self.imputation_dict[var] = float(X[var].mode()[0])
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._validate_imputation_dict(self.imputation_dict)
        self.is_fitted = True
        return self

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Processor is not fitted. Call 'fit' with appropriate arguments before using this method.")
        
        output_data = input_data.copy()
        for var, value in input_data.items():
            if var in self.imputation_dict and pd.isna(value):
                output_data[var] = self.imputation_dict[var]
        return output_data

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform input data by imputing missing values.
    
        Args:
            X: Input DataFrame or Series
        
        Returns:
            Transformed DataFrame or Series with imputed values
        """
        if not self.is_fitted:
            raise RuntimeError("Processor is not fitted. Call 'fit' with appropriate arguments before using this method.")

        # Handle Series input
        if isinstance(X, pd.Series):
            if X.name not in self.imputation_dict:
                raise ValueError(f"No imputation value found for series name: {X.name}")
            return X.fillna(self.imputation_dict[X.name])

        # Handle DataFrame input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas Series or DataFrame")

        # Make a copy to avoid modifying the input
        df = X.copy()
    
        # Apply imputation only to variables in imputation_dict and only to NaN values
        for var, impute_value in self.imputation_dict.items():
            if var in df.columns:
                # Create mask for NaN values
                nan_mask = df[var].isna()
                # Only replace NaN values
                df.loc[nan_mask, var] = impute_value
            
        return df

    def get_params(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "imputation_dict": self.imputation_dict,
            "strategy": self.strategy
        }