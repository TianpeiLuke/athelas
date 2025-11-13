# https://code.amazon.com/packages/OptimizerCorePythonLib/blobs/941a2003eb50fc3736e99746c6143e23be9c3e2d/--/src/optimizer_core_python_lib/transformer.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_map, columns_list):
        if not isinstance(categorical_map, dict):
            raise TypeError(
                f"Categorical Map is not dict type: {type(categorical_map)}. Please send a dict"
            )
        if not isinstance(columns_list, list):
            raise TypeError(
                f"Columns list is not list type: {type(columns_list)}. Please send a list"
            )
        self.categorical_map = categorical_map
        self.columns_list = columns_list
        self.index_column_map = self.__get_index_to_column_map(
            categorical_map=self.categorical_map, columns_list=self.columns_list
        )

    def __get_index_to_column_map(self, categorical_map, columns_list):
        """
        Returns a map with key = index and value = corresponding key in categorical_map.
        param arguments:
            categorical_map : {"A" : {"X" : 100}, "D" : {"Y": 200}}
            columns_list : ["A", "B", "C", "D", "E", "F"]
        return:
            index_column_map : {0: "A", 3: "D"}
        """
        index_column_map = {}
        for index in range(len(columns_list)):
            if columns_list[index] in categorical_map:
                index_column_map[index] = columns_list[index]
        return index_column_map

    def transform(self, input_data: np.ndarray) -> np.ndarray:
        if not isinstance(input_data, np.ndarray):
            raise TypeError(
                f"Input data is not np.ndarray : {type(input_data)}. Please send ndarray"
            )
        # TODO: Replace copy() - copy() is not compatible for large dataset
        transformed_data = input_data.copy()
        for key in self.index_column_map:
            for row in range(len(input_data)):
                try:
                    # Converting NaN type from float to str
                    if type(input_data[row][key]) == float:
                        input_data[row][key] = str(input_data[row][key])
                except IndexError:
                    raise ValueError(
                        f"Index: {key} is out of bounds for the input_data"
                    )

            # TODO: Fetching pre-determined value for a new key
            column_name = self.index_column_map[key]
            column_name_map = self.categorical_map[column_name]
            transformed_data[:, key] = np.vectorize(column_name_map.get, otypes=[str])(
                input_data[:, key]
            )
        return transformed_data
