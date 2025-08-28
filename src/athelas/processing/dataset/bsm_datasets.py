import os
import re
import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
from typing import Union, List, Tuple, Dict, Set, Pattern, Optional
from collections.abc import Callable
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch

# We no longer import Pipeline; instead we use our Processor and ComposedProcessor classes.
from .processors import Processor  # Base Processor classes include __rshift__ operator and ComposedProcessor


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')



class BSMDataset(Dataset):
    """
    Custom dataset for multimodal input supporting text, tabular, categorical,
    and Parquet/CSV/TSV file formats with per-column processing pipelines.
    """
    def __init__(self,
                 config: Dict[str, Union[str, List[str], int]],
                 file_dir: Optional[str] = None,
                 filename: Optional[str] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 processor_pipelines: Optional[Dict[str, Processor]] = None) -> None:
        """
        Args:
            config: Dict containing keys like 'text_name', 'label_name', etc.
            file_dir: Directory where data file is located.
            filename: Name of the data file (TSV, CSV, or Parquet).
            dataframe: Optional raw DataFrame for direct loading.
            processor_pipelines: Dictionary mapping field names to processing pipelines.
        """
        self.config = config
        self.header = config.get('header', 0)
        self.label_name = config.get('label_name')
        self.text_name = config.get('text_name')
        self.full_field_list = config.get('full_field_list')
        self.cat_field_list = config.get('cat_field_list', [])
        self.tab_field_list = config.get('tab_field_list')
        self.need_language_detect = config.get('need_language_detect', False)
        self.processor_pipelines = processor_pipelines or {}
        self.DataReader = None

        # If file is provided, load it based on extension
        if file_dir and filename:
            self.file_dir = file_dir
            self.filename = filename
            self.processed_filename = os.path.splitext(filename)[0] + "-processed" + os.path.splitext(filename)[1]
            self.ext = os.path.splitext(filename)[1].lower()
            self.load_data()
        elif dataframe is not None and isinstance(dataframe, pd.DataFrame):
            self.load_dataframe(dataframe)
        else:
            raise TypeError("Must provide either a file_dir + filename, or a dataframe.")

    def load_data(self, **kwargs):
        """
        Load data from file (CSV, TSV, or Parquet).
        """
        file_path = os.path.join(self.file_dir, self.filename)

        # Load Parquet
        if self.ext == '.parquet':
            print(f"Loading Parquet file: {self.filename}")
            self.DataReader = pd.read_parquet(file_path)

        # Load CSV or TSV
        elif self.ext in ['.csv', '.tsv']:
            sep = ',' if self.ext == '.csv' else '\t'
            print(f"Loading {self.ext.upper()} file: {self.filename} with sep='{sep}'")

            # Use custom field list if provided
            if self.full_field_list is None and 'full_field_list' not in kwargs:
                self.DataReader = pd.read_csv(file_path, sep=sep, header=self.header)
            else:
                fields = kwargs.get('full_field_list', self.full_field_list)
                self.DataReader = pd.read_csv(file_path, sep=sep, header=0, names=fields)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")

        self._postprocess_dataframe()

    def load_dataframe(self, dataframe):
        """
        Load data directly from a provided DataFrame.
        """
        if self.full_field_list and len(self.full_field_list) == len(dataframe.columns):
            dataframe.columns = self.full_field_list
        self.DataReader = dataframe.copy()
        self._postprocess_dataframe()

    def _postprocess_dataframe(self):
        """
        Convert types of each column based on its role (categorical vs numeric).
        """
        if self.cat_field_list:
            for col in self.DataReader.columns:
                if col in self.cat_field_list:
                    self.DataReader[col] = self.DataReader[col].astype(str).fillna('')
                else:
                    self.DataReader[col] = pd.to_numeric(self.DataReader[col], errors='coerce').fillna(-1.0)

    def fill_missing_value(self, **kwargs):
        """
        Ensure all fields are numeric or categorical with default fill values.
        """
        if self.DataReader is None:
            return

        # Update config values dynamically
        for key, value in kwargs.items():
            if key == 'label_name':
                self.label_name = value
            if key == 'cat_field_list':
                self.cat_field_list = value

        for feature in self.DataReader.columns:
            if self.label_name and feature == self.label_name:
                self.DataReader[feature] = pd.to_numeric(self.DataReader[feature], errors='coerce').fillna(0).astype(int)
            elif self.cat_field_list and feature in self.cat_field_list:
                self.DataReader[feature] = self.DataReader[feature].astype(str).fillna('')
            else:
                self.DataReader[feature] = pd.to_numeric(self.DataReader[feature], errors='coerce').fillna(-1.0)

    def add_pipeline(self, field_name: str, processor_pipeline: Processor):
        """
        Adds a processing pipeline for a specified field.
        The pipeline is built by composing Processors via the >> operator.
        For example, for the text field 'dialogue', you might have:
            pipeline = (HTMLNormalizerProcessor() >> EmojiRemoverProcessor() >>
                        TextNormalizationProcessor() >> DialogueSplitterProcessor() >>
                        DialogueChunkerProcessor(tokenizer, max_tokens=512) >>
                        TokenizationProcessor(tokenizer, add_special_tokens=True))
        which you then add via:
            dataset.add_pipeline("dialogue", pipeline)
        """
        if isinstance(field_name, str) and isinstance(processor_pipeline, Processor):
            self.processor_pipelines[field_name] = processor_pipeline
        else:
            raise TypeError("Expected str and Processor for field_name and processor_pipeline")

    def __getitem__(self, idx):
        """
        Fetch one row and apply processing pipelines (if defined).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.DataReader.iloc[idx].to_dict()

        # Apply all processors for each relevant field
        for field_name, pipeline in self.processor_pipelines.items():
            if field_name in row:
                processed_result = pipeline(row[field_name])
                row[field_name + '_processed'] = processed_result
                
            # Delete the original text field *after* processing
            if self.text_name and self.text_name == field_name:
                try:
                    del row[field_name]
                except KeyError:
                    self.logger.warning(f"Text field '{field_name}' not found in row. Skipping deletion.")
                    
        return row

    def __len__(self):
        """
        Returns number of samples.
        """
        return len(self.DataReader)

    # === Dynamic setters for modifying config ===
    def set_text_field_name(self, text_name: Union[str, List[str]]):
        if not isinstance(text_name, (str, list)):
            raise TypeError(f"Expected str or list for text_name, got {type(text_name)}")
        self.text_name = text_name

    def set_label_field_name(self, label_name: Union[str, List[str]]):
        if not isinstance(label_name, (str, list)):
            raise TypeError(f"Expected str or list for label_name, got {type(label_name)}")
        self.label_name = label_name

    def set_cat_field_list(self, cat_field_list: List[str]):
        if not isinstance(cat_field_list, list):
            raise TypeError(f"Expected list for cat_field_list, got {type(cat_field_list)}")
        self.cat_field_list = cat_field_list

    def set_full_field_list(self, full_field_list: List[str]):
        if not isinstance(full_field_list, list):
            raise TypeError(f"Expected list for full_field_list, got {type(full_field_list)}")
        self.full_field_list = full_field_list
