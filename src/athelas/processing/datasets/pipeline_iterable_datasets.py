"""
Streaming dataset implementation using PyTorch's IterableDataset.

This module provides PipelineIterableDataset, a memory-efficient streaming
alternative to PipelineDataset that loads data incrementally from shards.

Key Features:
- Fixed memory usage (loads one shard at a time)
- Multi-GPU/multi-worker support (automatic shard distribution)
- Same pipeline injection API as PipelineDataset
- Drop-in replacement with minimal code changes

Example:
    >>> from processing.datasets.pipeline_iterable_datasets import PipelineIterableDataset
    >>>
    >>> # Create streaming dataset
    >>> dataset = PipelineIterableDataset(
    ...     config=config,
    ...     file_dir="/data/train",  # Directory with part-*.parquet shards
    ... )
    >>>
    >>> # Add pipelines (same API as PipelineDataset)
    >>> dataset.add_pipeline("dialogue", text_pipeline)
    >>> dataset.add_pipeline("customer_id", categorical_pipeline)
    >>>
    >>> # Use with DataLoader (same as regular dataset)
    >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_batch)
"""

import os
import gc
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Iterator
import torch
from torch.utils.data import IterableDataset

from ..processors import Processor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

logger = logging.getLogger(__name__)


class PipelineIterableDataset(IterableDataset):
    """
    Streaming dataset for multimodal input with distributed training support.

    Memory-efficient alternative to PipelineDataset that loads data incrementally
    from multiple shard files. Maintains the same pipeline injection API for
    backward compatibility.

    **Distributed Training Support**:

    The dataset implements two-tier sharding for distributed training:

    1. **Rank-based sharding**: Shards are distributed across GPU ranks using
       round-robin assignment (rank 0 gets shards [0, world_size, 2*world_size, ...]).
       This ensures each GPU processes unique data in FSDP/DDP training.

    2. **Worker-based sharding**: Within each rank, shards are further distributed
       across DataLoader workers for parallel loading.

    **Example Usage**:

    Single GPU training:
        >>> dataset = PipelineIterableDataset(
        ...     config=config,
        ...     file_dir="/data/train",
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)

    Distributed training (FSDP):
        >>> # Same code! Distribution happens automatically
        >>> dataset = PipelineIterableDataset(
        ...     config=config,
        ...     file_dir="/data/train",
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>>
        >>> trainer = pl.Trainer(strategy=FSDPStrategy(), devices=8)
        >>> trainer.fit(model, loader)

    Epoch-aware shuffling:
        >>> for epoch in range(num_epochs):
        ...     dataset.set_epoch(epoch)  # Important for deterministic shuffling
        ...     for batch in loader:
        ...         # Training step

    Attributes:
        config: Configuration dictionary (same as PipelineDataset)
        processor_pipelines: Dictionary mapping field names to Processor pipelines
        shard_files: List of shard file paths to stream through
        shuffle_shards: Whether to shuffle shard order per epoch

    Key Differences from PipelineDataset:
        - Inherits from IterableDataset (not Dataset)
        - Implements __iter__() instead of __getitem__()
        - Loads shards incrementally (not all at once)
        - No __len__() by default (optional estimate available)
        - Automatic multi-GPU and multi-worker shard distribution
    """

    def __init__(
        self,
        config: Dict[str, Union[str, List[str], int]],
        file_dir: Optional[str] = None,
        filename: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        processor_pipelines: Optional[Dict[str, Processor]] = None,
        shard_pattern: str = "part-*.parquet",
        shuffle_shards: bool = False,
    ) -> None:
        """
        Initialize streaming dataset with same config as PipelineDataset.

        Args:
            config: Configuration dictionary containing:
                - label_name: Name of label column
                - text_name: Name of text column (for bimodal)
                - primary_text_name: Name of primary text (for trimodal)
                - secondary_text_name: Name of secondary text (for trimodal)
                - cat_field_list: List of categorical field names
                - tab_field_list: List of numerical field names
                - full_field_list: Complete list of field names
            file_dir: Directory containing shard files
            filename: Optional single file name (for backward compatibility)
            dataframe: Optional DataFrame for direct loading (testing only)
            processor_pipelines: Pre-configured processor pipelines
            shard_pattern: Glob pattern for finding shards (default: "part-*.parquet")
            shuffle_shards: Whether to shuffle shard order (default: False)

        Raises:
            TypeError: If neither file_dir nor dataframe is provided
            FileNotFoundError: If no shards found matching pattern

        Note:
            Data loading performance is optimized via DataLoader's num_workers
            parameter, not dataset-level prefetching. For best performance:
                loader = DataLoader(dataset, num_workers=4, prefetch_factor=2)
        """
        self.config = config
        self.header = config.get("header", 0)
        self.label_name = config.get("label_name")
        self.text_name = config.get("text_name")
        self.primary_text_name = config.get("primary_text_name")
        self.secondary_text_name = config.get("secondary_text_name")
        self.full_field_list = config.get("full_field_list")
        self.cat_field_list = config.get("cat_field_list", [])
        self.tab_field_list = config.get("tab_field_list")
        self.need_language_detect = config.get("need_language_detect", False)
        self.processor_pipelines = processor_pipelines or {}
        self.shuffle_shards = shuffle_shards

        # Find shard files based on input type
        if file_dir:
            self.file_dir = Path(file_dir)

            if filename:
                # Single file mode (backward compatible with PipelineDataset)
                file_path = self.file_dir / filename
                if file_path.exists():
                    self.shard_files = [file_path]
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
            else:
                # Multi-shard mode (new streaming behavior)
                self.shard_files = sorted(self.file_dir.glob(shard_pattern))

                if not self.shard_files:
                    raise FileNotFoundError(
                        f"No shards found in {file_dir} matching pattern '{shard_pattern}'"
                    )

            print(f"[PipelineIterableDataset] Found {len(self.shard_files)} shard(s)")

        elif dataframe is not None and isinstance(dataframe, pd.DataFrame):
            # DataFrame mode (for testing/compatibility)
            self._dataframe_mode = True
            self._temp_df = dataframe
            self.shard_files = []
            print(
                f"[PipelineIterableDataset] Loaded DataFrame with {len(dataframe)} rows"
            )
        else:
            raise TypeError("Must provide either file_dir or dataframe")

        # Initialize missing value handling rules
        self._missing_value_rules = {}

    def _load_shard(self, shard_path: Path) -> pd.DataFrame:
        """
        Load a single shard file.

        Supports Parquet, CSV, and TSV formats based on file extension.

        Args:
            shard_path: Path to shard file

        Returns:
            DataFrame containing shard data

        Raises:
            ValueError: If file format is not supported
        """
        ext = shard_path.suffix.lower()

        if ext == ".parquet":
            return pd.read_parquet(shard_path)
        elif ext == ".csv":
            if self.full_field_list is not None:
                return pd.read_csv(shard_path, header=0, names=self.full_field_list)
            else:
                return pd.read_csv(shard_path, header=self.header)
        elif ext == ".tsv":
            if self.full_field_list is not None:
                return pd.read_csv(
                    shard_path, sep="\t", header=0, names=self.full_field_list
                )
            else:
                return pd.read_csv(shard_path, sep="\t", header=self.header)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply type conversions to DataFrame columns.

        Same logic as PipelineDataset._postprocess_dataframe():
        - Categorical fields → string type, fill "" for missing
        - Numerical fields → numeric type, fill -1.0 for missing

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with converted types
        """
        if self.cat_field_list:
            for col in df.columns:
                if col in self.cat_field_list:
                    df[col] = df[col].astype(str).fillna("")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)

        # Apply missing value rules if set
        if self._missing_value_rules:
            for feature in df.columns:
                if feature == self.label_name:
                    df[feature] = (
                        pd.to_numeric(df[feature], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                elif self.cat_field_list and feature in self.cat_field_list:
                    df[feature] = df[feature].astype(str).fillna("")
                else:
                    df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(
                        -1.0
                    )

        return df

    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate through dataset with proper distributed sharding.

        Implements two-tier sharding:
        1. Rank-based sharding: Distribute shards across GPU ranks (FSDP/DDP)
        2. Worker-based sharding: Distribute shards across DataLoader workers

        Example:
            8 GPUs, 4 workers/GPU, 100 shards:
            - Rank 0 gets shards [0, 8, 16, 24, ..., 96]
            - Rank 0's worker 0 gets [0, 32, 64, 96]
            - Rank 0's worker 1 gets [8, 40, 72]
            - ...
            - Rank 1 gets shards [1, 9, 17, 25, ..., 97]
            - etc.

        Yields:
            Dict: Processed row with applied pipelines
        """
        # ============================================================
        # TIER 1: Rank-Based Sharding (NEW)
        # ============================================================
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Distribute shards across ranks using round-robin
            shards_for_this_rank = self.shard_files[rank::world_size]

            # Log distribution info (only once per rank)
            if not hasattr(self, "_logged_rank_info"):
                print(
                    f"[IterableDataset] Rank {rank}/{world_size}: "
                    f"Assigned {len(shards_for_this_rank)}/{len(self.shard_files)} shards"
                )
                self._logged_rank_info = True
        else:
            # Single GPU mode
            shards_for_this_rank = self.shard_files
            rank = 0
            world_size = 1

        # ============================================================
        # TIER 2: Worker-Based Sharding (ENHANCED)
        # ============================================================
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker: process all shards for this rank
            shards_to_process = shards_for_this_rank
            worker_id = 0
            num_workers = 1
        else:
            # Multiple workers: distribute rank's shards across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards_to_process = shards_for_this_rank[worker_id::num_workers]

            if not hasattr(self, "_logged_worker_info"):
                print(
                    f"[IterableDataset] Rank {rank}, Worker {worker_id}/{num_workers}: "
                    f"Processing {len(shards_to_process)} shards"
                )
                self._logged_worker_info = True

        # ============================================================
        # TIER 3: Shard Shuffling (ENHANCED)
        # ============================================================
        if self.shuffle_shards and shards_to_process:
            shards_list = list(shards_to_process)

            # Deterministic shuffle based on rank + worker + epoch
            # This ensures reproducibility while maintaining randomness
            shuffle_seed = (
                42  # Base seed
                + rank * 10000  # Rank offset
                + worker_id * 100  # Worker offset
                + getattr(self, "_current_epoch", 0)  # Epoch offset (set externally)
            )

            random.Random(shuffle_seed).shuffle(shards_list)
            shards_to_process = shards_list

            if not hasattr(self, "_logged_shuffle_info"):
                print(
                    f"[IterableDataset] Rank {rank}, Worker {worker_id}: "
                    f"Shuffled with seed {shuffle_seed}"
                )
                self._logged_shuffle_info = True

        # Handle DataFrame mode (for testing)
        if hasattr(self, "_dataframe_mode") and self._dataframe_mode:
            df = self._temp_df
            df = self._postprocess_dataframe(df)

            # Split DataFrame across workers if needed
            if num_workers > 1:
                df = df.iloc[worker_id::num_workers]

            # Yield rows
            for idx in range(len(df)):
                row = df.iloc[idx].to_dict()

                # Apply processor pipelines (same as PipelineDataset.__getitem__)
                for field_name, pipeline in self.processor_pipelines.items():
                    if field_name in row:
                        row[field_name] = pipeline(row[field_name])

                yield row

            return

        # ============================================================
        # TIER 4: Sequential Shard Loading
        # ============================================================
        # Note: DataLoader's num_workers provides superior parallel loading.
        # Dataset-level prefetching is redundant and causes thread conflicts.
        yield from self._iterate_sequential(shards_to_process)

    def _iterate_sequential(self, shards_to_process: List[Path]) -> Iterator[Dict]:
        """
        Sequential shard loading (original behavior).

        Used when prefetching is disabled or only one shard to process.

        Args:
            shards_to_process: List of shard paths to load

        Yields:
            Dict: Processed row with applied pipelines
        """
        for shard_idx, shard_path in enumerate(shards_to_process):
            # Load shard
            df = self._load_shard(shard_path)

            # Apply type conversions
            df = self._postprocess_dataframe(df)

            # Yield rows from shard
            for idx in range(len(df)):
                row = df.iloc[idx].to_dict()

                # Apply processor pipelines
                for field_name, pipeline in self.processor_pipelines.items():
                    if field_name in row:
                        row[field_name] = pipeline(row[field_name])

                yield row

            # Free memory after processing shard
            del df
            gc.collect()

    def __len__(self) -> int:
        """
        Return estimated dataset length.

        Note: This is an approximation based on the first shard.
        Actual length may vary if shards have different sizes or
        if data is filtered during processing.

        Returns:
            Estimated total number of rows across all shards
        """
        if hasattr(self, "_dataframe_mode") and self._dataframe_mode:
            return len(self._temp_df)

        if hasattr(self, "_estimated_length"):
            return self._estimated_length

        if self.shard_files:
            # Estimate based on first shard
            try:
                first_shard = self._load_shard(self.shard_files[0])
                rows_per_shard = len(first_shard)
                total_shards = len(self.shard_files)
                self._estimated_length = rows_per_shard * total_shards
                del first_shard
                gc.collect()
                return self._estimated_length
            except Exception:
                # If estimation fails, return unknown
                return 0

        return 0

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for deterministic shuffling.

        Should be called at the start of each epoch, similar to
        DistributedSampler.set_epoch().

        Args:
            epoch: Current epoch number

        Example:
            >>> dataset.set_epoch(epoch)
            >>> for batch in dataloader:
            >>>     # Training step
        """
        self._current_epoch = epoch

    def get_shard_distribution_info(self) -> Dict[str, any]:
        """
        Get diagnostic information about shard distribution.

        Returns:
            Dict containing:
                - total_shards: Total number of shards
                - shards_per_rank: Number of shards assigned to this rank
                - shards_per_worker: Number of shards per worker
                - rank: Current rank
                - world_size: Total number of ranks
                - worker_id: Current worker ID (if available)
                - num_workers: Total number of workers per rank
                - assigned_shards: List of shard files assigned to this rank/worker
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            shards_for_rank = self.shard_files[rank::world_size]
        else:
            rank = 0
            world_size = 1
            shards_for_rank = self.shard_files

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards_for_worker = shards_for_rank[worker_id::num_workers]
        else:
            worker_id = 0
            num_workers = 1
            shards_for_worker = shards_for_rank

        return {
            "total_shards": len(self.shard_files),
            "shards_per_rank": len(shards_for_rank),
            "shards_per_worker": len(shards_for_worker),
            "rank": rank,
            "world_size": world_size,
            "worker_id": worker_id,
            "num_workers": num_workers,
            "assigned_shards": [str(s) for s in shards_for_worker],
        }

    # =========================================================================
    # Pipeline Injection API (IDENTICAL to PipelineDataset)
    # =========================================================================

    def add_pipeline(self, field_name: str, processor_pipeline: Processor) -> None:
        """
        Add a processing pipeline for a specified field.

        IDENTICAL API to PipelineDataset.add_pipeline() for drop-in compatibility.

        The pipeline is built by composing Processors via the >> operator.
        For example:
            pipeline = (HTMLNormalizerProcessor() >>
                       EmojiRemoverProcessor() >>
                       TextNormalizationProcessor() >>
                       DialogueSplitterProcessor() >>
                       DialogueChunkerProcessor(tokenizer, max_tokens=512) >>
                       TokenizationProcessor(tokenizer))

        Args:
            field_name: Name of the field to process
            processor_pipeline: Processor or ComposedProcessor to apply

        Raises:
            TypeError: If arguments are not of expected types
        """
        if isinstance(field_name, str) and isinstance(processor_pipeline, Processor):
            self.processor_pipelines[field_name] = processor_pipeline
        else:
            raise TypeError(
                "Expected str and Processor for field_name and processor_pipeline"
            )

    def fill_missing_value(self, **kwargs) -> None:
        """
        Configure missing value handling rules.

        IDENTICAL API to PipelineDataset.fill_missing_value() for compatibility.

        Note: For IterableDataset, this sets rules to apply during iteration.
        The actual filling happens in _postprocess_dataframe() when each
        shard is loaded.

        Args:
            **kwargs: Configuration updates (label_name, cat_field_list, etc.)
        """
        # Update config values dynamically
        for key, value in kwargs.items():
            if key == "label_name":
                self.label_name = value
            if key == "cat_field_list":
                self.cat_field_list = value

        # Mark that missing value rules are active
        self._missing_value_rules = kwargs

    # =========================================================================
    # Dynamic Setters (IDENTICAL to PipelineDataset)
    # =========================================================================

    def set_text_field_name(self, text_name: Union[str, List[str]]) -> None:
        """Set text field name(s)."""
        if not isinstance(text_name, (str, list)):
            raise TypeError(
                f"Expected str or list for text_name, got {type(text_name)}"
            )
        self.text_name = text_name

    def set_label_field_name(self, label_name: Union[str, List[str]]) -> None:
        """Set label field name(s)."""
        if not isinstance(label_name, (str, list)):
            raise TypeError(
                f"Expected str or list for label_name, got {type(label_name)}"
            )
        self.label_name = label_name

    def set_cat_field_list(self, cat_field_list: List[str]) -> None:
        """Set categorical field list."""
        if not isinstance(cat_field_list, list):
            raise TypeError(
                f"Expected list for cat_field_list, got {type(cat_field_list)}"
            )
        self.cat_field_list = cat_field_list

    def set_full_field_list(self, full_field_list: List[str]) -> None:
        """Set full field list."""
        if not isinstance(full_field_list, list):
            raise TypeError(
                f"Expected list for full_field_list, got {type(full_field_list)}"
            )
        self.full_field_list = full_field_list
