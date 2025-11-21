---
tags:
  - design
  - implementation
  - native_pytorch
  - coding_guide
  - distributed_training
keywords:
  - PyTorch 2.x
  - DistributedDataParallel
  - FullyShardedDataParallel
  - Pipeline Parallelism
  - Trimodal BERT
  - code implementation
  - training scripts
  - NCCL
topics:
  - implementation details
  - distributed system coding
  - code examples
language: python
date of note: 2025-11-21
---

# Native PyTorch Implementation Plan

## 1. Overview

This document provides the technical implementation details for the strategy outlined in the [Native PyTorch Migration Strategy](./native_pytorch_migration_strategy.md). It includes complete code examples for model refactoring, training loops, and distributed configurations.

**Related Documents**:
- [Native PyTorch Migration Strategy](./native_pytorch_migration_strategy.md) - Theoretical foundations and distributed training mechanisms
- [Native PyTorch Hybrid Parallelism Implementation](./native_pytorch_hybrid_parallelism_implementation.md) - Advanced hybrid strategies (PP+DP, FSDP+PP, 3D parallelism)

## 2. Directory Structure

```text
projects/rnr_pytorch_bedrock/
├── docker/
│   └── lightning_models/          # Existing Lightning models
├── native_models/                 # New: Pure PyTorch models
│   ├── __init__.py
│   ├── native_text_bert.py        # Refactored TextBertBase
│   ├── native_tab_ae.py           # Refactored TabAE
│   ├── native_trimodal_bert.py    # Composite Model
│   └── config_classes.py          # Pydantic configs
├── native_train/                  # New: Training infrastructure
│   ├── __init__.py
│   ├── train_unified.py           # Main entry point (all strategies)
│   ├── train_ddp.py               # DDP-specific training
│   ├── train_fsdp.py              # FSDP-specific training
│   ├── train_pipeline.py          # Pipeline-specific training
│   ├── dist_utils.py              # Distributed helpers
│   └── data_utils.py              # DataLoader creation
└── configs/                       # Model/training configs
    ├── trimodal_base.json
    └── trimodal_large.json
```

## 3. Model Implementation

### 3.1 Configuration Classes (`native_models/config_classes.py`)

```python
"""Configuration classes using Pydantic for type safety."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class TextBertConfig(BaseModel):
    """Configuration for text BERT subnetwork."""
    text_name: str
    tokenizer: str = "bert-base-cased"
    hidden_common_dim: int = 256
    reinit_pooler: bool = False
    reinit_layers: int = 0
    text_input_ids_key: str = "input_ids"
    text_attention_mask_key: str = "attention_mask"
    dropout: float = 0.1


class TabularConfig(BaseModel):
    """Configuration for tabular subnetwork."""
    tab_field_list: List[str]
    hidden_common_dim: int = 256
    dropout: float = 0.1

    @field_validator("tab_field_list")
    @classmethod
    def validate_nonempty(cls, v: List[str], info: ValidationInfo) -> List[str]:
        if not v:
            raise ValueError("tab_field_list must not be empty")
        return v


class TrimodalBertConfig(BaseModel):
    """Configuration for trimodal BERT model."""
    # Data fields
    id_name: Optional[str] = None
    label_name: str
    primary_text_name: str
    secondary_text_name: str
    tab_field_list: Optional[List[str]] = None

    # Model architecture
    primary_tokenizer: str = "bert-base-cased"
    secondary_tokenizer: str = "bert-base-cased"
    hidden_common_dim: int = 256
    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.1

    # Task settings
    is_binary: bool = True
    num_classes: int = 2
    class_weights: List[float] = Field(default_factory=lambda: [1.0, 1.0])

    # Training
    lr: float = 2e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0

    @field_validator("num_classes")
    @classmethod
    def validate_num_classes(cls, value: int, info: ValidationInfo) -> int:
        if info.data.get("is_binary") and value != 2:
            raise ValueError("Binary classification requires num_classes=2")
        return value
```

### 3.2 Native Text BERT (`native_models/native_text_bert.py`)

```python
"""Pure PyTorch implementation of Text BERT subnetwork."""
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict

from .config_classes import TextBertConfig


class NativeTextBert(nn.Module):
    """Text BERT encoder with pooling and projection."""

    def __init__(self, config: TextBertConfig):
        super().__init__()
        self.config = config
        
        # Keys for accessing batch data
        self.text_input_ids_key = f"{config.text_name}_{config.text_input_ids_key}"
        self.text_attention_mask_key = f"{config.text_name}_{config.text_attention_mask_key}"
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(
            config.tokenizer,
            output_attentions=False
        )
        self._maybe_reinitialize()
        
        # Projection head
        self.output_bert_dim = self.bert.config.hidden_size
        self.output_text_dim = config.hidden_common_dim
        
        self.head_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.output_bert_dim, self.output_text_dim),
        )

    def _maybe_reinitialize(self):
        """Reinitialize pooler and/or top layers if configured."""
        if not self.config.reinit_pooler:
            return
            
        encoder = self.bert
        # Reinitialize pooler
        encoder.pooler.dense.weight.data.normal_(
            mean=0.0, std=encoder.config.initializer_range
        )
        encoder.pooler.dense.bias.data.zero_()
        
        # Reinitialize top N layers
        if self.config.reinit_layers > 0:
            for layer in encoder.encoder.layer[-self.config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(
                            mean=0.0, std=encoder.config.initializer_range
                        )
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through BERT and projection.
        
        Args:
            batch: Dict containing input_ids and attention_mask
                   Shape: [B, C, T] where C is num chunks
        
        Returns:
            Pooled and projected embeddings [B, hidden_common_dim]
        """
        input_ids = batch[self.text_input_ids_key]
        attention_mask = batch[self.text_attention_mask_key]
        
        # Reshape from [B, C, T] to [B*C, T]
        B, C, T = input_ids.shape
        input_ids = input_ids.view(B * C, T)
        attention_mask = attention_mask.view(B * C, T)
        
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B*C, hidden_size]
        
        # Average across chunks
        pooled = pooled.view(B, C, -1).mean(dim=1)  # [B, hidden_size]
        
        # Project to common dimension
        logits = self.head_layer(pooled)  # [B, hidden_common_dim]
        return logits
```

### 3.3 Native Tabular AE (`native_models/native_tab_ae.py`)

```python
"""Pure PyTorch implementation of Tabular Autoencoder."""
import torch
import torch.nn as nn
from typing import Dict, Union, List

from .config_classes import TabularConfig


class NativeTabAE(nn.Module):
    """Tabular embedding module with normalization."""

    def __init__(self, config: TabularConfig):
        super().__init__()
        self.config = config
        self.tab_field_list = config.tab_field_list
        
        input_dim = len(config.tab_field_list)
        hidden_dim = config.hidden_common_dim
        
        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_tab_dim = hidden_dim

    def combine_tab_data(
        self, batch: Dict[str, Union[torch.Tensor, List]]
    ) -> torch.Tensor:
        """
        Combine tabular fields into single tensor.
        
        Args:
            batch: Dict with tabular field values
        
        Returns:
            Combined tensor [B, input_tab_dim]
        """
        features = []
        device = next(self.parameters()).device
        
        for field in self.tab_field_list:
            if field not in batch:
                raise KeyError(f"Missing field '{field}' in batch")
            
            val = batch[field]
            if isinstance(val, list):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            elif isinstance(val, torch.Tensor):
                val = val.to(dtype=torch.float32, device=device)
            else:
                raise TypeError(f"Unsupported type for {field}: {type(val)}")
            
            if val.dim() == 1:
                val = val.unsqueeze(1)
            features.append(val)
        
        return torch.cat(features, dim=1)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            inputs: Either combined tensor or dict of fields
        
        Returns:
            Embedded features [B, hidden_common_dim]
        """
        if isinstance(inputs, dict):
            inputs = self.combine_tab_data(inputs)
        
        return self.embedding_layer(inputs)
```

### 3.4 Native Trimodal BERT (`native_models/native_trimodal_bert.py`)

```python
"""Pure PyTorch implementation of Trimodal BERT."""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .config_classes import TrimodalBertConfig, TextBertConfig, TabularConfig
from .native_text_bert import NativeTextBert
from .native_tab_ae import NativeTabAE


class NativeTrimodalBert(nn.Module):
    """Trimodal BERT with primary text, secondary text, and tabular branches."""

    def __init__(self, config: TrimodalBertConfig):
        super().__init__()
        self.config = config
        
        # Build subnetworks
        self.primary_text_subnetwork = self._build_text_subnetwork("primary")
        self.secondary_text_subnetwork = self._build_text_subnetwork("secondary")
        self.tab_subnetwork = self._build_tabular_subnetwork()
        
        # Fusion head
        primary_dim = self.primary_text_subnetwork.output_text_dim
        secondary_dim = self.secondary_text_subnetwork.output_text_dim
        tab_dim = self.tab_subnetwork.output_tab_dim if self.tab_subnetwork else 0
        
        total_dim = primary_dim + secondary_dim + tab_dim
        fusion_hidden_dim = config.fusion_hidden_dim
        
        self.final_merge_network = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(total_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(fusion_hidden_dim, config.num_classes),
        )
        
        # Loss function
        weights = torch.tensor(config.class_weights[:config.num_classes], dtype=torch.float)
        self.register_buffer("class_weights", weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def _build_text_subnetwork(self, text_type: str) -> NativeTextBert:
        """Build text subnetwork config."""
        if text_type == "primary":
            text_name = self.config.primary_text_name
            tokenizer = self.config.primary_tokenizer
        else:
            text_name = self.config.secondary_text_name
            tokenizer = self.config.secondary_tokenizer
        
        text_config = TextBertConfig(
            text_name=text_name,
            tokenizer=tokenizer,
            hidden_common_dim=self.config.hidden_common_dim,
        )
        return NativeTextBert(text_config)

    def _build_tabular_subnetwork(self) -> Optional[NativeTabAE]:
        """Build tabular subnetwork if configured."""
        if not self.config.tab_field_list:
            return None
        
        tab_config = TabularConfig(
            tab_field_list=self.config.tab_field_list,
            hidden_common_dim=self.config.hidden_common_dim,
        )
        return NativeTabAE(tab_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through all branches and fusion.
        
        Args:
            batch: Dict with all input data
        
        Returns:
            Logits [B, num_classes]
        """
        # Process each branch
        primary_out = self.primary_text_subnetwork(batch)
        secondary_out = self.secondary_text_subnetwork(batch)
        
        if self.tab_subnetwork is not None:
            tab_out = self.tab_subnetwork(batch)
        else:
            device = primary_out.device
            tab_out = torch.zeros((primary_out.size(0), 0), device=device)
        
        # Concatenate and fuse
        combined = torch.cat([primary_out, secondary_out, tab_out], dim=1)
        logits = self.final_merge_network(combined)
        return logits

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        if labels.dtype != torch.long:
            if labels.dim() > 1:  # One-hot encoded
                labels = labels.argmax(dim=1)
            labels = labels.long()
        return self.loss_fn(logits, labels)
```

## 4. Distributed Training Utilities (`native_train/dist_utils.py`)

```python
"""Distributed training utilities."""
import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed process group."""
    if not dist.is_initialized():
        # Get rank and world_size from environment (set by torchrun)
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
    
    return get_rank(), get_local_rank(), get_world_size()


def get_rank():
    """Get global rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    """Get local rank (GPU ID on current node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
```

## 5. DDP Training Script (`native_train/train_ddp.py`)

**See**: [DDP Theory](./native_pytorch_migration_strategy.md#31-distributed-data-parallel-ddp) for mechanism explanation

```python
"""DDP training implementation."""
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
import argparse
import json
from pathlib import Path

from native_models.native_trimodal_bert import NativeTrimodalBert
from native_models.config_classes import TrimodalBertConfig
from .dist_utils import (
    setup_distributed, cleanup_distributed, get_local_rank, is_main_process
)


def train_epoch_ddp(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
):
    """Train for one epoch with DDP."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        labels = batch[model.module.config.label_name]
        loss = model.module.compute_loss(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if is_main_process() and batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main_ddp(args):
    """Main DDP training function."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f"DDP Training on {world_size} GPUs")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrimodalBertConfig(**config_dict)
    
    # Create model
    model = NativeTrimodalBert(config).to(device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # Set True if some params unused
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )
    
    # Create dataloader with DistributedSampler
    # (Assume we have a function to create dataset)
    train_dataset = create_dataset(args.data_path, config)  # User-defined
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        avg_loss = train_epoch_ddp(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        if is_main_process():
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()
    
    main_ddp(args)
```

**Launch Command**:
```bash
torchrun --nproc_per_node=4 native_train/train_ddp.py \
    --config configs/trimodal_base.json \
    --data_path /path/to/data \
    --batch_size 32 \
    --epochs 10
```

## 6. FSDP Training Script (`native_train/train_fsdp.py`)

```python
"""FSDP training implementation."""
import torch
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers.models.bert.modeling_bert import BertLayer
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
import argparse
import json

from native_models.native_trimodal_bert import NativeTrimodalBert
from native_models.config_classes import TrimodalBertConfig
from .dist_utils import (
    setup_distributed, cleanup_distributed, get_local_rank, is_main_process
)


def get_fsdp_config():
    """Get FSDP configuration."""
    # Auto-wrap policy: wrap transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BertLayer},  # Wrap each BERT layer
    )
    
    # Mixed precision config
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,  # Parameters in BF16
        reduce_dtype=torch.bfloat16,  # Gradient reduction in BF16
        buffer_dtype=torch.bfloat16,  # Buffers in BF16
    )
    
    return {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO-3
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,  # Memory optimization
    }


def train_epoch_fsdp(
    model: FSDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
):
    """Train for one epoch with FSDP."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        labels = batch[model.module.config.label_name]
        loss = model.module.compute_loss(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (FSDP requires special handling)
        model.clip_grad_norm_(max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if is_main_process() and batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main_fsdp(args):
    """Main FSDP training function."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f"FSDP Training on {world_size} GPUs")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrimodalBertConfig(**config_dict)
    
    # Create model
    model = NativeTrimodalBert(config)
    
    # Wrap with FSDP
    fsdp_config = get_fsdp_config()
    model = FSDP(model, **fsdp_config)
    
    # Create optimizer (on sharded params)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )
    
    # Create dataloader
    train_dataset = create_dataset(args.data_path, config)
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = train_epoch_fsdp(model, train_loader, optimizer, scheduler, device, epoch)
        
        if is_main_process():
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint with FSDP state_dict API
            if (epoch + 1) % args.save_every == 0:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                    state_dict = model.state_dict()
                
                if is_main_process():
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                    }, f"checkpoint_epoch_{epoch}.pt")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()
    
    main_fsdp(args)
```

**Launch Command**:
```bash
torchrun --nproc_per_node=4 native_train/train_fsdp.py \
    --config configs/trimodal_large.json \
    --data_path /path/to/data \
    --batch_size 16 \
    --epochs 10
```

## 7. Pipeline Parallelism Training (`native_train/train_pipeline.py`)

```python
"""Pipeline parallelism using manual stage assignment."""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List
import argparse
import json

from native_models.native_trimodal_bert import NativeTrimodalBert
from native_models.config_classes import TrimodalBertConfig
from .dist_utils import setup_distributed, cleanup_distributed, get_local_rank, get_rank


class PipelineStage(nn.Module):
    """Wrapper for a pipeline stage."""
    
    def __init__(self, submodule: nn.Module, device: torch.device):
        super().__init__()
        self.submodule = submodule.to(device)
        self.device = device
    
    def forward(self, x):
        if isinstance(x, dict):
            x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return self.submodule(x)


def split_model_for_pipeline(model: NativeTrimodalBert, num_stages: int) -> List[nn.Module]:
    """
    Split trimodal model into pipeline stages.
    
    For 2 GPUs:
        Stage 0: Primary Text BERT
        Stage 1: Secondary Text BERT + Tabular + Fusion
    
    For 4 GPUs:
        Stage 0: Primary Text BERT (layers 0-5)
        Stage 1: Primary Text BERT (layers 6-11)
        Stage 2: Secondary Text BERT (layers 0-11)
        Stage 3: Tabular + Fusion
    
    For 8 GPUs:
        Stage 0: Primary BERT (layers 0-2)
        Stage 1: Primary BERT (layers 3-5)
        Stage 2: Primary BERT (layers 6-8)
        Stage 3: Primary BERT (layers 9-11) + Projection
        Stage 4: Secondary BERT (layers 0-2)
        Stage 5: Secondary BERT (layers 3-5)
        Stage 6: Secondary BERT (layers 6-8)
        Stage 7: Secondary BERT (layers 9-11) + Projection + Tabular + Fusion
    """
    if num_stages == 2:
        # Stage 0: Primary branch
        stage0 = model.primary_text_subnetwork
        
        # Stage 1: Everything else (needs custom wrapper)
        class Stage1Module(nn.Module):
            def __init__(self, secondary, tabular, fusion, config):
                super().__init__()
                self.secondary = secondary
                self.tabular = tabular
                self.fusion = fusion
                self.config = config
            
            def forward(self, batch_and_primary):
                batch, primary_out = batch_and_primary
                secondary_out = self.secondary(batch)
                
                if self.tabular:
                    tab_out = self.tabular(batch)
                else:
                    tab_out = torch.zeros((primary_out.size(0), 0), 
                                         device=primary_out.device)
                
                combined = torch.cat([primary_out, secondary_out, tab_out], dim=1)
                return self.fusion(combined)
        
        stage1 = Stage1Module(
            model.secondary_text_subnetwork,
            model.tab_subnetwork,
            model.final_merge_network,
            model.config
        )
        return [stage0, stage1]
    
    elif num_stages == 4:
        # For 4-GPU setup: Split primary BERT into 2 stages, keep secondary whole
        primary_bert = model.primary_text_subnetwork.bert
        primary_layers = primary_bert.encoder.layer
        
        # Stage 0: Primary BERT layers 0-5
        class PrimaryStage0(nn.Module):
            def __init__(self, embeddings, layers_0_5, config):
                super().__init__()
                self.embeddings = embeddings
                self.layers = nn.ModuleList(layers_0_5)
                self.config = config
            
            def forward(self, batch):
                input_ids = batch[f"{config.primary_text_name}_input_ids"]
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                B, C, T = input_ids.shape
                input_ids = input_ids.view(B * C, T)
                attention_mask = attention_mask.view(B * C, T)
                
                hidden_states = self.embeddings(input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, hidden_states
        
        stage0 = PrimaryStage0(
            primary_bert.embeddings,
            primary_layers[:6],
            model.config
        )
        
        # Stage 1: Primary BERT layers 6-11 + pooler + projection
        class PrimaryStage1(nn.Module):
            def __init__(self, layers_6_11, pooler, head_layer, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_6_11)
                self.pooler = pooler
                self.head_layer = head_layer
                self.config = config
            
            def forward(self, batch_and_hidden):
                batch, hidden_states = batch_and_hidden
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                pooled = self.pooler(hidden_states)
                B, C = pooled.shape[0], batch['B']  # Recover batch structure
                pooled = pooled.view(B, C, -1).mean(dim=1)
                output = self.head_layer(pooled)
                
                return batch, output
        
        stage1 = PrimaryStage1(
            primary_layers[6:],
            primary_bert.pooler,
            model.primary_text_subnetwork.head_layer,
            model.config
        )
        
        # Stage 2: Secondary BERT (full)
        class SecondaryStage(nn.Module):
            def __init__(self, secondary_subnet):
                super().__init__()
                self.secondary_subnet = secondary_subnet
            
            def forward(self, batch_and_primary):
                batch, primary_out = batch_and_primary
                secondary_out = self.secondary_subnet(batch)
                return batch, primary_out, secondary_out
        
        stage2 = SecondaryStage(model.secondary_text_subnetwork)
        
        # Stage 3: Tabular + Fusion
        class FusionStage(nn.Module):
            def __init__(self, tabular, fusion, config):
                super().__init__()
                self.tabular = tabular
                self.fusion = fusion
                self.config = config
            
            def forward(self, batch_primary_secondary):
                batch, primary_out, secondary_out = batch_primary_secondary
                
                if self.tabular:
                    tab_out = self.tabular(batch)
                else:
                    tab_out = torch.zeros((primary_out.size(0), 0), 
                                         device=primary_out.device)
                
                combined = torch.cat([primary_out, secondary_out, tab_out], dim=1)
                return self.fusion(combined)
        
        stage3 = FusionStage(
            model.tab_subnetwork,
            model.final_merge_network,
            model.config
        )
        
        return [stage0, stage1, stage2, stage3]
    
    elif num_stages == 8:
        # For 8-GPU setup: Split both BERTs into 4 stages each (3 layers per stage)
        primary_bert = model.primary_text_subnetwork.bert
        primary_layers = primary_bert.encoder.layer
        secondary_bert = model.secondary_text_subnetwork.bert
        secondary_layers = secondary_bert.encoder.layer
        
        # Stage 0: Primary BERT layers 0-2
        class PrimaryStage0(nn.Module):
            def __init__(self, embeddings, layers_0_2, config):
                super().__init__()
                self.embeddings = embeddings
                self.layers = nn.ModuleList(layers_0_2)
                self.config = config
            
            def forward(self, batch):
                input_ids = batch[f"{config.primary_text_name}_input_ids"]
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                B, C, T = input_ids.shape
                input_ids = input_ids.view(B * C, T)
                attention_mask = attention_mask.view(B * C, T)
                
                hidden_states = self.embeddings(input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, hidden_states
        
        # Stage 1: Primary BERT layers 3-5
        class PrimaryStage1(nn.Module):
            def __init__(self, layers_3_5, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_3_5)
                self.config = config
            
            def forward(self, batch_and_hidden):
                batch, hidden_states = batch_and_hidden
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, hidden_states
        
        # Stage 2: Primary BERT layers 6-8
        class PrimaryStage2(nn.Module):
            def __init__(self, layers_6_8, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_6_8)
                self.config = config
            
            def forward(self, batch_and_hidden):
                batch, hidden_states = batch_and_hidden
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, hidden_states
        
        # Stage 3: Primary BERT layers 9-11 + pooler + projection
        class PrimaryStage3(nn.Module):
            def __init__(self, layers_9_11, pooler, head_layer, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_9_11)
                self.pooler = pooler
                self.head_layer = head_layer
                self.config = config
            
            def forward(self, batch_and_hidden):
                batch, hidden_states = batch_and_hidden
                attention_mask = batch[f"{config.primary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                pooled = self.pooler(hidden_states)
                B, C = pooled.shape[0], batch['B']
                pooled = pooled.view(B, C, -1).mean(dim=1)
                primary_out = self.head_layer(pooled)
                
                return batch, primary_out
        
        # Stage 4: Secondary BERT layers 0-2
        class SecondaryStage0(nn.Module):
            def __init__(self, embeddings, layers_0_2, config):
                super().__init__()
                self.embeddings = embeddings
                self.layers = nn.ModuleList(layers_0_2)
                self.config = config
            
            def forward(self, batch_and_primary):
                batch, primary_out = batch_and_primary
                input_ids = batch[f"{config.secondary_text_name}_input_ids"]
                attention_mask = batch[f"{config.secondary_text_name}_attention_mask"]
                
                B, C, T = input_ids.shape
                input_ids = input_ids.view(B * C, T)
                attention_mask = attention_mask.view(B * C, T)
                
                hidden_states = self.embeddings(input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, primary_out, hidden_states
        
        # Stage 5: Secondary BERT layers 3-5
        class SecondaryStage1(nn.Module):
            def __init__(self, layers_3_5, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_3_5)
                self.config = config
            
            def forward(self, batch_primary_hidden):
                batch, primary_out, hidden_states = batch_primary_hidden
                attention_mask = batch[f"{config.secondary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, primary_out, hidden_states
        
        # Stage 6: Secondary BERT layers 6-8
        class SecondaryStage2(nn.Module):
            def __init__(self, layers_6_8, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_6_8)
                self.config = config
            
            def forward(self, batch_primary_hidden):
                batch, primary_out, hidden_states = batch_primary_hidden
                attention_mask = batch[f"{config.secondary_text_name}_attention_mask"]
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                return batch, primary_out, hidden_states
        
        # Stage 7: Secondary BERT layers 9-11 + pooler + projection + Tabular + Fusion
        class SecondaryStage3AndFusion(nn.Module):
            def __init__(self, layers_9_11, pooler, head_layer, tabular, fusion, config):
                super().__init__()
                self.layers = nn.ModuleList(layers_9_11)
                self.pooler = pooler
                self.head_layer = head_layer
                self.tabular = tabular
                self.fusion = fusion
                self.config = config
            
            def forward(self, batch_primary_hidden):
                batch, primary_out, hidden_states = batch_primary_hidden
                attention_mask = batch[f"{config.secondary_text_name}_attention_mask"]
                
                # Complete secondary BERT
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]
                
                pooled = self.pooler(hidden_states)
                B, C = pooled.shape[0], batch['B']
                pooled = pooled.view(B, C, -1).mean(dim=1)
                secondary_out = self.head_layer(pooled)
                
                # Process tabular
                if self.tabular:
                    tab_out = self.tabular(batch)
                else:
                    tab_out = torch.zeros((primary_out.size(0), 0), device=primary_out.device)
                
                # Fusion
                combined = torch.cat([primary_out, secondary_out, tab_out], dim=1)
                return self.fusion(combined)
        
        # Create all 8 stages
        stage0 = PrimaryStage0(primary_bert.embeddings, primary_layers[:3], model.config)
        stage1 = PrimaryStage1(primary_layers[3:6], model.config)
        stage2 = PrimaryStage2(primary_layers[6:9], model.config)
        stage3 = PrimaryStage3(
            primary_layers[9:],
            primary_bert.pooler,
            model.primary_text_subnetwork.head_layer,
            model.config
        )
        stage4 = SecondaryStage0(secondary_bert.embeddings, secondary_layers[:3], model.config)
        stage5 = SecondaryStage1(secondary_layers[3:6], model.config)
        stage6 = SecondaryStage2(secondary_layers[6:9], model.config)
        stage7 = SecondaryStage3AndFusion(
            secondary_layers[9:],
            secondary_bert.pooler,
            model.secondary_text_subnetwork.head_layer,
            model.tab_subnetwork,
            model.final_merge_network,
            model.config
        )
        
        return [stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7]
    
    else:
        raise NotImplementedError(f"Pipeline splitting for {num_stages} stages not implemented")


def train_epoch_pipeline(
    stages: List[PipelineStage],
    dataloader: DataLoader,
    optimizers: List[torch.optim.Optimizer],
    rank: int,
    num_microbatches: int = 4,
):
    """
    Train one epoch with simple pipeline parallelism (GPipe-style).
    
    This is a simplified implementation. For production, use torch.distributed.pipelining.
    """
    for stage in stages:
        stage.train()
    
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Split batch into microbatches
        batch_size = len(batch[list(batch.keys())[0]])
        microbatch_size = batch_size // num_microbatches
        
        microbatch_losses = []
        
        for micro_idx in range(num_microbatches):
            start_idx = micro_idx * microbatch_size
            end_idx = start_idx + microbatch_size
            
            # Create microbatch
            microbatch = {
                k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass through stages
            if rank == 0:
                # Stage 0: Process primary text
                with torch.no_grad() if micro_idx < num_microbatches - 1 else torch.enable_grad():
                    output = stages[0](microbatch)
                # Send to stage 1
                # (In real implementation, use torch.distributed.send)
                
            elif rank == 1:
                # Stage 1: Receive from stage 0, process rest
                # (In real implementation, use torch.distributed.recv)
                with torch.enable_grad():
                    output = stages[0]((microbatch, received_primary_output))
                
                # Compute loss
                labels = microbatch[stages[0].config.label_name]
                loss = nn.CrossEntropyLoss()(output, labels)
                microbatch_losses.append(loss)
        
        # Backward pass (accumulate gradients)
        if rank == 1:
            total_micro_loss = sum(microbatch_losses) / len(microbatch_losses)
            total_micro_loss.backward()
            
            # Gradient step
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            
            total_loss += total_micro_loss.item()
    
    return total_loss / len(dataloader)


def main_pipeline(args):
    """Main pipeline parallelism training."""
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Pipeline training on {world_size} GPUs")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrimodalBertConfig(**config_dict)
    
    # Create full model (each rank creates it, then splits)
    model = NativeTrimodalBert(config)
    
    # Split into stages
    raw_stages = split_model_for_pipeline(model, world_size)
    
    # Assign stage to this rank
    my_stage = PipelineStage(raw_stages[rank], device)
    
    # Create optimizer for this stage
    optimizer = AdamW(
        my_stage.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # NOTE: This is a simplified skeleton.
    # Production implementation should use torch.distributed.pipelining
    # with proper communication primitives
    
    print(f"[Rank {rank}] Stage {rank} initialized with {sum(p.numel() for p in my_stage.parameters())} parameters")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_microbatches", type=int, default=4)
    args = parser.parse_args()
    
    main_pipeline(args)
```

**Note**: The above is a simplified skeleton. For production Pipeline Parallelism, use:
- `torch.distributed.pipelining` (PyTorch 2.4+), OR
- `fairscale.nn.Pipe`, OR
- Manual implementation with `torch.distributed.send/recv`

**Launch Command**:
```bash
torchrun --nproc_per_node=2 native_train/train_pipeline.py \
    --config configs/trimodal_base.json \
    --data_path /path/to/data \
    --num_microbatches 4
```

## 8. Configuration Examples

### 8.1 Model Config (`configs/trimodal_base.json`)

```json
{
  "label_name": "label",
  "primary_text_name": "dialogue",
  "secondary_text_name": "shiptrack",
  "tab_field_list": ["feature_1", "feature_2", "feature_3"],
  "primary_tokenizer": "bert-base-cased",
  "secondary_tokenizer": "bert-base-cased",
  "hidden_common_dim": 256,
  "fusion_hidden_dim": 128,
  "fusion_dropout": 0.1,
  "is_binary": true,
  "num_classes": 2,
  "class_weights": [1.0, 2.0],
  "lr": 2e-5,
  "weight_decay": 0.01,
  "warmup_steps": 100
}
```

## 9. Summary

This implementation plan provides:

1.  **Pure PyTorch Models**: Fully decoupled from Lightning, using standard `nn.Module`.
2.  **DDP Support**: Simple replication for moderate-scale training.
3.  **FSDP Support**: Memory-efficient sharding for large models.
4.  **Pipeline Parallelism**: Stage-based splitting (skeleton provided).
5.  **Complete Code**: Ready-to-use training scripts with distributed utilities.

For full Pipeline Parallelism with 1F1B schedules and communication optimization, integrate `torch.distributed.pipelining` APIs available in PyTorch 2.4+.
