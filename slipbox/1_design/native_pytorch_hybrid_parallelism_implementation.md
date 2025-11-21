---
tags:
  - design
  - implementation
  - hybrid_parallelism
  - distributed_training
  - coding_guide
keywords:
  - PyTorch 2.x
  - Pipeline Parallelism
  - Data Parallelism
  - FSDP
  - Tensor Parallelism
  - 3D Parallelism
  - Multi-dimensional parallelism
topics:
  - implementation details
  - hybrid strategies
  - code examples
language: python
date of note: 2025-11-21
---

# Hybrid Parallelism Implementation Guide

## 1. Overview

This document provides complete implementation details for hybrid parallelism strategies, combining multiple parallelism dimensions (Data, Model, Pipeline, Tensor) for optimal distributed training.

**Related Documents**:
- [Native PyTorch Migration Strategy](./native_pytorch_migration_strategy.md) - Theoretical foundations
- [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md) - Base implementations

## 2. Directory Structure

```text
projects/rnr_pytorch_bedrock/
├── native_models/                 # Pure PyTorch models (from base implementation)
│   ├── __init__.py
│   ├── native_text_bert.py
│   ├── native_tab_ae.py
│   ├── native_trimodal_bert.py
│   └── config_classes.py
├── native_train/                  # Training infrastructure
│   ├── __init__.py
│   ├── hybrid_utils.py           # NEW: Hybrid parallelism utilities
│   ├── train_pp_dp.py            # NEW: Pipeline + Data Parallel
│   ├── train_fsdp_pp.py          # NEW: FSDP + Pipeline Parallel
│   ├── train_tp_dp.py            # NEW: Tensor + Data Parallel (optional)
│   ├── train_3d.py               # NEW: 3D Parallelism (DP+TP+PP)
│   ├── dist_utils.py             # Base distributed utilities
│   └── data_utils.py
└── configs/                       # Model/training configs
    ├── hybrid_pp_dp_8gpu.json    # NEW: PP+DP configuration
    ├── hybrid_fsdp_pp_8gpu.json  # NEW: FSDP+PP configuration
    └── hybrid_3d_16gpu.json      # NEW: 3D parallelism configuration
```

## 3. Hybrid Utilities (`native_train/hybrid_utils.py`)

```python
"""Utilities for hybrid parallelism setups."""
import os
import torch
import torch.distributed as dist
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class HybridConfig:
    """Configuration for hybrid parallelism dimensions."""
    pp_size: int = 1  # Pipeline parallel size
    tp_size: int = 1  # Tensor parallel size
    dp_size: int = 1  # Data parallel size (computed from world_size)
    
    def __post_init__(self):
        """Validate configuration."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        expected_dp = world_size // (self.pp_size * self.tp_size)
        if self.dp_size != expected_dp:
            self.dp_size = expected_dp


class HybridParallelContext:
    """
    Manages process groups and ranks for hybrid parallelism.
    
    Supports 3D parallelism: Data (DP) × Pipeline (PP) × Tensor (TP)
    """
    
    def __init__(self, pp_size: int = 1, tp_size: int = 1):
        """
        Initialize hybrid parallel context.
        
        Args:
            pp_size: Number of pipeline stages
            tp_size: Tensor parallel size (GPUs per layer split)
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.dp_size = self.world_size // (pp_size * tp_size)
        
        # Validate
        if self.world_size != pp_size * tp_size * self.dp_size:
            raise ValueError(
                f"world_size ({self.world_size}) != pp_size ({pp_size}) × "
                f"tp_size ({tp_size}) × dp_size ({self.dp_size})"
            )
        
        # Calculate this rank's position in 3D grid
        self.pp_rank = self._get_pp_rank()
        self.tp_rank = self._get_tp_rank()
        self.dp_rank = self._get_dp_rank()
        
        # Create process groups
        self.dp_group = self._create_dp_group()
        self.pp_group = self._create_pp_group()
        self.tp_group = self._create_tp_group()
        
    def _get_pp_rank(self) -> int:
        """Get this rank's pipeline parallel rank (which stage)."""
        return self.rank // (self.tp_size * self.dp_size)
    
    def _get_tp_rank(self) -> int:
        """Get this rank's tensor parallel rank (which shard)."""
        return (self.rank // self.dp_size) % self.tp_size
    
    def _get_dp_rank(self) -> int:
        """Get this rank's data parallel rank (which replica)."""
        return self.rank % self.dp_size
    
    def _create_dp_group(self) -> dist.ProcessGroup:
        """
        Create data parallel process group.
        
        All ranks with same pp_rank and tp_rank.
        """
        dp_groups = []
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                ranks = [
                    pp_rank * (self.tp_size * self.dp_size) + 
                    tp_rank * self.dp_size + dp_rank
                    for dp_rank in range(self.dp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_dp_group = group
                dp_groups.append(group)
        
        return my_dp_group
    
    def _create_pp_group(self) -> dist.ProcessGroup:
        """
        Create pipeline parallel process group.
        
        All ranks with same tp_rank and dp_rank (across stages).
        """
        pp_groups = []
        for tp_rank in range(self.tp_size):
            for dp_rank in range(self.dp_size):
                ranks = [
                    pp_rank * (self.tp_size * self.dp_size) + 
                    tp_rank * self.dp_size + dp_rank
                    for pp_rank in range(self.pp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_pp_group = group
                pp_groups.append(group)
        
        return my_pp_group
    
    def _create_tp_group(self) -> dist.ProcessGroup:
        """
        Create tensor parallel process group.
        
        All ranks with same pp_rank and dp_rank (across TP shards).
        """
        tp_groups = []
        for pp_rank in range(self.pp_size):
            for dp_rank in range(self.dp_size):
                ranks = [
                    pp_rank * (self.tp_size * self.dp_size) + 
                    tp_rank * self.dp_size + dp_rank
                    for tp_rank in range(self.tp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_tp_group = group
                tp_groups.append(group)
        
        return my_tp_group
    
    def is_pp_first_stage(self) -> bool:
        """Check if this rank is in the first pipeline stage."""
        return self.pp_rank == 0
    
    def is_pp_last_stage(self) -> bool:
        """Check if this rank is in the last pipeline stage."""
        return self.pp_rank == self.pp_size - 1
    
    def get_pp_prev_rank(self) -> Optional[int]:
        """Get the rank of the previous pipeline stage (same TP, DP position)."""
        if self.is_pp_first_stage():
            return None
        return self.rank - (self.tp_size * self.dp_size)
    
    def get_pp_next_rank(self) -> Optional[int]:
        """Get the rank of the next pipeline stage (same TP, DP position)."""
        if self.is_pp_last_stage():
            return None
        return self.rank + (self.tp_size * self.dp_size)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    def is_dp_main_process(self) -> bool:
        """Check if this is the main process in its DP group."""
        return self.dp_rank == 0
    
    def __repr__(self) -> str:
        return (
            f"HybridParallelContext(world_size={self.world_size}, rank={self.rank}, "
            f"pp={self.pp_rank}/{self.pp_size}, tp={self.tp_rank}/{self.tp_size}, "
            f"dp={self.dp_rank}/{self.dp_size})"
        )


def get_pp_dp_grid(world_size: int, pp_size: int) -> Tuple[int, int]:
    """
    Calculate PP and DP dimensions for 2D grid.
    
    Args:
        world_size: Total number of GPUs
        pp_size: Number of pipeline stages
    
    Returns:
        (pp_size, dp_size) tuple
    """
    if world_size % pp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by pp_size ({pp_size})")
    
    dp_size = world_size // pp_size
    return pp_size, dp_size


def setup_hybrid_pp_dp(pp_size: int) -> HybridParallelContext:
    """
    Setup 2D hybrid parallelism (PP + DP).
    
    Args:
        pp_size: Number of pipeline stages
    
    Returns:
        HybridParallelContext with tp_size=1
    """
    return HybridParallelContext(pp_size=pp_size, tp_size=1)


def setup_hybrid_tp_dp(tp_size: int) -> HybridParallelContext:
    """
    Setup 2D hybrid parallelism (TP + DP).
    
    Args:
        tp_size: Tensor parallel size
    
    Returns:
        HybridParallelContext with pp_size=1
    """
    return HybridParallelContext(pp_size=1, tp_size=tp_size)


def setup_3d_parallel(pp_size: int, tp_size: int) -> HybridParallelContext:
    """
    Setup 3D parallelism (DP + PP + TP).
    
    Args:
        pp_size: Number of pipeline stages
        tp_size: Tensor parallel size
    
    Returns:
        HybridParallelContext with all dimensions
    """
    return HybridParallelContext(pp_size=pp_size, tp_size=tp_size)
```

## 4. Pipeline + Data Parallel (`native_train/train_pp_dp.py`)

```python
"""
Training with Pipeline Parallelism + Data Parallelism.

Example: 8 GPUs = 2 PP stages × 4 DP replicas per stage
"""
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
from .hybrid_utils import setup_hybrid_pp_dp
from .dist_utils import setup_distributed, cleanup_distributed


def split_model_pp_dp(model: NativeTrimodalBert, pp_size: int, pp_rank: int):
    """
    Split model for pipeline parallelism.
    
    For pp_size=2:
        Stage 0: Primary BERT
        Stage 1: Secondary BERT + Tabular + Fusion
    """
    if pp_size == 2:
        if pp_rank == 0:
            # Stage 0: Primary text branch
            return model.primary_text_subnetwork
        else:
            # Stage 1: Everything else
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
            
            return Stage1Module(
                model.secondary_text_subnetwork,
                model.tab_subnetwork,
                model.final_merge_network,
                model.config
            )
    else:
        raise NotImplementedError(f"pp_size={pp_size} not implemented")


def train_epoch_pp_dp(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hybrid_ctx,
    num_microbatches: int = 4
):
    """
    Train one epoch with PP + DP.
    
    Note: This is a simplified skeleton. Production should use
    torch.distributed.pipelining with proper communication.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Split batch into microbatches for pipeline
        microbatch_losses = []
        
        for micro_idx in range(num_microbatches):
            # Create microbatch
            microbatch = extract_microbatch(batch, micro_idx, num_microbatches)
            
            # Forward through pipeline stages
            if hybrid_ctx.is_pp_first_stage():
                # Stage 0: Process and send to next stage
                output = model(microbatch)
                # Send to next stage via torch.distributed.send
                # (Simplified - real implementation needs proper P2P)
                
            elif hybrid_ctx.is_pp_last_stage():
                # Last stage: Receive, process, compute loss
                # Receive from prev stage via torch.distributed.recv
                # (Simplified)
                output = model((microbatch, received_input))
                
                # Compute loss
                labels = microbatch[model.module.config.label_name]
                loss = model.module.compute_loss(output, labels)
                microbatch_losses.append(loss)
        
        # Backward pass
        if hybrid_ctx.is_pp_last_stage():
            total_micro_loss = sum(microbatch_losses) / len(microbatch_losses)
            
            # DDP will handle gradient synchronization within DP group
            optimizer.zero_grad()
            total_micro_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_micro_loss.item()
    
    return total_loss / len(dataloader)


def extract_microbatch(batch, micro_idx, num_microbatches):
    """Extract a microbatch slice from the full batch."""
    batch_size = len(batch[list(batch.keys())[0]])
    microbatch_size = batch_size // num_microbatches
    start_idx = micro_idx * microbatch_size
    end_idx = start_idx + microbatch_size
    
    return {
        k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def main_pp_dp(args):
    """Main training function for PP + DP."""
    # Setup distributed
    setup_distributed()
    
    # Setup hybrid context
    hybrid_ctx = setup_hybrid_pp_dp(pp_size=args.pp_size)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    if hybrid_ctx.is_main_process():
        print(f"PP+DP Training: {hybrid_ctx}")
        print(f"  PP stages: {hybrid_ctx.pp_size}")
        print(f"  DP replicas per stage: {hybrid_ctx.dp_size}")
        print(f"  Total throughput: {hybrid_ctx.dp_size}× batch size")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrimodalBertConfig(**config_dict)
    
    # Create model and get this rank's stage
    model = NativeTrimodalBert(config)
    model_stage = split_model_pp_dp(model, hybrid_ctx.pp_size, hybrid_ctx.pp_rank)
    model_stage = model_stage.to(device)
    
    # Wrap with DDP using DP group
    model_stage = DDP(
        model_stage,
        process_group=hybrid_ctx.dp_group,
        device_ids=[torch.cuda.current_device()]
    )
    
    # Create optimizer for this stage
    optimizer = AdamW(
        model_stage.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Create dataloader with DistributedSampler for DP dimension
    train_dataset = create_dataset(args.data_path, config)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=hybrid_ctx.dp_size,
        rank=hybrid_ctx.dp_rank,
        shuffle=True
    )
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
        avg_loss = train_epoch_pp_dp(
            model_stage, train_loader, optimizer, scheduler,
            hybrid_ctx, args.num_microbatches
        )
        
        if hybrid_ctx.is_main_process():
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint (only from main process)
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_stage.module.state_dict(),
                }, checkpoint_path)
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--pp_size", type=int, default=2, help="Number of pipeline stages")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()
    
    main_pp_dp(args)
```

**Launch Command**:
```bash
# 8 GPUs: 2 PP stages × 4 DP replicas
torchrun --nproc_per_node=8 native_train/train_pp_dp.py \
    --config configs/hybrid_pp_dp_8gpu.json \
    --data_path /path/to/data \
    --pp_size 2 \
    --batch_size 32 \
    --num_microbatches 4
```

## 5. FSDP + Pipeline Parallel (`native_train/train_fsdp_pp.py`)

```python
"""
Training with FSDP + Pipeline Parallelism.

Each pipeline stage uses FSDP for parameter sharding.
Example: 8 GPUs = 2 PP stages × FSDP(4 GPUs) per stage
"""
import torch
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bert.modeling_bert import BertLayer
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import json

from native_models.native_trimodal_bert import NativeTrimodalBert
from native_models.config_classes import TrimodalBertConfig
from .hybrid_utils import setup_hybrid_pp_dp
from .dist_utils import setup_distributed, cleanup_distributed


def get_fsdp_config_for_hybrid(hybrid_ctx):
    """
    Get FSDP configuration for hybrid setup.
    
    FSDP operates within each DP group of a pipeline stage.
    """
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BertLayer},
    )
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    return {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "process_group": hybrid_ctx.dp_group,  # FSDP within DP group
        "device_id": torch.cuda.current_device(),
    }


def main_fsdp_pp(args):
    """Main training function for FSDP + PP."""
    # Setup distributed
    setup_distributed()
    
    # Setup hybrid context
    hybrid_ctx = setup_hybrid_pp_dp(pp_size=args.pp_size)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    if hybrid_ctx.is_main_process():
        print(f"FSDP+PP Training: {hybrid_ctx}")
        print(f"  PP stages: {hybrid_ctx.pp_size}")
        print(f"  FSDP shards per stage: {hybrid_ctx.dp_size}")
        print(f"  Memory efficiency: ~{hybrid_ctx.dp_size}× reduction per stage")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrimodalBertConfig(**config_dict)
    
    # Create model and get this rank's stage
    model = NativeTrimodalBert(config)
    model_stage = split_model_pp_dp(model, hybrid_ctx.pp_size, hybrid_ctx.pp_rank)
    
    # Wrap with FSDP using DP group
    fsdp_config = get_fsdp_config_for_hybrid(hybrid_ctx)
    model_stage = FSDP(model_stage, **fsdp_config)
    
    # Create optimizer
    optimizer = AdamW(
        model_stage.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Create dataloader with DistributedSampler
    train_dataset = create_dataset(args.data_path, config)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=hybrid_ctx.dp_size,
        rank=hybrid_ctx.dp_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop (similar to PP+DP but with FSDP)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        # Training logic here
        pass
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    main_fsdp_pp(args)
```

**Launch Command**:
```bash
# 8 GPUs: 2 PP stages × FSDP(4 shards) per stage
torchrun --nproc_per_node=8 native_train/train_fsdp_pp.py \
    --config configs/hybrid_fsdp_pp_8gpu.json \
    --data_path /path/to/data \
    --pp_size 2 \
    --batch_size 16
```

## 6. Configuration Examples

### 6.1 PP+DP Configuration (`configs/hybrid_pp_dp_8gpu.json`)

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
  "warmup_steps": 100,
  "_comments": {
    "hybrid_config": "8 GPUs = 2 PP stages × 4 DP replicas",
    "throughput": "4× compared to single GPU",
    "memory_per_gpu": "Full stage + optimizer state"
  }
}
```

### 6.2 FSDP+PP Configuration (`configs/hybrid_fsdp_pp_8gpu.json`)

```json
{
  "label_name": "label",
  "primary_text_name": "dialogue",
  "secondary_text_name": "shiptrack",
  "tab_field_list": ["feature_1", "feature_2", "feature_3"],
  "primary_tokenizer": "bert-large-cased",
  "secondary_tokenizer": "bert-large-cased",
  "hidden_common_dim": 512,
  "fusion_hidden_dim": 256,
  "fusion_dropout": 0.1,
  "is_binary": true,
  "num_classes": 2,
  "class_weights": [1.0, 2.0],
  "lr": 1e-5,
  "weight_decay": 0.01,
  "warmup_steps": 200,
  "_comments": {
    "hybrid_config": "8 GPUs = 2 PP stages × FSDP(4 shards) per stage",
    "model_variant": "bert-large for larger capacity",
    "memory_efficiency": "~4× reduction per stage via FSDP"
  }
}
```

### 6.3 3D Parallelism Configuration (`configs/hybrid_3d_16gpu.json`)

```json
{
  "label_name": "label",
  "primary_text_name": "dialogue",
  "secondary_text_name": "shiptrack",
  "tab_field_list": ["feature_1", "feature_2", "feature_3"],
  "primary_tokenizer": "roberta-large",
  "secondary_tokenizer": "roberta-large",
  "hidden_common_dim": 1024,
  "fusion_hidden_dim": 512,
  "fusion_dropout": 0.1,
  "is_binary": true,
  "num_classes": 2,
  "class_weights": [1.0, 2.0],
  "lr": 5e-6,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "_comments": {
    "hybrid_config": "16 GPUs = 4 PP × 2 TP × 2 DP",
    "model_variant": "roberta-large with custom layers",
    "use_case": "Extreme model (>1B params)"
  }
}
```

## 7. Deployment Recommendations

### 7.1 Small-Scale (8 GPUs)

| Model Size | Setup | Config | Throughput | Memory/GPU |
|------------|-------|--------|------------|------------|
| bert-base (~220M) | PP=2 × DP=4 | `hybrid_pp_dp_8gpu.json` | 4× | Medium |
| bert-large (~680M) | PP=2 × FSDP=4 | `hybrid_fsdp_pp_8gpu.json` | 1-2× | Low |
| roberta-large (>1B) | PP=4 × FSDP=2 | Custom | ~1× | Very Low |

**Recommendation**: For bert-base, prefer pure DDP for simplicity. Use PP+DP only when testing pipeline infrastructure or preparing for larger scales.

### 7.2 Medium-Scale (16 GPUs)

| Model Size | Setup | Config | Throughput | Memory/GPU |
|------------|-------|--------|------------|------------|
| bert-base | PP=2 × DP=8 | `hybrid_pp_dp_16gpu.json` | 8× | Medium |
| bert-large | PP=2 × DP=8 or PP=4 × DP=4 | Custom | 4-8× | Medium |
| roberta-large (>1B) | PP=4 × FSDP=4 | `hybrid_fsdp_pp_16gpu.json` | 2-4× | Low |
| Extreme (>2B) | PP=4 × TP=2 × DP=2 | `hybrid_3d_16gpu.json` | 1-2× | Very Low |

**Recommendation**: PP=2 × DP=8 provides excellent throughput scaling for medium models. For very large models, add TP or use FSDP+PP.

### 7.3 Large-Scale (32+ GPUs)

| Model Size | Setup | Config | Throughput | Memory/GPU |
|------------|-------|--------|------------|------------|
| bert-large | PP=2 × DP=16 | Custom | 16× | Medium |
| roberta-large | PP=4 × DP=8 or PP=4 × FSDP=8 | Custom | 8× | Low-Medium |
| Extreme (5B-10B) | PP=8 × TP=2 × DP=2 | Custom 3D | 2-4× | Very Low |
| Extreme (>10B) | PP=8 × TP=4 × DP=1-2 | Custom 3D | 1-2× | Minimal |

**Recommendation**: At this scale, carefully profile to find optimal PP/TP/DP balance. Communication becomes critical—use NVLink/InfiniBand.

## 8. Performance Tuning Guidelines

### 8.1 Process Group Optimization

```python
# Use NCCL backend for GPU communication
dist.init_process_group(backend="nccl", init_method="env://")

# For FSDP, consider limiting all-gathers
fsdp_config = {
    "limit_all_gathers": True,  # Reduce memory spikes
    "use_orig_params": False,   # Enable parameter flattening
}

# For Pipeline, tune microbatch count
# Rule of thumb: num_microbatches = pp_size * 2 to 4
num_microbatches = pp_size * 4
```

### 8.2 Memory Optimization

```python
# Gradient checkpointing for BERT layers
from torch.utils.checkpoint import checkpoint

class CheckpointedBertLayer(nn.Module):
    def forward(self, hidden_states, attention_mask):
        return checkpoint(
            self.original_forward,
            hidden_states,
            attention_mask,
            use_reentrant=False
        )

# Mixed precision with BF16 (preferred) or FP16
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
```

### 8.3 Communication Efficiency

**DP Dimension**:
- Use DDP's bucket system (default 25MB buckets)
- Set `find_unused_parameters=False` unless necessary
- Consider `gradient_as_bucket_view=True` to save memory

**PP Dimension**:
- Maximize microbatches to reduce pipeline bubbles
- Use 1F1B schedule instead of GPipe
- Profile stage compute times—balance is critical

**TP Dimension**:
- Requires fast inter-GPU communication (NVLink)
- Minimize TP size unless layer width demands it
- Consider sequence parallelism for transformers

### 8.4 Profiling and Debugging

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3),
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        # Training step
        prof.step()

# Analyze
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("trace.json")
```

## 9. Summary and Best Practices

### 9.1 Key Takeaways

1. **Hybrid parallelism is unnecessary for small models** (<500M params) on 2-8 GPUs—use pure DDP.

2. **PP + DP is effective** when:
   - Model doesn't fit on single GPU
   - You want throughput scaling via DP
   - PP_size is small (2-4) to minimize bubbles

3. **FSDP + PP is powerful** when:
   - Model is very large (>1B params)
   - Memory is the primary constraint
   - Each pipeline stage can benefit from sharding

4. **3D parallelism** is only needed for:
   - Models >10B parameters
   - 16+ GPUs with fast interconnects
   - When layer width demands TP

5. **Process group management** is critical:
   - Each parallelism dimension needs its own process group
   - Communication happens within groups only
   - Incorrect groups cause deadlocks or wrong results

### 9.2 Implementation Checklist

- [ ] Start with pure DDP baseline
- [ ] Profile memory and throughput
- [ ] Identify bottleneck (memory vs compute vs communication)
- [ ] Choose hybrid strategy based on bottleneck
- [ ] Implement `HybridParallelContext` for process groups
- [ ] Split model appropriately for PP dimension
- [ ] Wrap model stages with DDP or FSDP for DP dimension
- [ ] Create DistributedSampler using DP dimension
- [ ] Test with small microbatch counts first
- [ ] Profile and tune microbatch size, PP stages, DP replicas
- [ ] Monitor GPU utilization, memory, and communication overhead
- [ ] Scale up gradually and re-tune at each scale

### 9.3 Common Pitfalls

**Process Groups**:
- ❌ Using wrong process group for DDP/FSDP
- ✅ Always pass correct `process_group` parameter

**Pipeline Parallelism**:
- ❌ Unbalanced stages (one stage much slower)
- ✅ Profile per-stage compute time and balance

**FSDP**:
- ❌ Forgetting to use `use_orig_params=False` for optimizer compatibility
- ✅ Follow official FSDP configuration patterns

**Memory**:
- ❌ Assuming linear memory scaling
- ✅ Account for activation memory, gradient accumulation

**Communication**:
- ❌ Ignoring network topology
- ✅ Use NVLink-aware placement for TP dimension

### 9.4 Future Enhancements

For production deployments, consider:

1. **Sequence Parallelism**: Split sequence dimension for very long inputs
2. **Zero Redundancy Optimizer (ZeRO)**: Use ZeRO-3 with DeepSpeed
3. **FlashAttention**: Reduce memory for attention layers
4. **Async Pipeline**: Overlap communication and computation
5. **Expert Parallelism**: For mixture-of-experts models
6. **Activation Offloading**: CPU offload for extreme memory pressure

### 9.5 Reference Resources

**PyTorch Docs**:
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [torch.distributed.pipelining](https://pytorch.org/docs/stable/distributed.pipelining.html)

**Papers**:
- GPipe: [Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- ZeRO: [Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- Megatron-LM: [Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)

**Frameworks**:
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Advanced ZeRO implementation
- [FairScale](https://github.com/facebookresearch/fairscale): Pipeline and FSDP utilities
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): Tensor + Pipeline parallelism reference

## 10. Related Documents

- [Native PyTorch Migration Strategy](./native_pytorch_migration_strategy.md) - Theoretical foundations and comparison
- [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md) - Base DDP/FSDP/PP implementations
