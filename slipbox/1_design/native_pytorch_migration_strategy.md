---
tags:
  - design
  - native_pytorch
  - ddp
  - fsdp
  - pipeline_parallelism
  - tensor_parallelism
  - distributed_training
keywords:
  - PyTorch 2.x
  - DistributedDataParallel
  - FullyShardedDataParallel
  - Pipeline Parallelism
  - Trimodal BERT
  - collective communication
  - ZeRO
topics:
  - model migration
  - distributed system design
  - training strategy
language: python
date of note: 2025-11-21
---

# Native PyTorch Migration Strategy: DDP, FSDP, and Pipeline Parallelism

## 1. Executive Summary

This document outlines the strategy for migrating the Trimodal BERT model from PyTorch Lightning (`pl.LightningModule`) to native PyTorch 2+. The primary objective is to gain fine-grained control over distributed training strategies, specifically enabling:

1.  **Distributed Data Parallel (DDP)** for standard scaling.
2.  **Fully Sharded Data Parallel (FSDP)** for large-model efficiency.
3.  **Pipeline Parallelism (PP)** for splitting the model across multiple GPUs (specifically 2-4 GPUs).
4.  **Tensor Parallelism (TP)** for splitting individual layers (future-proofing).

## 2. Model Architecture: Trimodal BERT

### 2.1 Current Lightning Implementation

The existing `TrimodalBert` model in `projects/rnr_pytorch_bedrock/docker/lightning_models/trimodal/pl_trimodal_bert.py` consists of:

1.  **Primary Text Branch** (`TextBertBase`):
    *   Pre-trained BERT encoder (12 layers, 768 hidden dim for `bert-base-cased`).
    *   Pooling layer extracting `[CLS]` token.
    *   Linear projection to `hidden_common_dim`.
    *   Processes dialogue/chat text.

2.  **Secondary Text Branch** (`TextBertBase`):
    *   Independent BERT encoder (same architecture).
    *   Processes shiptrack or other auxiliary text.

3.  **Tabular Branch** (`TabAE`):
    *   Simple MLP: `LayerNorm -> Linear(input_dim, hidden_common_dim) -> ReLU`.
    *   Processes numerical features.

4.  **Fusion Head**:
    *   Concatenates outputs: `[primary_dim + secondary_dim + tab_dim]`.
    *   Two-layer MLP with dropout.
    *   Final classification layer: `Linear(fusion_hidden_dim, num_classes)`.

### 2.2 Training Orchestration (Current)

*   Uses `pl.Trainer` with FSDP strategy configured via `strategy="fsdp"`.
*   Lightning handles:
    *   Distributed process group initialization.
    *   Automatic gradient synchronization.
    *   Optimizer state sharding.
    *   Checkpointing and logging.

### 2.3 Refactoring Goals

Remove Lightning dependencies to enable:
*   Direct control over distributed wrappers (DDP, FSDP, Pipeline).
*   Custom training loops with explicit forward/backward passes.
*   Fine-grained profiling and optimization.
*   Easier integration with non-Lightning tooling.

**See**: [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md) for complete code examples of the refactored models.

## 3. Distributed Training Theory

### 3.1 Distributed Data Parallel (DDP)

#### 3.1.1 Mechanism

DDP replicates the entire model on each GPU and splits the input batch across processes:

1.  **Initialization**:
    *   `torch.distributed.init_process_group("nccl", ...)` creates a process group.
    *   Each process loads a full copy of the model.
    *   Model is wrapped: `model = DDP(model, device_ids=[local_rank])`.

2.  **Forward Pass**:
    *   Each process receives a different slice of the global batch.
    *   Forward pass is independent (no communication).

3.  **Backward Pass**:
    *   Gradients are computed locally.
    *   DDP registers hooks on parameters to trigger all-reduce.
    *   **All-Reduce**: Uses ring-based or tree-based algorithms to average gradients across all processes.
    *   Result: All processes have identical averaged gradients.

4.  **Optimizer Step**:
    *   Each process updates its local copy with the averaged gradients.
    *   Models remain synchronized.

#### 3.1.2 Components

*   **Process Group**: Logical grouping of processes (ranks).
*   **Backend**: `nccl` (GPU), `gloo` (CPU).
*   **Bucket Gradient Aggregation**: DDP groups gradients into buckets (~25MB) for efficient all-reduce.

#### 3.1.3 Use Case

*   Model fits comfortably on single GPU.
*   Goal: Increase effective batch size by distributing data.
*   Memory overhead: Full model + optimizer state per GPU.

### 3.2 Fully Sharded Data Parallel (FSDP)

#### 3.2.1 Mechanism

FSDP shards model parameters, gradients, and optimizer states across GPUs. It's inspired by ZeRO-3 (DeepSpeed).

**Key Idea**: Each GPU only holds a fraction (1/N) of the model parameters permanently. During forward/backward, layers are temporarily "unsharded" (all-gathered) as needed, then immediately discarded.

1.  **Initialization**:
    *   Model is wrapped recursively: `FSDP(model, auto_wrap_policy=...)`.
    *   Parameters are sharded across the process group.
    *   Each rank holds `params[rank::world_size]`.

2.  **Forward Pass (Layer-wise)**:
    *   **Pre-Forward All-Gather**: Before computing a layer, all-gather its full parameters from all ranks.
    *   **Compute**: Execute the layer with full parameters.
    *   **Discard**: Free the full parameters (keep only local shard).

3.  **Backward Pass (Layer-wise)**:
    *   **Pre-Backward All-Gather**: Re-gather full parameters for gradient computation.
    *   **Compute Gradients**: Compute gradients w.r.t. full parameters.
    *   **Reduce-Scatter**: Average gradients across ranks and scatter so each rank gets its shard.
    *   **Discard**: Free full gradients (keep only local gradient shard).

4.  **Optimizer Step**:
    *   Each rank updates its parameter shard.
    *   Optimizer states are also sharded (1/N memory per rank).

#### 3.2.2 Components

*   **Sharding Strategy**: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (DDP-like).
*   **Auto-Wrap Policy**: Determines which submodules to wrap. Common policies:
    *   `transformer_auto_wrap_policy`: Wraps transformer layers (e.g., `BertLayer`).
    *   `size_based_auto_wrap_policy`: Wraps modules exceeding a parameter count threshold.
*   **Mixed Precision**: Uses `MixedPrecision` object to specify compute/reduce dtypes (FP16/BF16).
*   **CPU Offload**: Optionally offload parameters/gradients to CPU.

#### 3.2.3 Use Case

*   Model is too large for single GPU (even with gradient checkpointing).
*   Goal: Maximize model capacity while maintaining memory efficiency.
*   Memory overhead: ~1/N of model + optimizer state per GPU.
*   Trade-off: Increased communication (all-gather/reduce-scatter) vs. memory savings.

### 3.3 Pipeline Parallelism (PP)

#### 3.3.1 Mechanism

Pipeline Parallelism splits the model into sequential stages, each assigned to a different GPU. Input batches are divided into microbatches that flow through stages in a pipelined fashion.

**Key Idea**: While GPU 0 processes microbatch 2, GPU 1 can simultaneously process microbatch 1 (which finished on GPU 0). This overlaps computation across stages.

1.  **Model Partitioning**:
    *   Model is divided into stages: `Stage0 -> Stage1 -> ... -> StageN`.
    *   Each stage is a sequential subset of layers.
    *   Stage `i` is assigned to GPU `i`.

2.  **Forward Pass (GPipe Schedule)**:
    *   Split batch into M microbatches.
    *   GPU 0 processes all M microbatches sequentially, sending outputs to GPU 1.
    *   GPU 1 waits for first microbatch from GPU 0, processes it, sends to GPU 2, etc.
    *   Results in a "fill-up" phase where pipeline gradually fills.

3.  **Backward Pass**:
    *   Once all microbatches complete forward on final stage, backward begins.
    *   Gradients flow in reverse: Stage N -> Stage N-1 -> ... -> Stage 0.
    *   Each stage accumulates gradients from all microbatches.

4.  **1F1B Schedule (Interleaved)**:
    *   More efficient than GPipe: interleaves forward and backward passes.
    *   After a stage completes forward for a microbatch, it immediately does backward for a previous microbatch.
    *   Reduces memory (fewer activations stored) and improves GPU utilization.

#### 3.3.2 Components

*   **PipelineStage**: Wrapper around a stage's submodule + device assignment.
*   **Schedule**: Defines the execution order (GPipe, 1F1B, Interleaved 1F1B).
*   **Microbatches**: Number of splits. More microbatches = better pipeline efficiency but higher memory for activations.
*   **Activation Checkpointing**: Reduce memory by recomputing activations during backward.

#### 3.3.3 Use Case for Trimodal BERT (2-4 GPUs)

**2-GPU Setup**:
*   **Stage 0 (GPU 0)**: Primary Text BERT (12 layers, ~110M params).
*   **Stage 1 (GPU 1)**: Secondary Text BERT + Tabular + Fusion Head (~110M + small).
*   Issue: Imbalanced stages. GPU 1 has more computation.

**4-GPU Setup (Balanced)**:
*   **Stage 0 (GPU 0)**: Primary Text BERT Layers 1-6.
*   **Stage 1 (GPU 1)**: Primary Text BERT Layers 7-12.
*   **Stage 2 (GPU 2)**: Secondary Text BERT Layers 1-12.
*   **Stage 3 (GPU 3)**: Tabular + Fusion Head + Loss.
*   Better balance, but Stage 2 is still heavy. Could split Secondary BERT similarly.

**Alternative: Branch Parallelism**:
*   Since Primary and Secondary branches are independent until fusion, we can use a "DAG-style" pipeline:
    *   GPU 0: Primary Text BERT (all layers).
    *   GPU 1: Secondary Text BERT (all layers).
    *   GPU 0 or 1: Tabular + Fusion (after gathering outputs).

#### 3.3.4 Trade-offs

*   **Pros**: Enables training models that don't fit on single GPU due to layer depth.
*   **Cons**:
    *   Pipeline bubbles (idle time during fill-up and drain phases).
    *   Increased latency per step.
    *   Complex to balance stages.
    *   Requires careful tuning of microbatch count.

### 3.4 Tensor Parallelism (TP)

#### 3.4.1 Mechanism

TP splits individual layers (e.g., matrix multiplications) across GPUs. For a linear layer `Y = XW`, the weight matrix `W` is partitioned column-wise or row-wise across devices.

**Column-Wise Parallelism**:
*   Split `W` into `[W0, W1, ..., Wn]` along columns.
*   Each GPU holds `Wi` and computes `Yi = X * Wi`.
*   Final output: `Y = [Y0, Y1, ..., Yn]` (concatenated).
*   All-gather required to reconstruct full output for next layer.

**Row-Wise Parallelism**:
*   Split `W` along rows: `W = [W0; W1; ...; Wn]`.
*   Each GPU computes partial result with local rows.
*   All-reduce required to sum partial outputs.

#### 3.4.2 Components

*   **DeviceMesh**: Defines the topology of devices (e.g., 2D mesh for TP + DP).
*   **DTensor (Distributed Tensor)**: Abstraction for tensors sharded across devices.
*   **Parallelization Plan**: Specifies which layers use colwise/rowwise/etc.

#### 3.4.3 Use Case

*   Extremely large layers (e.g., LLaMA with 8192 hidden dim).
*   Goal: Reduce per-GPU memory for individual layer weights.
*   Trade-off: High communication overhead (all-gather/all-reduce per layer).
*   Best combined with other strategies (e.g., TP + FSDP).

## 4. Hybrid Parallelism: Combining Strategies

### 4.1 Overview

**Hybrid Parallelism** combines multiple parallelism strategies to leverage their complementary strengths. This approach is essential for scaling to very large models and GPU clusters.

**Key Insight**: Different parallelism dimensions address different bottlenecks:
- **Data Parallelism (DP/DDP)**: Scales throughput via batch distribution
- **Model Parallelism (PP/TP/FSDP)**: Scales model capacity when memory-constrained
- **Hybrid**: Combines both for optimal resource utilization

### 4.2 Common Hybrid Patterns

#### 4.2.1 Pipeline + Data Parallel (PP + DP)

**Configuration**: 2D partitioning across pipeline stages and data replicas.

```text
Example: 8 GPUs with 2 pipeline stages
┌─────────────────────┬─────────────────────┐
│  Stage 0 Replicas   │  Stage 1 Replicas   │
├─────────────────────┼─────────────────────┤
│ GPU 0: Stage0 Rep0  │ GPU 4: Stage1 Rep0  │
│ GPU 1: Stage0 Rep1  │ GPU 5: Stage1 Rep1  │
│ GPU 2: Stage0 Rep2  │ GPU 6: Stage1 Rep2  │
│ GPU 3: Stage0 Rep3  │ GPU 7: Stage1 Rep3  │
└─────────────────────┴─────────────────────┘
```

**Mechanism**:
1. Model is split into pipeline stages (e.g., 2 stages).
2. Each stage is replicated across multiple GPUs (e.g., 4 replicas).
3. **Within stage**: DDP synchronizes gradients across replicas.
4. **Across stages**: Pipeline communication for activations/gradients.

**Process Groups**:
- **DP Group**: All GPUs with same pipeline stage (e.g., `[0,1,2,3]` for Stage 0).
- **PP Group**: GPUs with same replica ID across stages (e.g., `[0,4]` for Replica 0).

**Benefits**:
- Scales throughput by N_replicas × batch_size.
- Each pipeline stage can handle larger intermediate activations.
- Efficient for models with modest depth but large batch requirements.

**Trade-offs**:
- Pipeline bubbles still exist (reduced by more microbatches).
- Memory per GPU: Full stage + optimizer state.
- Communication: All-reduce within DP group + P2P between stages.

#### 4.2.2 Tensor + Data Parallel (TP + DP)

**Configuration**: Layers split via TP, model replicated via DP.

```text
Example: 8 GPUs with TP=2, DP=4
┌──────────────────────────────────────┐
│        DP Replica 0                  │
│  ┌──────────────┬──────────────┐     │
│  │ GPU 0: TP 0  │ GPU 1: TP 1  │     │
│  └──────────────┴──────────────┘     │
├──────────────────────────────────────┤
│        DP Replica 1                  │
│  ┌──────────────┬──────────────┐     │
│  │ GPU 2: TP 0  │ GPU 3: TP 1  │     │
│  └──────────────┴──────────────┘     │
├──────────────────────────────────────┤
│  (Replicas 2 and 3 similarly)        │
└──────────────────────────────────────┘
```

**Mechanism**:
1. Large layers (e.g., attention, FFN) split column/row-wise via TP.
2. Entire model+TP group replicated via DP.
3. **Within TP group**: All-reduce/all-gather for tensor ops.
4. **Across DP replicas**: All-reduce for gradient synchronization.

**Benefits**:
- Handles extremely wide layers (e.g., 8192-dim embeddings).
- Scales throughput via DP dimension.
- Memory per GPU: `Model_size / TP_size`.

**Trade-offs**:
- High communication within TP group (every layer).
- Requires fast inter-GPU communication (NVLink preferred).
- Complex to implement and debug.

#### 4.2.3 FSDP + Pipeline Parallel (FSDP + PP)

**Configuration**: Parameter sharding within each pipeline stage.

```text
Example: 4 GPUs with 2 pipeline stages, FSDP within each
┌────────────────────┬────────────────────┐
│  Stage 0 (FSDP)    │  Stage 1 (FSDP)    │
├────────────────────┼────────────────────┤
│ GPU 0: Shard 0/2   │ GPU 2: Shard 0/2   │
│ GPU 1: Shard 1/2   │ GPU 3: Shard 1/2   │
└────────────────────┴────────────────────┘
```

**Mechanism**:
1. Model split into pipeline stages.
2. Each stage's parameters sharded via FSDP across assigned GPUs.
3. **Within stage**: FSDP all-gather/reduce-scatter.
4. **Across stages**: Pipeline communication.

**Benefits**:
- Extreme memory efficiency: `Model_size / (PP_size × FSDP_size)`.
- Can train very deep models with limited memory.

**Trade-offs**:
- Complex communication patterns.
- Pipeline bubbles + FSDP communication overhead.
- Requires careful tuning of both PP and FSDP configurations.

#### 4.2.4 3D Parallelism (DP + TP + PP)

**Configuration**: Partitioning across all three dimensions (used for 100B+ models).

```text
Example: 64 GPUs = 4 PP × 4 TP × 4 DP
┌─────────────────────────────────────────────┐
│ Pipeline Stage 0 (GPUs 0-15)                │
│  ┌────────────────────────────────────────┐ │
│  │ DP Replica 0 (GPUs 0-3):   TP 0-3     │ │
│  │ DP Replica 1 (GPUs 4-7):   TP 0-3     │ │
│  │ DP Replica 2 (GPUs 8-11):  TP 0-3     │ │
│  │ DP Replica 3 (GPUs 12-15): TP 0-3     │ │
│  └────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ (Stages 1-3 similarly structured)           │
└─────────────────────────────────────────────┘
```

**Mechanism**:
1. **PP**: Split model depth-wise into stages.
2. **TP**: Split large layers within each stage.
3. **DP**: Replicate across DP dimension for throughput.

**Benefits**:
- Maximum scalability for extremely large models (GPT-3, LLaMA-2).
- Flexible tuning of each dimension based on bottleneck.

**Trade-offs**:
- Most complex to implement and tune.
- Requires sophisticated communication optimization.
- Best suited for models >10B parameters.

### 4.3 Hybrid Strategy Selection

#### 4.3.1 Decision Factors

| Factor | Consideration |
|--------|---------------|
| **Model Size** | >10GB single GPU → Consider model parallelism |
| **Layer Width** | Very wide layers (>4096) → TP helpful |
| **Model Depth** | Very deep (>48 layers) → PP helpful |
| **GPU Count** | 2-4 GPUs → Simple hybrid; 8+ → Complex hybrid |
| **Memory/GPU** | Limited → FSDP or PP; Ample → DDP or TP+DP |
| **Throughput Goal** | High → Maximize DP dimension |
| **Network** | Fast (NVLink) → TP feasible; Slow → Prefer DP/PP |

#### 4.3.2 Recommended Configurations by GPU Count

**8 GPUs**:
- **Small Models** (<500M params): `DP=8` (pure DDP)
- **Medium Models** (500M-2B): `PP=2 × DP=4` or `TP=2 × DP=4`
- **Large Models** (2B-10B): `FSDP=8` or `PP=2 × FSDP=4`

**16 GPUs**:
- **Medium Models**: `PP=2 × DP=8` or `TP=2 × DP=8`
- **Large Models**: `PP=4 × DP=4` or `FSDP=16` with subgroups
- **Very Large**: `PP=4 × TP=2 × DP=2`

**32+ GPUs**:
- **Large Models** (10B-50B): `PP=4 × TP=2 × DP=4`
- **Extreme Models** (50B+): `PP=8 × TP=4 × DP=1-2` with FSDP

### 4.4 Hybrid Strategies for Trimodal BERT

#### 4.4.1 Standard Configuration (bert-base-cased, ~220M params)

**2 GPUs**:
- **Recommended**: Pure DDP
- **Alternative**: PP=2 (only if memory-constrained)

**4 GPUs**:
- **Recommended**: Pure DDP
- **Alternative**: PP=2 × DP=2 for 2× throughput

**8 GPUs**:
- **Recommended**: Pure DDP (8× throughput)
- **Alternative**: PP=2 × DP=4 if testing pipeline infrastructure

#### 4.4.2 Large Variant (bert-large, ~680M params)

**8 GPUs**:
- **Recommended**: `PP=2 × DP=4`
  - Stage 0 (4 GPUs): Primary BERT-large
  - Stage 1 (4 GPUs): Secondary BERT-large + Fusion
  - Each stage uses DDP across 4 replicas
  - Achieves 4× throughput with good memory efficiency

**16 GPUs**:
- **Recommended**: `PP=2 × DP=8`
  - 8× throughput with pipeline stages
- **Alternative**: `TP=2 × DP=8` 
  - Split attention/FFN layers, replicate across 8 groups

#### 4.4.3 Extreme Variant (roberta-large + custom layers, >1B params)

**8 GPUs**:
- **Recommended**: `PP=2 × FSDP=4`
  - Stage 0 (4 GPUs): Primary encoder with FSDP sharding
  - Stage 1 (4 GPUs): Secondary encoder + fusion with FSDP

**16 GPUs**:
- **Recommended**: `PP=4 × FSDP=4`
  - 4 pipeline stages, each using FSDP across 4 GPUs
  - Maximum memory efficiency

### 4.5 Implementation Considerations

#### 4.5.1 Process Group Management

For hybrid parallelism, must create separate process groups:

```python
import torch.distributed as dist

def setup_hybrid_groups(world_size, pp_size, tp_size):
    """
    Setup process groups for 3D parallelism.
    
    world_size = pp_size × tp_size × dp_size
    """
    rank = dist.get_rank()
    
    # Calculate position in 3D grid
    pp_rank = rank // (world_size // pp_size)
    tp_rank = (rank // (world_size // (pp_size * tp_size))) % tp_size
    dp_rank = rank % (world_size // (pp_size * tp_size))
    
    # Create DP groups (all ranks with same pp_rank and tp_rank)
    dp_groups = create_dp_groups(world_size, pp_size, tp_size)
    
    # Create TP groups (all ranks with same pp_rank and dp_rank)
    tp_groups = create_tp_groups(world_size, pp_size, tp_size)
    
    # Create PP groups (all ranks with same tp_rank and dp_rank)
    pp_groups = create_pp_groups(world_size, pp_size, tp_size)
    
    return dp_groups[dp_rank], tp_groups[tp_rank], pp_groups[pp_rank]
```

#### 4.5.2 Memory-Communication Trade-off

| Strategy | Memory/GPU | Communication | Throughput | Complexity |
|----------|------------|---------------|------------|------------|
| Pure DP | Highest | Lowest | Highest | Lowest |
| DP + PP | Medium-High | Medium | High | Medium |
| DP + TP | Medium | High | Medium-High | High |
| DP + FSDP | Low | Medium | Medium | Medium |
| 3D (DP+TP+PP) | Lowest | Highest | Medium | Highest |

#### 4.5.3 Debugging and Profiling

Hybrid strategies require careful profiling:
- **Pipeline bubbles**: Monitor GPU utilization across stages
- **Communication overhead**: Profile all-reduce, all-gather times
- **Load balancing**: Ensure stages have similar compute time
- **Memory usage**: Track peak memory per GPU

Tools:
- `torch.profiler` with distributed support
- NVIDIA Nsight Systems for GPU traces
- Custom logging of communication volume per group

## 5. Hybrid Strategies for 2-4 GPUs (Simplified)

For the Trimodal BERT model on 2-4 GPUs, we recommend:

### 5.1 Small-Medium Models (<1B params)

*   **2 GPUs**: Pure DDP (simple, efficient).
*   **4 GPUs**: Pure DDP for maximum throughput.
*   **Alternative**: PP=2 × DP=2 for testing hybrid setup.

### 5.2 Large Models (>1B params)

*   **2 GPUs**: FSDP with aggressive sharding.
*   **4 GPUs**: PP=2 × DP=2 (balanced hybrid) or FSDP + Gradient Checkpointing.

### 5.3 Experimental (Deep Pipelines)

*   **2-4 GPUs**: Pipeline Parallelism with 1F1B schedule.
*   Suitable for models with sequential depth (e.g., 48-layer transformers).

### 5.4 Recommendation Matrix

| Model Size | 2 GPUs | 4 GPUs | 8 GPUs |
|------------|--------|--------|--------|
| <500M (bert-base) | DDP | DDP | DDP |
| 500M-1B (bert-large) | DDP or FSDP | PP=2×DP=2 | PP=2×DP=4 |
| 1B-5B | FSDP | PP=2×FSDP=2 | PP=2×FSDP=4 |
| 5B+ | FSDP + CPU offload | PP=2×FSDP=2 | PP=4×FSDP=2 or 3D |

## 6. Implementation References

For detailed code examples and step-by-step implementation:

- [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md) - Base implementations (DDP, FSDP, Pipeline Parallelism)
- [Native PyTorch Hybrid Parallelism Implementation](./native_pytorch_hybrid_parallelism_implementation.md) - Advanced hybrid strategies (PP+DP, FSDP+PP, 3D parallelism)
