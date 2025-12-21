# Distributed Training Utilities

## Summary
The `dist_utils` module provides utility functions for distributed training in PyTorch. It includes functions for managing process groups, gathering data across processes, synchronizing processes, and monitoring GPU memory usage. These utilities are essential for efficient distributed training of deep learning models.

## Key Functions

### Process Group Management
- `get_world_size(group=None)`: Returns the total number of processes in the distributed training
- `get_rank()`: Returns the rank of the current process
- `is_main_process()`: Returns True if the current process is the main process (rank 0)
- `synchronize()`: Synchronizes all processes (barrier)
- `create_local_process_group(num_workers_per_machine)`: Creates a process group for workers on the same machine
- `get_local_process_group()`: Returns the local process group
- `get_local_rank()`: Returns the rank of the current process within its local group
- `get_local_size()`: Returns the size of the local process group

### Data Gathering
- `all_gather(data, group=None, safe_clone=True)`: Gathers data from all processes
- `gather(data, dst=0, group=None)`: Gathers data from all processes to a specific destination process
- `reduce_dict(input_dict, average=True)`: Reduces dictionary values across all processes

### Random Seed Management
- `shared_random_seed()`: Returns a random seed that is shared across all processes

### GPU Memory Monitoring
- `print_gpu_memory_usage(device_id)`: Prints memory usage statistics for a specific GPU
- `print_gpu_memory_stats(device_id)`: Prints detailed memory statistics for a specific GPU

## Usage in Models
This utility module is used by various model classes in the MODS_BSM system to:

1. Gather predictions and labels from all processes during validation and testing
2. Synchronize processes during distributed training
3. Monitor GPU memory usage for debugging and optimization
4. Ensure consistent random seeds across processes

## Example Usage
```python
import torch
import torch.distributed as dist
from dist_utils import all_gather, synchronize, is_main_process, print_gpu_memory_usage

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Train model
model.train()
# ... training code ...

# Gather predictions from all processes during validation
model.eval()
predictions = []
labels = []
for batch in val_dataloader:
    with torch.no_grad():
        preds = model(batch)
        predictions.extend(preds.detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())

# Gather predictions from all processes
all_predictions = all_gather(predictions)
all_labels = all_gather(labels)

# Only the main process computes and logs metrics
if is_main_process():
    # Flatten gathered lists
    all_predictions = [p for sublist in all_predictions for p in sublist]
    all_labels = [l for sublist in all_labels for l in sublist]
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    print(f"Validation metrics: {metrics}")

# Monitor GPU memory usage
if is_main_process():
    print_gpu_memory_usage(torch.cuda.current_device())

# Synchronize all processes before continuing
synchronize()
```
