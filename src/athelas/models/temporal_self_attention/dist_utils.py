#!/usr/bin/env python3
"""
Distributed training utilities for TSA Lightning models.

This module provides utilities for distributed training, including
functions for gathering data across multiple GPUs and getting rank information.
Based on the patterns from pytorch_bsm/lightning_models/dist_utils.py.
"""

import torch
import torch.distributed as dist
from typing import List, Any, Optional


def is_dist_available_and_initialized() -> bool:
    """
    Check if distributed training is available and initialized.
    
    Returns:
        True if distributed training is available and initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Get the number of processes in the distributed training group.
    
    Returns:
        World size (number of processes)
    """
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process in the distributed training group.
    
    Returns:
        Rank of the current process (0 if not distributed)
    """
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    
    Returns:
        True if this is the main process, False otherwise
    """
    return get_rank() == 0


def all_gather(data: List[Any]) -> List[List[Any]]:
    """
    Gather data from all processes in the distributed training group.
    
    This function collects data from all processes and returns a list
    containing the data from each process. If distributed training is
    not available, it returns the data wrapped in a single-element list.
    
    Args:
        data: List of data to gather from the current process
        
    Returns:
        List of lists, where each inner list contains the data from one process
        
    Example:
        # On rank 0: data = [1, 2, 3]
        # On rank 1: data = [4, 5, 6]
        # Result: [[1, 2, 3], [4, 5, 6]]
    """
    if not is_dist_available_and_initialized():
        return [data]
    
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # Convert data to tensor for gathering
    # We'll use pickle to serialize the data
    import pickle
    import io
    
    # Serialize the data
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    data_bytes = buffer.getvalue()
    
    # Convert to tensor
    data_tensor = torch.ByteTensor(list(data_bytes))
    
    # Get the size of the data tensor
    size_tensor = torch.tensor([len(data_tensor)], dtype=torch.long)
    
    # Gather sizes from all processes
    size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)
    
    # Get the maximum size
    max_size = max(size.item() for size in size_list)
    
    # Pad the data tensor to the maximum size
    if len(data_tensor) < max_size:
        padding = torch.zeros(max_size - len(data_tensor), dtype=torch.uint8)
        data_tensor = torch.cat([data_tensor, padding])
    
    # Gather data from all processes
    gathered_tensors = [torch.zeros_like(data_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, data_tensor)
    
    # Deserialize the data
    gathered_data = []
    for i, tensor in enumerate(gathered_tensors):
        # Trim to the actual size
        actual_size = size_list[i].item()
        trimmed_tensor = tensor[:actual_size]
        
        # Convert back to bytes and deserialize
        data_bytes = bytes(trimmed_tensor.tolist())
        buffer = io.BytesIO(data_bytes)
        try:
            deserialized_data = pickle.load(buffer)
            gathered_data.append(deserialized_data)
        except Exception as e:
            print(f"Warning: Failed to deserialize data from rank {i}: {e}")
            gathered_data.append([])
    
    return gathered_data


def all_gather_object(obj: Any) -> List[Any]:
    """
    Gather objects from all processes in the distributed training group.
    
    This is a simpler version of all_gather that works with single objects
    rather than lists.
    
    Args:
        obj: Object to gather from the current process
        
    Returns:
        List containing the object from each process
    """
    if not is_dist_available_and_initialized():
        return [obj]
    
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    
    # Use PyTorch's built-in all_gather_object if available
    if hasattr(dist, 'all_gather_object'):
        gathered_objects = [None] * world_size
        dist.all_gather_object(gathered_objects, obj)
        return gathered_objects
    else:
        # Fallback to our custom implementation
        return all_gather([obj])


def barrier():
    """
    Synchronize all processes in the distributed training group.
    
    This function blocks until all processes in the group have reached this point.
    If distributed training is not available, this function does nothing.
    """
    if is_dist_available_and_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce a tensor across all processes in the distributed training group.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("mean", "sum", "max", "min")
        
    Returns:
        Reduced tensor
    """
    if not is_dist_available_and_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    # Clone the tensor to avoid modifying the original
    reduced_tensor = tensor.clone()
    
    if op == "mean":
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        reduced_tensor /= world_size
    elif op == "sum":
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    return reduced_tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast an object from the source process to all other processes.
    
    Args:
        obj: Object to broadcast (only used on the source process)
        src: Source process rank
        
    Returns:
        The broadcasted object
    """
    if not is_dist_available_and_initialized():
        return obj
    
    world_size = get_world_size()
    if world_size == 1:
        return obj
    
    # Use PyTorch's built-in broadcast_object_list if available
    if hasattr(dist, 'broadcast_object_list'):
        obj_list = [obj] if get_rank() == src else [None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]
    else:
        # Fallback implementation using pickle
        import pickle
        import io
        
        if get_rank() == src:
            # Serialize the object
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            data_bytes = buffer.getvalue()
            data_tensor = torch.ByteTensor(list(data_bytes))
            size_tensor = torch.tensor([len(data_tensor)], dtype=torch.long)
        else:
            data_tensor = torch.ByteTensor()
            size_tensor = torch.tensor([0], dtype=torch.long)
        
        # Broadcast the size
        dist.broadcast(size_tensor, src=src)
        
        # Prepare tensor for receiving data
        if get_rank() != src:
            data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
        
        # Broadcast the data
        dist.broadcast(data_tensor, src=src)
        
        # Deserialize the object
        if get_rank() != src:
            data_bytes = bytes(data_tensor.tolist())
            buffer = io.BytesIO(data_bytes)
            obj = pickle.load(buffer)
        
        return obj


def setup_for_distributed(is_master: bool):
    """
    Setup utilities for distributed training.
    
    This function can be used to disable printing on non-master processes
    or perform other setup tasks for distributed training.
    
    Args:
        is_master: Whether this process is the master process
    """
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def cleanup_distributed():
    """
    Clean up distributed training resources.
    
    This function should be called at the end of distributed training
    to properly clean up resources.
    """
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


# Context manager for distributed training
class DistributedContext:
    """
    Context manager for distributed training setup and cleanup.
    
    Example:
        with DistributedContext():
            # Distributed training code here
            pass
    """
    
    def __init__(self, setup_print: bool = True):
        """
        Initialize the distributed context.
        
        Args:
            setup_print: Whether to setup print function for distributed training
        """
        self.setup_print = setup_print
        self.is_master = is_main_process()
    
    def __enter__(self):
        if self.setup_print:
            setup_for_distributed(self.is_master)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Note: We don't call cleanup_distributed here as it might interfere
        # with Lightning's distributed training management
        pass
