import functools
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None

def get_world_size(group=None):
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)

def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return

    # Check if CUDA is available and PyTorch is compiled with CUDA support
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        # Ensure the current device is set
        try:
            torch.cuda.current_device()
        except AssertionError:
            # Handle cases where CUDA is not properly configured
            pass
    dist.barrier()

def get_local_process_group():
    assert _LOCAL_PROCESS_GROUP is not None, (
        "Local process group not initialized. "
        "Use `create_local_process_group()` after `torch.distributed.init_process_group()`."
    )
    return _LOCAL_PROCESS_GROUP

def get_local_rank() -> int:
    return dist.get_rank(group=get_local_process_group()) if dist.is_available() and dist.is_initialized() else 0

def get_local_size() -> int:
    return dist.get_world_size(group=get_local_process_group()) if dist.is_available() and dist.is_initialized() else 1

@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None, "Local process group already created!"
    world_size = get_world_size()
    assert world_size % num_workers_per_machine == 0, "World size must be divisible by num_workers_per_machine"
    num_machines = world_size // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine

    for i in range(num_machines):
        ranks = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg

@functools.lru_cache()
def _get_global_gloo_group():
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    return dist.group.WORLD


def all_gather(data, group=None, safe_clone=True):
    if get_world_size() == 1:
        return [data]
    group = group or _get_global_gloo_group()
    if safe_clone and isinstance(data, torch.Tensor):
        data = data.clone().detach().cpu()
    output = [None for _ in range(get_world_size(group))]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    if get_world_size() == 1:
        return [data]
    group = group or _get_global_gloo_group()
    rank = dist.get_rank(group)
    output = [None for _ in range(get_world_size(group))] if rank == dst else None
    dist.gather_object(data, output, dst=dst, group=group)
    return output if rank == dst else []

def shared_random_seed():
    local_seed = np.random.randint(0, 2**31)
    seeds = all_gather(local_seed)
    return seeds[0]

def reduce_dict(input_dict, average=True):
    if get_world_size() < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys])
        dist.reduce(values, dst=0)
        if is_main_process() and average:
            values /= get_world_size()
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict

#------------------ Monitor GPU usage ----------------------
def print_gpu_memory_usage(device_id):
    if torch.cuda.is_available():
        print(f"--- GPU {device_id} Memory Usage ---")
        print(f"  Allocated: {torch.cuda.memory_allocated(device=device_id) / 1024**2:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(device=device_id) / 1024**2:.2f} MB")
        print(f"  Free:      {torch.cuda.memory_reserved(device=device_id) / 1024**2 - torch.cuda.memory_allocated(device=device_id) / 1024**2:.2f} MB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device=device_id) / 1024**2:.2f} MB")
        print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(device=device_id) / 1024**2:.2f} MB")

        
def print_gpu_memory_stats(device_id):
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats(device=device_id)
        print(f"--- GPU {device_id} Memory Stats ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")