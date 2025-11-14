#!/usr/bin/env python3
"""
Native PyTorch Learning Rate Schedulers

This module provides native PyTorch implementations of learning rate schedulers
that replace the transformers package dependency, specifically:
- Linear warmup with decay
- Constant warmup
- Custom lambda-based schedulers

These implementations provide identical functionality to transformers schedulers
but use only native PyTorch components.
"""

import math
from typing import Union, Callable
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        lr_end: The end LR.
        power: Power factor.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    timescale: int = None, 
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with an inverse square-root learning rate, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        timescale: Time scale. Defaults to `num_warmup_steps`.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if timescale is None:
        timescale = num_warmup_steps

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return min(1.0, 1.0 / math.sqrt(max(current_step, timescale) / timescale))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Convenience mapping for easy scheduler selection
SCHEDULER_REGISTRY = {
    "linear": get_linear_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "inverse_sqrt": get_inverse_sqrt_schedule_with_warmup,
}


def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int = None,
    **kwargs
) -> LambdaLR:
    """
    Get a learning rate scheduler by name.
    
    Args:
        scheduler_type: Type of scheduler ("linear", "constant", "cosine", "polynomial", "inverse_sqrt")
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps (required for some schedulers)
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        The requested scheduler
        
    Raises:
        ValueError: If scheduler_type is not supported
    """
    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Available: {list(SCHEDULER_REGISTRY.keys())}")
    
    scheduler_fn = SCHEDULER_REGISTRY[scheduler_type]
    
    # Handle schedulers that don't need num_training_steps
    if scheduler_type in ["constant", "inverse_sqrt"]:
        return scheduler_fn(optimizer, num_warmup_steps, **kwargs)
    else:
        if num_training_steps is None:
            raise ValueError(f"num_training_steps is required for {scheduler_type} scheduler")
        return scheduler_fn(optimizer, num_warmup_steps, num_training_steps, **kwargs)
