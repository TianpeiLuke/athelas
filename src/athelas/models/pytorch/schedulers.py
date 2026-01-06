"""
Learning Rate Scheduler Utilities

Common learning rate scheduler patterns for deep learning training.

**Core Concept:**
Provides utility functions for creating and configuring learning rate schedulers.
Particularly useful for transformer-based models that benefit from warmup
and various decay schedules.

**Common Patterns:**

1. **Linear Warmup + Linear Decay:**
   - Gradually increase LR during warmup
   - Then linearly decrease to 0
   - Standard for BERT and GPT fine-tuning

2. **Constant Warmup:**
   - Gradually increase LR during warmup
   - Then keep constant
   - Useful for continued pretraining

3. **Cosine Annealing:**
   - Smooth decay following cosine curve
   - Can help avoid local minima

**References:**
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- "Accurate, Large Minibatch SGD" (Goyal et al., 2017) - Warmup rationale
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from typing import Optional


def create_bert_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    schedule_type: str = "linear",
) -> LRScheduler:
    """
    Create learning rate scheduler for BERT-style training.
    
    Supports various warmup + decay strategies commonly used for
    transformer fine-tuning and pretraining.
    
    Args:
        optimizer: PyTorch optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps (gradual LR increase)
        schedule_type: Type of schedule after warmup:
            - "linear": Linear decay to 0 (default, standard for BERT)
            - "constant": Constant LR after warmup
            - "cosine": Cosine annealing after warmup
            
    Returns:
        scheduler: Learning rate scheduler
        
    Example:
        >>> from torch.optim import AdamW
        >>> optimizer = AdamW(model.parameters(), lr=2e-5)
        >>> 
        >>> # Standard BERT schedule: warmup + linear decay
        >>> scheduler = create_bert_scheduler(
        ...     optimizer,
        ...     num_training_steps=10000,
        ...     num_warmup_steps=1000,
        ...     schedule_type="linear"
        ... )
        >>> 
        >>> # In training loop
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()  # Update LR
        ...     optimizer.zero_grad()
    """
    schedule_type = schedule_type.lower()
    
    if schedule_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif schedule_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
    elif schedule_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. "
            f"Supported: 'linear', 'constant', 'cosine'"
        )
    
    return scheduler


def create_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    schedule_after_warmup: str = "linear",
    total_steps: Optional[int] = None,
) -> LRScheduler:
    """
    Create scheduler with warmup period.
    
    Convenience wrapper around create_bert_scheduler with clearer naming.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        schedule_after_warmup: Schedule type after warmup ("linear", "constant", "cosine")
        total_steps: Total training steps (required for linear/cosine)
        
    Returns:
        scheduler: Learning rate scheduler
        
    Example:
        >>> # Warmup for 1000 steps, then linear decay
        >>> scheduler = create_warmup_scheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     schedule_after_warmup="linear",
        ...     total_steps=10000
        ... )
    """
    if schedule_after_warmup in ["linear", "cosine"] and total_steps is None:
        raise ValueError(
            f"total_steps required for schedule_after_warmup='{schedule_after_warmup}'"
        )
    
    return create_bert_scheduler(
        optimizer,
        num_training_steps=total_steps or warmup_steps,  # For constant schedule
        num_warmup_steps=warmup_steps,
        schedule_type=schedule_after_warmup
    )


def get_scheduler_config_for_lightning(
    scheduler: LRScheduler,
    interval: str = "step",
    frequency: int = 1,
    monitor: Optional[str] = None,
) -> dict:
    """
    Create scheduler configuration dict for PyTorch Lightning.
    
    PyTorch Lightning requires schedulers to be returned as a dict
    with specific keys from configure_optimizers().
    
    Args:
        scheduler: PyTorch learning rate scheduler
        interval: When to step the scheduler ("step" or "epoch")
        frequency: How often to step (default: 1)
        monitor: Metric to monitor for ReduceLROnPlateau (optional)
        
    Returns:
        config: Scheduler configuration dict for Lightning
        
    Example:
        >>> class MyModel(pl.LightningModule):
        ...     def configure_optimizers(self):
        ...         optimizer = AdamW(self.parameters(), lr=2e-5)
        ...         scheduler = create_bert_scheduler(
        ...             optimizer,
        ...             num_training_steps=self.trainer.estimated_stepping_batches,
        ...             num_warmup_steps=1000
        ...         )
        ...         scheduler_config = get_scheduler_config_for_lightning(
        ...             scheduler, interval="step"
        ...         )
        ...         return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
    """
    config = {
        "scheduler": scheduler,
        "interval": interval,  # "step" or "epoch"
        "frequency": frequency,
    }
    
    if monitor is not None:
        config["monitor"] = monitor
    
    return config


def calculate_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.1
) -> int:
    """
    Calculate number of warmup steps from ratio.
    
    Common practice is to use 10% of total steps for warmup.
    
    Args:
        total_steps: Total number of training steps
        warmup_ratio: Fraction of steps to use for warmup (default: 0.1 = 10%)
        
    Returns:
        warmup_steps: Number of warmup steps
        
    Example:
        >>> # 10% warmup of 10000 steps = 1000 steps
        >>> warmup_steps = calculate_warmup_steps(10000, warmup_ratio=0.1)
        >>> print(warmup_steps)
        1000
        
        >>> # 5% warmup
        >>> warmup_steps = calculate_warmup_steps(10000, warmup_ratio=0.05)
        >>> print(warmup_steps)
        500
    """
    return int(total_steps * warmup_ratio)
