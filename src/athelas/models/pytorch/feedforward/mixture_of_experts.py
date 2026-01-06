"""
Mixture of Experts (MoE) Module

Sparse expert routing for specialized processing in attention mechanisms.

**Core Concept:**
Routes inputs to multiple expert networks based on learned gating scores.
Each expert specializes in different patterns, enabling the model to handle
diverse transaction types in fraud detection. Essential for TSA feedforward networks.

**Architecture:**
- Multiple expert networks: Specialized feedforward networks
- Gating network: Learns routing decisions
- Soft routing: Weighted combination of all experts
- Load balancing: Implicit through softmax gating

**Parameters:**
- dim (int): Input/output dimension
- num_experts (int): Number of expert networks (typically 3-7)
- hidden_dim (int): Hidden dimension for each expert
- second_policy_train (str): Training policy (for compatibility)
- second_policy_eval (str): Evaluation policy (for compatibility)

**Forward Signature:**
Input:
  - x: [L, B, E] or [B, E] - Input features

Output:
  - output: Same shape as input - Expert-processed features

**Dependencies:**
- torch.nn.Linear → Expert networks and gating
- torch.nn.functional.softmax → Gating normalization

**Used By:**
- temporal_self_attention_pytorch.pytorch.blocks.attention_layer → Feedforward in attention blocks

**Alternative Approaches:**
- Standard MLP → Single expert, less capacity
- Hard routing → More efficient but less stable
- Switch Transformers → Top-k routing for efficiency

**Usage Example:**
```python
from temporal_self_attention_pytorch.pytorch.feedforward import MixtureOfExperts

moe = MixtureOfExperts(dim=128, num_experts=5, hidden_dim=512)

# Process attention outputs
x = torch.randn(50, 32, 128)  # [L, B, E]
output = moe(x)  # [50, 32, 128]
```

**References:**
- "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (Shazeer et al., 2017)
- "GShard: Scaling Giant Models with Conditional Computation" (Lepikhin et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts module for sparse expert routing.

    Routes inputs through multiple specialized expert networks
    using learned gating scores for soft routing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_dim: int,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
    ):
        """
        Initialize MixtureOfExperts.

        Args:
            dim: Input/output dimension
            num_experts: Number of expert networks (typically 3-7)
            hidden_dim: Hidden dimension for each expert
            second_policy_train: Training policy (for compatibility, unused)
            second_policy_eval: Evaluation policy (for compatibility, unused)
        """
        super().__init__()

        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Expert networks - each is a 2-layer MLP
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network - learns routing decisions
        self.gate = nn.Linear(dim, num_experts)

        # Policy parameters (for compatibility with original MoE)
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor [L, B, E] or [B, E]

        Returns:
            output: Expert-processed tensor with same shape as input
        """
        original_shape = x.shape

        # Flatten if needed for processing
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))  # [L*B, E]

        # Compute gate scores and normalize with softmax
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [N, num_experts]

        # Process through all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # Each is [N, E]

        # Stack expert outputs: [N, E, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        # Weighted combination of expert outputs
        # gate_scores: [N, num_experts] -> [N, 1, num_experts]
        # expert_outputs: [N, E, num_experts]
        # Result: [N, E]
        output = torch.sum(expert_outputs * gate_scores.unsqueeze(-2), dim=-1)

        # Restore original shape
        output = output.view(original_shape)

        return output
