"""
Concatenation Fusion

Simple concatenation-based fusion for combining multiple modalities.

**Core Concept:**
The simplest multi-modal fusion strategy: concatenate features from different
modalities and project to target dimension through ReLU + Linear. This serves
as a strong baseline that often performs competitively with more complex fusion
mechanisms while being computationally efficient.

**Architecture:**
1. Concatenate all modality features along feature dimension
2. ReLU activation for non-linearity
3. Linear projection to output dimension

**Parameters:**
- input_dims (List[int]): List of input dimensions for each modality
- output_dim (int): Target output dimension
- use_activation (bool): Whether to use ReLU before projection (default: True)

**Forward Signature:**
Input:
  - *features: Variable number of tensors (B, D_i) - One per modality
  
Output:
  - fused: (B, output_dim) - Concatenated and projected features

**Dependencies:**
- torch.nn.Linear → Projection layer
- torch.nn.ReLU → Activation function

**Used By:**
- athelas.models.lightning.bimodal.pl_bimodal_cnn → Simple bimodal fusion
- Any model requiring simple concatenation fusion
- Baseline for comparing against complex fusion mechanisms

**Alternative Approaches:**
- athelas.models.pytorch.fusion.cross_attention_fusion → Attention-based
- athelas.models.pytorch.fusion.gate_fusion → Gating-based
- athelas.models.pytorch.fusion.mixture_of_experts → Routing-based
- Simple concatenation without projection → When dimensions match

**Usage Example:**
```python
from athelas.models.pytorch.fusion import ConcatenationFusion

# Create concatenation fusion for text (768-d) + tabular (128-d) → 256-d
fusion = ConcatenationFusion(
    input_dims=[768, 128],
    output_dim=256,
    use_activation=True
)

# Fuse text and tabular features
text_features = torch.randn(32, 768)  # (batch, text_dim)
tab_features = torch.randn(32, 128)   # (batch, tab_dim)

fused = fusion(text_features, tab_features)
# Output: (32, 256)
```

**Multi-Modal Example:**
```python
# For 3 modalities: text, tabular, image
fusion = ConcatenationFusion(
    input_dims=[768, 128, 2048],  # text, tab, image
    output_dim=512
)

fused = fusion(text_feat, tab_feat, img_feat)
# Output: (32, 512)
```

**Implementation Notes:**
- Most simple and interpretable fusion strategy
- No learnable interaction between modalities (unlike attention/gating)
- Computationally efficient (just concat + linear)
- Often serves as strong baseline in multi-modal learning
- Can handle arbitrary number of modalities

**When to Use:**
- ✅ As baseline before trying complex fusion
- ✅ When modalities are already well-aligned
- ✅ When computational efficiency is critical
- ✅ When interpretability is important
- ❌ When modalities need to interact (use attention/gating)

**References:**
- "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2019)
- "Deep Multimodal Learning" (Ngiam et al., 2011) - Early fusion strategies
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation-based multi-modal fusion.
    
    Concatenates features from multiple modalities and projects to target
    dimension. Serves as a strong baseline for multi-modal fusion.
    """
    
    def __init__(
        self, 
        input_dims: List[int], 
        output_dim: int,
        use_activation: bool = True
    ):
        """
        Initialize ConcatenationFusion.
        
        Args:
            input_dims: List of input dimensions for each modality
            output_dim: Target output dimension
            use_activation: Whether to use ReLU before projection (default: True)
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.num_modalities = len(input_dims)
        self.total_input_dim = sum(input_dims)
        self.use_activation = use_activation
        
        # Fusion network: ReLU → Linear (or just Linear)
        if use_activation:
            self.fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.total_input_dim, output_dim)
            )
        else:
            self.fusion = nn.Linear(self.total_input_dim, output_dim)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Concatenate and fuse multiple modality features.
        
        Args:
            *features: Variable number of tensors (B, D_i) - one per modality
            
        Returns:
            fused: (B, output_dim) - Fused features
            
        Raises:
            ValueError: If number of features doesn't match num_modalities
            ValueError: If feature dimensions don't match input_dims
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality features, "
                f"got {len(features)}"
            )
        
        # Validate dimensions
        for i, (feat, expected_dim) in enumerate(zip(features, self.input_dims)):
            if feat.shape[1] != expected_dim:
                raise ValueError(
                    f"Modality {i} has dimension {feat.shape[1]}, "
                    f"expected {expected_dim}"
                )
        
        # Concatenate along feature dimension
        concatenated = torch.cat(features, dim=1)  # (B, sum(D_i))
        
        # Project to output dimension
        fused = self.fusion(concatenated)  # (B, output_dim)
        
        return fused
    
    def __repr__(self) -> str:
        activation_str = "ReLU + " if self.use_activation else ""
        return (
            f"ConcatenationFusion(input_dims={self.input_dims}, "
            f"output_dim={self.output_dim}, "
            f"fusion={activation_str}Linear)"
        )


def validate_modality_features(
    features: Tuple[torch.Tensor, ...],
    expected_dims: List[int],
    modality_names: List[str] = None
) -> None:
    """
    Utility function to validate modality feature dimensions.
    
    Checks that:
    1. Number of features matches expected number of modalities
    2. Each feature has correct dimension
    3. All features have same batch size
    
    Args:
        features: Tuple of feature tensors
        expected_dims: Expected dimension for each modality
        modality_names: Optional names for better error messages
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> text_feat = torch.randn(32, 768)
        >>> tab_feat = torch.randn(32, 128)
        >>> validate_modality_features(
        ...     (text_feat, tab_feat),
        ...     [768, 128],
        ...     ["text", "tabular"]
        ... )
    """
    if modality_names is None:
        modality_names = [f"modality_{i}" for i in range(len(expected_dims))]
    
    if len(features) != len(expected_dims):
        raise ValueError(
            f"Expected {len(expected_dims)} modalities, got {len(features)}"
        )
    
    batch_sizes = [feat.shape[0] for feat in features]
    if len(set(batch_sizes)) > 1:
        raise ValueError(
            f"Inconsistent batch sizes across modalities: {batch_sizes}"
        )
    
    for feat, exp_dim, name in zip(features, expected_dims, modality_names):
        if feat.dim() != 2:
            raise ValueError(
                f"{name} features should be 2D (batch, dim), got shape {feat.shape}"
            )
        if feat.shape[1] != exp_dim:
            raise ValueError(
                f"{name} has dimension {feat.shape[1]}, expected {exp_dim}"
            )
