---
tags:
  - analysis
  - comparison
  - attention-mechanisms
  - pytorch
keywords:
  - MultiHeadAttention
  - self-attention
  - cross-attention
  - pytorch components
topics:
  - deep learning
  - attention mechanisms
  - component design
language: python
date of note: 2026-01-05
---

# MultiHeadAttention vs nn.MultiheadAttention Comparison

## Overview

Comparison between our custom `MultiHeadAttention` (in `pytorch/attention/`) and PyTorch's built-in `nn.MultiheadAttention` used in fusion components. Understanding the difference is crucial for choosing the right component.

## Key Difference: Self-Attention vs Cross-Attention

### Our MultiHeadAttention → Self-Attention Only

**Location:** `src/athelas/models/pytorch/attention/multihead_attention.py`

**Design:**
```python
class MultiHeadAttention(nn.Module):
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x is the ONLY input
        # Q, K, V all derived from x internally by each AttentionHead
        head_outputs = [head(x, attn_mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.proj(concatenated)
        return output
```

**Key Characteristics:**
- ✅ Designed for **self-attention** (token attends to other tokens in same sequence)
- ✅ Single input `x` → Q, K, V computed from same source
- ✅ Custom implementation using `AttentionHead` modules
- ✅ Manual head concatenation
- ✅ Good for learning: shows attention mechanism internals
- ❌ Only supports self-attention pattern

**Use Cases:**
- Transformer encoder (tokens attending to each other)
- Names3Risk model (characters attending to other characters)
- Any case where Q, K, V come from the same input

---

### PyTorch's nn.MultiheadAttention → Both Self & Cross-Attention

**Location:** `torch.nn.MultiheadAttention` (PyTorch built-in)

**Design:**
```python
class nn.MultiheadAttention:
    def forward(
        self,
        query: torch.Tensor,      # Q can be from different source
        key: torch.Tensor,        # K can be from different source
        value: torch.Tensor,      # V can be from different source
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        ...
    ):
        # Highly optimized C++/CUDA implementation
        # Supports both self-attention and cross-attention
        return output, attention_weights
```

**Key Characteristics:**
- ✅ Supports **both self-attention AND cross-attention**
- ✅ Separate Q, K, V inputs (more flexible)
- ✅ Highly optimized (C++/CUDA backend)
- ✅ Industry-standard implementation
- ✅ More options (dropout, bias, add_zero_attn, etc.)
- ✅ Returns attention weights for visualization

**Use Cases:**
- **Cross-attention**: Text queries attend to tabular keys/values
- **Self-attention**: Can also do self-attention (pass same input 3 times)
- Transformer decoder (queries from decoder, keys/values from encoder)
- Multi-modal fusion (one modality queries another)

---

## Detailed Comparison Table

| Feature | Our MultiHeadAttention | PyTorch nn.MultiheadAttention |
|---------|----------------------|------------------------------|
| **Primary Use** | Self-attention only | Self + Cross-attention |
| **Input Signature** | `forward(x, mask)` | `forward(query, key, value, ...)` |
| **Q, K, V Source** | All from same `x` | Can be from different sources |
| **Implementation** | Custom Python (AttentionHead) | Optimized C++/CUDA |
| **Performance** | Slower (pure Python) | Faster (C++/CUDA backend) |
| **Flexibility** | Limited to self-attention | Highly flexible |
| **Attention Weights** | Not returned | Can return via `need_weights=True` |
| **Dropout** | Via AttentionHead | Built-in parameter |
| **Batch First** | ✓ (implicit) | ✓ (via `batch_first=True`) |
| **Educational Value** | High (shows internals) | Lower (black box) |
| **Production Ready** | For self-attention | For all attention types |

---

## Code Examples

### Example 1: Self-Attention with Our Component

```python
from athelas.models.pytorch.attention import MultiHeadAttention

# Our component - self-attention only
mha = MultiHeadAttention(embedding_dim=256, n_heads=8)

# Single input - tokens attend to each other
x = torch.randn(32, 50, 256)  # (batch, seq_len, dim)
output = mha(x)  # (32, 50, 256)

# Q, K, V all computed from x internally
```

### Example 2: Self-Attention with PyTorch Built-in

```python
import torch.nn as nn

# PyTorch built-in - pass same input 3 times for self-attention
mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

x = torch.randn(32, 50, 256)
# Pass x as query, key, AND value for self-attention
output, weights = mha(query=x, key=x, value=x)  # (32, 50, 256)
```

### Example 3: Cross-Attention (Only PyTorch Built-in)

```python
import torch.nn as nn

# Cross-attention: text queries attend to tabular keys/values
cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

text_features = torch.randn(32, 1, 256)  # Query from text
tab_features = torch.randn(32, 1, 256)   # Keys/Values from tabular

# Text attends to tabular (cross-attention)
output, weights = cross_attn(
    query=text_features,   # Q from text
    key=tab_features,      # K from tabular
    value=tab_features     # V from tabular
)  # (32, 1, 256)

# This is NOT possible with our MultiHeadAttention!
```

---

## Why We Use Each Component

### When to Use Our MultiHeadAttention

✅ **Use our custom component when:**
1. You need **self-attention** only
2. You want to understand attention mechanism internals
3. You're working on educational/research code
4. Performance is not critical
5. You need custom AttentionHead modifications

**Examples:**
- Transformer encoder blocks
- Names3Risk character self-attention
- Text sequence self-attention

### When to Use PyTorch nn.MultiheadAttention

✅ **Use PyTorch built-in when:**
1. You need **cross-attention** (Q, K, V from different sources)
2. You want **maximum performance** (production code)
3. You need attention weights for visualization
4. You want industry-standard implementation
5. You need advanced features (dropout, bias options, etc.)

**Examples:**
- Cross-modal fusion (text ↔ tabular)
- Transformer decoder (decoder queries, encoder keys/values)
- Any multi-modal attention mechanism

---

## CrossAttentionFusion Example

### Why We Use nn.MultiheadAttention in Fusion Components

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        # Use PyTorch built-in for cross-attention
        self.text2tab = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.tab2text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, text_seq, tab_seq):
        # Text queries attend to tabular keys/values (cross-attention)
        t2t_out, _ = self.text2tab(
            query=text_seq,    # Q from text modality
            key=tab_seq,       # K from tabular modality
            value=tab_seq      # V from tabular modality
        )
        
        # Tabular queries attend to text keys/values (cross-attention)
        tab2_out, _ = self.tab2text(
            query=tab_seq,     # Q from tabular modality
            key=text_seq,      # K from text modality
            value=text_seq     # V from text modality
        )
        
        return text_enhanced, tab_enhanced
```

**Why not use our MultiHeadAttention?**
- ❌ Our component only accepts single input `x`
- ❌ Cannot separate Q from one modality, K/V from another
- ❌ Not designed for cross-attention pattern

---

## Performance Comparison

### Benchmark Results (Approximate)

| Operation | Our MultiHeadAttention | PyTorch nn.MultiheadAttention |
|-----------|----------------------|------------------------------|
| Forward Pass | 100ms | 10-20ms (5-10x faster) |
| Backward Pass | 150ms | 20-30ms (5-7x faster) |
| Memory Usage | Higher (Python objects) | Lower (optimized buffers) |
| GPU Utilization | ~60% | ~90% |

**Note:** Times are illustrative. Actual performance depends on hardware, batch size, sequence length, etc.

---

## Implementation Details

### Our MultiHeadAttention Architecture

```
Input: x (B, L, D)
    ↓
Split into n_heads AttentionHead modules
    ↓
Each head computes:
  1. Q = x, K = x, V = x (self-attention)
  2. Attention(Q, K, V) = softmax(QK^T/√d)V
    ↓
Concatenate head outputs
    ↓
Output projection: Linear(D, D)
    ↓
Output: (B, L, D)
```

### PyTorch nn.MultiheadAttention Architecture

```
Input: query, key, value (can be different!)
    ↓
Optimized C++/CUDA implementation:
  1. Linear projections for Q, K, V
  2. Split into num_heads
  3. Scaled dot-product attention per head
  4. Concatenate heads
  5. Output projection
    ↓
Optional: Return attention weights
    ↓
Output: (B, L, D), Optional[attention_weights]
```

---

## Migration Guide

### If You Want Cross-Attention

**Don't do this:**
```python
# ❌ Wrong - our component doesn't support cross-attention
from athelas.models.pytorch.attention import MultiHeadAttention
mha = MultiHeadAttention(embedding_dim=256, n_heads=8)
# No way to pass separate text and tabular!
```

**Do this:**
```python
# ✅ Correct - use PyTorch built-in
import torch.nn as nn
cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
output, _ = cross_attn(query=text, key=tabular, value=tabular)
```

### If You Want Self-Attention

**Both work, but different trade-offs:**

```python
# Option 1: Our component (educational, self-attention only)
from athelas.models.pytorch.attention import MultiHeadAttention
mha = MultiHeadAttention(embedding_dim=256, n_heads=8)
output = mha(x)  # Simple interface

# Option 2: PyTorch built-in (production, more flexible)
import torch.nn as nn
mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
output, weights = mha(query=x, key=x, value=x)  # More verbose but flexible
```

---

## Recommendations

### For New Code

1. **Cross-attention (multi-modal fusion):**
   - ✅ Use `nn.MultiheadAttention`
   - Reason: Only option that supports separate Q, K, V

2. **Self-attention (production):**
   - ✅ Use `nn.MultiheadAttention` with same input 3 times
   - Reason: Better performance, more features

3. **Self-attention (educational/research):**
   - ✅ Use our `MultiHeadAttention`
   - Reason: Clearer implementation, easier to modify

### For Existing Code

- **Keep our MultiHeadAttention** in transformer blocks (self-attention)
- **Use nn.MultiheadAttention** for all fusion components (cross-attention)
- **Don't mix**: Inconsistent interfaces make code harder to understand

---

## Summary

### Quick Reference

| Need | Component to Use |
|------|------------------|
| Self-attention (same sequence) | Either (prefer ours for clarity) |
| Cross-attention (different sources) | **Only** PyTorch nn.MultiheadAttention |
| Production performance | PyTorch nn.MultiheadAttention |
| Educational code | Our MultiHeadAttention |
| Attention weights | PyTorch nn.MultiheadAttention |
| Multi-modal fusion | PyTorch nn.MultiheadAttention |

### The Bottom Line

- **Our MultiHeadAttention** = Self-attention only, educational, clear implementation
- **PyTorch nn.MultiheadAttention** = Universal (self + cross), production-ready, optimized

**In fusion components, we MUST use PyTorch's built-in because we need cross-attention!**

---

## Related Documents

- [CrossAttentionFusion](../../src/athelas/models/pytorch/fusion/cross_attention_fusion.py) - Uses nn.MultiheadAttention
- [Our MultiHeadAttention](../../src/athelas/models/pytorch/attention/multihead_attention.py) - Custom self-attention
- [PyTorch Fusion Component Reuse Analysis](./2026-01-05_pytorch_fusion_component_reuse_analysis.md)
