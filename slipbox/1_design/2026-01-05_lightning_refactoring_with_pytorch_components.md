---
tags:
  - design
  - refactoring
  - lightning
  - pytorch-components
keywords:
  - code simplification
  - component reuse
  - lightning modules
  - refactoring plan
topics:
  - software architecture
  - code organization
  - technical debt reduction
language: python
date of note: 2026-01-05
---

# Lightning Module Refactoring with PyTorch Components

## Overview

Design document for refactoring Lightning modules to use newly extracted PyTorch components. This will significantly reduce code duplication, improve maintainability, and make the codebase more modular.

## Refactoring Opportunities

### 1. Text Models (`models/lightning/text/`)

#### pl_text_cnn.py → Use CNNEncoder

**Current State:** ~200 lines with custom CNN implementation

**Refactoring:**
```python
# BEFORE: Custom CNN implementation
class TextCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        # Custom embedding
        self.embeddings = nn.Embedding(vocab_size, self.embed_size)
        
        # Custom conv layers with complex logic
        self.convs = nn.ModuleList([...])  # ~50 lines
        
        # Custom dimension calculations
        self.conv_output_dims = {...}
        self.conv_input_dims = {...}

# AFTER: Use CNNEncoder component
from athelas.models.pytorch.blocks import CNNEncoder

class TextCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        # Use CNNEncoder - all complexity handled
        self.text_encoder = CNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=word_embeddings.shape[1],
            kernel_sizes=config.get("kernel_size", [3, 5, 7]),
            num_channels=config.get("num_channels", [100, 100]),
            num_layers=config.get("num_layers", 2),
            output_dim=self.num_classes,
            dropout=config.get("dropout_keep", 0.5),
            max_seq_len=config.get("max_sen_len", 512)
        )
        # Set pretrained embeddings
        self.text_encoder.embeddings.weight = nn.Parameter(word_embeddings)
```

**Benefits:**
- ✅ ~50 lines of CNN logic removed
- ✅ Dimension calculations handled by CNNEncoder
- ✅ Easier to test (encoder tested separately)
- ✅ Consistent with other models

**Complexity Reduction:** HIGH (50+ lines → 10 lines)

---

### 2. Bimodal Models (`models/lightning/bimodal/`)

#### pl_bimodal_cnn.py → Use CNNEncoder + TabularEmbedding + ConcatenationFusion

**Current State:** ~250 lines with custom implementations

**Refactoring:**
```python
# BEFORE: Multiple custom components
class BimodalCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        # Custom text subnetwork (duplicates TextCNN)
        self.text_subnetwork = TextCNN(config, vocab_size, word_embeddings)
        
        # Custom tabular subnetwork
        self.tab_subnetwork = TabAE(config)
        
        # Custom fusion (simple concat + linear)
        self.final_merge_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(tab_dim + text_dim, self.num_classes),
        )

# AFTER: Use atomic components
from athelas.models.pytorch.blocks import CNNEncoder
from athelas.models.pytorch.embeddings import TabularEmbedding
from athelas.models.pytorch.fusion import ConcatenationFusion

class BimodalCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        # Text encoder
        self.text_encoder = CNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=word_embeddings.shape[1],
            kernel_sizes=config.get("kernel_size", [3, 5, 7]),
            num_channels=config.get("num_channels", [100, 100]),
            output_dim=config.get("text_output_dim", 256)
        )
        
        # Tabular encoder
        tab_input_dim = sum([...])  # from tab_field_list
        self.tab_encoder = TabularEmbedding(
            input_dim=tab_input_dim,
            hidden_dim=config.get("tab_output_dim", 128)
        )
        
        # Fusion
        self.fusion = ConcatenationFusion(
            input_dims=[256, 128],  # text_dim, tab_dim
            output_dim=self.num_classes,
            use_activation=True
        )
    
    def forward(self, batch):
        text_features = self.text_encoder(batch[self.text_name])
        tab_features = self.tab_encoder(
            combine_tabular_fields(batch, self.tab_field_list, self.device)
        )
        return self.fusion(text_features, tab_features)
```

**Benefits:**
- ✅ No need to instantiate full TextCNN Lightning module
- ✅ Clear separation: encoding vs fusion
- ✅ TabularEmbedding reusable across models
- ✅ ConcatenationFusion makes fusion strategy explicit

**Complexity Reduction:** MEDIUM (clearer architecture, ~30 lines saved)

---

#### pl_bimodal_cross_attn.py → Use CrossAttentionFusion

**Current State:** ~200 lines with custom cross-attention

**Refactoring:**
```python
# BEFORE: Custom cross-attention implementation
class BimodalCrossAttn(pl.LightningModule):
    def __init__(self, config):
        # Duplicate cross-attention logic
        self.cross_attn = nn.MultiheadAttention(...)
        self.norm = nn.LayerNorm(...)
        # Manual forward implementation

# AFTER: Use CrossAttentionFusion
from athelas.models.pytorch.fusion import CrossAttentionFusion

class BimodalCrossAttn(pl.LightningModule):
    def __init__(self, config):
        self.text_encoder = ...  # BERT or other
        self.tab_encoder = TabularEmbedding(...)
        
        # Use extracted component
        self.fusion = CrossAttentionFusion(
            hidden_dim=config.get("hidden_dim", 256),
            num_heads=config.get("num_heads", 4)
        )
    
    def forward(self, batch):
        text_feat = self.text_encoder(batch[self.text_name])
        tab_feat = self.tab_encoder(...)
        
        # Simple fusion call
        text_enhanced, tab_enhanced = self.fusion(text_feat, tab_feat)
        return torch.cat([text_enhanced, tab_enhanced], dim=1)
```

**Benefits:**
- ✅ ~40 lines of attention logic removed
- ✅ Attention mechanism tested independently
- ✅ Easy to swap fusion strategies

**Complexity Reduction:** HIGH (~40 lines → ~5 lines for fusion)

---

#### pl_bimodal_gate_fusion.py → Use GateFusion

**Current State:** Custom gating implementation

**Refactoring:**
```python
# BEFORE: Custom gating
class BimodalGateFusion(pl.LightningModule):
    def __init__(self, config):
        self.gate_net = nn.Sequential(
            nn.Linear(...),
            nn.LayerNorm(...),
            nn.Sigmoid()
        )
        # Manual gating logic

# AFTER: Use GateFusion
from athelas.models.pytorch.fusion import GateFusion

class BimodalGateFusion(pl.LightningModule):
    def __init__(self, config):
        self.text_encoder = ...
        self.tab_encoder = ...
        
        self.fusion = GateFusion(
            text_dim=config.get("text_dim", 768),
            tab_dim=config.get("tab_dim", 128),
            fusion_dim=config.get("fusion_dim", 256)
        )
    
    def forward(self, batch):
        text_feat = self.text_encoder(...)
        tab_feat = self.tab_encoder(...)
        return self.fusion(text_feat, tab_feat)
```

**Benefits:**
- ✅ ~25 lines of gating logic removed
- ✅ Consistent gating across models

**Complexity Reduction:** MEDIUM (~25 lines → ~3 lines)

---

#### pl_bimodal_moe.py → Use MixtureOfExperts

**Current State:** Custom MoE routing

**Refactoring:**
```python
# BEFORE: Custom MoE
class BimodalMoE(pl.LightningModule):
    def __init__(self, config):
        self.router = nn.Sequential(nn.Linear(...), nn.Softmax(...))
        # Manual routing logic

# AFTER: Use MixtureOfExperts
from athelas.models.pytorch.fusion import MixtureOfExperts

class BimodalMoE(pl.LightningModule):
    def __init__(self, config):
        self.text_encoder = ...
        self.tab_encoder = ...
        
        self.fusion = MixtureOfExperts(
            text_dim=config.get("text_dim", 768),
            tab_dim=config.get("tab_dim", 128),
            fusion_dim=config.get("fusion_dim", 256)
        )
```

**Complexity Reduction:** MEDIUM (~20 lines → ~3 lines)

---

### 3. Trimodal Models (`models/lightning/trimodal/`)

#### pl_trimodal_cross_attn.py → Use BidirectionalCrossAttention

**Current State:** Complex bidirectional attention with FFN

**Refactoring:**
```python
# BEFORE: Custom implementation (~80 lines)
class TrimodalCrossAttn(pl.LightningModule):
    def __init__(self, config):
        self.attn_p2s = nn.MultiheadAttention(...)
        self.attn_s2p = nn.MultiheadAttention(...)
        self.ffn_primary = nn.Sequential(...)  # ~10 lines
        self.ffn_secondary = nn.Sequential(...)  # ~10 lines
        self.norm_primary = nn.LayerNorm(...)
        # ... many more layers

# AFTER: Use BidirectionalCrossAttention
from athelas.models.pytorch.fusion import BidirectionalCrossAttention

class TrimodalCrossAttn(pl.LightningModule):
    def __init__(self, config):
        # Primary/Secondary text fusion
        self.text_fusion = BidirectionalCrossAttention(
            d_model=config.get("d_model", 100),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
    
    def forward(self, batch):
        primary_text = ...
        secondary_text = ...
        
        primary_out, secondary_out, attn_weights = self.text_fusion(
            primary_text, secondary_text
        )
```

**Benefits:**
- ✅ ~60 lines of attention + FFN logic removed
- ✅ Attention weights available for analysis
- ✅ Uses MLPBlock internally (component reuse)

**Complexity Reduction:** VERY HIGH (~80 lines → ~5 lines)

---

### 4. Tabular Models (`models/lightning/tabular/`)

#### pl_tab_ae.py → Use TabularEmbedding

**Current State:** Custom tabular embedding

**Refactoring:**
```python
# BEFORE: Custom embedding
class TabAE(pl.LightningModule):
    def __init__(self, config):
        self.embedding = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU()
        )

# AFTER: Use TabularEmbedding
from athelas.models.pytorch.embeddings import TabularEmbedding, combine_tabular_fields

class TabAE(pl.LightningModule):
    def __init__(self, config):
        self.embedding = TabularEmbedding(
            input_dim=config.get("tab_dim"),
            hidden_dim=config.get("hidden_dim")
        )
    
    def combine_tab_data(self, batch):
        return combine_tabular_fields(
            batch, 
            self.tab_field_list, 
            self.device
        )
```

**Complexity Reduction:** LOW (already simple, but now reusable)

---

## Refactoring Strategy

### Phase 1: High-Impact, Low-Risk (Recommended Start)

**Priority 1:** Text and Bimodal CNN models
- `pl_text_cnn.py` → CNNEncoder
- `pl_bimodal_cnn.py` → CNNEncoder + TabularEmbedding + ConcatenationFusion

**Rationale:**
- High code duplication reduction
- CNNEncoder well-tested
- Clear 1:1 component mapping

**Estimated Effort:** 2-4 hours
**Lines Saved:** ~100-150 lines
**Risk:** Low (components are drop-in replacements)

---

### Phase 2: Fusion Mechanisms

**Priority 2:** Fusion-specific models
- `pl_bimodal_cross_attn.py` → CrossAttentionFusion
- `pl_bimodal_gate_fusion.py` → GateFusion
- `pl_bimodal_moe.py` → MixtureOfExperts
- `pl_trimodal_cross_attn.py` → BidirectionalCrossAttention

**Rationale:**
- Eliminates duplicated fusion logic
- Makes fusion strategy explicit and swappable
- Improves testability

**Estimated Effort:** 3-5 hours
**Lines Saved:** ~150-200 lines
**Risk:** Low-Medium (need to verify attention weights match)

---

### Phase 3: Embeddings

**Priority 3:** Tabular models
- `pl_tab_ae.py` → TabularEmbedding

**Rationale:**
- Standardizes tabular encoding
- Lower impact (less duplication currently)

**Estimated Effort:** 1-2 hours
**Lines Saved:** ~20-30 lines
**Risk:** Low

---

## Implementation Example: pl_bimodal_cnn.py Refactoring

### Before (Current)

```python
class BimodalCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        super().__init__()
        # ~250 lines total
        
        # Text subnetwork (duplicates entire TextCNN class)
        self.text_subnetwork = TextCNN(config, vocab_size, word_embeddings)
        
        # Tabular subnetwork
        self.tab_subnetwork = TabAE(config) if self.tab_field_list else None
        
        # Get dimensions
        text_dim = self.text_subnetwork.output_text_dim
        tab_dim = self.tab_subnetwork.output_tab_dim if self.tab_subnetwork else 0
        
        # Fusion network
        self.final_merge_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(tab_dim + text_dim, self.num_classes),
        )
    
    def forward(self, batch):
        tab_data = (
            self.tab_subnetwork.combine_tab_data(batch) 
            if self.tab_subnetwork 
            else None
        )
        return self._forward_impl(batch, tab_data)
    
    def _forward_impl(self, batch, tab_data):
        input_ids = batch[self.text_name]
        text_out = self.text_subnetwork(input_ids)
        tab_out = (
            self.tab_subnetwork(tab_data.float())
            if tab_data is not None
            else torch.zeros((text_out.size(0), 0), device=self.device)
        )
        combined = torch.cat([text_out, tab_out], dim=1)
        return self.final_merge_network(combined)
```

### After (With PyTorch Components)

```python
from athelas.models.pytorch.blocks import CNNEncoder
from athelas.models.pytorch.embeddings import TabularEmbedding, combine_tabular_fields
from athelas.models.pytorch.fusion import ConcatenationFusion

class BimodalCNN(pl.LightningModule):
    def __init__(self, config, vocab_size, word_embeddings):
        super().__init__()
        # ~150 lines total (40% reduction)
        
        # Text encoder - clean component
        self.text_encoder = CNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=word_embeddings.shape[1],
            kernel_sizes=config.get("kernel_size", [3, 5, 7]),
            num_channels=config.get("num_channels", [100, 100]),
            num_layers=config.get("num_layers", 2),
            output_dim=config.get("hidden_common_dim", 100),
            dropout=config.get("dropout_keep", 0.5),
            max_seq_len=config.get("max_sen_len", 512)
        )
        self.text_encoder.embeddings.weight = nn.Parameter(word_embeddings)
        
        # Tabular encoder - clean component
        if self.tab_field_list:
            tab_input_dim = len(self.tab_field_list)  # Simplified
            self.tab_encoder = TabularEmbedding(
                input_dim=tab_input_dim,
                hidden_dim=config.get("hidden_common_dim", 100)
            )
        else:
            self.tab_encoder = None
        
        # Fusion - explicit strategy
        text_dim = self.text_encoder.output_dim
        tab_dim = self.tab_encoder.hidden_dim if self.tab_encoder else 0
        
        self.fusion = ConcatenationFusion(
            input_dims=[text_dim, tab_dim] if tab_dim > 0 else [text_dim],
            output_dim=self.num_classes,
            use_activation=True
        )
    
    def forward(self, batch):
        # Clean, readable forward pass
        text_feat = self.text_encoder(batch[self.text_name])
        
        if self.tab_encoder:
            tab_data = combine_tabular_fields(batch, self.tab_field_list, self.device)
            tab_feat = self.tab_encoder(tab_data)
            logits = self.fusion(text_feat, tab_feat)
        else:
            logits = self.fusion(text_feat)
        
        return logits
```

**Improvements:**
- ✅ 40% line reduction (~250 → ~150 lines)
- ✅ Clear component boundaries
- ✅ No nested Lightning modules (TextCNN, TabAE)
- ✅ Explicit fusion strategy (easy to swap)
- ✅ Cleaner forward() method
- ✅ Components testable independently

---

## Testing Strategy

### 1. Backward Compatibility Tests

For each refactored module:

```python
def test_backward_compatibility():
    """Ensure refactored model produces same outputs as original."""
    
    # Load old model
    old_model = OldBimodalCNN(config, vocab_size, embeddings)
    old_model.load_state_dict(checkpoint)
    
    # Load new model (with components)
    new_model = NewBimodalCNN(config, vocab_size, embeddings)
    new_model.load_state_dict(checkpoint, strict=False)  # Some param names may change
    
    # Test on same batch
    batch = create_test_batch()
    
    with torch.no_grad():
        old_output = old_model(batch)
        new_output = new_model(batch)
    
    # Outputs should be identical (or very close)
    assert torch.allclose(old_output, new_output, atol=1e-5)
```

### 2. Component-Level Tests

```python
def test_cnn_encoder_equivalence():
    """Test CNNEncoder produces same outputs as original TextCNN."""
    
    # Original implementation
    original = TextCNN(config, vocab_size, embeddings)
    
    # New component
    encoder = CNNEncoder(vocab_size=vocab_size, ...)
    encoder.embeddings.weight = original.embeddings.weight
    
    tokens = torch.randint(0, vocab_size, (32, 50))
    
    with torch.no_grad():
        original_out = original(tokens)
        encoder_out = encoder(tokens)
    
    assert torch.allclose(original_out, encoder_out)
```

---

## Migration Checklist

For each Lightning module to refactor:

- [ ] Identify atomic components needed
- [ ] Create test to verify output equivalence
- [ ] Refactor __init__() to use components
- [ ] Refactor forward() to use components
- [ ] Run backward compatibility tests
- [ ] Update model documentation
- [ ] Verify training works (small test run)
- [ ] Check ONNX export still works
- [ ] Update any model-specific visualizations

---

## Benefits Summary

### Code Quality
- ✅ **40-60% line reduction** in model implementations
- ✅ **Eliminates duplication** across similar models
- ✅ **Clear separation of concerns** (encoding vs fusion vs classification)
- ✅ **Testable components** (test encoder, fusion, classifier independently)

### Maintainability
- ✅ **Single source of truth** for CNN encoder, fusion mechanisms, etc.
- ✅ **Easier to add new models** (compose existing components)
- ✅ **Easier to swap strategies** (change fusion mechanism in 1 line)
- ✅ **Consistent implementations** across models

### Developer Experience
- ✅ **Clearer code** (component names describe function)
- ✅ **Less cognitive load** (understand components separately)
- ✅ **Reusable knowledge** (learn CNNEncoder once, use everywhere)
- ✅ **Easier experimentation** (swap CNNEncoder ↔ LSTMEncoder)

---

## Risks and Mitigation

### Risk 1: Breaking Existing Checkpoints

**Mitigation:**
- Keep original implementations as `_legacy` modules
- Provide checkpoint migration scripts
- Add `strict=False` option when loading state dicts
- Version checkpoints with model architecture info

### Risk 2: Output Differences Due to Numeric Precision

**Mitigation:**
- Use comprehensive backward compatibility tests
- Allow small tolerance (1e-5) for floating point differences
- Document any expected differences
- Test on real data, not just synthetic

### Risk 3: ONNX Export Compatibility

**Mitigation:**
- Test ONNX export after each refactoring
- Keep export wrappers if needed
- Document any ONNX-specific requirements

---

## Recommended Approach

### Week 1: Foundation
1. Create backward compatibility test framework
2. Refactor `pl_text_cnn.py` as pilot project
3. Verify all tests pass
4. Document lessons learned

### Week 2-3: Bimodal Models
1. Refactor `pl_bimodal_cnn.py`
2. Refactor `pl_bimodal_cross_attn.py`
3. Refactor `pl_bimodal_gate_fusion.py`
4. Refactor `pl_bimodal_moe.py`

### Week 4: Trimodal & Cleanup
1. Refactor `pl_trimodal_cross_attn.py`
2. Refactor `pl_tab_ae.py`
3. Mark old implementations as legacy
4. Update documentation

---

## Success Metrics

- [ ] **Code reduction**: 40%+ line reduction in Lightning modules
- [ ] **Test coverage**: 100% backward compatibility tests pass
- [ ] **Training**: Models train to same accuracy as before
- [ ] **Inference**: ONNX export works
- [ ] **Developer satisfaction**: Easier to understand and modify

---

## Related Documents

- [PyTorch Fusion Component Reuse Analysis](../3_analysis/2026-01-05_pytorch_fusion_component_reuse_analysis.md)
- [PyTorch Module Reorganization Design](./pytorch_module_reorganization_design.md)
- [Zettelkasten Knowledge Management Principles](../7_resources/zettelkasten_knowledge_management_principles.md)
