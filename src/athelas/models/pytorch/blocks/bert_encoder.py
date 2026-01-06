"""
BERT-Based Text Encoder

Transformer-based encoder using pretrained BERT models for text encoding.

**Core Concept:**
Leverages pretrained BERT models (or variants like RoBERTa, DistilBERT) for
encoding text sequences. Handles chunked inputs (multiple text segments per sample)
by encoding each chunk and averaging the representations. Provides optional
layer reinitialization for domain adaptation.

**Architecture:**
1. Pretrained BERT model (from HuggingFace Transformers)
2. Pooling: Use [CLS] token representation (pooler_output)
3. Chunk aggregation: Average across multiple chunks if provided
4. Optional projection head to target dimension
5. Optional layer reinitialization for fine-tuning

**Parameters:**
- model_name (str): HuggingFace model identifier (default: "bert-base-cased")
- output_dim (int): Output dimension (optional projection)
- dropout (float): Dropout probability for projection head (default: 0.1)
- reinit_pooler (bool): Whether to reinitialize pooler layer (default: False)
- reinit_layers (int): Number of top layers to reinitialize (default: 0)
- gradient_checkpointing (bool): Enable gradient checkpointing for memory (default: False)

**Forward Signature:**
Input:
  - input_ids: (B, C, T) or (B, T) - Token IDs (C=chunks, T=seq_len)
  - attention_mask: (B, C, T) or (B, T) - Attention mask
  
Output:
  - encoded: (B, output_dim) or (B, hidden_size) - Encoded representations

**Dependencies:**
- transformers.AutoModel → Pretrained BERT models
- transformers.AutoConfig → Model configuration
- torch.nn.Linear → Optional projection head
- torch.nn.Dropout → Regularization

**Used By:**
- athelas.models.lightning.text.pl_bert → BERT text classifier
- athelas.models.lightning.bimodal.pl_bimodal_bert → Bimodal BERT model
- Any model requiring BERT-based text encoding

**Alternative Approaches:**
- athelas.models.pytorch.blocks.lstm_encoder → Recurrent encoding
- athelas.models.pytorch.blocks.cnn_encoder → Convolutional encoding
- athelas.models.pytorch.blocks.transformer_encoder → Custom transformer

**Usage Example:**
```python
from athelas.models.pytorch.blocks import BertEncoder

# Standard BERT encoder
encoder = BertEncoder(
    model_name="bert-base-cased",
    output_dim=256,
    dropout=0.1
)

# With chunked inputs (multiple segments per sample)
input_ids = torch.randint(0, 30522, (32, 4, 128))  # (batch, chunks, seq_len)
attention_mask = torch.ones(32, 4, 128)

encoded = encoder(input_ids, attention_mask)  # (32, 256)

# Single sequence per sample
input_ids = torch.randint(0, 30522, (32, 128))  # (batch, seq_len)
attention_mask = torch.ones(32, 128)

encoded = encoder(input_ids, attention_mask)  # (32, 256)
```

**Domain Adaptation:**
```python
# Reinitialize top layers for domain adaptation
encoder = BertEncoder(
    model_name="bert-base-cased",
    output_dim=256,
    reinit_pooler=True,
    reinit_layers=2  # Reinitialize top 2 transformer layers
)
```

**Implementation Notes:**
- Handles both single sequences (B, T) and chunked inputs (B, C, T)
- Chunked inputs are averaged after encoding (useful for long documents)
- Pooler output uses [CLS] token representation
- Optional layer reinitialization for domain-specific fine-tuning
- Gradient checkpointing available for large models

**When to Use:**
- ✅ Transfer learning from pretrained language models
- ✅ Text classification, sentiment analysis, NER, etc.
- ✅ When you have limited labeled data (leverage pretraining)
- ✅ When you need contextual embeddings

**References:**
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
- HuggingFace Transformers: https://huggingface.co/transformers/
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig


class BertEncoder(nn.Module):
    """
    BERT-based text encoder with optional projection and chunked input support.
    
    Encodes text using pretrained BERT models, with support for multiple
    text chunks per sample and optional domain adaptation via layer reinitialization.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        reinit_pooler: bool = False,
        reinit_layers: int = 0,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize BertEncoder.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "bert-base-cased")
            output_dim: Optional output dimension (adds projection head)
            dropout: Dropout probability for projection head
            reinit_pooler: Whether to reinitialize BERT pooler layer
            reinit_layers: Number of top transformer layers to reinitialize (0 = none)
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()
        self.model_name = model_name
        self.reinit_pooler = reinit_pooler
        self.reinit_layers = reinit_layers
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(
            model_name,
            output_attentions=False
        )
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        
        # Optional layer reinitialization for domain adaptation
        if reinit_pooler or reinit_layers > 0:
            self._reinitialize_layers()
        
        # Optional projection head
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bert_hidden_size, output_dim)
            )
            self.output_dim = output_dim
        else:
            self.projection = None
            self.output_dim = self.bert_hidden_size
    
    def _reinitialize_layers(self):
        """
        Reinitialize pooler and/or top transformer layers.
        
        Useful for domain adaptation when fine-tuning pretrained models
        on domain-specific data.
        """
        encoder = self.bert
        initializer_range = encoder.config.initializer_range
        
        # Reinitialize pooler (projection of [CLS] token)
        if self.reinit_pooler and hasattr(encoder, 'pooler'):
            encoder.pooler.dense.weight.data.normal_(
                mean=0.0, std=initializer_range
            )
            encoder.pooler.dense.bias.data.zero_()
            # Ensure gradients are enabled
            for p in encoder.pooler.parameters():
                p.requires_grad = True
        
        # Reinitialize top transformer layers
        if self.reinit_layers > 0 and hasattr(encoder, 'encoder'):
            for layer in encoder.encoder.layer[-self.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=initializer_range)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text sequences.
        
        Handles both single sequences and chunked inputs (multiple segments per sample).
        
        Args:
            input_ids: (B, T) or (B, C, T) - Token IDs
                      B=batch, C=chunks, T=sequence_length
            attention_mask: (B, T) or (B, C, T) - Attention mask (1=attend, 0=ignore)
            
        Returns:
            encoded: (B, output_dim) - Encoded representations
        """
        # Handle chunked inputs (B, C, T)
        if input_ids.dim() == 3:
            B, C, T = input_ids.shape
            # Flatten chunks into batch dimension
            input_ids = input_ids.view(B * C, T)
            attention_mask = attention_mask.view(B * C, T)
            
            # Check for empty chunks (all padding)
            valid_mask = attention_mask.sum(dim=1) > 0
            if not valid_mask.any():
                raise ValueError("All input chunks in batch are empty!")
            
            # Encode all chunks
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use pooler output ([CLS] token representation)
            pooled = outputs.pooler_output  # (B*C, hidden_size)
            
            # Reshape and average across chunks
            pooled = pooled.view(B, C, -1).mean(dim=1)  # (B, hidden_size)
        
        # Handle single sequence per sample (B, T)
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled = outputs.pooler_output  # (B, hidden_size)
        
        # Optional projection
        if self.projection is not None:
            encoded = self.projection(pooled)
        else:
            encoded = pooled
        
        return encoded
    
    def __repr__(self) -> str:
        proj_str = f", output_dim={self.output_dim}" if self.projection else ""
        reinit_str = f", reinit={self.reinit_layers}L" if self.reinit_layers > 0 else ""
        return (
            f"BertEncoder(model={self.model_name}, "
            f"hidden_size={self.bert_hidden_size}{proj_str}{reinit_str})"
        )


def get_bert_config(model_name: str = "bert-base-cased") -> AutoConfig:
    """
    Utility to inspect BERT model configuration.
    
    Useful for determining hidden size, number of layers, etc.
    before instantiating the full model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        config: Model configuration object
        
    Example:
        >>> config = get_bert_config("bert-base-cased")
        >>> print(f"Hidden size: {config.hidden_size}")
        Hidden size: 768
        >>> print(f"Num layers: {config.num_hidden_layers}")
        Num layers: 12
    """
    return AutoConfig.from_pretrained(model_name)


def create_bert_optimizer_groups(
    model: nn.Module,
    weight_decay: float = 0.01
) -> list:
    """
    Create optimizer parameter groups for BERT models.
    
    Separates parameters into two groups:
    1. Parameters with weight decay (weights)
    2. Parameters without weight decay (bias, LayerNorm)
    
    This is the standard approach from the BERT paper for fine-tuning,
    preventing overfitting while maintaining training stability.
    
    Args:
        model: PyTorch model (typically contains BERT)
        weight_decay: Weight decay value for applicable parameters (default: 0.01)
        
    Returns:
        param_groups: List of parameter group dicts for optimizer
        
    Example:
        >>> from torch.optim import AdamW
        >>> encoder = BertEncoder(model_name="bert-base-cased", output_dim=256)
        >>> param_groups = create_bert_optimizer_groups(encoder, weight_decay=0.01)
        >>> optimizer = AdamW(param_groups, lr=2e-5, eps=1e-8)
        
        >>> # Use in Lightning module
        >>> class MyModel(pl.LightningModule):
        ...     def configure_optimizers(self):
        ...         param_groups = create_bert_optimizer_groups(self, self.weight_decay)
        ...         optimizer = AdamW(param_groups, lr=self.lr, eps=self.adam_epsilon)
        ...         return optimizer
    """
    # Parameters that should NOT have weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    # Split parameters into groups
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return optimizer_grouped_parameters
