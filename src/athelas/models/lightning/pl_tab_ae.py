import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl  # Or torch.nn.Module if not training independently
from typing import Dict, Union, List

from pydantic import BaseModel, Field, computed_field, field_validator, ValidationInfo


class TabularEmbeddingConfig(BaseModel):
    tab_field_list: List[str]
    hidden_common_dim: int
    is_binary: bool = True  # Added for clarity (though not used)
    num_classes: int = 2  # Added for clarity (though not used)
    
    @computed_field  # Pydantic v2: computed instead of validator
    @property
    def input_tab_dim(self) -> int:
        return len(self.tab_field_list)

    @property
    def output_tab_dim(self) -> int:
        return self.hidden_common_dim

    @field_validator("tab_field_list")
    @classmethod
    def validate_tab_field_list(cls, v: List[str], info: ValidationInfo) -> List[str]:
        if not v:
            raise ValueError("tab_field_list must not be empty")
        return v


class TabularEmbeddingModule(pl.LightningModule): 
    def __init__(self, config: Union[Dict, TabularEmbeddingConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = TabularEmbeddingConfig(**config)
        self.config = config.model_dump()

        input_dim = self.config["input_tab_dim"]
        hidden_dim = self.config["hidden_common_dim"]

        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.output_tab_dim = hidden_dim
        self.save_hyperparameters(self.config)

        
    def combine_tab_data(self, batch: Dict[str, Union[torch.Tensor, List]]) -> torch.Tensor:
        """
        Combines tabular fields into a single tensor of shape [B, input_tab_dim]
        """
        features = []
        device = next(self.parameters()).device
        for field in self.config["tab_field_list"]:
            if field not in batch:
                raise KeyError(f"Missing field '{field}' in batch during tabular combination")

            val = batch[field]
            if isinstance(val, list):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            elif isinstance(val, torch.Tensor):
                val = val.to(dtype=torch.float32, device=device)
            else:
                raise TypeError(f"Unsupported type for field {field}: {type(val)}")
            if val.dim() == 1:
                val = val.unsqueeze(1)
            features.append(val)
        return torch.cat(features, dim=1)
    

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Returns embedding vector from tabular input.
        """
        if isinstance(inputs, dict):
            inputs = self.combine_tab_data(inputs)
        
        if inputs.shape[1] != self.embedding_layer[1].in_features:
            raise ValueError(f"Expected input with {self.embedding_layer[1].in_features} features, got {inputs.shape[1]}")

        return self.embedding_layer(inputs)
    
    def __str__(self):
        return f"TabAE(input_dim={self.embedding_layer[1].in_features}, output_dim={self.output_tab_dim})"


class TabAE(TabularEmbeddingModule):  # Inherit for combine_tab_data and config
    def __init__(self, config: Union[Dict, TabularEmbeddingConfig]):
        super().__init__(config)
