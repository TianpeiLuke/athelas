from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class TransformerConfig:
    embedding_size: int = 128
    dropout_rate: float = 0.2
    hidden_size: int = 256
    n_tab_features: int = 100
    n_embed: int = 4000
    n_blocks: int = 8
    n_heads: int = 8
    block_size: int = 100


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.embedding_size % config.n_heads == 0
        self.head_size = config.embedding_size // config.n_heads
        self.qkv = nn.Linear(config.embedding_size, 3 * self.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        wei = q @ k.mT * self.head_size**-0.5

        if attn_mask is not None:
            wei = wei.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attn_mask=None):
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embedding_size)
        self.ln2 = nn.LayerNorm(config.embedding_size)

    def forward(self, x, attn_mask=None):
        x = x + self.sa(self.ln1(x), attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4 * config.hidden_size, 4 * config.hidden_size),
            nn.ReLU(),
            nn.Linear(4 * config.hidden_size, 4 * config.hidden_size),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x):
        return x + self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Linear(config.embedding_size, 1)

    def forward(self, x, mask=None):
        scores = self.attention(x)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(scores, dim=1)  # Shape: [batch, seq_len, 1]

        return torch.sum(weights * x, dim=1)


class TextProjection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.n_embed, config.embedding_size)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embedding_size)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])

        self.tab_layer = nn.Linear(config.n_tab_features, config.hidden_size)
        self.attention_pooling = AttentionPooling(config)
        self.proj = nn.Linear(config.embedding_size, 2 * config.hidden_size)

    def forward(self, X_text, attn_mask=None):
        B, T = X_text.shape

        tok_emb = self.token_embedding_table(X_text)
        pos_emb = self.position_embedding_table(torch.arange(T, device=X_text.device))
        emb = tok_emb + pos_emb

        for block in self.blocks:
            emb = block(emb, attn_mask)

        text_pooled = self.attention_pooling(emb)
        text_output = self.proj(text_pooled)
        return text_output


class Transformer2Risk(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.text_projection = TextProjection(config)

        self.tab_projection = nn.Sequential(
            nn.BatchNorm1d(config.n_tab_features),
            nn.Linear(config.n_tab_features, 2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(2 * config.hidden_size, 2 * config.hidden_size),
            nn.LayerNorm(2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )

        self.net = nn.Sequential(
            ResidualBlock(config),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(config),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(config),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(config),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(4 * config.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch, attn_mask=None):
        X_text = batch["text"]
        X_tab = batch["tabular"]
        attn_mask = batch.get("attn_mask")

        text_hidden = self.text_projection(X_text, attn_mask=attn_mask)
        tab_hidden = self.tab_projection(X_tab)

        return self.net(torch.cat([text_hidden, tab_hidden], dim=-1))

    def create_collate_fn(self, pad_token):
        def collate_fn(batch):

            texts = [item["text"][: self.block_size] for item in batch]
            tabs = [item["tabular"] for item in batch]
            labels = [item["label"] for item in batch]

            texts_padded = pad_sequence(
                texts, batch_first=True, padding_value=pad_token
            )
            tabs_stacked = torch.stack(tabs)
            labels_stacked = torch.stack(labels)

            attn_mask = texts_padded != pad_token

            return {
                "text": texts_padded,
                "tabular": tabs_stacked,
                "label": labels_stacked,
                "attn_mask": attn_mask,
            }

        return collate_fn
