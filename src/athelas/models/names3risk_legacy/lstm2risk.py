from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class LSTMConfig:
    embedding_size: int = 16
    dropout_rate: float = 0.2
    hidden_size: int = 128
    n_tab_features: int = 100
    n_embed: int = 4000
    n_lstm_layers: int = 4


class AttentionPooling(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.attention = nn.Linear(2 * config.hidden_size, 1)

    def forward(self, sequence, lengths):
        # Compute attention scores
        scores = self.attention(sequence)  # Shape: [batch, seq_len, 1]

        if lengths is not None:
            # Create and apply mask to remove any padding influence
            mask = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Get normalized weights and apply them
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights * sequence, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.LayerNorm(4 * config.hidden_size),
            nn.Linear(4 * config.hidden_size, 16 * config.hidden_size),
            nn.ReLU(),
            nn.Linear(16 * config.hidden_size, 4 * config.hidden_size),
        )

    def forward(self, x):
        return x + self.ffwd(x)


class TextProjection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.n_embed, config.embedding_size)

        self.lstm = nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            num_layers=config.n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout_rate,
        )
        self.lstm_pooling = AttentionPooling(config)
        self.lstm_norm = nn.LayerNorm(2 * config.hidden_size)

    def forward(self, tokens, lengths=None):
        embedding = self.token_embedding_table(tokens)

        if lengths is None:
            lstm_output, _ = self.lstm(embedding)
        else:
            packed_embedding = nn.utils.rnn.pack_padded_sequence(
                embedding, lengths.cpu(), batch_first=True, enforce_sorted=True
            )

            # Run LSTM
            packed_output, _ = self.lstm(packed_embedding)

            # Unpack sequence
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

        # Apply attention and layer norm
        lstm_hidden = self.lstm_pooling(lstm_output, lengths)
        lstm_hidden = self.lstm_norm(lstm_hidden)
        return lstm_hidden


class LSTM2Risk(nn.Module):

    def __init__(self, config: LSTMConfig):
        super().__init__()

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
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        X_text = batch["text"]
        X_tab = batch["tabular"]
        lengths = batch.get("text_length")

        lstm_hidden = self.text_projection(X_text, lengths=lengths)
        tab_hidden = self.tab_projection(X_tab)

        # Combine and predict
        return self.net(torch.cat([lstm_hidden, tab_hidden], dim=-1))

    @staticmethod
    def create_collate_fn(pad_token):
        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            tabs = [item["tabular"] for item in batch]
            labels = [item["label"] for item in batch]
            lengths = torch.tensor([len(text) for text in texts])

            lengths, sort_idx = lengths.sort(descending=True)
            texts = [texts[i] for i in sort_idx]
            tabs = [tabs[i] for i in sort_idx]
            labels = [labels[i] for i in sort_idx]

            texts_padded = pad_sequence(
                texts, batch_first=True, padding_value=pad_token
            )

            tabs_stacked = torch.stack(tabs)
            labels_stacked = torch.stack(labels)

            return {
                "text": texts_padded,
                "text_length": lengths,
                "tabular": tabs_stacked,
                "label": labels_stacked,
            }

        return collate_fn
