import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from TemporalMultiheadAttentionDelta import TemporalMultiheadAttention
from mixture_of_experts import MoE


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TimeEncode, self).__init__()

        self.time_dim = time_dim

        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)

    def forward(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(2)
        t = t.permute(1, 0, 2)

        return t


class TimeEncoder(torch.nn.Module):
    def __init__(self, time_dim):
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        self.periodic = nn.Linear(1, time_dim - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(2)
        t = t.permute(1, 0, 2)

        return t


def compute_FM_parallel(feature_embedding):
    summed_features_emb = torch.sum(feature_embedding, dim=-2)
    summed_features_emb_square = torch.square(summed_features_emb)

    # _________ square_sum part _____________
    squared_features_emb = torch.square(feature_embedding)
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)

    # ________ FM __________ for items level in each event
    FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

    return FM


class FeatureAggregation(torch.nn.Module):
    def __init__(self, num_feature):
        super(FeatureAggregation, self).__init__()

        self.dim_embed = num_feature

        self.encoder = nn.Sequential(
            nn.Linear(num_feature, num_feature // 2),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 2, num_feature // 4),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 4, num_feature // 8),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 8, num_feature // 16),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 16, num_feature // 32),
            nn.LeakyReLU(),
            nn.Linear(num_feature // 32, 1),
        )

    def forward(self, x):
        encode = self.encoder(x)

        return encode


class AttentionLayer(torch.nn.Module):
    """A simple multi-head attention layer inspired by Vaswani et al."""

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads=1,
        dropout=0.1,
        use_moe=True,
        num_experts=5,
        use_time_seq=True,
    ):
        super().__init__()

        # parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.use_time_seq = use_time_seq

        # main blocks
        if self.use_time_seq:
            self.multi_attn = TemporalMultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
        else:
            self.multi_attn = nn.modules.MultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)
        if use_moe:
            self.feedforward = MoE(
                dim=dim_embed,
                num_experts=num_experts,
                hidden_dim=dim_attn_feedforward,
                second_policy_train="random",
                second_policy_eval="random",
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, attn_mask=None, key_padding_mask=None
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (L, B, E)
        Where B is the batch size, L is the sequence length and E is embedding dimension
        :return: a tensor of dimension (L, B, E)
        """

        # multihead attention
        if self.use_time_seq:
            x2, _ = self.multi_attn(
                x, x, x, time, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        else:
            x2, _ = self.multi_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )

        # add & norm
        x = x + self.dropout1(x2)
        x = self.layer_norm1(x)

        # feedforward
        x2 = self.feedforward(
            x
        )  # (L, B, E) since the feed forward simple expand then reduce back to original dimension

        # add & norm
        x = x + self.dropout2(x2)
        x = self.layer_norm2(x)

        return x


class AttentionLayerPreNorm(torch.nn.Module):
    """A simple multi-head attention layer inspired by Vaswani et al."""

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads=1,
        dropout=0.1,
        use_moe=True,
        num_experts=5,
    ):
        super().__init__()

        # parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads

        # main blocks
        self.multi_attn = nn.modules.MultiheadAttention(
            dim_embed, num_heads, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)
        if use_moe:
            self.feedforward = MoE(
                dim=dim_embed,
                num_experts=num_experts,
                hidden_dim=dim_attn_feedforward,
                second_policy_train="random",
                second_policy_eval="random",
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )

    def forward(
        self, x: torch.Tensor, attn_mask=None, key_padding_mask=None
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (L, B, E)
        Where B is the batch size, L is the sequence length and E is embedding dimension
        :return: a tensor of dimension (L, B, E)
        """

        # pre norm
        x2 = self.layer_norm1(x)

        # multihead attention
        x2, _ = self.multi_attn(
            x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # add
        x = x + self.dropout1(x2)

        # pre norm
        x2 = self.layer_norm2(x)

        # feedforward
        x2 = self.feedforward(
            x2
        )  # (L, B, E) since the feed forward simple expand then reduce back to original dimension

        # add & norm
        x = x + self.dropout2(x2)

        return x


class OrderAttentionLayer(torch.nn.Module):
    """Multilayer order attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_embedding: int,
        seq_len: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        embedding_table: nn.Module,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        emb_tbl_use_bias=1,
        use_moe=True,
        num_experts=5,
        use_time_seq=True,
        return_seq=False,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.return_seq = return_seq
        self.use_time_seq = use_time_seq

        # main blocks
        self.dummy_order = nn.Parameter(torch.rand(1, dim_embed))

        embedding_table_dim = dim_embed // 2

        self.embedding_table_dim = embedding_table_dim

        #         self.embedding = nn.Embedding(n_embedding+2, embedding_table_dim, padding_idx=0)
        self.embedding = embedding_table
        self.layer_norm_feature = nn.LayerNorm(int(embedding_table_dim * 2))

        # stack multiple attention layers
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayer(
                    dim_embed,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    use_time_seq=use_time_seq,
                )
                for _ in range(n_layers_order)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, embedding_table_dim)
            )
            if emb_tbl_use_bias
            else None
        )

        self.feature_aggregation_cat = FeatureAggregation(n_cat_features)
        self.feature_aggregation_num = FeatureAggregation(n_num_features)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, D)
        Where B is the batch size, L is the sequence length and D is the number of features
        :return: a tensor of dimension (B, n_classes)
        """

        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # embedding for one hot encoded categorical features, linear embedding for numerical features
        # (B, L, D) => (B, L, E)
        cat_indices = x_cat.int()
        # (B, L, D, E)
        x_cat_all = self.embedding(cat_indices)

        x_cat = self.feature_aggregation_cat(x_cat_all.permute(0, 1, 3, 2)).squeeze(-1)

        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num_all = self.embedding(num_indices) * (x_num[..., None])
        #         x_num = torch.mean(x_num_all, dim=-2)
        x_num = self.feature_aggregation_num(x_num_all.permute(0, 1, 3, 2)).squeeze(-1)

        x = torch.cat([x_cat, x_num], dim=-1)

        x = x.permute(1, 0, 2)  # attention layer takes matrix in order of (L, B, E)

        x = self.layer_norm_feature(x)

        dummy = self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)
        x = self.layer_norm(x)

        if self.use_time_seq:
            time_seq = torch.cat([time_seq, torch.zeros([B, 1, 1]).to(x.device)], dim=1)
            time_seq = time_seq.permute(1, 0, 2)
        else:
            time_seq = None

        # multilayer attention
        # x: (B, L, E)
        # attn_mask shape (L, L), will broadcasting to (B, L, L)
        # key_padding_mask shape (B, L)

        # multilayer attention
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)

        if not self.return_seq:
            x = torch.transpose(x, 0, 1)[:, -1, :]
        else:
            x = torch.transpose(x, 0, 1)

        return x


# Attention Classifier with one hot encoded categorical features as input
class FeatureAttentionLayer(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_embedding: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        embedding_table: nn.Module,
        embedding_table_engineered: nn.Module,
        num_heads=1,
        dropout=0.1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
        use_moe=True,
        num_experts=5,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_embedding = n_embedding
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads

        # main blocks
        embedding_table_dim = dim_embed // 2

        self.embedding_table_dim = embedding_table_dim

        self.embedding = embedding_table

        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    embedding_table_dim,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, embedding_table_dim)
            )
            if emb_tbl_use_bias
            else None
        )

        # self.embedding_engineered = nn.Embedding(n_engineered_num_features+1, embedding_table_dim, padding_idx=0)
        self.embedding_engineered = embedding_table_engineered
        self.layer_norm_engineered = nn.LayerNorm(embedding_table_dim)
        if self.n_engineered_num_features > 0:
            self.engineered_emb_tbl_bias = (
                nn.Parameter(
                    torch.randn(self.n_engineered_num_features, embedding_table_dim)
                )
                if emb_tbl_use_bias and self.n_engineered_num_features > 0
                else None
            )

    def forward(
        self, x_cat: torch.Tensor, x_num: torch.Tensor, x_engineered: torch.Tensor
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, D)
        Where B is the batch size, L is the sequence length and D is the number of features
        :return: a tensor of dimension (B, n_classes)
        """

        if x_engineered is not None:
            x_engineered = x_engineered.float()

        # embedding for one hot encoded categorical features, linear embedding for numerical features

        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # (B, L, D) => (B, L, E)
        cat_indices = x_cat.int()
        # (B, L, D, E)
        x_cat_all = self.embedding(cat_indices)

        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num_all = self.embedding(num_indices) * (x_num[..., None])

        # fc layer

        x_cat_last = x_cat_all[:, -1, :, :]
        x_num_last = x_num_all[:, -1, :, :]

        x_last = torch.cat([x_cat_last, x_num_last], dim=1)
        if self.emb_tbl_bias is not None:
            x_last = x_last + self.emb_tbl_bias[None]

        if self.n_engineered_num_features > 0:
            engineered_indices = torch.arange(1, self.n_engineered_num_features + 1).to(
                x_cat.device
            )
            #             print(self.embedding_engineered(engineered_indices).shape)
            #             print((x_engineered[..., None]).shape)
            x_engineered_emb = (
                self.embedding_engineered(engineered_indices)
                * (x_engineered[..., None])
            )
            if self.emb_tbl_bias is not None:
                x_engineered_emb = x_engineered_emb + self.engineered_emb_tbl_bias[None]
            x_last = torch.cat(
                [
                    x_last,
                    x_engineered_emb,
                    self.embedding_engineered(
                        torch.zeros([B, 1]).int().to(x_cat.device)
                    ),
                ],
                dim=1,
            )
        else:
            x_last = torch.cat(
                [
                    x_last,
                    self.embedding_engineered(
                        torch.zeros([B, 1]).int().to(x_cat.device)
                    ),
                ],
                dim=1,
            )

        x_last = x_last.permute(1, 0, 2)
        x_last = self.layer_norm_engineered(x_last)
        #         print(x_last.shape)
        for att_layer_feature in self.layer_stack_feature:
            x_last = att_layer_feature(x_last, None, None)

        x_last = torch.transpose(x_last, 0, 1)[:, -1, :]  # dim_emb/2

        return x_last
