import torch
import torch.nn as nn
from basic_blocks import OrderAttentionLayer, FeatureAttentionLayer


# Attention Classifier with one hot encoded categorical features as input
class OrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embedding_table: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
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
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embedding_table = dim_embedding_table
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers_order = n_layers_order
        self.n_layers_feature = n_layers_feature
        self.emb_tbl_use_bias = emb_tbl_use_bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.use_time_seq = use_time_seq
        self.return_seq = return_seq

        dim_embed = 2 * dim_embedding_table
        self.dim_embed = dim_embed

        # main blocks
        self.embedding = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )

        self.order_attention = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.num_heads,
            self.dropout,
            self.n_layers_order,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
            self.use_time_seq,
            self.return_seq,
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            self.dropout,
            self.n_layers_feature,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
        )

        self.layer_norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)

        if self.use_mlp:
            self.MLP = MLPBlock(
                self.n_num_features + self.n_engineered_num_features,
                1024,
                dim_embedding_table,
                0.1,
            )
            self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
            self.clf = nn.Sequential(
                #             nn.LayerNorm(dim_embed+dim_embedding_table),
                nn.Linear(dim_embed + dim_embedding_table + dim_embedding_table, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, n_classes),
            )
        else:
            self.clf = nn.Sequential(
                # nn.LayerNorm(dim_embed+dim_embedding_table),
                nn.Linear(dim_embed + dim_embedding_table, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, n_classes),
            )

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        x_engineered: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, D)
        Where B is the batch size, L is the sequence length and D is the number of features
        :return: a tensor of dimension (B, n_classes)
        """

        # Order attention
        if self.use_time_seq:
            x = self.order_attention(
                x_cat, x_num, time_seq, attn_mask, key_padding_mask
            )
        else:
            x = self.order_attention(x_cat, x_num, None, attn_mask, key_padding_mask)

        # Feature attention
        x_last = self.feature_attention(x_cat, x_num, x_engineered)

        # MLP
        if self.use_mlp:
            x_mlp = self.MLP(torch.cat([x_num[:, -1, :], x_engineered], dim=-1))
            x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        if self.use_mlp:
            ensemble = torch.cat([x, x_last, x_mlp], dim=-1)
        else:
            ensemble = torch.cat([x, x_last], dim=-1)
        #         scores = self.fc(ensemble)
        scores = self.clf(ensemble)

        return scores, ensemble


class TwoSeqMoEOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embedding_table: int,
        dim_attn_feedforward: int,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
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
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embedding_table = dim_embedding_table
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers_order = n_layers_order
        self.n_layers_feature = n_layers_feature
        self.emb_tbl_use_bias = emb_tbl_use_bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.use_time_seq = use_time_seq
        self.return_seq = return_seq

        dim_embed = 2 * dim_embedding_table
        self.dim_embed = dim_embed

        # main blocks
        self.embedding = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )

        # output dimension of order attention is (B, dim_embed)
        self.order_attention_cid = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.num_heads,
            self.dropout,
            self.n_layers_order,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
            self.use_time_seq,
            self.return_seq,
        )

        self.order_attention_ccid = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.num_heads,
            self.dropout,
            self.n_layers_order,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
            self.use_time_seq,
            self.return_seq,
        )

        # Gate function use a separate embedding table, with smaller size.
        self.embedding_gate = nn.Embedding(n_embedding + 2, 16, padding_idx=0)
        self.gate_emb = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            32,  # self.dim_embed,
            128,  # self.dim_attn_feedforward,
            self.embedding_gate,
            1,  # self.num_heads,
            self.dropout,
            1,  # self.n_layers_order,
            self.emb_tbl_use_bias,
            0,  # self.use_moe,
            1,  # self.num_experts,
            False,  # self.use_time_seq,
            False,  # self.return_seq,
        )
        self.gate_score = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            self.dropout,
            self.n_layers_feature,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
        )

        self.layer_norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)

        self.clf = nn.Sequential(
            # nn.LayerNorm(dim_embed + dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.n_classes),
        )

    def forward(
        self,
        x_seq_cat_cid: torch.Tensor,
        x_seq_num_cid: torch.Tensor,
        time_seq_cid: torch.Tensor,
        x_seq_cat_ccid: torch.Tensor,
        x_seq_num_ccid: torch.Tensor,
        time_seq_ccid: torch.Tensor,
        x_engineered: torch.Tensor,
        attn_mask=None,
        key_padding_mask_cid=None,
        key_padding_mask_ccid=None,
    ) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, D)
        Where B is the batch size, L is the sequence length and D is the number of features
        :return: a tensor of dimension (B, n_classes)
        """

        B, L, D = x_seq_cat_cid.shape

        # Gate function
        gate_emb_cid = self.gate_emb(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        gate_emb_ccid = self.gate_emb(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )

        gate_scores_raw = self.gate_score(
            torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1)
        )
        gate_scores = gate_scores_raw.clone()
        gate_scores[
            (torch.sum(key_padding_mask_ccid, dim=1) == 50).nonzero().squeeze(-1), 1
        ] = 0

        ccid_keep_idx = (
            (gate_scores[:, 1] > 0.05).nonzero().squeeze(-1).to(x_seq_cat_cid.device)
        )

        x_ccid = torch.zeros([B, self.dim_embed]).to(x_seq_cat_cid.device)

        # Order attention
        if self.use_time_seq:
            x_cid = self.order_attention_cid(
                x_seq_cat_cid,
                x_seq_num_cid,
                time_seq_cid,
                attn_mask,
                key_padding_mask_cid,
            )
            if len(ccid_keep_idx) > 0:
                x_ccid[ccid_keep_idx, :] = self.order_attention_ccid(
                    x_seq_cat_ccid[ccid_keep_idx, :, :],
                    x_seq_num_ccid[ccid_keep_idx, :, :],
                    time_seq_ccid[ccid_keep_idx, :],
                    attn_mask,
                    key_padding_mask_ccid[ccid_keep_idx, :],
                )
        else:
            x_cid = self.order_attention_cid(
                x_seq_cat_cid, x_seq_num_cid, None, attn_mask, key_padding_mask_cid
            )
            if len(ccid_keep_idx) > 0:
                x_ccid[ccid_keep_idx, :] = self.order_attention_ccid(
                    x_seq_cat_ccid[ccid_keep_idx, :, :],
                    x_seq_num_ccid[ccid_keep_idx, :, :],
                    None,
                    attn_mask,
                    key_padding_mask_ccid[ccid_keep_idx, :],
                )

        # Feature attention: cid and ccid have the save input as current order
        x_last = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # Ensemble embeddings
        ensemble_order = torch.einsum(
            "i,ij->ij", gate_scores[:, 0], x_cid
        ) + torch.einsum("i,ij->ij", gate_scores[:, 1], x_ccid)
        ensemble_order = self.layer_norm(ensemble_order)

        ensemble = torch.cat([ensemble_order, x_last], dim=-1)
        scores = self.clf(ensemble)

        return scores, ensemble
