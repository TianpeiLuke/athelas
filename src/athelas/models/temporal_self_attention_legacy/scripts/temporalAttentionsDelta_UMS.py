from basic_blocks import *


# ========
class TwoViewUMSFusionOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
        cva_mask=True,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            False,
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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            False,
        )

        #         # whether to use mask for cva
        #         self.cva_mask = cva_mask
        #         # cva attn for acc, src should be acc, target should be payment
        #         self.cva_acc_attn = nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=1, batch_first=True)
        #         self.cva_acc_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        #         # cva attn for acc, src should be acc, target should be payment
        #         self.cva_pay_attn = nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=1, batch_first=True)
        #         self.cva_pay_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        #         # linear layer to reduce the dimension
        #         self.cva_linear = nn.Linear(in_features=2*self.dim_embed, out_features=self.dim_embed)

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        #         self.fc = torch.nn.Linear(dim_embed+dim_embedding_table, n_classes)
        #         self.clf = nn.Sequential(
        #             nn.LayerNorm(dim_embed+dim_embedding_table),
        # #                                          nn.Linear(dim_embed+dim_embedding_table, dim_embed),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(dropout),
        #                                          nn.Linear(dim_embed+dim_embedding_table, n_classes))
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        # if self.use_mlp:
        #     self.MLP = MLPBlock(self.n_num_features+self.n_engineered_num_features, 1024, dim_embedding_table, 0.1)
        #     self.clf = nn.Sequential(
        #         #             nn.LayerNorm(dim_embed+dim_embedding_table),
        #         nn.Linear(dim_embed+dim_embedding_table+dim_embedding_table, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(1024, n_classes))
        # else:

        self.clf_acc = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

        self.clf_pay = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
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

        # Order attention
        x_cid = self.order_attention_cid(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        x_ccid = self.order_attention_ccid(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )

        #         # CVA Fusion
        #         if self.cva_mask:
        #             x_cid_fused, _ = self.cva_acc_attn.forward(query=x_ccid[:, [-1], :],
        #                                                        key=x_cid, value=x_cid,
        #                                                        key_padding_mask=key_padding_mask_cid)
        #             x_ccid_fused, _ = self.cva_pay_attn.forward(query=x_cid[:, [-1], :],
        #                                                        key=x_ccid, value=x_ccid,
        #                                                        key_padding_mask=key_padding_mask_ccid)
        #         else:
        #             x_cid_fused, _ = self.cva_acc_attn.forward(query=x_ccid[:, [-1], :],
        #                                                        key=x_cid, value=x_cid,
        #                                                        key_padding_mask=None)
        #             x_ccid_fused, _ = self.cva_pay_attn.forward(query=x_cid[:, [-1], :],
        #                                                        key=x_ccid, value=x_ccid,
        #                                                        key_padding_mask=None)

        #         # CVA skip conn & norm
        #         x_cid_fused += x_cid[:, [-1], :]
        #         x_cid_fused = self.cva_acc_norm(x_cid_fused)  # still have the dummy dimension
        #         x_ccid_fused += x_ccid[:, [-1], :]
        #         x_ccid_fused = self.cva_pay_norm(x_ccid_fused)  # still have the dummy dimension

        #         # CVA dimension reduction
        #         x = self.cva_linear.forward(torch.cat([x_cid_fused[:, -1, :], x_ccid_fused[:, -1, :]], dim=-1))

        # Feature attention on acc only
        x_last = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # MLP TODO: commented out
        # if self.use_mlp:
        #     x_mlp = self.MLP(torch.cat([x_num[:,-1,:], x_engineered], dim=-1))
        #     x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        # if self.use_mlp:
        #     ensemble = torch.cat([x,x_last, x_mlp], dim=-1)
        # else:

        #         print(x_cid.shape, x_last.shape)

        ensemble_cid = torch.cat([x_cid, x_last], dim=-1)
        #  scores = self.fc(ensemble)
        scores_cid = self.clf_acc(ensemble_cid)

        ensemble_ccid = torch.cat([x_ccid, x_last], dim=-1)
        #  scores = self.fc(ensemble)
        scores_ccid = self.clf_acc(ensemble_ccid)

        scores = torch.log(torch.exp(scores_cid) + torch.exp(scores_ccid))

        #         print(scores.shape)

        #         print(scores_cid.shape, scores_ccid.shape, scores.shape)
        #         print(scores_cid, scores_ccid, scores)

        return scores, ensemble_cid


########### Feature Attention Classifier ###########


class FeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

        self.embedding = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        if self.use_mlp:
            self.MLP = MLPBlock(
                self.n_num_features + self.n_engineered_num_features,
                1024,
                dim_embedding_table,
                0.1,
            )
            self.clf = nn.Sequential(
                #             nn.LayerNorm(dim_embed+dim_embedding_table),
                nn.Linear(dim_embed + dim_embedding_table + dim_embedding_table, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, n_classes),
            )
        else:
            self.clf = nn.Sequential(
                #             nn.LayerNorm(dim_embed+dim_embedding_table),
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

        # Feature attention
        x_last = self.feature_attention(x_cat, x_num, x_engineered)

        # MLP
        if self.use_mlp:
            x_mlp = self.MLP(torch.cat([x_num[:, -1, :], x_engineered], dim=-1))
            x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        if self.use_mlp:
            ensemble = torch.cat([x_last, x_mlp], dim=-1)
        else:
            ensemble = x_last
        #         scores = self.fc(ensemble)
        scores = self.clf(ensemble)

        return scores, ensemble


######################## Other Legacy Models #############################


# Attention Classifier with one hot encoded categorical features as input
class OrderAttentionNormOneHotClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        dim_embed: int,
        dim_time_embed: int,
        dim_attn_feedforward: int,
        use_time_seq: bool,
        num_heads=1,
        dropout=0.1,
        n_layers=1,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.dim_embed = dim_embed
        self.dim_time_embed = dim_time_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_time_seq = use_time_seq
        self.num_heads = num_heads

        # main blocks
        self.dummy_order = nn.Parameter(torch.rand(1, dim_embed))

        self.embedding = nn.Embedding(n_embedding + 1, dim_embed // 2, padding_idx=0)
        self.layer_norm_feature = nn.LayerNorm(dim_embed)

        # stack multiple attention layers
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayer(
                    dim_embed,
                    dim_time_embed,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_time = nn.LayerNorm(dim_time_embed)
        self.layer_norm = nn.LayerNorm(dim_embed)
        self.fc = torch.nn.Linear(dim_embed, n_classes)

    #         self.time_encoder = TimeEncoder(dim_time_embed)
    # self.time_encoder = TimeEncode(dim_time_embed)

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
        x_cat = torch.mean(self.embedding(cat_indices), dim=-2)

        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num = torch.mean(self.embedding(num_indices) * (x_num[..., None]), dim=-2)

        x = torch.cat([x_cat, x_num], dim=-1)

        x = x.permute(1, 0, 2)  # attention layer takes matrix in order of (L, B, E)

        x = self.layer_norm_feature(x)

        dummy = self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)
        x = self.layer_norm(x)

        #         print(time_seq.shape, torch.zeros([B,1,1]).shape)
        time_seq = torch.cat([time_seq, torch.zeros([B, 1, 1]).to(x.device)], dim=1)
        time_seq = time_seq.permute(1, 0, 2)

        # multilayer attention
        # x: (B, L, E)
        # attn_mask shape (L, L), will broadcasting to (B, L, L)
        # key_padding_mask shape (B, L)

        # multilayer attention
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)

        x = torch.transpose(x, 0, 1)[:, -1, :]

        # fc layer
        scores = self.fc(x)

        return scores, x


# Attention Classifier with one hot encoded categorical features as input
class OrderStaticFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_time_embed: int,
        dim_attn_feedforward: int,
        use_time_seq: bool,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_time_embed = dim_time_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_time_seq = use_time_seq
        self.num_heads = num_heads

        # main blocks
        self.dummy_order = nn.Parameter(torch.rand(1, dim_embed))

        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

        self.embedding = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )
        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )
        self.layer_norm_feature = nn.LayerNorm(int(dim_embedding_table * 2))
        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        self.fc = torch.nn.Linear(dim_embed + dim_embedding_table, n_classes)
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        # stack multiple attention layers
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayer(
                    dim_embed,
                    dim_time_embed,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_order)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_time = nn.LayerNorm(dim_time_embed)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
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

        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # embedding for one hot encoded categorical features, linear embedding for numerical features
        # (B, L, D) => (B, L, E)
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)
        x_cat = torch.mean(x_cat_all, dim=-2)

        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num_all = self.embedding(num_indices) * (x_num[..., None])
        x_num = torch.mean(x_num_all, dim=-2)

        x = torch.cat([x_cat, x_num], dim=-1)

        x = x.permute(1, 0, 2)  # attention layer takes matrix in order of (L, B, E)

        x = self.layer_norm_feature(x)

        dummy = self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)
        x = self.layer_norm(x)

        # multilayer attention
        # x: (B, L, E)
        # attn_mask shape (L, L), will broadcasting to (B, L, L)
        # key_padding_mask shape (B, L)

        time_seq = torch.cat([time_seq, torch.zeros([B, 1, 1]).to(x.device)], dim=1)
        time_seq = time_seq.permute(1, 0, 2)

        # multilayer attention
        # x: (B, L, E)
        # attn_mask shape (L, L), will broadcasting to (B, L, L)
        # key_padding_mask shape (B, L)

        # multilayer attention
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)

        x = torch.transpose(x, 0, 1)[:, -1, :]

        # fc layer

        engineered_indices = torch.arange(1, self.n_engineered_num_features + 1).to(
            x_engineered.device
        )
        x_engineered_emb = (
            self.embedding_engineered(engineered_indices) * (x_engineered[..., None])
        )
        if self.emb_tbl_bias is not None:
            x_engineered_emb = x_engineered_emb + self.engineered_emb_tbl_bias[None]
        x_last = torch.cat(
            [
                x_engineered_emb,
                self.embedding_engineered(
                    torch.zeros([B, 1]).int().to(x_engineered.device)
                ),
            ],
            dim=1,
        )

        x_last = x_last.permute(1, 0, 2)
        x_last = self.layer_norm_engineered(x_last)

        for att_layer_feature in self.layer_stack_feature:
            x_last = att_layer_feature(x_last, None, None)

        x_last = torch.transpose(x_last, 0, 1)[:, -1, :]

        ensemble = torch.cat([x, x_last], dim=-1)
        scores = self.fc(ensemble)

        return scores, ensemble


class TwoViewMLPFusionOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
        )

        self.mlp_fusion = nn.Sequential(
            nn.Linear(in_features=2 * self.dim_embed, out_features=4 * self.dim_embed),
            nn.ReLU(),
            nn.Linear(in_features=4 * self.dim_embed, out_features=self.dim_embed),
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        #         self.fc = torch.nn.Linear(dim_embed+dim_embedding_table, n_classes)
        #         self.clf = nn.Sequential(
        #             nn.LayerNorm(dim_embed+dim_embedding_table),
        # #                                          nn.Linear(dim_embed+dim_embedding_table, dim_embed),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(dropout),
        #                                          nn.Linear(dim_embed+dim_embedding_table, n_classes))
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        # if self.use_mlp:
        #     self.MLP = MLPBlock(self.n_num_features+self.n_engineered_num_features, 1024, dim_embedding_table, 0.1)
        #     self.clf = nn.Sequential(
        #         #             nn.LayerNorm(dim_embed+dim_embedding_table),
        #         nn.Linear(dim_embed+dim_embedding_table+dim_embedding_table, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(1024, n_classes))
        # else:

        self.clf = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
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

        # Order attention
        x_cid = self.order_attention_cid(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        x_ccid = self.order_attention_ccid(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )

        # Linear Fusion
        x = self.mlp_fusion(torch.cat([x_cid, x_ccid], dim=-1))

        # Feature attention
        x_last = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # MLP TODO: commented out
        # if self.use_mlp:
        #     x_mlp = self.MLP(torch.cat([x_num[:,-1,:], x_engineered], dim=-1))
        #     x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        # if self.use_mlp:
        #     ensemble = torch.cat([x,x_last, x_mlp], dim=-1)
        # else:

        ensemble = torch.cat([x, x_last], dim=-1)
        #  scores = self.fc(ensemble)
        scores = self.clf(ensemble)

        return scores, ensemble


class TwoViewCVAFusionOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
        cva_mask=True,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            True,
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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            True,
        )

        # whether to use mask for cva
        self.cva_mask = cva_mask
        # cva attn for acc, src should be acc, target should be payment
        self.cva_acc_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_acc_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        # cva attn for acc, src should be acc, target should be payment
        self.cva_pay_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_pay_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        # linear layer to reduce the dimension
        self.cva_linear = nn.Linear(
            in_features=2 * self.dim_embed, out_features=self.dim_embed
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        #         self.fc = torch.nn.Linear(dim_embed+dim_embedding_table, n_classes)
        #         self.clf = nn.Sequential(
        #             nn.LayerNorm(dim_embed+dim_embedding_table),
        # #                                          nn.Linear(dim_embed+dim_embedding_table, dim_embed),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(dropout),
        #                                          nn.Linear(dim_embed+dim_embedding_table, n_classes))
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        # if self.use_mlp:
        #     self.MLP = MLPBlock(self.n_num_features+self.n_engineered_num_features, 1024, dim_embedding_table, 0.1)
        #     self.clf = nn.Sequential(
        #         #             nn.LayerNorm(dim_embed+dim_embedding_table),
        #         nn.Linear(dim_embed+dim_embedding_table+dim_embedding_table, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(1024, n_classes))
        # else:

        self.clf = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
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

        # Order attention
        x_cid = self.order_attention_cid(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        x_ccid = self.order_attention_ccid(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )

        # CVA Fusion
        if self.cva_mask:
            x_cid_fused, _ = self.cva_acc_attn.forward(
                query=x_ccid[:, [-1], :],
                key=x_cid,
                value=x_cid,
                key_padding_mask=key_padding_mask_cid,
            )
            x_ccid_fused, _ = self.cva_pay_attn.forward(
                query=x_cid[:, [-1], :],
                key=x_ccid,
                value=x_ccid,
                key_padding_mask=key_padding_mask_ccid,
            )
        else:
            x_cid_fused, _ = self.cva_acc_attn.forward(
                query=x_ccid[:, [-1], :], key=x_cid, value=x_cid, key_padding_mask=None
            )
            x_ccid_fused, _ = self.cva_pay_attn.forward(
                query=x_cid[:, [-1], :], key=x_ccid, value=x_ccid, key_padding_mask=None
            )

        # CVA skip conn & norm
        x_cid_fused += x_cid[:, [-1], :]
        x_cid_fused = self.cva_acc_norm(x_cid_fused)  # still have the dummy dimension
        x_ccid_fused += x_ccid[:, [-1], :]
        x_ccid_fused = self.cva_pay_norm(x_ccid_fused)  # still have the dummy dimension

        # CVA dimension reduction
        x = self.cva_linear.forward(
            torch.cat([x_cid_fused[:, -1, :], x_ccid_fused[:, -1, :]], dim=-1)
        )

        # Feature attention on acc only
        x_last = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # MLP TODO: commented out
        # if self.use_mlp:
        #     x_mlp = self.MLP(torch.cat([x_num[:,-1,:], x_engineered], dim=-1))
        #     x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        # if self.use_mlp:
        #     ensemble = torch.cat([x,x_last, x_mlp], dim=-1)
        # else:

        ensemble = torch.cat([x, x_last], dim=-1)
        #  scores = self.fc(ensemble)
        scores = self.clf(ensemble)

        return scores, ensemble


class TwoViewCVADoubleFusionOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            True,
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
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            True,
        )

        # whether to use mask for cva        # cva attn for acc, src should be acc, target should be payment
        self.cva_acc_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_acc_norm = nn.LayerNorm(normalized_shape=self.dim_embed)

        # cva attn for acc, src should be acc, target should be payment
        self.cva_pay_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_pay_norm = nn.LayerNorm(normalized_shape=self.dim_embed)

        # linear layer to reduce the dimension
        self.cva_order_linear = nn.Linear(
            in_features=2 * self.dim_embed, out_features=self.dim_embed
        )

        # cva attn for order attention output, src should be order att, target should be feature att
        self.cva_order_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_order_norm = nn.LayerNorm(normalized_shape=self.dim_embed)

        # linear layer to expand the dimension
        self.cva_feat_linear = nn.Linear(
            in_features=self.dim_embed // 2, out_features=self.dim_embed
        )

        # cva attn for feature attention output
        self.cva_feat_attn = nn.MultiheadAttention(
            embed_dim=self.dim_embed, num_heads=1, batch_first=True
        )
        self.cva_feat_norm = nn.LayerNorm(normalized_shape=self.dim_embed)

        # linear layer to reduce the dimension
        self.cva_feat_order_linear = nn.Linear(
            in_features=2 * self.dim_embed,
            out_features=(self.dim_embed + self.dim_embed // 2),
        )

        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
            return_seq=False,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        #         self.fc = torch.nn.Linear(dim_embed+dim_embedding_table, n_classes)
        #         self.clf = nn.Sequential(
        #             nn.LayerNorm(dim_embed+dim_embedding_table),
        # #                                          nn.Linear(dim_embed+dim_embedding_table, dim_embed),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(dropout),
        #                                          nn.Linear(dim_embed+dim_embedding_table, n_classes))
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        # if self.use_mlp:
        #     self.MLP = MLPBlock(self.n_num_features+self.n_engineered_num_features, 1024, dim_embedding_table, 0.1)
        #     self.clf = nn.Sequential(
        #         #             nn.LayerNorm(dim_embed+dim_embedding_table),
        #         nn.Linear(dim_embed+dim_embedding_table+dim_embedding_table, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(1024, n_classes))
        # else:

        self.clf = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
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

        # Order attention
        x_cid = self.order_attention_cid(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        x_ccid = self.order_attention_ccid(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )
        # CVA Fusion for acc and pay
        x_cid_fused, _ = self.cva_acc_attn.forward(
            query=x_ccid[:, [-1], :],
            key=x_cid,
            value=x_cid,
            key_padding_mask=key_padding_mask_cid,
        )
        x_ccid_fused, _ = self.cva_pay_attn.forward(
            query=x_cid[:, [-1], :],
            key=x_ccid,
            value=x_ccid,
            key_padding_mask=key_padding_mask_ccid,
        )
        # CVA for acc and pay skip connection & norm
        x_cid_fused += x_cid[:, [-1], :]
        x_cid_fused = self.cva_acc_norm(x_cid_fused)  # still have the dummy dimension
        x_ccid_fused += x_ccid[:, [-1], :]
        x_ccid_fused = self.cva_acc_norm(x_ccid_fused)  # still have the dummy dimension
        # CVA dimension reduction
        x_order = self.cva_order_linear.forward(
            torch.cat([x_cid_fused[:, -1, :], x_ccid_fused[:, -1, :]], dim=-1)
        )

        # Feature attention on acc only
        x_feat = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)
        x_feat = self.cva_feat_linear.forward(x_feat)

        # CVA Fusion for order and feature
        # no need to use padding mask because input is length one dummy sequence
        x_order_fused, _ = self.cva_order_attn.forward(
            query=x_feat[:, None, :],
            key=x_order[:, None, :],
            value=x_order[:, None, :],
            key_padding_mask=None,
        )
        x_feature_fused, _ = self.cva_feat_attn.forward(
            query=x_order[:, None, :],
            key=x_feat[:, None, :],
            value=x_feat[:, None, :],
            key_padding_mask=None,
        )

        # CVA for order and feature skip connection & norm
        x_order_fused += x_order[:, None, :]
        x_order_fused = self.cva_order_norm(
            x_order_fused
        )  # still have the dummy dimension
        x_feature_fused += x_feat[:, None, :]
        x_feature_fused = self.cva_feat_norm(
            x_feature_fused
        )  # still have the dummy dimension

        ensemble = self.cva_feat_order_linear.forward(
            torch.cat([x_order_fused[:, 0, :], x_feature_fused[:, 0, :]], dim=-1)
        )

        # MLP TODO: commented out
        # if self.use_mlp:
        #     x_mlp = self.MLP(torch.cat([x_num[:,-1,:], x_engineered], dim=-1))
        #     x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        # if self.use_mlp:
        #     ensemble = torch.cat([x,x_last, x_mlp], dim=-1)
        # else:

        # ensemble = torch.cat([x, x_last], dim=-1)
        #  scores = self.fc(ensemble)
        scores = self.clf(ensemble)

        return scores, ensemble


class TwoSepViewUMSFusionOrderFeatureAttentionClassifier(torch.nn.Module):
    """Multilayer attention model"""

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        use_mlp=0,
        num_heads=1,
        dropout=0.1,
        n_layers_order=1,
        n_layers_feature=1,
        emb_tbl_use_bias=1,
        cva_mask=True,
    ):
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads

        # main blocks
        dim_embedding_table = dim_embed // 2

        self.dim_embedding_table = dim_embedding_table

        self.embedding_acc = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )
        self.embedding_pay = nn.Embedding(
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
            self.embedding_acc,
            self.num_heads,
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            False,
        )

        self.order_attention_ccid = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding_pay,
            self.num_heads,
            dropout,
            n_layers_order,
            emb_tbl_use_bias,
            False,
        )

        #         # whether to use mask for cva
        #         self.cva_mask = cva_mask
        #         # cva attn for acc, src should be acc, target should be payment
        #         self.cva_acc_attn = nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=1, batch_first=True)
        #         self.cva_acc_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        #         # cva attn for acc, src should be acc, target should be payment
        #         self.cva_pay_attn = nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=1, batch_first=True)
        #         self.cva_pay_norm = nn.LayerNorm(normalized_shape=self.dim_embed)
        #         # linear layer to reduce the dimension
        #         self.cva_linear = nn.Linear(in_features=2*self.dim_embed, out_features=self.dim_embed)

        self.embedding_engineered_acc = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )
        self.embedding_engineered_pay = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        self.feature_attention_acc = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding_acc,
            self.embedding_engineered_acc,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.feature_attention_pay = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding_pay,
            self.embedding_engineered_pay,
            self.num_heads,
            dropout,
            n_layers_feature,
            emb_tbl_use_bias,
        )

        self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)
        #         self.fc = torch.nn.Linear(dim_embed+dim_embedding_table, n_classes)
        #         self.clf = nn.Sequential(
        #             nn.LayerNorm(dim_embed+dim_embedding_table),
        # #                                          nn.Linear(dim_embed+dim_embedding_table, dim_embed),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(dropout),
        #                                          nn.Linear(dim_embed+dim_embedding_table, n_classes))
        self.layer_stack_feature = nn.ModuleList(
            [
                #             AttentionLayer(dim_embed, dim_attn_feedforward, num_heads, dropout=dropout)
                AttentionLayerPreNorm(
                    dim_embedding_table,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, dim_embedding_table)
            )
            if emb_tbl_use_bias
            else None
        )
        self.engineered_emb_tbl_bias = (
            nn.Parameter(torch.randn(n_engineered_num_features, dim_embedding_table))
            if emb_tbl_use_bias
            else None
        )

        # if self.use_mlp:
        #     self.MLP = MLPBlock(self.n_num_features+self.n_engineered_num_features, 1024, dim_embedding_table, 0.1)
        #     self.clf = nn.Sequential(
        #         #             nn.LayerNorm(dim_embed+dim_embedding_table),
        #         nn.Linear(dim_embed+dim_embedding_table+dim_embedding_table, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(1024, n_classes))
        # else:

        self.clf_acc = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

        self.clf_pay = nn.Sequential(
            # nn.LayerNorm(dim_embed+dim_embedding_table),
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
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

        # Order attention
        x_cid = self.order_attention_cid(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        x_ccid = self.order_attention_ccid(
            x_seq_cat_ccid,
            x_seq_num_ccid,
            time_seq_ccid,
            attn_mask,
            key_padding_mask_ccid,
        )

        #         # CVA Fusion
        #         if self.cva_mask:
        #             x_cid_fused, _ = self.cva_acc_attn.forward(query=x_ccid[:, [-1], :],
        #                                                        key=x_cid, value=x_cid,
        #                                                        key_padding_mask=key_padding_mask_cid)
        #             x_ccid_fused, _ = self.cva_pay_attn.forward(query=x_cid[:, [-1], :],
        #                                                        key=x_ccid, value=x_ccid,
        #                                                        key_padding_mask=key_padding_mask_ccid)
        #         else:
        #             x_cid_fused, _ = self.cva_acc_attn.forward(query=x_ccid[:, [-1], :],
        #                                                        key=x_cid, value=x_cid,
        #                                                        key_padding_mask=None)
        #             x_ccid_fused, _ = self.cva_pay_attn.forward(query=x_cid[:, [-1], :],
        #                                                        key=x_ccid, value=x_ccid,
        #                                                        key_padding_mask=None)

        #         # CVA skip conn & norm
        #         x_cid_fused += x_cid[:, [-1], :]
        #         x_cid_fused = self.cva_acc_norm(x_cid_fused)  # still have the dummy dimension
        #         x_ccid_fused += x_ccid[:, [-1], :]
        #         x_ccid_fused = self.cva_pay_norm(x_ccid_fused)  # still have the dummy dimension

        #         # CVA dimension reduction
        #         x = self.cva_linear.forward(torch.cat([x_cid_fused[:, -1, :], x_ccid_fused[:, -1, :]], dim=-1))

        # Feature attention on acc only
        x_last_acc = self.feature_attention_acc(
            x_seq_cat_cid, x_seq_num_cid, x_engineered
        )
        x_last_pay = self.feature_attention_pay(
            x_seq_cat_ccid, x_seq_num_ccid, x_engineered
        )

        # MLP TODO: commented out
        # if self.use_mlp:
        #     x_mlp = self.MLP(torch.cat([x_num[:,-1,:], x_engineered], dim=-1))
        #     x_mlp = self.layer_norm_engineered(x_mlp)

        # Ensemble embeddings
        # if self.use_mlp:
        #     ensemble = torch.cat([x,x_last, x_mlp], dim=-1)
        # else:

        #         print(x_cid.shape, x_last.shape)

        ensemble_cid = torch.cat([x_cid, x_last_acc], dim=-1)
        #  scores = self.fc(ensemble)
        scores_cid = self.clf_acc(ensemble_cid)

        ensemble_ccid = torch.cat([x_ccid, x_last_pay], dim=-1)
        #  scores = self.fc(ensemble)
        scores_ccid = self.clf_acc(ensemble_ccid)

        scores = torch.log(torch.exp(scores_cid) + torch.exp(scores_ccid))

        #         print(scores.shape)

        #         print(scores_cid.shape, scores_ccid.shape, scores.shape)
        #         print(scores_cid, scores_ccid, scores)

        return scores, ensemble_cid
