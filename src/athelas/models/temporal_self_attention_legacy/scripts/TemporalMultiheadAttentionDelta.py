import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import _VF
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Callable, List, Optional, Tuple

from torch.overrides import (
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
    handle_torch_function,
)


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TimeEncode, self).__init__()

        self.time_dim = time_dim

        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.bias[1:]))
        out1 = F.linear(tt, self.weight[0:1, :], self.bias[0:1])
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(2)
        t = t.permute(1, 0, 2)

        return t


class TemporalMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(TemporalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        #         self.weight_time = nn.Parameter(torch.empty((time_dim, 1)))
        #         self.bias_time = nn.Parameter(torch.empty(time_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        #             init.kaiming_uniform_(self.weight_time, a=math.sqrt(5))
        #             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_time)
        #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #             init.uniform_(self.bias_time, -bound, bound)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        #             init.kaiming_uniform_(self.weight_time, a=math.sqrt(5))
        #             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_time)
        #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #             init.uniform_(self.bias_time, -bound, bound)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        time: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shapes for inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
              source sequence length.

              If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
              length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
              the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return temporal_multi_head_attention_forward(
                query,
                key,
                value,
                time,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return temporal_multi_head_attention_forward(
                query,
                key,
                value,
                time,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )


#
# multihead attention
#

# def _in_projection_packed(
#     q: Tensor,
#     k: Tensor,
#     v: Tensor,
#     w: Tensor,
#     b: Optional[Tensor] = None,
# ) -> List[Tensor]:
#     r"""
#     Performs the in-projection step of the attention operation, using packed weights.
#     Output is a triple containing projection tensors for query, key and value.
#     Args:
#         q, k, v: query, key and value tensors to be projected. For self-attention,
#             these are typically the same tensor; for encoder-decoder attention,
#             k and v are typically the same tensor. (We take advantage of these
#             identities for performance if they are present.) Regardless, q, k and v
#             must share a common embedding dimension; otherwise their shapes may vary.
#         w: projection weights for q, k and v, packed into a single tensor. Weights
#             are packed along dimension 0, in q, k, v order.
#         b: optional projection biases for q, k and v, packed into a single tensor
#             in q, k, v order.
#     Shape:
#         Inputs:
#         - q: :math:`(..., E)` where E is the embedding dimension
#         - k: :math:`(..., E)` where E is the embedding dimension
#         - v: :math:`(..., E)` where E is the embedding dimension
#         - w: :math:`(E * 3, E)` where E is the embedding dimension
#         - b: :math:`E * 3` where E is the embedding dimension
#         Output:
#         - in output list :math:`[q', k', v']`, each output tensor will have the
#             same shape as the corresponding input tensor.
#     """
#     E = q.size(-1)
#     if k is v:
#         if q is k:
#             # self-attention
#             return linear(q, w, b).chunk(3, dim=-1)
#         else:
#             # encoder-decoder attention
#             w_q, w_kv = w.split([E, E * 2])
#             if b is None:
#                 b_q = b_kv = None
#             else:
#                 b_q, b_kv = b.split([E, E * 2])
#             return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
#     else:
#         w_q, w_k, w_v = w.chunk(3)
#         if b is None:
#             b_q = b_k = b_v = None
#         else:
#             b_q, b_k, b_v = b.chunk(3)
#         return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# def _in_projection(
#     q: Tensor,
#     k: Tensor,
#     v: Tensor,
#     w_q: Tensor,
#     w_k: Tensor,
#     w_v: Tensor,
#     b_q: Optional[Tensor] = None,
#     b_k: Optional[Tensor] = None,
#     b_v: Optional[Tensor] = None,
# ) -> Tuple[Tensor, Tensor, Tensor]:
#     r"""
#     Performs the in-projection step of the attention operation. This is simply
#     a triple of linear projections, with shape constraints on the weights which
#     ensure embedding dimension uniformity in the projected outputs.
#     Output is a triple containing projection tensors for query, key and value.
#     Args:
#         q, k, v: query, key and value tensors to be projected.
#         w_q, w_k, w_v: weights for q, k and v, respectively.
#         b_q, b_k, b_v: optional biases for q, k and v, respectively.
#     Shape:
#         Inputs:
#         - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
#             number of leading dimensions.
#         - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
#             number of leading dimensions.
#         - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
#             number of leading dimensions.
#         - w_q: :math:`(Eq, Eq)`
#         - w_k: :math:`(Eq, Ek)`
#         - w_v: :math:`(Eq, Ev)`
#         - b_q: :math:`(Eq)`
#         - b_k: :math:`(Eq)`
#         - b_v: :math:`(Eq)`
#         Output: in output triple :math:`(q', k', v')`,
#          - q': :math:`[Qdims..., Eq]`
#          - k': :math:`[Kdims..., Eq]`
#          - v': :math:`[Vdims..., Eq]`
#     """
#     Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
#     assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
#     assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
#     assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
#     assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
#     assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
#     assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
#     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

# def _scaled_dot_product_attention(
#     q: Tensor,
#     k: Tensor,
#     v: Tensor,
#     attn_mask: Optional[Tensor] = None,
#     dropout_p: float = 0.0,
# ) -> Tuple[Tensor, Tensor]:
#     r"""
#     Computes scaled dot product attention on query, key and value tensors, using
#     an optional attention mask if passed, and applying dropout if a probability
#     greater than 0.0 is specified.
#     Returns a tensor pair containing attended values and attention weights.
#     Args:
#         q, k, v: query, key and value tensors. See Shape section for shape details.
#         attn_mask: optional tensor containing mask values to be added to calculated
#             attention. May be 2D or 3D; see Shape section for details.
#         dropout_p: dropout probability. If greater than 0.0, dropout is applied.
#     Shape:
#         - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
#             and E is embedding dimension.
#         - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
#             and E is embedding dimension.
#         - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
#             and E is embedding dimension.
#         - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
#             shape :math:`(Nt, Ns)`.
#         - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
#             have shape :math:`(B, Nt, Ns)`
#     """
#     B, Nt, E = q.shape
#     q = q / math.sqrt(E)
#     # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
#     attn = torch.bmm(q, k.transpose(-2, -1))
#     if attn_mask is not None:
#         attn += attn_mask
#     attn = softmax(attn, dim=-1)
#     if dropout_p > 0.0:
#         attn = dropout(attn, p=dropout_p)
#     # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
#     output = torch.bmm(attn, v)
#     return output, attn

# def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
#                      key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
#     # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
#     # and returns if the input is batched or not.
#     # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

#     # Shape check.
#     if query.dim() == 3:
#         # Batched Inputs
#         is_batched = True
#         assert key.dim() == 3 and value.dim() == 3, \
#             ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
#              f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
#         if key_padding_mask is not None:
#             assert key_padding_mask.dim() == 2, \
#                 ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
#                  f" but found {key_padding_mask.dim()}-D tensor instead")
#         if attn_mask is not None:
#             assert attn_mask.dim() in (2, 3), \
#                 ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
#                  f" but found {attn_mask.dim()}-D tensor instead")
#     elif query.dim() == 2:
#         # Unbatched Inputs
#         is_batched = False
#         assert key.dim() == 2 and value.dim() == 2, \
#             ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
#              f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

#         if key_padding_mask is not None:
#             assert key_padding_mask.dim() == 1, \
#                 ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
#                  f" but found {key_padding_mask.dim()}-D tensor instead")

#         if attn_mask is not None:
#             assert attn_mask.dim() in (2, 3), \
#                 ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
#                  f" but found {attn_mask.dim()}-D tensor instead")
#             if attn_mask.dim() == 3:
#                 expected_shape = (num_heads, query.shape[0], key.shape[0])
#                 assert attn_mask.shape == expected_shape, \
#                     (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
#     else:
#         raise AssertionError(
#             f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

#     return is_batched

# def multi_head_attention_forward(
#     query: Tensor,
#     key: Tensor,
#     value: Tensor,
#     time: Tensor,
#     embed_dim_to_check: int,
#     num_heads: int,
#     in_proj_weight: Tensor,
#     weight_time: Tensor,
#     bias_time: Tensor,
#     in_proj_bias: Optional[Tensor],
#     bias_k: Optional[Tensor],
#     bias_v: Optional[Tensor],
#     add_zero_attn: bool,
#     dropout_p: float,
#     out_proj_weight: Tensor,
#     out_proj_bias: Optional[Tensor],
#     training: bool = True,
#     key_padding_mask: Optional[Tensor] = None,
#     need_weights: bool = True,
#     attn_mask: Optional[Tensor] = None,
#     use_separate_proj_weight: bool = False,
#     q_proj_weight: Optional[Tensor] = None,
#     k_proj_weight: Optional[Tensor] = None,
#     v_proj_weight: Optional[Tensor] = None,
#     static_k: Optional[Tensor] = None,
#     static_v: Optional[Tensor] = None,
#     average_attn_weights: bool = True,
# ) -> Tuple[Tensor, Optional[Tensor]]:
#     r"""
#     Args:
#         query, key, value: map a query and a set of key-value pairs to an output.
#             See "Attention Is All You Need" for more details.
#         embed_dim_to_check: total dimension of the model.
#         num_heads: parallel attention heads.
#         in_proj_weight, in_proj_bias: input projection weight and bias.
#         bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
#         add_zero_attn: add a new batch of zeros to the key and
#                        value sequences at dim=1.
#         dropout_p: probability of an element to be zeroed.
#         out_proj_weight, out_proj_bias: the output projection weight and bias.
#         training: apply dropout if is ``True``.
#         key_padding_mask: if provided, specified padding elements in the key will
#             be ignored by the attention. This is an binary mask. When the value is True,
#             the corresponding value on the attention layer will be filled with -inf.
#         need_weights: output attn_output_weights.
#         attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
#             the batches while a 3D mask allows to specify a different mask for the entries of each batch.
#         use_separate_proj_weight: the function accept the proj. weights for query, key,
#             and value in different forms. If false, in_proj_weight will be used, which is
#             a combination of q_proj_weight, k_proj_weight, v_proj_weight.
#         q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
#         static_k, static_v: static key and value used for attention operators.
#         average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
#             Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
#             when ``need_weights=True.``. Default: True
#     Shape:
#         Inputs:
#         - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
#           If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
#           will be unchanged. If a BoolTensor is provided, the positions with the
#           value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
#         - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
#           3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
#           S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
#           positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#           while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#           are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#           is provided, it will be added to the attention weight.
#         - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#           N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
#         - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#           N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
#         Outputs:
#         - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
#           E is the embedding dimension.
#         - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
#           attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
#           :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
#           :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
#           head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
#     """

#     tt = time.permute(1, 0, 2)
#     tt = tt.unsqueeze(-1)
#     out2 = torch.sin(F.linear(tt, weight_time[1:,:], bias_time[1:]))
#     out1 = F.linear(tt, weight_time[0:1,:], bias_time[0:1])
#     t = torch.cat([out1, out2], -1)
#     t = t.squeeze(2)
#     t = t.permute(1, 0, 2)

#     query = query+t
#     key = key+t
#     value = value + t

#     is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

#     # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
#     # is batched, run the computation and before returning squeeze the
#     # batch dimension so that the output doesn't carry this temporary batch dimension.
#     if not is_batched:
#         # unsqueeze if the input is unbatched
#         query = query.unsqueeze(1)
#         key = key.unsqueeze(1)
#         value = value.unsqueeze(1)
#         if key_padding_mask is not None:
#             key_padding_mask = key_padding_mask.unsqueeze(0)

#     # set up shape vars
#     tgt_len, bsz, embed_dim = query.shape
#     src_len, _, _ = key.shape
#     assert embed_dim == embed_dim_to_check, \
#         f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
#     if isinstance(embed_dim, torch.Tensor):
#         # embed_dim can be a tensor when JIT tracing
#         head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
#     else:
#         head_dim = embed_dim // num_heads
#     assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
#     if use_separate_proj_weight:
#         # allow MHA to have different embedding dimensions when separate projection weights are used
#         assert key.shape[:2] == value.shape[:2], \
#             f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
#     else:
#         assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

#     #
#     # compute in-projection
#     #
#     if not use_separate_proj_weight:
#         q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
#     else:
#         assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
#         assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
#         assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
#         if in_proj_bias is None:
#             b_q = b_k = b_v = None
#         else:
#             b_q, b_k, b_v = in_proj_bias.chunk(3)
#         q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

#     # prep attention mask
#     if attn_mask is not None:
#         if attn_mask.dtype == torch.uint8:
#             warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
#             attn_mask = attn_mask.to(torch.bool)
#         else:
#             assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
#                 f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
#         # ensure attn_mask's dim is 3
#         if attn_mask.dim() == 2:
#             correct_2d_size = (tgt_len, src_len)
#             if attn_mask.shape != correct_2d_size:
#                 raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
#             attn_mask = attn_mask.unsqueeze(0)
#         elif attn_mask.dim() == 3:
#             correct_3d_size = (bsz * num_heads, tgt_len, src_len)
#             if attn_mask.shape != correct_3d_size:
#                 raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
#         else:
#             raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

#     # prep key padding mask
#     if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
#         warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
#         key_padding_mask = key_padding_mask.to(torch.bool)

#     # add bias along batch dimension (currently second)
#     if bias_k is not None and bias_v is not None:
#         assert static_k is None, "bias cannot be added to static key."
#         assert static_v is None, "bias cannot be added to static value."
#         k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
#         v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
#         if attn_mask is not None:
#             attn_mask = pad(attn_mask, (0, 1))
#         if key_padding_mask is not None:
#             key_padding_mask = pad(key_padding_mask, (0, 1))
#     else:
#         assert bias_k is None
#         assert bias_v is None

#     #
#     # reshape q, k, v for multihead attention and make em batch first
#     #
#     q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#     if static_k is None:
#         k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
#     else:
#         # TODO finish disentangling control flow so we don't do in-projections when statics are passed
#         assert static_k.size(0) == bsz * num_heads, \
#             f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
#         assert static_k.size(2) == head_dim, \
#             f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
#         k = static_k
#     if static_v is None:
#         v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
#     else:
#         # TODO finish disentangling control flow so we don't do in-projections when statics are passed
#         assert static_v.size(0) == bsz * num_heads, \
#             f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
#         assert static_v.size(2) == head_dim, \
#             f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
#         v = static_v

#     # add zero attention along batch dimension (now first)
#     if add_zero_attn:
#         zero_attn_shape = (bsz * num_heads, 1, head_dim)
#         k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
#         v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
#         if attn_mask is not None:
#             attn_mask = pad(attn_mask, (0, 1))
#         if key_padding_mask is not None:
#             key_padding_mask = pad(key_padding_mask, (0, 1))

#     # update source sequence length after adjustments
#     src_len = k.size(1)

#     # merge key padding and attention masks
#     if key_padding_mask is not None:
#         assert key_padding_mask.shape == (bsz, src_len), \
#             f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
#         key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
#             expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
#         if attn_mask is None:
#             attn_mask = key_padding_mask
#         elif attn_mask.dtype == torch.bool:
#             attn_mask = attn_mask.logical_or(key_padding_mask)
#         else:
#             attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

#     # convert mask to float
#     if attn_mask is not None and attn_mask.dtype == torch.bool:
#         new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
#         new_attn_mask.masked_fill_(attn_mask, float("-inf"))
#         attn_mask = new_attn_mask

#     # adjust dropout probability
#     if not training:
#         dropout_p = 0.0


#     #
#     # (deep breath) calculate attention and out projection
#     #
#     attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
#     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
#     attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
#     attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

#     if need_weights:
#         # optionally average attention weights over heads
#         attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#         if average_attn_weights:
#             attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

#         if not is_batched:
#             # squeeze the output if input was unbatched
#             attn_output = attn_output.squeeze(1)
#             attn_output_weights = attn_output_weights.squeeze(0)
#         return attn_output, attn_output_weights
#     else:
#         if not is_batched:
#             # squeeze the output if input was unbatched
#             attn_output = attn_output.squeeze(1)
#         return attn_output, None


def temporal_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    time: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    #     weight_time: Tensor,
    #     bias_time: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    #     dim_time_embed = weight_time.shape[0]
    #     layer_norm_time = nn.LayerNorm(dim_time_embed)

    #     tt = time.permute(1, 0, 2)
    #     tt = tt.unsqueeze(-1)
    #     out2 = torch.sin(F.linear(tt, weight_time[1:,:], bias_time[1:]))
    #     out1 = F.linear(tt, weight_time[0:1,:], bias_time[0:1])
    #     t = torch.cat([out1, out2], -1)
    #     t = t.squeeze(2)
    # #     t = t.permute(1, 0, 2)
    # #     print(t.shape)
    # #     t = layer_norm(t,(dim_time_embed,))
    # #     print(t.shape)
    # #     alpha_time = F.cosine_similarity(t[..., None, :, :], t[..., :, None, :], dim=-1)
    # #     print(t)
    # #     alpha_time = torch.cdist(t,t)
    # #     alpha_time = torch.exp(alpha_time/1000)
    # #     print('alpha',alpha_time.shape)

    # Plain
    tt = time.permute(1, 0, 2)
    #     print(weight_time.shape)
    alpha_time = tt - tt.view([tt.shape[0], tt.shape[2], tt.shape[1]])
    #     print(tt[0,:,:])
    #     for iii in range(51):
    #         print('delta_time',iii, alpha_time[0,iii,:])
    #     alpha_time = weight_time[0,0]*alpha_time + bias_time[0]
    #     alpha_time = torch.exp(alpha_time/alpha_time.max())
    #     alpha_time = torch.exp(alpha_time/1000000)
    alpha_time = torch.exp(alpha_time / 1000000)

    #     print('alpha',alpha_time)

    tgt_len, bsz, embed_dim = query.size()

    #     query = layer_norm(query+t,(embed_dim,))
    #     key = layer_norm(key+t,(embed_dim,))
    #     value = layer_norm(value + t,(embed_dim,))

    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (
            key is value or torch.equal(key, value)
        ):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), (
            "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            )
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None
    #     print('q before: ',q.shape)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    #     print('q after: ', q.shape)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    #     print(q.shape,k.shape)
    #     attn_output_weights = torch.bmm(q, k.transpose(1, 2))+torch.bmm(q, t.transpose(1, 2))+torch.bmm(k, t.transpose(1, 2))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

    #     print('alpha_time_0',alpha_time[0,:,:])
    #     print(alpha_time.max(),alpha_time.min())
    #     print('alpha_time',alpha_time[0,:,-1])
    #     print('alpha_time1',alpha_time[0,-1,:])
    #     print('alpha_time2',alpha_time[0,-2,:])
    #     for iii in range(51):
    #         print('alpha_time',iii, alpha_time[0,iii,:])
    #     print('attn_output_weights1',attn_output_weights[0,:,-1])
    #     print('attn_output_weights11',attn_output_weights[0,-1,:])
    #     print(alpha_time.shape)
    #     print(alpha_time)
    #     alpha_time = torch.repeat_interleave(alpha_time, num_heads, dim=0)
    alpha_time = (
        alpha_time.unsqueeze(1)
        .expand(-1, num_heads, -1, -1)
        .contiguous()
        .view(bsz * num_heads, tgt_len, src_len)
    )
    #     print(alpha_time.shape)
    #     print(alpha_time)
    #     print(attn_output_weights.shape)
    #     print(v.shape)
    attn_output_weights = alpha_time * attn_output_weights
    #     print('attn_output_weights2',attn_output_weights[0,:,-1])
    #     print('attn_output_weights22',attn_output_weights[0,-1,:])
    #     print('attention weight',attn_output_weights.shape)
    #     print(attn_output_weights[0,:,-1].shape)
    #     for iii in range(51):
    #         print('attn_output_weights_softmaxt',iii, attn_output_weights[0,iii,:])
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    #     print('attn_output_weights3',attn_output_weights[0,:,-1])
    #     print('attn_output_weights4',attn_output_weights[0,-1,:])
    #     print('attn_output_weights5',attn_output_weights[0,-2,:])
    #     for iii in range(51):
    #         print('attn_output_weights_softmaxt',iii, attn_output_weights[0,iii,:])
    #     print('attn_output_weights_sum',torch.sum(attn_output_weights,dim=-1))
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)


def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[int] = None,
) -> Tensor:
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


# Activation functions
def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, but got {}".format(p)
        )
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )


def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            layer_norm,
            (input,),
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    return torch.layer_norm(
        input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled
    )
