# class MultiHeadAttention(torch.nn.Module):
#     """Typical multi-head self-attention attention using pytorch's module."""

#     def __init__(self, dim_token, nheads, dropout=0.0):
#         super().__init__()

#         self.to_q = torch.nn.Linear(dim_token, dim_token)
#         self.to_kv = torch.nn.Linear(dim_token, 2 * dim_token, bias=False)

#         self.mha = torch.nn.MultiheadAttention(
#             embed_dim=dim_token,
#             num_heads=nheads,
#             dropout=dropout,
#             batch_first=True,
#         )

#     def forward(self, x, mask):
#         """
#         Args:
#             x: Input sequence, shape [b, n, dim_token]
#             mask: binary mask, shape [b, n]

#         Returns:
#             Updated sequence, shape [b, n, dim_token]
#         """
#         query = self.to_q(x)  # [b, n, dim_token]
#         key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, dim_token]
#         return (
#             self.mha(
#                 query=query,
#                 key=key,
#                 value=value,
#                 key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
#                 need_weights=False,
#                 is_causal=False,
#             )[0]
#             * mask[..., None]
#         )  # [b, n, dim_token]


# class MultiHeadBiasedAttention(torch.nn.Module):
#     """Multi-head self-attention with pair bias, based on openfold."""

#     def __init__(self, dim_token, dim_pair, nheads, dropout=0.0):
#         super().__init__()

#         self.row_attn_pair_bias = MSARowAttentionWithPairBias(
#             c_m=dim_token,
#             c_z=dim_pair,
#             c_hidden=int(dim_token // nheads),  # Per head dimension
#             no_heads=nheads,
#         )

#     def forward(self, x, pair_rep, mask):
#         """
#         Args:
#             x: Input sequence, shape [b, n, dim_token]
#             pair_rep: Pair representation, shape [b, n, n, dim_pair]
#             mask: Binary mask, shape [b, n]

#         Returns:
#             Updated sequence representation, shape [b, n, dim_token]
#         """
#         # Add extra dimension for MSA, unused here but required by openfold
#         x = einops.rearrange(x, "b n d -> b () n d")  # [b, 1, n, dim_token]
#         mask = einops.rearrange(mask, "b n -> b () n") * 1.0  # float [b, 1, n]
#         x = self.row_attn_pair_bias(x, pair_rep, mask)  # [b, 1, n, dim_token]
#         x = x * mask[..., None]
#         x = einops.rearrange(
#             x, "b () n c -> b n c"
#         )  # Remove extra dimension [b, n, dim_token]
#         return x


# class MultiHeadAttentionADALN(torch.nn.Module):
#     """Typical multi-head self-attention with adaptive layer norm applied to input
#     and adaptive scaling applied to output."""

#     def __init__(self, dim_token, nheads, dim_cond, dropout=0.0):
#         super().__init__()
#         self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
#         self.mha = MultiHeadAttention(
#             dim_token=dim_token, nheads=nheads, dropout=dropout
#         )
#         self.scale_output = AdaptiveOutputScale(
#             dim=dim_token, dim_cond=dim_cond
#         )

#     def forward(self, x, cond, mask):
#         """
#         Args:
#             x: Input sequence representation, shape [b, n, dim_token]
#             cond: Conditioning variables, shape [b, n, dim_cond]
#             mask: Binary mask, shape [b, n]

#         Returns:
#             Updated sequence representation, shape [b, n, dim_token].
#         """
#         x = self.adaln(x, cond, mask)
#         x = self.mha(x, mask)
#         x = self.scale_output(x, cond, mask)
#         return x * mask[..., None]


# class MultiHeadBiasedAttentionADALN(torch.nn.Module):
#     """Pair biased multi-head self-attention with adaptive layer norm applied to input
#     and adaptive scaling applied to output."""

#     def __init__(self, dim_token, dim_pair, nheads, dim_cond, dropout=0.0):
#         super().__init__()
#         self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
#         self.mha = MultiHeadBiasedAttention(
#             dim_token=dim_token, dim_pair=dim_pair, nheads=nheads, dropout=dropout
#         )
#         self.scale_output = AdaptiveOutputScale(
#             dim=dim_token, dim_cond=dim_cond
#         )

#     def forward(self, x, pair_rep, cond, mask):
#         """
#         Args:
#             x: Input sequence representation, shape [b, n, dim_token]
#             cond: Conditioning variables, shape [b, n, dim_cond]
#             pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
#             mask: Binary mask, shape [b, n]

#         Returns:
#             Updated sequence representation, shape [b, n, dim_token].
#         """
#         x = self.adaln(x, cond, mask)
#         x = self.mha(x, pair_rep, mask)
#         x = self.scale_output(x, cond, mask)
#         return x * mask[..., None]
