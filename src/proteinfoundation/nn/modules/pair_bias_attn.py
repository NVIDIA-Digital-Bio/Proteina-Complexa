# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm, AdaptiveOutputScale

# Large negative value for masking (not -inf to avoid gradient issues)
NEG_INF = -1e4

# Try to import cuequivariance for fused attention
try:
    from cuequivariance_torch import attention_pair_bias as cuet_attention_pair_bias

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False
    cuet_attention_pair_bias = None


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kawrgs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...") if exists(pair_feats) else 0
        q, k, v, g = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g))
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h)
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Tensor | None) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class FlashPairBiasAttention(nn.Module):
    """
    Pair bias attention using PyTorch's scaled_dot_product_attention for Flash Attention.

    Uses the flash attention kernel via F.scaled_dot_product_attention, which provides
    significant memory and speed improvements. The pair bias is passed through the
    attn_mask parameter.
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head

        # Combined projection for Q, K, V, and Gate
        self.node_norm = nn.LayerNorm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)

        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim), bias=bias)

        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with flash attention kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Boolean tensor of node adjacencies (b, n, n).

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        # Get target dtype for flash attention (handles autocast)
        target_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else node_feats.dtype

        node_feats = self.node_norm(node_feats)
        h = self.heads

        # Compute Q, K, V
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        # Reshape for attention: [b, n, (h d)] -> [b, h, n, d]
        q, k, v, g = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v, g))

        # Compute pair bias: [b, n, n, pair_dim] -> [b, h, n, n]
        if exists(pair_feats) and exists(self.to_bias):
            pair_feats = self.pair_norm(pair_feats)
            attn_bias = rearrange(self.to_bias(pair_feats), "b i j h -> b h i j")
            attn_bias = attn_bias.to(target_dtype)
        else:
            attn_bias = None

        # Apply mask to bias
        if exists(mask):
            # mask is [b, n, n] boolean
            mask_expanded = rearrange(mask, "b i j -> b () i j")  # [b, 1, n, n]
            if attn_bias is not None:
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)
            else:
                # Create mask-only bias
                attn_bias = torch.zeros(
                    node_feats.shape[0],
                    h,
                    node_feats.shape[1],
                    node_feats.shape[1],
                    device=node_feats.device,
                    dtype=target_dtype,
                )
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)

        # Flash attention via scaled_dot_product_attention
        # Note: attn_mask is added to attention scores before softmax
        o = F.scaled_dot_product_attention(
            q.contiguous().to(target_dtype),
            k.contiguous().to(target_dtype),
            v.contiguous().to(target_dtype),
            attn_mask=attn_bias.contiguous() if attn_bias is not None else None,
        )

        # Apply gating (matches PairBiasAttention)
        o_gated = torch.sigmoid(g) * o

        # Reshape and project out: [b, h, n, d] -> [b, n, (h d)]
        out = rearrange(o_gated, "b h n d -> b n (h d)", h=h)
        return self.to_out_node(out)


class CuEqPairBiasAttention(nn.Module):
    """
    Pair bias attention using cuEquivariance fused kernel.

    Uses cuequivariance_torch.attention_pair_bias for optimized computation.
    This is a drop-in replacement for PairBiasAttention that uses fused CUDA kernels.

    Note: The cuequivariance kernel handles:
    - Pair feature normalization (via w_ln_z, b_ln_z)
    - Pair bias projection (via w_proj_z)
    - Gate computation (via w_proj_g, b_proj_g)
    - Output projection (via w_proj_o, b_proj_o)
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if not CUET_AVAILABLE:
            raise ImportError(
                "cuequivariance_torch is required for CuEqPairBiasAttention. "
                "Install it or use attention_type='torch' or 'flash'."
            )

        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.node_norm = nn.LayerNorm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))

        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        if exists(pair_dim):
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
        else:
            self.pair_norm, self.to_bias = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with cuEquivariance fused kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Token mask of shape (b, n) - cuequivariance computes pair mask internally.

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        h = self.heads
        node_feats_normed = self.node_norm(node_feats)

        # Compute Q, K, V (matching original PairBiasAttention flow)
        q, k, v = self.to_qkv(node_feats_normed).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        # Reshape for cuequivariance: [b, n, (h d)] -> [b, h, n, d]
        # This matches the original commented code: "b ... (h d) -> b h ... d"
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v))

        # Use cuequivariance fused kernel
        # Note: cuet expects raw pair_feats (pre-normalization) and applies LayerNorm internally
        # Note: cuet expects 1D token mask [b, n] and computes pair mask internally
        # Returns tuple (output, z_proj) - we only need the output
        result = cuet_attention_pair_bias(
            s=node_feats_normed,  # Normalized node features
            q=q,
            k=k,
            v=v,
            z=pair_feats,  # Raw pair features (cuet applies pair_norm internally)
            mask=mask,  # [b, n] token mask
            num_heads=self.heads,
            w_proj_z=self.to_bias.weight if self.to_bias is not None else None,
            w_proj_g=self.to_g.weight,
            w_proj_o=self.to_out_node.weight,
            w_ln_z=self.pair_norm.weight if self.pair_norm is not None else None,
            b_ln_z=self.pair_norm.bias if self.pair_norm is not None else None,
            b_proj_z=None,  # to_bias has no bias
            b_proj_g=self.to_g.bias if self.to_g.bias is not None else None,
            b_proj_o=(self.to_out_node.bias if self.to_out_node.bias is not None else None),
        )
        # cuet returns (output, z_proj) tuple - extract the output tensor
        out = result[0] if isinstance(result, tuple) else result
        return out


class CuEqMultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention using cuEquivariance fused kernel.

    Uses CuEqPairBiasAttention for optimized computation.
    """

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = CuEqPairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        # Note: CuEqPairBiasAttention expects 1D token mask [b, n],
        # cuequivariance kernel computes pair mask internally
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=mask)  # Pass 1D mask
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class FlashMultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention using Flash Attention kernel.

    Uses FlashPairBiasAttention which leverages F.scaled_dot_product_attention
    for improved memory efficiency and speed.
    """

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = FlashPairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class CrossAttention(nn.Module):
    """
    Symmetric cross attention with gating.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        dim_head_a: int,
        dim_head_b: int,
        heads: int,
        bias: bool,
        qkln: bool,
        **kwargs,
    ):
        super().__init__()
        inner_dim_a = dim_head_a * heads
        inner_dim_b = dim_head_b * heads
        self.dim_a, self.dim_b = dim_a, dim_b
        self.heads, self.scale_b = (
            heads,
            dim_head_b**-0.5,
        )
        self.to_q_a = nn.Linear(dim_a, inner_dim_b, bias=bias)
        self.to_g_a = nn.Linear(dim_a, inner_dim_a)
        self.to_v_b = nn.Linear(dim_b, inner_dim_a, bias=bias)
        self.to_k_b = nn.Linear(dim_b, inner_dim_b, bias=bias)

        self.to_out_node_a = nn.Linear(inner_dim_a, dim_a)
        self.node_norm_a = nn.LayerNorm(dim_a)
        self.node_norm_b = nn.LayerNorm(dim_b)
        self.q_layer_norm_a = nn.LayerNorm(inner_dim_b) if qkln else nn.Identity()
        self.k_layer_norm_b = nn.LayerNorm(inner_dim_b) if qkln else nn.Identity()

    def forward(
        self,
        feat_a: Tensor,
        feat_b: Tensor,
        mask_a_b: Tensor | None,
    ) -> Tensor:
        """Multi-head scalar attention Layer

        :param feat_a: scalar features of shape (b,na,d_a)
        :param feat_b: scalar features of shape (b,nb,d_b)
        :param mask_a_b: boolean tensor of node adjacencies (b, na, nb)
        :return:
            updated scalar features of shape (b,na,d_a)
        """
        feat_a, feat_b, h = (
            self.node_norm_a(feat_a),
            self.node_norm_b(feat_b),
            self.heads,
        )
        q_a = self.to_q_a(feat_a)
        q_a = self.q_layer_norm_a(q_a)
        k_b = self.to_k_b(feat_b)
        k_b = self.k_layer_norm_b(k_b)
        v_b = self.to_v_b(feat_b)
        g_a = self.to_g_a(feat_a)
        q_a, k_b, v_b, g_a = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h),
            (q_a, k_b, v_b, g_a),
        )
        attn_feats_a = self._attn(q_a, k_b, v_b, self.scale_b, mask_a_b)
        attn_feats_a = rearrange(torch.sigmoid(g_a) * attn_feats_a, "b h n d -> b n (h d)", h=h)
        return self.to_out_node_a(attn_feats_a)

    def _attn(self, q, k, v, scale, mask: Tensor | None) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


def get_multihead_attention_adaln(attention_type: str = "naive"):
    """Factory function to get the appropriate MultiHeadBiasedAttentionADALN_MM class.

    Args:
        attention_type: One of 'naive', 'flash', or 'cuequivariance'.
            - 'naive': Standard PyTorch implementation (PairBiasAttention)
            - 'flash': Flash Attention via scaled_dot_product_attention
            - 'cuequivariance': Fused cuEquivariance kernel

    Returns:
        The appropriate attention class.

    Raises:
        ValueError: If an unknown attention_type is provided.
        ImportError: If 'cuequivariance' is requested but not installed.
    """
    if attention_type == "naive":
        return MultiHeadBiasedAttentionADALN_MM
    elif attention_type == "flash":
        return FlashMultiHeadBiasedAttentionADALN_MM
    elif attention_type == "cuequivariance":
        if not CUET_AVAILABLE:
            raise ImportError(
                "cuequivariance_torch is required for attention_type='cuequivariance'. "
                "Install it or use attention_type='naive' or 'flash'."
            )
        return CuEqMultiHeadBiasedAttentionADALN_MM
    else:
        raise ValueError(
            f"Unknown attention_type: {attention_type}. Expected one of: 'naive', 'flash', 'cuequivariance'."
        )


class MultiHeadCrossAttentionADALN_MM(torch.nn.Module):
    """Multi-head cross-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token_a, dim_token_b, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head_a = int(dim_token_a // nheads)
        dim_head_b = int(dim_token_b // nheads)
        self.adaln_a = AdaptiveLayerNorm(dim=dim_token_a, dim_cond=dim_cond)
        self.ln_b = torch.nn.LayerNorm(dim_token_b)
        self.mha = CrossAttention(
            dim_a=dim_token_a,
            dim_b=dim_token_b,
            dim_head_a=dim_head_a,
            dim_head_b=dim_head_b,
            heads=nheads,
            bias=True,
            qkln=use_qkln,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token_a, dim_cond=dim_cond)

    def forward(self, a, b, cond, mask_a, mask_b):
        """
        Args:
            a: Input sequence representation, shape [b, na, dim_token_a]
            b: Input atom representation, shape [b, nb, dim_token_b]
            cond: conditioning variables, shape [b, na, dim_cond]
            mask_a: binary mask, shape [b, na]
            mask_b: binary mask, shape [b, nb]

        Returns:
            Updated sequence representation, shape [b, na, dim_token_a].
        """
        mask_a_b = mask_a[:, :, None] * mask_b[:, None, :]  # [b, na, nb]

        a = self.adaln_a(a, cond, mask_a)
        b = self.ln_b(b) * mask_b[..., None]
        a = self.mha(a, b, mask_a_b)
        a = self.scale_output(a, cond, mask_a)
        return a * mask_a[..., None]
