import torch

from proteinfoundation.nn.modules.pair_bias_attn import (
    MultiHeadBiasedAttentionADALN_MM,
    MultiHeadCrossAttentionADALN_MM,
)
from proteinfoundation.nn.modules.seq_transition_af3 import TransitionADALN


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_attn_pair_bias,
        use_qkln,
        dropout=0.0,
        expansion_factor=4,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            # logger.info("MHA and transition are residual, but with parallel track. Setting transition to non-residual.")
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasedAttentionADALN_MM(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )

        self.transition = TransitionADALN(dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor)

    def _apply_mha(self, x, pair_rep, cond, mask):
        x_attn = self.mhba(x, pair_rep, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


class MultiheadCrossAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token_a: Token dimension in sequence representation.
        dim_token_b: Token dimension in sequence representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token_a,
        dim_token_b,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        use_qkln,
        expansion_factor=4,
    ):
        super().__init__()
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadCrossAttentionADALN_MM(
            dim_token_a=dim_token_a,
            dim_token_b=dim_token_b,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )

        self.transition_a = TransitionADALN(dim=dim_token_a, dim_cond=dim_cond, expansion_factor=expansion_factor)

    def _apply_mha(self, a, b, cond, mask_a, mask_b):
        a_attn = self.mhba(a, b, cond, mask_a, mask_b)
        if self.residual_mha:
            a_attn = a_attn + a
        return a_attn * mask_a[..., None]

    def _apply_transition(self, x, cond, mask, transition):
        x_tr = transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, a, b, cond, mask_a, mask_b):
        """
        Args:
            a: Input sequence representation, shape [b, n, dim_token]
            b: Input atom representation, shape [b, na, dim_atom]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            atom_mask: binary mask, shape [b, na]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
            Updated atom representation, shape [b, na, dim_atom].
        """
        a = a * mask_a[..., None]
        b = b * mask_b[..., None]
        a = self._apply_mha(a, b, cond, mask_a, mask_b)
        a = self._apply_transition(a, cond, mask_a, self.transition_a)
        return a * mask_a[..., None]
